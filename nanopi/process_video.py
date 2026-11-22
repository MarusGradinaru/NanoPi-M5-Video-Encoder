#!/usr/bin/env python3
"""Detect target-class video segments by sampling frames with the RK3576 NPU."""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from rknn_batch import (
    batch_size_from_model_path,
    resolve_feature_model_path,
    validate_batch_size,
)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
FEATURE_COUNT = 2048
SOC_THERMAL_ROOT = Path("/sys/class/thermal")
SOC_THERMAL_TYPES = {
    "little-core-thermal",
    "bigcore-thermal",
    "gpu-thermal",
    "ddr-thermal",
    "npu-thermal",
}
BATCH_SIZE = 4
# Target interval; actual progress updates are checked after completed inference batches.
PROGRESS_UPDATE_INTERVAL_SECONDS = 1.0
VALIDATION_WINDOW = 5
VALIDATION_VOTES = 4
VALIDATION_STREAK = 2
MIN_SEGMENT_TIME_MS = 1000
MIN_SEGMENT_RUN = 3
SHOWINFO_TIMESTAMP_PATTERN = re.compile(
    r"\bpts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)

DOSATTRIB_XATTR = "user.DOSATTRIB"
DOSATTRIB_HEADER = bytes.fromhex("0000050005000000")
DOSATTRIB_HIDDEN = 0x02
DOSATTRIB_ARCHIVE = 0x20

# Prefer the tested Rockchip-enabled FFmpeg build when it is installed.
# Processing requires an FFmpeg build with MPP and RGA support.
ROCKCHIP_FFMPEG = Path("/opt/rkmpp-stack/bin/ffmpeg")
ROCKCHIP_FFPROBE = Path("/opt/rkmpp-stack/bin/ffprobe")
FFMPEG_BINARY = str(ROCKCHIP_FFMPEG) if ROCKCHIP_FFMPEG.is_file() else "ffmpeg"
FFPROBE_BINARY = str(ROCKCHIP_FFPROBE) if ROCKCHIP_FFPROBE.is_file() else "ffprobe"


def discover_soc_thermal_sensors(thermal_root: Path = SOC_THERMAL_ROOT) -> list[Path]:
    """Find the temperature files for the five monitored RK3576 SoC thermal zones."""
    sensors: list[Path] = []

    for type_path in sorted(thermal_root.glob("thermal_zone*/type")):
        try:
            sensor_type = type_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            continue

        if sensor_type in SOC_THERMAL_TYPES:
            sensors.append(type_path.with_name("temp"))

    return sensors


def read_hottest_soc_temperature(sensors: list[Path]) -> int | None:
    """Return the hottest readable SoC thermal-zone value in millidegrees Celsius."""
    readings: list[int] = []

    for temperature_path in sensors:
        try:
            value = int(temperature_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError, UnicodeError):
            continue
        readings.append(value)

    return max(readings) if readings else None


def format_soc_temperature(millidegrees: int | None) -> str:
    """Format a thermal-zone reading as whole degrees Celsius."""
    return str(int(millidegrees / 1000)) if millidegrees is not None else "--"


class ConsoleTee:
    """Mirror regular console output to a log while allowing live progress to bypass it."""

    def __init__(self, terminal, log_file) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        written = self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text) if written is None else written

    def write_terminal_only(self, text: str) -> int:
        """Write transient progress text to the terminal without recording it."""
        written = self.terminal.write(text)
        return len(text) if written is None else written

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


_ACTIVE_OUTPUT_LOG = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None


def close_output_log() -> None:
    """Restore console streams and close the optional output log."""
    global _ACTIVE_OUTPUT_LOG, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if _ACTIVE_OUTPUT_LOG is None:
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, OSError, ValueError):
        pass

    if _ORIGINAL_STDOUT is not None:
        sys.stdout = _ORIGINAL_STDOUT
    if _ORIGINAL_STDERR is not None:
        sys.stderr = _ORIGINAL_STDERR

    try:
        _ACTIVE_OUTPUT_LOG.flush()
        _ACTIVE_OUTPUT_LOG.close()
    finally:
        _ACTIVE_OUTPUT_LOG = None
        _ORIGINAL_STDOUT = None
        _ORIGINAL_STDERR = None


def install_output_log(path: Path) -> None:
    """Capture displayed stdout/stderr, excluding transient live progress updates."""
    global _ACTIVE_OUTPUT_LOG, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    close_output_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("w", encoding="utf-8", errors="replace", buffering=1)
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    _ACTIVE_OUTPUT_LOG = log_file
    sys.stdout = ConsoleTee(_ORIGINAL_STDOUT, log_file)
    sys.stderr = ConsoleTee(_ORIGINAL_STDERR, log_file)
    atexit.register(close_output_log)


def write_live_progress(text: str) -> None:
    """Write a transient progress update to the terminal, never to log-class-work.txt."""
    stream = sys.stdout
    terminal_writer = getattr(stream, "write_terminal_only", None)
    if callable(terminal_writer):
        terminal_writer(text)
    else:
        stream.write(text)
    stream.flush()


def run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and raise a readable error if it fails."""
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        executable = command[0]
        raise RuntimeError(
            f"Executable '{executable}' was not found. Install it or add it to PATH."
        ) from exc

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        if len(details) > 7000:
            details = details[-7000:]
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{' '.join(command)}\n\n{details}"
        )
    return result


def as_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def as_frame_rate(value: object) -> Fraction | None:
    """Parse a positive FFprobe frame rate while preserving its exact rational value."""
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        rate = Fraction(text)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return rate if rate > 0 else None


def probe_video(
    input_file: Path,
) -> tuple[Decimal, Decimal, Decimal, Decimal | None, Decimal | None, int, int, Fraction]:
    """Return timeline data, duration, source dimensions, and source frame rate."""
    command = [
        FFPROBE_BINARY, "-v", "error",
        "-show_entries",
        "format=start_time,duration:stream=index,codec_type,start_time,time_base,duration,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", str(input_file),
    ]
    result = run_checked(command)

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe did not return valid JSON.") from exc

    streams = info.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video_stream is None:
        raise RuntimeError("ffprobe did not find a video stream.")

    video_start = as_decimal(video_stream.get("start_time"))
    audio_start = as_decimal(audio_stream.get("start_time")) if audio_stream else None
    format_info = info.get("format", {})
    format_start = as_decimal(format_info.get("start_time"))
    duration = as_decimal(format_info.get("duration"))

    if format_start is None:
        stream_starts = [
            value for value in (as_decimal(s.get("start_time")) for s in streams)
            if value is not None
        ]
        if not stream_starts:
            raise RuntimeError("Could not determine the timeline start time from ffprobe.")
        format_start = min(stream_starts)

    if video_start is None:
        raise RuntimeError("Could not determine the video stream start time; no guessed offset will be used.")

    if duration is None:
        duration = as_decimal(video_stream.get("duration"))

    try:
        video_width = int(video_stream["width"])
        video_height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Could not determine the source video dimensions with ffprobe.") from exc
    if video_width < 1 or video_height < 1:
        raise RuntimeError("ffprobe returned invalid source video dimensions.")

    # Prefer the stream's average frame rate; fall back to its nominal/base rate.
    # Keep the exact rational value (for example, 30000/1001) for FFmpeg.
    source_fps = as_frame_rate(video_stream.get("avg_frame_rate"))
    if source_fps is None:
        source_fps = as_frame_rate(video_stream.get("r_frame_rate"))
    if source_fps is None:
        raise RuntimeError(
            "Could not determine a valid source frame rate from ffprobe "
            "(avg_frame_rate or r_frame_rate)."
        )

    offset = video_start - format_start
    return (
        format_start, video_start, offset, audio_start, duration,
        video_width, video_height, source_fps,
    )


def print_video_header(
    input_file: Path,
    video_info: tuple[Decimal, Decimal, Decimal, Decimal | None, Decimal | None, int, int, Fraction],
    manual_offset: Decimal | None,
    *,
    verbose: bool = False,
) -> Decimal:
    """Print input video metadata and timeline information."""
    (
        format_start,
        video_start,
        calculated_offset,
        audio_start,
        duration,
        source_width,
        source_height,
        source_fps,
    ) = video_info
    applied_offset = manual_offset if manual_offset is not None else calculated_offset

    print(f"[Video] {input_file}")
    duration_text = f"{duration} s" if duration is not None else "unknown"
    fps_text = (
        f"{source_fps.numerator} fps"
        if source_fps.denominator == 1
        else f"{float(source_fps):.3f} fps ({source_fps.numerator}/{source_fps.denominator})"
    )
    print(
        f"[Video] Source: {source_width}x{source_height}; "
        f"frame rate: {fps_text}; duration: {duration_text}"
    )
    if verbose:
        print(f"[Timeline] format.start_time: {format_start} s")
        print(f"[Timeline] video.start_time:  {video_start} s")
        if audio_start is not None:
            print(f"[Timeline] audio.start_time:  {audio_start} s")
        print(f"[Timeline] Calculated offset: {calculated_offset:+.6f} s")
        print(f"[Timeline] Applied offset:    {applied_offset:+.6f} s")
    if duration is None:
        print("[Warning] Video duration is unavailable; realtime factor and final EOF timing may be unavailable.")
    return applied_offset


def print_model_header(
    available_classes: list[str],
    target_class: str,
    threshold: float,
    extractor_path: Path,
    batch_size: int,
    *,
    verbose: bool = False,
) -> None:
    """Print the selected target configuration and optional model diagnostics."""
    if verbose:
        print()
        print(
            f"[Model] Selected RKNN extractor: {extractor_path.name} "
            f"(batch size {batch_size})"
        )
        print(f"[Model] Available classes: {', '.join(available_classes)}")
    print(f"[Model] Target class: {target_class}; probability threshold: {threshold:.3f}")
    if verbose:
        print("[Backend] Native C++ ResNetEngine (MPP + RGA + RKNN Zero-Copy Multi-Batch).")


def print_summary(
    *,
    analysed_count: int,
    inference_calls: int,
    validated_count: int,
    review_count: int,
    rejected_count: int,
    suppressed_count: int,
    target_class: str,
    time_range_ms: tuple[int, int] | None,
    processing_seconds: float,
    duration: Decimal | None,
    frame_read_seconds: float,
    packing_seconds: float,
    inference_seconds: float,
    classifier_seconds: float,
    review_dir_created: bool,
    validated_dir_created: bool,
    work_log_created: bool,
    frame_log_created: bool,
) -> None:
    """Print the final processing summary in grouped sections."""
    separator = "  " + "-" * 53

    print("[Summary]")
    print(separator)
    print(f"  {'Validated segments:':<23} {validated_count:,} ({target_class})")
    print(f"  {'Review segments:':<23} {review_count:,}")
    print(f"  {'Rejected candidates:':<23} {rejected_count:,}")
    print(f"  {'Suppressed detections:':<23} {suppressed_count:,}")

    print(separator)
    print(f"  {'Analysed frames:':<23} {analysed_count:,}")
    print(f"  {'Inference calls:':<23} {inference_calls:,}")
    if time_range_ms is not None:
        start_text = format_segment_timestamp(time_range_ms[0])
        end_text = format_segment_timestamp(time_range_ms[1])
        time_range_text = f"{start_text} - {end_text}"
    else:
        time_range_text = "n/a"
    print(f"  {'Analysed time range:':<23} {time_range_text}")
    print(f"  {'Processing time:':<23} {format_elapsed(processing_seconds)}")
    if duration is not None and duration > 0 and processing_seconds > 0:
        realtime_text = f"{float(duration) / processing_seconds:.2f}x"
    else:
        realtime_text = "n/a"
    print(f"  {'Realtime factor:':<23} {realtime_text}")

    print(separator)
    print(f"  {'Frame read time:':<23} {frame_read_seconds:.2f} s")
    print(f"  {'RGB24 batch packing:':<23} {packing_seconds:.2f} s")
    print(f"  {'ResNet inference:':<23} {inference_seconds:.2f} s")
    print(f"  {'Logistic classifier:':<23} {classifier_seconds:.2f} s")

    print(separator)
    if validated_dir_created:
        print(f"  {'Validated seg. path:':<23} PROJ_ROOT/seg_validated")
    if review_dir_created:
        print(f"  {'Review seg. path:':<23} PROJ_ROOT/seg_review")
    if work_log_created:
        print(f"  {'Console output log:':<23} PROJ_ROOT/log-class-work.txt")
    if frame_log_created:
        print(f"  {'Frame classification log:':<23} PROJ_ROOT/log-class-frames.txt")
    print(separator)

def format_timestamp_stem(total_ms: int) -> str:
    """Format a non-negative millisecond timestamp as HH-MM-SS-mmm."""
    if total_ms < 0:
        raise ValueError("A segment timestamp is negative.")
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}-{minutes:02d}-{whole_seconds:02d}-{milliseconds:03d}"


def shell_single_quote(value: str) -> str:
    """Quote a path as one POSIX-shell argument."""
    return "'" + value.replace("'", "'\\''") + "'"


def set_samba_hidden_attribute(path: Path) -> None:
    """Set the Samba DOS Hidden attribute while preserving existing DOSATTRIB data."""
    try:
        blob = bytearray(os.getxattr(path, DOSATTRIB_XATTR))
    except OSError as exc:
        if exc.errno != errno.ENODATA:
            raise RuntimeError(
                f"Could not read Samba hidden attribute for '{path}': {exc}"
            ) from exc

        # Samba treats a file without stored DOS attributes as Archive by default.
        # Create the known Samba 4.17 DOSATTRIB blob with Archive + Hidden flags.
        nt_time = int((time.time() + 11644473600) * 10_000_000)
        blob = bytearray(DOSATTRIB_HEADER)
        blob += struct.pack("<II", 0x11, DOSATTRIB_ARCHIVE | DOSATTRIB_HIDDEN)
        blob += struct.pack("<Q", nt_time)
    else:
        if len(blob) < 16 or bytes(blob[:8]) != DOSATTRIB_HEADER:
            raise RuntimeError(
                f"Unrecognized '{DOSATTRIB_XATTR}' format on '{path}'; refusing to modify it."
            )
        valid_flags, attributes = struct.unpack_from("<II", blob, 8)
        # Preserve all unrelated attribute bits and bytes.
        struct.pack_into(
            "<II", blob, 8,
            valid_flags | 0x01,
            attributes | DOSATTRIB_HIDDEN,
        )

    try:
        os.setxattr(path, DOSATTRIB_XATTR, bytes(blob))
        check = os.getxattr(path, DOSATTRIB_XATTR)
    except OSError as exc:
        raise RuntimeError(
            f"Could not set Samba hidden attribute on '{path}': {exc}"
        ) from exc

    if len(check) < 16 or bytes(check[:8]) != DOSATTRIB_HEADER:
        raise RuntimeError(f"Samba hidden attribute verification failed for '{path}'.")
    valid_flags, attributes = struct.unpack_from("<II", check, 8)
    if not (valid_flags & 0x01 and attributes & DOSATTRIB_HIDDEN):
        raise RuntimeError(f"Samba Hidden flag was not retained for '{path}'.")


def create_project_artifacts(output_dir: Path, input_file: Path) -> None:
    """Silently ensure the relative source symlink and project deletion scripts exist."""
    if not input_file.suffix:
        raise RuntimeError(
            f"The source video has no filename extension: {input_file}. "
            "A source_link.<extension> symlink is required by encode_segments.py."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = input_file.resolve(strict=False)
    source_link_name = f"source_link{input_file.suffix}"
    source_link = output_dir / source_link_name
    relative_target = os.path.relpath(str(source_file), start=str(output_dir.resolve()))

    # Keep a correct relative link; repair a missing, dangling, absolute, or stale
    # generated link. A non-symlink collision is not ours to overwrite.
    if source_link.is_symlink():
        try:
            current_target = source_link.resolve(strict=False)
            current_link_text = os.readlink(source_link)
        except OSError as exc:
            raise RuntimeError(f"Could not inspect source symlink '{source_link}': {exc}") from exc
        if current_target != source_file or current_link_text != relative_target:
            try:
                source_link.unlink()
                source_link.symlink_to(relative_target)
            except OSError as exc:
                raise RuntimeError(f"Could not repair relative source symlink '{source_link}': {exc}") from exc
    elif source_link.exists():
        raise RuntimeError(
            f"Cannot create source symlink because this path already exists and is not a symlink: "
            f"{source_link}"
        )
    else:
        try:
            source_link.symlink_to(relative_target)
        except OSError as exc:
            raise RuntimeError(f"Could not create relative source symlink '{source_link}': {exc}") from exc

    delete_sh_path = output_dir / "delete_source.sh"
    delete_bat_path = output_dir / "delete_source.bat"
    delete_sh_content = "\n".join([
        "#!/bin/bash",
        'cd "$(dirname "$0")" || exit 1',
        f"t=$(readlink -f {source_link_name})",
        f'[[ -L {source_link_name} && -f $t ]] || {{ echo "Invalid symlink or target."; exit 1; }}',
        'echo "$t"',
        'read -rn1 -p "Are you sure you want to delete the source file ? [Y/n]: " a; echo',
        '[[ $a == [Yy] || -z $a ]] || { echo "Cancelled."; exit; }',
        f'if rm -- "$t" {source_link_name}; then echo "Deleted."',
        'rm -f delete_source.bat delete_source.sh; else echo "Failed."; fi',
        "",
    ]).encode("utf-8")

    # Avoid rewriting unchanged files, but repair missing or outdated generated
    # launchers. Temporary files keep replacements atomic on the same filesystem.
    def ensure_file(path: Path, content: bytes) -> None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError(f"Cannot create project artifact because the path is not a regular file: {path}")
        try:
            current_content = path.read_bytes() if path.exists() else None
            if current_content != content:
                fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(output_dir))
                temporary_path = Path(temporary_name)
                try:
                    with os.fdopen(fd, "wb") as temporary_file:
                        fd = -1
                        temporary_file.write(content)
                    temporary_path.chmod(0o775)
                    os.replace(temporary_path, path)
                finally:
                    if fd >= 0:
                        os.close(fd)
                    temporary_path.unlink(missing_ok=True)
            path.chmod(0o775)
        except OSError as exc:
            raise RuntimeError(f"Could not create or configure project artifact '{path}': {exc}") from exc

    ensure_file(delete_sh_path, delete_sh_content)
    set_samba_hidden_attribute(delete_sh_path)

    # The path is fixed in this project's BAT so the launcher can be run from Total Commander.
    remote_script_path = str(delete_sh_path.resolve())
    remote_script_argument = shell_single_quote(remote_script_path)
    delete_bat_content = f'@ssh -tt rkvenc-nas "bash {remote_script_argument}" & pause\n'
    ensure_file(delete_bat_path, delete_bat_content.replace("\n", "\r\n").encode("utf-8"))

def format_segment_timestamp(total_ms: int) -> str:
    """Format a timestamp as HH:MM:SS.milliseconds."""
    if total_ms < 0:
        raise ValueError("Cannot format a negative timestamp.")
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def read_exact_frame(stream: BinaryIO, frame_size: int) -> bytes:
    """Read one complete raw RGB24 frame; return a short tail only at EOF."""
    frame = bytearray(frame_size)
    view = memoryview(frame)
    received = 0
    while received < frame_size:
        count = stream.readinto(view[received:])
        if not count:
            break
        received += count
    if received == 0:
        return b""
    return bytes(view[:received])


def ffmpeg_stderr_reader(
    stream: BinaryIO,
    timestamp_queue: "queue.Queue[Decimal | None]",
    stderr_tail: "deque[str]",
) -> None:
    """Drain FFmpeg diagnostics and queue one showinfo PTS for each output frame."""
    try:
        while True:
            raw_line = stream.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                stderr_tail.append(line)
            if "showinfo" in line:
                match = SHOWINFO_TIMESTAMP_PATTERN.search(line)
                if match:
                    try:
                        timestamp_queue.put(Decimal(match.group(1)))
                    except InvalidOperation:
                        continue
    finally:
        timestamp_queue.put(None)


def get_ffmpeg_frame_timestamp(
    timestamp_queue: "queue.Queue[Decimal | None]",
    process: subprocess.Popen[bytes],
) -> Decimal:
    """Retrieve the showinfo timestamp paired with the next raw RGB24 frame."""
    try:
        timestamp = timestamp_queue.get(timeout=60.0)
    except queue.Empty as exc:
        raise RuntimeError(
            "Timed out waiting for the FFmpeg showinfo timestamp matching a raw RGB24 frame."
        ) from exc
    if timestamp is None:
        raise RuntimeError(
            "FFmpeg closed its timestamp stream while raw RGB24 frames were still arriving "
            f"(process status: {process.poll()})."
        )
    return timestamp


def seconds_to_milliseconds(seconds: Decimal) -> int:
    return int((seconds * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def save_rgb24_segment_thumbnail(
    frame_data: bytes,
    start_time: Decimal,
    end_time: Decimal,
    output_dir: Path,
    *,
    jpeg_quality: int = 90,
) -> Path:
    """Save the first detected RGB24 frame using the project's start/end filename format."""
    from PIL import Image

    start_ms = seconds_to_milliseconds(start_time)
    end_ms = seconds_to_milliseconds(end_time)
    if end_ms <= start_ms:
        raise RuntimeError(
            "A detected segment has a non-positive duration after millisecond rounding: "
            f"{format_timestamp_stem(start_ms)} - {format_timestamp_stem(end_ms)}."
        )

    filename = (
        f"{format_timestamp_stem(start_ms)} - {format_timestamp_stem(end_ms)}.jpg"
    )
    output_path = output_dir / filename
    if output_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite an existing class-segment thumbnail: {output_path}"
        )

    temporary_path = output_dir / f".{filename}.{os.getpid()}.tmp"
    try:
        image = Image.frombytes("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), frame_data)
        image.save(temporary_path, format="JPEG", quality=jpeg_quality, optimize=False)
        if output_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite an existing class-segment thumbnail: {output_path}"
            )
        temporary_path.replace(output_path)
    except (OSError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save class-segment thumbnail '{output_path}': {exc}") from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def build_frame_classification_log_header(
    class_names: list[str],
    target_index: int,
    used_class_indices: set[int],
) -> str:
    """Build a compact legend containing only classes present in logged top-four rows."""
    if len(class_names) > 0x1000:
        raise ValueError(
            "The frame log uses 12-bit hexadecimal class IDs and supports at most 4096 classes."
        )
    if not 0 <= target_index < len(class_names):
        raise ValueError("Target class index is outside the classifier class list.")

    legend_indices = sorted(
        index for index in used_class_indices
        if index != target_index and 0 <= index < len(class_names)
    )
    lines = [
        "# Frame classification debug log\n",
        "# Columns: index, timestamp, band, class=probability (top 4x, target-class always included)\n",
        "#\n",
        "# Class legend:\n",
        f"#   XXX {class_names[target_index]}\n",
    ]
    for class_index in legend_indices:
        lines.append(f"#   {class_index:03X} {class_names[class_index]}\n")
    lines.append("#\n")
    return "".join(lines)


def format_frame_classification_log_line(
    frame_index: int,
    timestamp: Decimal,
    probabilities,
    class_names: list[str],
    target_index: int,
    threshold: float,
    review_threshold: float,
    used_class_indices: set[int] | None = None,
) -> str:
    """Format the four highest-probability classes, always including the target."""
    target_probability = float(probabilities[target_index])
    if target_probability >= threshold:
        band = "[ HIGH ]"
    elif review_threshold > 0.0 and target_probability >= review_threshold:
        band = "[REVIEW]"
    else:
        band = "[ NONE ]"

    # Keep the target plus the three strongest non-target classes, then sort all
    # selected entries together by their actual probability (not rounded text).
    other_indices = sorted(
        (
            class_index
            for class_index in range(len(class_names))
            if class_index != target_index and float(probabilities[class_index]) > 0.0
        ),
        key=lambda class_index: (-float(probabilities[class_index]), class_index),
    )[:3]
    selected_indices = [target_index, *other_indices]
    selected_indices.sort(
        key=lambda class_index: (-float(probabilities[class_index]), class_index)
    )
    if used_class_indices is not None:
        used_class_indices.update(selected_indices)

    def class_field(class_code: str, probability: float) -> str:
        return f"{class_code:>3} = {probability * 100.0:6.2f} %"

    fields = [
        class_field(
            "XXX" if class_index == target_index else f"{class_index:03X}",
            float(probabilities[class_index]),
        )
        for class_index in selected_indices
    ]
    fields.extend([""] * (4 - len(fields)))
    class_columns = " | ".join(field.ljust(14) for field in fields)
    timestamp_text = format_segment_timestamp(seconds_to_milliseconds(timestamp))
    return f"{frame_index:06d}  {timestamp_text} {band}  {class_columns}\n"


def finalize_frame_classification_log(
    body_file,
    body_path: Path,
    output_path: Path,
    class_names: list[str],
    target_index: int,
    used_class_indices: set[int],
) -> None:
    """Prepend the completed dynamic legend and atomically publish the frame log."""
    try:
        body_file.flush()
        os.fsync(body_file.fileno())
    finally:
        body_file.close()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        output_file = os.fdopen(
            descriptor, "w", encoding="utf-8", errors="replace", buffering=1024 * 1024
        )
        descriptor = -1
        with output_file, body_path.open("r", encoding="utf-8", errors="replace") as body:
            output_file.write(
                build_frame_classification_log_header(
                    class_names, target_index, used_class_indices
                )
            )
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        # mkstemp creates mode 0600; match normal open(..., "w") permissions
        # so the final log remains accessible under the process' configured umask.
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(temporary_path, 0o666 & ~current_umask)
        os.replace(temporary_path, output_path)
        body_path.unlink(missing_ok=True)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def extract_classified_segments(
    input_file: Path,
    output_dir: Path,
    model_parameters,
    target_class: str,
    threshold: float,
    review_threshold: float,
    feature_model_path: Path,
    batch_size: int,
    sampling_ms: int,
    deint: int,
    window_size: int,
    confirm_votes: int,
    confirm_streak: int,
    min_duration_ms: int,
    min_run_frames: int,
    manual_offset: Decimal | None,
    video_info: tuple[Decimal, Decimal, Decimal, Decimal | None, Decimal | None, int, int, Fraction] | None = None,
    verbose: bool = False,
    log_frames: bool = False,
) -> tuple[list[Path], dict[str, object]]:
    """Classify sampled frames into VALID/REVIEW segments using temporal voting."""
    import numpy as np
    try:
        from PIL import Image as _PillowImage  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to save class-segment thumbnails. "
            "Install it with: python -m pip install pillow."
        ) from exc

    (
        format_start,
        video_start,
        calculated_offset,
        audio_start,
        duration,
        source_width,
        source_height,
        source_fps,
    ) = video_info if video_info is not None else probe_video(input_file)
    offset = manual_offset if manual_offset is not None else calculated_offset

    if duration is not None and duration <= 0:
        raise RuntimeError(f"ffprobe returned an invalid video duration: {duration}.")
    if batch_size < 1:
        raise ValueError("Inference batch size must be a positive integer.")
    if not (review_threshold == 0.0 or 0.0 < review_threshold < threshold):
        raise ValueError(
            "Review threshold must be 0 (disabled) or greater than 0 and lower than the main threshold."
        )
    if not 5 <= sampling_ms <= 1000:
        raise ValueError("Sampling interval must be between 5 and 1000 milliseconds.")
    if window_size < 1:
        raise ValueError("Validation window must contain at least one sample.")
    if not 1 <= confirm_votes <= window_size:
        raise ValueError("Validation votes must be between 1 and the window size.")
    if not 1 <= confirm_streak <= window_size:
        raise ValueError("Validation streak must be between 1 and the window size.")
    if min_duration_ms < 1:
        raise ValueError("Minimum segment duration must be at least 1 millisecond.")
    if min_run_frames < 1:
        raise ValueError("Minimum consecutive matching-category samples must be at least one.")

    try:
        target_index = model_parameters[4].index(target_class)
    except ValueError as exc:
        raise RuntimeError(
            f"Target class '{target_class}' is not present in the classifier model."
        ) from exc
    mean, scale, coef, intercept, class_names = model_parameters[:5]
    if log_frames and len(class_names) > 0x1000:
        raise ValueError(
            "The frame log uses 12-bit hexadecimal class IDs and supports at most 4096 classes."
        )

    seg_validated_dir = output_dir / "seg_validated"
    seg_review_dir = output_dir / "seg_review"
    seg_validated_dir.mkdir(parents=True, exist_ok=True)
    review_enabled = review_threshold > 0.0
    if review_enabled:
        seg_review_dir.mkdir(parents=True, exist_ok=True)

    if offset > 0:
        setpts_expr = f"PTS+{offset:.9f}/TB"
    elif offset < 0:
        setpts_expr = f"PTS-{abs(offset):.9f}/TB"
    else:
        setpts_expr = "PTS"

    # Select frames without rebuilding a CFR timeline, preserving source PTS.
    # Convert the requested interval to seconds for FFmpeg's timestamp-based selector.
    # A tiny tolerance avoids floating-point edge cases at exact sampling boundaries.
    sampling_interval = Decimal(sampling_ms) / Decimal(1000)
    sampling_tolerance = min(Decimal("0.000001"), sampling_interval / Decimal(1000))
    minimum_sampling_interval = sampling_interval - sampling_tolerance
    select_interval = format(minimum_sampling_interval, "f")

    # Optional source-rate normalization. Keep it disabled while the sampler
    # works directly with the full cadence produced by the MPP decoder.
    filter_graph = (
        # f"fps={source_fps.numerator}/{source_fps.denominator}:round=near,"
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{select_interval})',"
        f"scale_rkrga=w={IMAGE_WIDTH}:h={IMAGE_HEIGHT}:format=rgb24,"
        "hwmap=mode=read,"
        "format=rgb24,"
        "settb=AVTB,"
        f"setpts={setpts_expr},"
        "showinfo"
    )
    command_prefix = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostdin",
        "-loglevel", "info",
        "-nostats",
        "-hwaccel", "rkmpp",
        "-hwaccel_output_format", "drm_prime",
        "-deint", str(deint),
        "-i", str(input_file),
        "-map", "0:v:0",
        "-vf", filter_graph,
        "-an", "-sn", "-dn",
        "-fps_mode", "passthrough",
        "-c:v", "rawvideo",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
    ]

    engine_python_dir = Path(__file__).resolve().parent / "resnet_engine" / "python"
    engine_python_text = str(engine_python_dir)
    if engine_python_text in sys.path:
        sys.path.remove(engine_python_text)
    sys.path.insert(0, engine_python_text)
    try:
        import resnet_engine
    except ImportError as exc:
        raise RuntimeError(
            "The ResNetEngine Python extension could not be imported. Build the updated shared "
            "library and Python extension with nanopi/resnet_engine/src/build-resnet-engine-python-ext.sh."
        ) from exc

    frame_log_path = output_dir / "log-class-frames.txt" if log_frames else None
    frame_log_temp_path: Path | None = None
    frame_log_file = None
    used_class_indices: set[int] = {target_index}
    if frame_log_path is not None:
        descriptor: int | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".log-class-frames.",
                suffix=".body.tmp",
                dir=output_dir,
            )
            frame_log_temp_path = Path(temporary_name)
            frame_log_file = os.fdopen(
                descriptor, "w", encoding="utf-8", errors="replace", buffering=1024 * 1024
            )
            descriptor = None
        except (OSError, ValueError) as exc:
            if descriptor is not None:
                os.close(descriptor)
            if frame_log_file is not None:
                frame_log_file.close()
                frame_log_file = None
            if frame_log_temp_path is not None:
                frame_log_temp_path.unlink(missing_ok=True)
                frame_log_temp_path = None
            raise RuntimeError(
                f"Could not create temporary frame classification log in '{output_dir}': {exc}"
            ) from exc

    engine = None
    process: subprocess.Popen[bytes] | None = None
    stderr_thread: threading.Thread | None = None
    raw_read_fd: int | None = None
    raw_write_fd: int | None = None
    raw_stream: BinaryIO | None = None
    timestamp_queue: "queue.Queue[Decimal | None]" = queue.Queue()
    stderr_tail: "deque[str]" = deque(maxlen=80)

    frame_count = 0
    inference_calls = 0
    segment_count = 0
    previous_timestamp: Decimal | None = None
    first_timestamp: Decimal | None = None
    last_timestamp: Decimal | None = None
    # Each sample stores (timestamp, RGB24 frame, target probability, category).
    # Categories are mutually exclusive: "valid", "review", or "none".
    # The rolling window confirms category transitions; candidate runs are
    # validated separately before becoming active segments.
    validation_window: deque[tuple[Decimal, bytes, float, str]] = deque(maxlen=window_size)
    candidate_category: str | None = None
    candidate_start: Decimal | None = None
    candidate_last_timestamp: Decimal | None = None
    candidate_frame: bytes | None = None
    candidate_probability = 0.0
    candidate_frame_count = 0
    active_category: str | None = None
    active_start: Decimal | None = None
    active_last_timestamp: Decimal | None = None
    active_frame: bytes | None = None
    active_probability = 0.0
    candidates_rejected = 0
    suppressed_detections = 0  # Counts suppressed groups, not individual positive samples.
    pending_suppressed_timestamps: list[Decimal] = []
    suppressed_gap_count = 0
    review_segment_count = 0
    review_paths: list[Path] = []
    pending_segment_messages: list[str] = []
    progress_line_displayed = False
    progress_last_inference_calls: int | None = None
    progress_last_update_time: float | None = None
    progress_is_tty = sys.stdout.isatty()
    soc_thermal_sensors = discover_soc_thermal_sensors()
    maximum_soc_temperature_millidegrees: int | None = None
    read_seconds = 0.0
    packing_seconds = 0.0
    inference_seconds = 0.0
    classifier_seconds = 0.0
    started = 0.0
    saved_paths: list[Path] = []
    frame_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3
    raw_batch: list[bytes] = []
    timestamp_batch: list[Decimal] = []

    def build_progress_line() -> str:
        nonlocal maximum_soc_temperature_millidegrees

        current_soc_temperature = read_hottest_soc_temperature(soc_thermal_sensors)
        if current_soc_temperature is not None:
            if (maximum_soc_temperature_millidegrees is None
                    or current_soc_temperature > maximum_soc_temperature_millidegrees):
                maximum_soc_temperature_millidegrees = current_soc_temperature

        current_temperature_text = format_soc_temperature(current_soc_temperature)
        maximum_temperature_text = format_soc_temperature(maximum_soc_temperature_millidegrees)

        elapsed = max(0.001, time.perf_counter() - started)
        position_text = (
            format_segment_timestamp(seconds_to_milliseconds(last_timestamp))
            if last_timestamp is not None else "00:00:00.000"
        )
        if duration is not None and duration > 0 and last_timestamp is not None:
            percent = min(100.0, max(0.0, float(last_timestamp / duration * Decimal(100))))
            progress_position = f"{position_text} ({percent:.1f}%)"
        else:
            progress_position = position_text
        realtime = (
            float(last_timestamp) / elapsed
            if last_timestamp is not None and elapsed > 0 else 0.0
        )
        if frame_count > 1 and first_timestamp is not None and last_timestamp is not None:
            # The arithmetic mean of consecutive PTS differences telescopes to
            # the total timestamp span divided by the number of processed intervals.
            average_interval_ms = (
                (last_timestamp - first_timestamp) * Decimal(1000)
                / Decimal(frame_count - 1)
            )
            average_interval_text = f"{average_interval_ms:.1f} ms"
        else:
            average_interval_text = "n/a"
        return (
            f"[Progress] {frame_count:,} frames "
            + (f"({average_interval_text}) | {inference_calls:,} infs " if verbose else "") +
            f"| seg: {segment_count} / {review_segment_count} | {progress_position} | {realtime:.1f}x RT "
            f"| SoC {current_temperature_text}/{maximum_temperature_text} °C"
        )

    def write_progress_line() -> None:
        nonlocal progress_line_displayed, progress_last_inference_calls
        nonlocal progress_last_update_time
        line = build_progress_line()
        progress_last_inference_calls = inference_calls
        if progress_is_tty:
            # Replace the current line instead of scrolling the terminal.
            write_live_progress("\r\x1b[2K" + line)
            progress_line_displayed = True
        else:
            # Keep redirected output and log files readable; terminals get in-place updates.
            write_live_progress(line + "\n")
            progress_line_displayed = False
        progress_last_update_time = time.perf_counter()

    def flush_segment_messages() -> None:
        nonlocal progress_line_displayed
        if not pending_segment_messages:
            return
        if progress_is_tty and progress_line_displayed:
            write_live_progress("\r\x1b[2K")
            progress_line_displayed = False
        for message in pending_segment_messages:
            is_valid_segment = message.startswith(f"[Progress] Detected {target_class} segment:")
            if verbose or is_valid_segment:
                print(message, flush=True)
        pending_segment_messages.clear()
        # A segment message may occur between scheduled updates. Redraw progress
        # immediately so that it remains the last visible terminal line.
        write_progress_line()

    def finish_progress_line() -> None:
        nonlocal progress_line_displayed
        if progress_is_tty and progress_line_displayed:
            # Erase the live progress snapshot before the final summary is printed.
            write_live_progress("\r\x1b[2K")
            progress_line_displayed = False

    def candidate_coverage_ms(
        frame_count: int,
        first_time: Decimal,
        last_time: Decimal,
    ) -> Decimal:
        """Estimate covered duration, counting the final sample's frame interval."""
        if frame_count < 2:
            return Decimal(0)
        timestamp_span_ms = (last_time - first_time) * Decimal(1000)
        average_interval_ms = timestamp_span_ms / Decimal(frame_count - 1)
        return timestamp_span_ms + average_interval_ms

    def category_for_probability(probability: float) -> str:
        """Map a target probability to one mutually exclusive validation category."""
        if probability >= threshold:
            return "valid"
        if review_threshold > 0.0 and probability >= review_threshold:
            return "review"
        return "none"

    def reset_candidate() -> None:
        nonlocal candidate_category, candidate_start, candidate_last_timestamp, candidate_frame
        nonlocal candidate_probability, candidate_frame_count
        candidate_category = None
        candidate_start = None
        candidate_last_timestamp = None
        candidate_frame = None
        candidate_probability = 0.0
        candidate_frame_count = 0

    def flush_suppressed_group() -> None:
        nonlocal suppressed_detections, suppressed_gap_count
        if not pending_suppressed_timestamps:
            return

        first_suppressed = pending_suppressed_timestamps[0]
        last_suppressed = pending_suppressed_timestamps[-1]
        sample_count = len(pending_suppressed_timestamps)
        sample_label = "sample" if sample_count == 1 else "samples"
        suppressed_threshold_label = (
            f"review threshold {review_threshold:.2f}"
            if review_enabled else f"target threshold {threshold:.2f}"
        )
        pending_segment_messages.append(
            f"[Progress] Suppressed {target_class} detection group: "
            f"{format_segment_timestamp(seconds_to_milliseconds(first_suppressed))} - "
            f"{format_segment_timestamp(seconds_to_milliseconds(last_suppressed))} "
            f"({sample_count} {sample_label} >= {suppressed_threshold_label}; "
            f"no candidate was formed from this group)"
        )
        suppressed_detections += 1
        pending_suppressed_timestamps.clear()
        suppressed_gap_count = 0

    def reject_candidate(reason: str) -> None:
        nonlocal candidates_rejected
        if candidate_start is None:
            return
        coverage_ms = (
            candidate_coverage_ms(
                candidate_frame_count,
                candidate_start,
                candidate_last_timestamp,
            )
            if candidate_last_timestamp is not None else Decimal(0)
        )
        category_label = "REVIEW" if candidate_category == "review" else "VALID"
        pending_segment_messages.append(
            f"[Progress] Rejected {target_class} {category_label} candidate at "
            f"{format_segment_timestamp(seconds_to_milliseconds(candidate_start))} "
            f"({candidate_frame_count} consecutive matching samples, "
            f"{coverage_ms:.0f} ms coverage; {reason})"
        )
        candidates_rejected += 1
        reset_candidate()

    def promote_candidate() -> None:
        nonlocal active_category, active_start, active_last_timestamp, active_frame
        nonlocal active_probability
        if (candidate_category is None or candidate_start is None
                or candidate_last_timestamp is None or candidate_frame is None):
            raise RuntimeError("Cannot validate an incomplete class-segment candidate.")
        active_category = candidate_category
        active_start = candidate_start
        active_last_timestamp = candidate_last_timestamp
        active_frame = candidate_frame
        active_probability = candidate_probability
        reset_candidate()

    def candidate_is_valid() -> bool:
        if candidate_start is None or candidate_last_timestamp is None:
            return False
        if candidate_frame_count < min_run_frames:
            return False
        coverage_ms = candidate_coverage_ms(
            candidate_frame_count,
            candidate_start,
            candidate_last_timestamp,
        )
        return coverage_ms >= Decimal(min_duration_ms)

    def close_active_segment(end_time: Decimal) -> None:
        nonlocal active_category, active_start, active_last_timestamp, active_frame
        nonlocal active_probability, segment_count, review_segment_count
        if active_start is None or active_frame is None or active_category is None:
            return

        start_ms = seconds_to_milliseconds(active_start)
        end_ms = seconds_to_milliseconds(end_time)
        if end_ms <= start_ms:
            raise RuntimeError(
                "A detected class segment has a non-positive duration after millisecond rounding: "
                f"{format_timestamp_stem(start_ms)} - {format_timestamp_stem(end_ms)}."
            )

        if active_category == "valid":
            destination_dir = seg_validated_dir
            output_path = save_rgb24_segment_thumbnail(
                active_frame, active_start, end_time, destination_dir
            )
            segment_count += 1
            saved_paths.append(output_path)
            message_prefix = f"Detected {target_class} segment"
        else:
            destination_dir = seg_review_dir
            output_path = save_rgb24_segment_thumbnail(
                active_frame, active_start, end_time, destination_dir
            )
            review_segment_count += 1
            review_paths.append(output_path)
            message_prefix = f"Review {target_class} segment"

        pending_segment_messages.append(
            f"[Progress] {message_prefix}: "
            f"{format_segment_timestamp(start_ms)} - {format_segment_timestamp(end_ms)} "
            f"(start confidence {active_probability * 100.0:.1f}%)"
        )
        active_category = None
        active_start = None
        active_last_timestamp = None
        active_frame = None
        active_probability = 0.0

    def confirmed_trailing_category(
        samples: list[tuple[Decimal, bytes, float, str]],
    ) -> tuple[str, int] | None:
        """Return the category confirmed by votes and a matching trailing streak."""
        if not samples:
            return None
        category = samples[-1][3]
        trailing_count = 0
        for sample in reversed(samples):
            if sample[3] != category:
                break
            trailing_count += 1
        votes = sum(1 for sample in samples if sample[3] == category)
        if votes >= confirm_votes and trailing_count >= confirm_streak:
            return category, trailing_count
        return None

    def open_candidate_from_window(
        category: str,
        trailing_count: int,
        *,
        from_active_transition: bool = False,
    ) -> None:
        """Start a candidate from the final consecutive run of its confirmed category."""
        nonlocal candidate_category, candidate_start, candidate_last_timestamp
        nonlocal candidate_frame, candidate_probability, candidate_frame_count
        nonlocal suppressed_gap_count

        samples = list(validation_window)
        if trailing_count < 1 or trailing_count > len(samples):
            raise RuntimeError("Category transition produced an invalid trailing sample count.")
        first_candidate_sample = samples[-trailing_count]

        # The trailing run is assigned to the candidate. Any older positive
        # detections remain in the suppressed group and are reported separately.
        if not from_active_transition:
            if len(pending_suppressed_timestamps) < trailing_count:
                raise RuntimeError(
                    "Suppressed-detection tracking lost samples needed to open a candidate."
                )
            del pending_suppressed_timestamps[-trailing_count:]
        flush_suppressed_group()
        suppressed_gap_count = 0

        candidate_category = category
        candidate_start = first_candidate_sample[0]
        candidate_frame = first_candidate_sample[1]
        candidate_probability = first_candidate_sample[2]
        candidate_last_timestamp = samples[-1][0]
        candidate_frame_count = trailing_count
        if candidate_is_valid():
            promote_candidate()

    def confirmed_active_transition(
        active: str,
        samples: list[tuple[Decimal, bytes, float, str]],
    ) -> tuple[str, int] | None:
        """Confirm a change away from an active category, respecting its lower tiers."""
        if not samples:
            return None

        current_category = samples[-1][3]
        if current_category == active:
            return None

        if active == "valid":
            # Both REVIEW and NONE are below VALID. Count them together when
            # deciding whether the high-confidence segment has really ended.
            lower_votes = sum(1 for sample in samples if sample[3] != "valid")
            trailing_lower = 0
            for sample in reversed(samples):
                if sample[3] == "valid":
                    break
                trailing_lower += 1
            if lower_votes < confirm_votes or trailing_lower < confirm_streak:
                return None

            # Open a REVIEW candidate only when REVIEW itself has a confirmed
            # trailing run. Otherwise this is a confirmed transition out of VALID
            # without a sufficiently stable destination category.
            exact_transition = confirmed_trailing_category(samples)
            if exact_transition is not None and exact_transition[0] in ("review", "none"):
                return exact_transition
            return "mixed_lower", trailing_lower

        # REVIEW is above NONE. A transition to NONE closes it; a confirmed
        # transition to VALID hands off to a separate high-confidence candidate.
        transition = confirmed_trailing_category(samples)
        if transition is not None and transition[0] != active:
            return transition
        return None

    def queue_unassigned_positive_samples(
        samples: list[tuple[Decimal, bytes, float, str]],
        after_timestamp: Decimal,
        trailing_category: str | None,
        trailing_count: int,
    ) -> None:
        """Track positive samples after the old segment that are not in a new candidate."""
        suffix_start = len(samples) - trailing_count if trailing_category in ("valid", "review") else len(samples)
        for index, sample in enumerate(samples):
            sample_time, _frame, _probability, sample_category = sample
            if (sample_time > after_timestamp and index < suffix_start
                    and sample_category in ("valid", "review")):
                pending_suppressed_timestamps.append(sample_time)

    def process_sample(frame_data: bytes, timestamp: Decimal, probability: float) -> None:
        """Validate mutually exclusive VALID, REVIEW, and NONE category transitions."""
        nonlocal candidate_last_timestamp, candidate_frame_count
        nonlocal active_last_timestamp, suppressed_gap_count

        category = category_for_probability(probability)
        validation_window.append((timestamp, frame_data, probability, category))
        samples = list(validation_window)

        if active_category is not None:
            # A segment's own category advances its endpoint. Lower categories are
            # tolerated temporarily; a VALID segment treats REVIEW and NONE as a
            # combined lower tier when confirming its exit.
            if category == active_category:
                active_last_timestamp = timestamp

            transition = confirmed_active_transition(active_category, samples)
            if transition is None:
                return

            next_category, trailing_count = transition
            if active_last_timestamp is None:
                raise RuntimeError("Active class segment has no matching-category endpoint.")

            previous_end = active_last_timestamp
            candidate_category_for_suffix = (
                next_category if next_category in ("valid", "review") else None
            )
            queue_unassigned_positive_samples(
                samples,
                previous_end,
                candidate_category_for_suffix,
                trailing_count if candidate_category_for_suffix is not None else 0,
            )
            close_active_segment(previous_end)

            if next_category not in ("valid", "review"):
                # NONE or a mixed lower-tier transition closes the active segment.
                # Keep unassigned positive fragments grouped across the state reset;
                # age the group by consecutive NONE samples rather than by deque contents.
                suppressed_gap_count = 0
                for sample in reversed(samples):
                    if sample[0] <= previous_end:
                        break
                    if sample[3] == "none":
                        suppressed_gap_count += 1
                    else:
                        break
                if pending_suppressed_timestamps and suppressed_gap_count >= window_size:
                    flush_suppressed_group()
                validation_window.clear()
                return

            # A confirmed transition to the other positive category closes the old
            # segment and seeds a separate candidate from the new category's suffix.
            open_candidate_from_window(
                next_category, trailing_count, from_active_transition=True
            )
            return

        if candidate_category is not None:
            if category == candidate_category:
                candidate_last_timestamp = timestamp
                candidate_frame_count += 1
                if candidate_is_valid():
                    promote_candidate()
                return

            reject_candidate("matching-category run was interrupted before validation")
            # The interrupting sample now belongs to the next unassigned detection
            # group and may begin a candidate of its own category.

        if category != "none":
            pending_suppressed_timestamps.append(timestamp)
            suppressed_gap_count = 0
        elif pending_suppressed_timestamps:
            suppressed_gap_count += 1

        transition = confirmed_trailing_category(samples)
        if transition is not None:
            next_category, trailing_count = transition
            if next_category in ("valid", "review"):
                open_candidate_from_window(next_category, trailing_count)
                return

        if pending_suppressed_timestamps and suppressed_gap_count >= window_size:
            flush_suppressed_group()

    def process_batch() -> None:
        nonlocal frame_count, inference_calls, packing_seconds, inference_seconds
        nonlocal classifier_seconds

        if not raw_batch:
            return
        batch_count = len(raw_batch)
        packing_started = time.perf_counter()
        packed_frames = b"".join(raw_batch)
        rgb24 = np.frombuffer(packed_frames, dtype=np.uint8).reshape(
            batch_count, IMAGE_HEIGHT, IMAGE_WIDTH, 3
        )
        packing_seconds += time.perf_counter() - packing_started

        inference_started = time.perf_counter()
        features = np.asarray(engine.infer_rgb24(rgb24), dtype=np.float32)
        inference_seconds += time.perf_counter() - inference_started
        inference_calls += 1
        if features.shape != (batch_count, FEATURE_COUNT):
            raise RuntimeError(
                f"ResNetEngine returned feature shape {features.shape}; "
                f"expected ({batch_count}, {FEATURE_COUNT})."
            )
        if not np.isfinite(features).all():
            raise RuntimeError("ResNetEngine returned NaN or infinite feature values.")

        classifier_started = time.perf_counter()
        logits = ((features - mean) / scale) @ coef.T + intercept
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        classifier_seconds += time.perf_counter() - classifier_started
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Classifier produced NaN or infinite probabilities.")

        # The complete batch has been classified by this point. Count it before
        # emitting any segment messages so their accompanying progress is current.
        frame_count += batch_count
        if frame_log_file is not None:
            first_frame_index = frame_count - batch_count + 1
            batch_used_class_indices: set[int] = set()
            frame_log_text = "".join(
                format_frame_classification_log_line(
                    first_frame_index + index,
                    timestamp,
                    probabilities[index],
                    class_names,
                    target_index,
                    threshold,
                    review_threshold,
                    batch_used_class_indices,
                )
                for index, timestamp in enumerate(timestamp_batch)
            )
            try:
                # One write per inference batch; the 1 MiB text buffer combines
                # these writes into much larger filesystem operations.
                frame_log_file.write(frame_log_text)
                used_class_indices.update(batch_used_class_indices)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not write temporary frame classification log '{frame_log_temp_path}': {exc}"
                ) from exc

        for index, (frame_data, timestamp) in enumerate(zip(raw_batch, timestamp_batch)):
            target_probability = float(probabilities[index, target_index])
            process_sample(frame_data, timestamp, target_probability)

        raw_batch.clear()
        timestamp_batch.clear()

        if pending_segment_messages:
            flush_segment_messages()
        elif (
            progress_last_update_time is None
            or time.perf_counter() - progress_last_update_time
            >= PROGRESS_UPDATE_INTERVAL_SECONDS
        ):
            write_progress_line()

    try:
        engine = resnet_engine.ResNetEngine(
            str(feature_model_path),
            batch_size=batch_size,
            npu_core_count=2,
        )
        engine_info = engine.info
        if engine_info.batch_size != batch_size:
            raise RuntimeError(
                f"ResNetEngine initialized with batch size {engine_info.batch_size}; expected {batch_size}."
            )
        if (engine_info.input_width, engine_info.input_height, engine_info.input_channels) != (
            IMAGE_WIDTH, IMAGE_HEIGHT, 3
        ):
            raise RuntimeError(
                f"ResNetEngine input is {engine_info.input_width}x{engine_info.input_height}x"
                f"{engine_info.input_channels}; expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}x3."
            )
        if engine_info.feature_count != FEATURE_COUNT:
            raise RuntimeError(
                f"ResNetEngine returns {engine_info.feature_count} features; expected {FEATURE_COUNT}."
            )
        if not callable(getattr(engine, "infer_rgb24", None)):
            raise RuntimeError(
                "The loaded ResNetEngine Python extension does not expose infer_rgb24(). "
                "Rebuild the shared library and Python extension from the updated sources."
            )
        if verbose:
            print(f"[Engine] {engine_info}")

        raw_read_fd, raw_write_fd = os.pipe()
        command = command_prefix + [f"pipe:{raw_write_fd}"]
        if verbose:
            print(
                f"[ProgUpd] Progress update target: "
                f"every {PROGRESS_UPDATE_INTERVAL_SECONDS:.1f} s "
                "(checked after each completed inference batch)."
            )
        started = time.perf_counter()
        if verbose: print()
        print(
            f"[FFmpeg/ResNet] Starting FFmpeg hardware decode, "
            f"streaming {IMAGE_WIDTH}x{IMAGE_HEIGHT} RGB24 frames to ResNet classification...",
            flush=True,
        )
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(raw_write_fd,),
                close_fds=True,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not start FFmpeg executable '{FFMPEG_BINARY}': {exc}. "
                "Install the Rockchip-enabled build or make sure ffmpeg is available in PATH."
            ) from exc
        os.close(raw_write_fd)
        raw_write_fd = None
        raw_stream = os.fdopen(raw_read_fd, "rb", buffering=0)
        raw_read_fd = None

        if process.stderr is None:
            raise RuntimeError("Could not capture FFmpeg stderr for frame timestamp synchronization.")
        stderr_thread = threading.Thread(
            target=ffmpeg_stderr_reader,
            args=(process.stderr, timestamp_queue, stderr_tail),
            name="process-video-ffmpeg-stderr",
            daemon=True,
        )
        stderr_thread.start()
        write_progress_line()

        while True:
            read_started = time.perf_counter()
            frame_data = read_exact_frame(raw_stream, frame_size)
            read_seconds += time.perf_counter() - read_started
            if not frame_data:
                break
            if len(frame_data) != frame_size:
                raise RuntimeError(
                    f"FFmpeg ended with an incomplete RGB24 frame: received {len(frame_data):,} "
                    f"of {frame_size:,} bytes."
                )

            timestamp = get_ffmpeg_frame_timestamp(timestamp_queue, process)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise RuntimeError(
                    f"Non-increasing FFmpeg frame timestamp: {timestamp} after {previous_timestamp}."
                )
            previous_timestamp = timestamp
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            raw_batch.append(frame_data)
            timestamp_batch.append(timestamp)
            if len(raw_batch) >= batch_size:
                process_batch()

        # Run the last incomplete batch; infer_rgb24() pads internally and discards padded outputs.
        process_batch()

        return_code = process.wait()
        stderr_thread.join()
        if return_code != 0:
            detail = "\n".join(stderr_tail).strip()
            if len(detail) > 7000:
                detail = detail[-7000:]
            raise RuntimeError(
                f"FFmpeg RGB24 streaming failed with exit code {return_code}."
                + (f"\n{detail}" if detail else "")
            )
        if frame_count == 0:
            raise RuntimeError("FFmpeg completed successfully but produced no RGB24 frames.")

        # Every decoded raw frame must have exactly one corresponding showinfo timestamp.
        extra_timestamps = 0
        while True:
            try:
                queued_timestamp = timestamp_queue.get_nowait()
            except queue.Empty:
                break
            if queued_timestamp is not None:
                extra_timestamps += 1
        if extra_timestamps:
            raise RuntimeError(
                f"FFmpeg produced {extra_timestamps} extra showinfo timestamp(s) without matching raw frames."
            )

        if candidate_start is not None:
            reject_candidate("end of video reached before validation")

        if active_start is not None and active_category is not None:
            trailing_matching = 0
            for sample in reversed(validation_window):
                if sample[3] != active_category:
                    break
                trailing_matching += 1

            # Extend to EOF only when the final category state is itself confirmed.
            # Otherwise stop at the last sample matching the active category.
            if trailing_matching >= confirm_streak:
                if duration is None:
                    raise RuntimeError(
                        "The video duration is unavailable, so the final open class segment "
                        "cannot be closed at EOF safely."
                    )
                final_end = duration
            else:
                final_end = active_last_timestamp

            if final_end is None:
                raise RuntimeError("The final active class segment has no matching-category endpoint.")
            close_active_segment(final_end)

        # A final group may not have had enough following samples to age out
        # of the rolling validation window before EOF.
        flush_suppressed_group()

        if pending_segment_messages:
            flush_segment_messages()

        if frame_log_file is not None:
            try:
                frame_log_file.flush()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not flush frame classification log '{frame_log_path}': {exc}"
                ) from exc

        # Always leave the user with a final progress snapshot, even when the
        # last inference call did not land exactly on the configured interval.
        if progress_last_inference_calls != inference_calls:
            write_progress_line()
        finish_progress_line()
        elapsed = time.perf_counter() - started
        stats = {
            "frames_analysed": frame_count,
            "inference_calls": inference_calls,
            "segments_found": segment_count,
            "review_segments": review_segment_count,
            "review_paths": review_paths,
            "candidates_rejected": candidates_rejected,
            "suppressed_detections": suppressed_detections,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "stream_elapsed": elapsed,
            "frame_read_seconds": read_seconds,
            "rgb24_packing_seconds": packing_seconds,
            "resnet_inference_seconds": inference_seconds,
            "classifier_seconds": classifier_seconds,
        }
        return saved_paths, stats

    finally:
        if sys.exc_info()[0] is KeyboardInterrupt:
            print(
                "\n[Aborted] Cleaning up resources, please wait...",
                flush=True,
            )        
        # A processing error can leave the live progress line active. Clear it
        # before the outer exception handler prints the error on a fresh line.
        finish_progress_line()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if raw_stream is not None:
            raw_stream.close()
        if raw_read_fd is not None:
            os.close(raw_read_fd)
        if raw_write_fd is not None:
            os.close(raw_write_fd)
        if process is not None and process.stderr is not None:
            process.stderr.close()
        if stderr_thread is not None:
            stderr_thread.join(timeout=5.0)
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        if frame_log_file is not None and frame_log_temp_path is not None and frame_log_path is not None:
            try:
                finalize_frame_classification_log(
                    frame_log_file,
                    frame_log_temp_path,
                    frame_log_path,
                    class_names,
                    target_index,
                    used_class_indices,
                )
                frame_log_file = None
                frame_log_temp_path = None
            except Exception as exc:
                print(
                    f"[Warning] Could not finalize frame classification log '{frame_log_path}': {exc}. "
                    f"The partial frame data is retained in '{frame_log_temp_path}'.",
                    file=sys.stderr,
                    flush=True,
                )


def load_multiclass_parameters(model_path: Path):
    import numpy as np

    try:
        with np.load(model_path, allow_pickle=False) as data:
            mean = data["scaler_mean"].astype(np.float32)
            scale = data["scaler_scale"].astype(np.float32)
            coef = data["classifier_coef"].astype(np.float32)
            intercept = data["classifier_intercept"].astype(np.float32)
            class_names = data["class_names"].astype(str).tolist()
            class_indices = data["classifier_classes"].astype(np.int64)
            width = int(data["image_width"].item())
            height = int(data["image_height"].item())
            feature_count = int(data["feature_count"].item())
            class_count = int(data["number_of_classes"].item())
    except (KeyError, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load multiclass classifier: {exc}") from exc

    if class_count < 2:
        raise RuntimeError("This script requires a multiclass model with at least three classes.")
    if (len(class_names) != class_count or len(set(class_names)) != class_count
            or coef.shape != (class_count, feature_count)
            or intercept.shape != (class_count,)
            or mean.shape != (feature_count,)
            or scale.shape != (feature_count,)
            or not np.array_equal(class_indices, np.arange(class_count))
            or feature_count != FEATURE_COUNT):
        raise RuntimeError("Multiclass model metadata or parameter shapes are incompatible.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("Invalid feature scaling parameters.")
    return mean, scale, coef, intercept, class_names, width, height


def format_elapsed(seconds: float) -> str:
    """Format elapsed wall-clock time as HH:MM:SS.mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


class UppercaseUsageHelpFormatter(argparse.HelpFormatter):
    """Format the usage heading with a capital U."""

    def _format_usage(self, usage, actions, groups, prefix):
        return super()._format_usage(usage, actions, groups, "Usage: ")


def main() -> int:
    total_started = time.perf_counter()
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Detect target-class segments by continuously sampling video frames and "
            "classifying them with the RK3576 NPU."            
        ),
        usage="%(prog)s --target-class <name> [OPTIONS] video.ts",
        add_help=False,
        formatter_class=lambda prog: UppercaseUsageHelpFormatter(prog, max_help_position=28, width=118),
    )
    parser.add_argument(
        "video",
        type=Path,
        metavar="video.ts",
        help="Input video file.",
    )

    model_group = parser.add_argument_group("Model files")
    model_group.add_argument(
        "--classifier",
        type=Path,
        default=script_dir / ".." / "resnet" / "models" / "resnet50v2_logistic_classifier.npz",
        metavar="<path>",
        help="Multiclass classifier parameters (default: ../resnet/models/resnet50v2_logistic_classifier.npz).",
    )
    model_group.add_argument(
        "--extractor",
        type=Path,
        default=None,
        metavar="<path>",
        help="RKNN feature-extractor model (default: selected automatically from --batch-size).",
    )

    classification_group = parser.add_argument_group("Classification options")
    classification_group.add_argument(
        "--target-class",
        required=True,
        metavar="<name>",
        help="Class name to detect; must exist in the classifier.",
    )
    classification_group.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        metavar="<n>",
        help="Minimum target-class probability for automatic validation (default: 0.75).",
    )
    classification_group.add_argument(
        "--review-thr",
        type=float,
        default=0.0,
        metavar="<n>",
        help="Lower probability threshold for manual-review segments (default: 0; 0 disables review).",
    )
    classification_group.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="<n>",
        help="Images per inference batch (default: 4). Selects the RKNN model unless --extractor is set.",
    )

    validation_group = parser.add_argument_group("Validation options")
    validation_group.add_argument(
        "--window",
        type=int,
        default=VALIDATION_WINDOW,
        metavar="<n>",
        help="Samples in the transition-voting window (default: 5).",
    )
    validation_group.add_argument(
        "--votes",
        type=int,
        default=VALIDATION_VOTES,
        metavar="<n>",
        help="Matching-category samples required in the window (default: 4).",
    )
    validation_group.add_argument(
        "--streak",
        type=int,
        default=VALIDATION_STREAK,
        metavar="<n>",
        help="Consecutive matching samples required at the window end (default: 2).",
    )
    validation_group.add_argument(
        "--min-time",
        type=int,
        default=MIN_SEGMENT_TIME_MS,
        metavar="<ms>",
        help="Minimum continuous positive coverage in milliseconds (default: 1000).",
    )
    validation_group.add_argument(
        "--min-run",
        type=int,
        default=MIN_SEGMENT_RUN,
        metavar="<n>",
        help="Minimum consecutive matching-category samples to validate a segment (default: 3).",
    )

    video_group = parser.add_argument_group("Video and output options")
    video_group.add_argument(
        "--sampling",
        type=int,
        default=160,
        metavar="<n>",
        help="Minimum interval between sampled frames in milliseconds (default: 160; range: 5-1000).",
    )
    video_group.add_argument(
        "--deint",
        type=int,
        choices=(0, 1),
        default=1,
        metavar="<bool>",
        help="MPP hardware deinterlacing: 0=off, 1=on (default: 1).",
    )
    video_group.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="<path>",
        help="Working directory (default: folder named after the input video).",
    )
    video_group.add_argument(
        "--offset",
        type=Decimal,
        default=None,
        metavar="<n>",
        help="Manual timestamp offset in seconds (default: automatic).",
    )

    console_group = parser.add_argument_group("Console output options")
    console_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show all detection events and low-level diagnostic messages (default: concise output).",
    )
    console_group.add_argument(
        "--log-work",
        action="store_true",
        help="Save displayed console output to log-class-work.txt (excluding live progress updates).",
    )
    console_group.add_argument(
        "--log-frames",
        action="store_true",
        help="Save one classification-probability record per sampled frame to log-class-frames.txt.",
    )

    help_group = parser.add_argument_group("Help")
    help_group.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )
    args = parser.parse_args()

    # Validate numeric options before checking files or starting processing.
    if not 0.0 < args.threshold < 1.0:
        print("[Error] --threshold must be between 0 and 1.", file=sys.stderr)
        return 2
    if not (args.review_thr == 0.0 or 0.0 < args.review_thr < args.threshold):
        print(
            "[Error] --review-thr must be 0 (disabled) or greater than 0 and lower than --threshold.",
            file=sys.stderr,
        )
        return 2
    if not 5 <= args.sampling <= 1000:
        print("[Error] --sampling must be between 5 and 1000 milliseconds.", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("[Error] --batch-size must be a positive integer.", file=sys.stderr)
        return 2
    if args.window < 1:
        print("[Error] --window must be at least 1.", file=sys.stderr)
        return 2
    if not 1 <= args.votes <= args.window:
        print("[Error] --votes must be between 1 and --window.", file=sys.stderr)
        return 2
    if not 1 <= args.streak <= args.window:
        print("[Error] --streak must be between 1 and --window.", file=sys.stderr)
        return 2
    if args.min_time < 1:
        print("[Error] --min-time must be at least 1 millisecond.", file=sys.stderr)
        return 2
    if args.min_run < 1:
        print("[Error] --min-run must be at least 1.", file=sys.stderr)
        return 2

    input_file = args.video.expanduser().resolve()
    if not input_file.is_file():
        print(f"[Error] Input video not found: {input_file}", file=sys.stderr)
        return 2
    if not input_file.suffix:
        print(
            f"[Error] The source video must have a filename extension so its source_link can match it: {input_file}",
            file=sys.stderr,
        )
        return 2

    output_dir = (
        args.out.expanduser().resolve()
        if args.out is not None
        else input_file.with_name(input_file.stem)
    )
    if output_dir.exists():
        if not output_dir.is_dir():
            print(f"[Error] Output path exists and is not a directory: {output_dir}", file=sys.stderr)
            return 2
        if any(output_dir.iterdir()):
            print(
                f"[Error] Working directory already exists and is not empty: {output_dir}\n"
                "Choose another directory with --out or move/delete the existing directory first.",
                file=sys.stderr,
            )
            return 2

    if args.log_work:
        try:
            install_output_log(output_dir / "log-class-work.txt")
        except OSError as exc:
            print(f"[Error] Could not create output log: {exc}", file=sys.stderr)
            return 2

    classifier_path = args.classifier.expanduser().resolve()
    if not classifier_path.is_file():
        print(f"[Error] Classifier model not found: {classifier_path}", file=sys.stderr)
        return 2

    models_dir = script_dir.parent / "resnet" / "models"
    try:
        if args.extractor is not None:
            extractor_path = resolve_feature_model_path(
                models_dir,
                explicit_path=args.extractor,
            )
            # An explicitly selected model is authoritative. Its filename sets the batch size.
            args.batch_size = batch_size_from_model_path(extractor_path)
        else:
            args.batch_size = validate_batch_size(args.batch_size)
            extractor_path = resolve_feature_model_path(
                models_dir,
                batch_size=args.batch_size,
            )
    except ValueError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    if not extractor_path.is_file():
        print(
            f"[Error] RKNN feature model not found: {extractor_path}\n"
            "Place the required RKNN feature model in '../resnet/models' or specify --extractor.",
            file=sys.stderr,
        )
        return 2
    if args.batch_size < 1:
        print("[Error] The effective batch size must be a positive integer.", file=sys.stderr)
        return 2

    try:
        video_info = probe_video(input_file)
        (
            _format_start, _video_start, _calculated_offset, _audio_start,
            video_duration, _source_width, _source_height, source_fps,
        ) = video_info
        print_video_header(input_file, video_info, args.offset, verbose=args.verbose)

        model_parameters = load_multiclass_parameters(classifier_path)
        available_classes = model_parameters[4]
        if args.target_class not in available_classes:
            raise RuntimeError(
                f"Unknown target class '{args.target_class}'. "
                f"Available classes: {', '.join(available_classes)}"
            )

        print_model_header(
            available_classes,
            args.target_class,
            args.threshold,
            extractor_path,
            args.batch_size,
            verbose=args.verbose,
        )

        if args.verbose:
            print(
                f"[Sampling] Requested interval: {args.sampling} ms; MPP deinterlacing: {args.deint}; "
                f"source rate: {source_fps.numerator}/{source_fps.denominator} fps; "
                f"ResNetEngine input: {IMAGE_WIDTH}x{IMAGE_HEIGHT} RGB24"
            )
            review_category_text = (
                f"REVIEW >= {args.review_thr:.3f}; NONE below review threshold"
                if args.review_thr > 0.0
                else "REVIEW disabled; NONE below VALID threshold"
            )
            print(
                f"[Validator] Categories: VALID >= {args.threshold:.3f}; "
                f"{review_category_text}"
            )
            print(
                f"[Validator] Window: {args.window}; transition: {args.votes}/{args.window} "
                f"matching-category votes + {args.streak} consecutive; segment minimum: "
                f"{args.min_time} ms + {args.min_run} consecutive matching samples"
            )

        processing_started = time.perf_counter()
        validated_dir = output_dir / "seg_validated"
        segment_paths, class_all_stats = extract_classified_segments(
            input_file,
            output_dir,
            model_parameters,
            args.target_class,
            args.threshold,
            args.review_thr,
            extractor_path,
            args.batch_size,
            args.sampling,
            args.deint,
            args.window,
            args.votes,
            args.streak,
            args.min_time,
            args.min_run,
            args.offset,
            video_info,
            verbose=args.verbose,
            log_frames=args.log_frames,
        )
        print(
            "[FFmpeg/ResNet] Finished FFmpeg hardware decode and ResNet classification "
            f"in {format_elapsed(float(class_all_stats['stream_elapsed']))}.",
            flush=True,
        )
        print()
        create_project_artifacts(output_dir, input_file)
        processing_elapsed = time.perf_counter() - processing_started

        first_timestamp = class_all_stats["first_timestamp"]
        last_timestamp = class_all_stats["last_timestamp"]
        time_range = None
        if isinstance(first_timestamp, Decimal) and isinstance(last_timestamp, Decimal):
            time_range = (
                seconds_to_milliseconds(first_timestamp),
                seconds_to_milliseconds(last_timestamp),
            )
        print_summary(
            analysed_count=int(class_all_stats["frames_analysed"]),
            inference_calls=int(class_all_stats["inference_calls"]),
            validated_count=len(segment_paths),
            review_count=int(class_all_stats["review_segments"]),
            rejected_count=int(class_all_stats["candidates_rejected"]),
            suppressed_count=int(class_all_stats["suppressed_detections"]),
            target_class=args.target_class,
            time_range_ms=time_range,
            processing_seconds=processing_elapsed,
            duration=video_duration,
            frame_read_seconds=float(class_all_stats["frame_read_seconds"]),
            packing_seconds=float(class_all_stats["rgb24_packing_seconds"]),
            inference_seconds=float(class_all_stats["resnet_inference_seconds"]),
            classifier_seconds=float(class_all_stats["classifier_seconds"]),
            review_dir_created=(args.review_thr > 0.0 and (output_dir / "seg_review").is_dir()),
            validated_dir_created=validated_dir.is_dir(),
            work_log_created=args.log_work and (output_dir / "log-class-work.txt").is_file(),
            frame_log_created=args.log_frames and (output_dir / "log-class-frames.txt").is_file(),
        )
        print(
            "\n[Done] Target-class segment detection completed. "
            f"Total runtime: {format_elapsed(time.perf_counter() - total_started)}"
        )
        return 0

    except (RuntimeError, ValueError) as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        print(f"[Timing] Total runtime until error: {format_elapsed(time.perf_counter() - total_started)}")
        return 1
    except KeyboardInterrupt:
        print("\n[Error] Interrupted by user.", file=sys.stderr)
        print(f"[Timing] Total runtime until interruption: {format_elapsed(time.perf_counter() - total_started)}")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
