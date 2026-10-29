#!/usr/bin/env python3
"""Extract video scene cuts and classify presenter scenes using RK3576 NPU or optional CPU."""

from __future__ import annotations

print("[Start] Initializing used libraries...")

import argparse
import csv
import errno
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
FEATURE_COUNT = 2048
BATCH_SIZE = 16
TIMESTAMP_PATTERN = re.compile(r"^(\d+)-(\d{2})-(\d{2})-(\d{3})$")

DOSATTRIB_XATTR = "user.DOSATTRIB"
DOSATTRIB_HEADER = bytes.fromhex("0000050005000000")
DOSATTRIB_HIDDEN = 0x02
DOSATTRIB_ARCHIVE = 0x20

# Prefer the tested Rockchip-enabled FFmpeg build when it is installed. The scene
# extraction filter graph below requires an FFmpeg build with MPP and RGA support.
ROCKCHIP_FFMPEG = Path("/opt/rkmpp-stack/bin/ffmpeg")
ROCKCHIP_FFPROBE = Path("/opt/rkmpp-stack/bin/ffprobe")
FFMPEG_BINARY = str(ROCKCHIP_FFMPEG) if ROCKCHIP_FFMPEG.is_file() else "ffmpeg"
FFPROBE_BINARY = str(ROCKCHIP_FFPROBE) if ROCKCHIP_FFPROBE.is_file() else "ffprobe"


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


def probe_video(
    input_file: Path,
) -> tuple[Decimal, Decimal, Decimal, Decimal | None, Decimal | None, int, int]:
    """Return timeline data, duration, and source video dimensions."""
    command = [
        FFPROBE_BINARY, "-v", "error",
        "-show_entries",
        "format=start_time,duration:stream=index,codec_type,start_time,time_base,duration,width,height",
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

    offset = video_start - format_start
    return format_start, video_start, offset, audio_start, duration, video_width, video_height


def format_timestamp_stem(total_ms: int) -> str:
    """Format milliseconds as HH-MM-SS-mmm without a file extension."""
    if total_ms < 0:
        raise ValueError("A scene timestamp is negative.")
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}-{minutes:02d}-{whole_seconds:02d}-{milliseconds:03d}"


def format_image_timestamp(seconds: Decimal) -> str:
    total_ms = int((seconds * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return f"{format_timestamp_stem(total_ms)}.jpg"


def parse_timestamp_ms(path: Path) -> int:
    """Read the start time from either a timestamp-only or interval image filename."""
    start_text = path.stem.split(" - ", 1)[0]
    match = TIMESTAMP_PATTERN.fullmatch(start_text)
    if not match:
        raise ValueError(
            f"Image filename does not start with HH-MM-SS-mmm timestamp: {path.name}"
        )
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Image filename contains an invalid timestamp: {path.name}")
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds


def assign_scene_interval_filenames(scenes_dir: Path, duration: Decimal | None) -> list[Path]:
    """Rename extracted timestamp thumbnails to start/end interval filenames."""
    image_paths = sorted(
        (path for path in scenes_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
        key=parse_timestamp_ms,
    )
    if not image_paths:
        return []
    if duration is None or duration <= 0:
        raise RuntimeError(
            "Could not determine a valid video duration. Interval filenames require an end time "
            "for the final scene; no thumbnails were renamed."
        )

    duration_ms = int((duration * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    rename_plan: list[tuple[Path, Path]] = []
    for index, image_path in enumerate(image_paths):
        start_ms = parse_timestamp_ms(image_path)
        if index + 1 < len(image_paths):
            end_ms = parse_timestamp_ms(image_paths[index + 1])
        else:
            end_ms = duration_ms
        if end_ms <= start_ms:
            raise RuntimeError(
                f"Cannot determine a valid end time for scene '{image_path.name}': "
                f"start={format_timestamp_stem(start_ms)}, end={format_timestamp_stem(end_ms)}."
            )
        target = image_path.with_name(
            f"{format_timestamp_stem(start_ms)} - {format_timestamp_stem(end_ms)}.jpg"
        )
        if target.exists() and target != image_path:
            raise RuntimeError(f"Cannot rename scene image; target already exists: {target}")
        rename_plan.append((image_path, target))

    for source, target in rename_plan:
        source.rename(target)

    return sorted((target for _, target in rename_plan), key=parse_timestamp_ms)


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
    """Create a relative source symlink and project-local source deletion scripts."""
    if not input_file.suffix:
        raise RuntimeError(
            f"The source video has no filename extension: {input_file}. "
            "A source_link.<extension> symlink is required by encode_segments.py."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_link_name = f"source_link{input_file.suffix}"
    source_link = output_dir / source_link_name
    if source_link.exists() or source_link.is_symlink():
        raise RuntimeError(f"Source link path already exists in the project folder: {source_link}")

    relative_target = os.path.relpath(str(input_file), start=str(output_dir))
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
        f'rm -- "$t" {source_link_name} && echo "Deleted." || echo "Failed."',
        "",
    ])
    try:
        delete_sh_path.write_text(delete_sh_content, encoding="utf-8", newline="\n")
        delete_sh_path.chmod(0o775)
        set_samba_hidden_attribute(delete_sh_path)
    except OSError as exc:
        raise RuntimeError(f"Could not create or configure source deletion script '{delete_sh_path}': {exc}") from exc

    # The path is fixed in this project's BAT so the launcher can be run from Total Commander.
    remote_script_path = str(delete_sh_path.resolve())
    remote_script_argument = shell_single_quote(remote_script_path)
    delete_bat_content = f'@ssh -tt nanopi-encoder "bash {remote_script_argument}" & pause\n'
    try:
        # Use Windows CRLF line endings for cmd.exe and the requested executable/group mode.
        delete_bat_path.write_bytes(delete_bat_content.replace("\n", "\r\n").encode("utf-8"))
        delete_bat_path.chmod(0o775)
    except OSError as exc:
        raise RuntimeError(f"Could not create or configure Windows source deletion launcher '{delete_bat_path}': {exc}") from exc

    print(f"\n[Project] Created source symlink: {source_link.name} -> {relative_target}")
    print("[Project] Created source deletion launchers: delete_source.sh, delete_source.bat")

def format_segment_timestamp(total_ms: int) -> str:
    """Format a timestamp as HH:MM:SS.milliseconds."""
    if total_ms < 0:
        raise ValueError("Cannot format a negative timestamp.")
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def extract_scene_frames(
    input_file: Path,
    output_dir: Path,
    scene_threshold: float,
    image_width: int,
    manual_offset: Decimal | None,
) -> list[Path]:
    (
        format_start,
        video_start,
        calculated_offset,
        audio_start,
        duration,
        source_width,
        source_height,
    ) = probe_video(input_file)
    offset = manual_offset if manual_offset is not None else calculated_offset

    if image_width % 2 != 0:
        raise ValueError("Thumbnail width must be an even number when using RGA YUV420 output.")
    image_height = max(2, int(round((image_width * source_height / source_width) / 2.0)) * 2)

    print(f"[Video] {input_file}")
    print(f"[Timeline] format.start_time: {format_start} s")
    print(f"[Timeline] video.start_time:  {video_start} s")
    if audio_start is not None:
        print(f"[Timeline] audio.start_time:  {audio_start} s")
    print(f"[Timeline] Calculated offset: {calculated_offset:+.6f} s")
    print(f"[Timeline] Applied offset:    {offset:+.6f} s")
    if duration is not None:
        print(f"[Video] Duration: {duration} s")
    else:
        print("[Warning] Video duration is unavailable; a final open segment may have no end time.")

    if offset > 0:
        setpts_expr = f"PTS+{offset:.9f}/TB"
    elif offset < 0:
        setpts_expr = f"PTS-{abs(offset):.9f}/TB"
    else:
        setpts_expr = "PTS"

    # Keep scaling on the RGA hardware path. Convert DRM frames to a software-readable
    # pixel format before adjusting timestamps and running scdet, matching the tested
    # MPP -> RGA -> hwmap -> format -> setpts -> scdet filter order.
    filter_graph = (
        f"scale_rkrga=w={image_width}:h={image_height}:format=yuv420p:"
        "force_yuv=8bit:force_chroma=420p,"
        "hwmap=mode=read,"
        "format=yuv420p,"
        f"setpts={setpts_expr},"
        f"scdet=t={scene_threshold}:s=1,"
        "metadata=mode=print:key=lavfi.scd.time:file=scene_times_raw.txt"
    )

    # This FFmpeg build's software MJPEG encoder tries an unsupported swscale
    # conversion for frames coming from the MPP -> RGA -> hwmap path. Use the
    # Rockchip MPP MJPEG encoder instead; it accepts yuv420p directly and writes
    # one JPEG packet per selected frame (image2 sequence).
    encoder_result = run_checked([FFMPEG_BINARY, "-hide_banner", "-encoders"])
    encoder_listing = encoder_result.stdout + encoder_result.stderr
    if not re.search(r"\bmjpeg_rkmpp\b", encoder_listing):
        raise RuntimeError(
            f"FFmpeg binary '{FFMPEG_BINARY}' does not provide the mjpeg_rkmpp encoder. "
            "Check it with: /opt/rkmpp-stack/bin/ffmpeg -hide_banner -encoders | grep rkmpp"
        )

    # Let FFmpeg select the MPP decoder from the input codec. Keep deinterlacing
    # disabled during scene detection so the original frame cadence/timestamps are retained;
    # deinterlacing, when required, is handled later by encode_segments.py.
    with tempfile.TemporaryDirectory(prefix="ffmpeg_scene_extract_") as temp_name:
        temp_dir = Path(temp_name)
        command = [
            FFMPEG_BINARY, "-hide_banner", "-nostdin", "-loglevel", "error",
            "-hwaccel", "rkmpp", "-hwaccel_output_format", "drm_prime",
            "-deint", "0",
            "-i", str(input_file),
            "-map", "0:v:0",
            "-vf", filter_graph,
            "-an", "-fps_mode", "vfr", "-c:v", "mjpeg_rkmpp", "-qp_init", "80",
            "-f", "image2", "frame_%06d.jpg",
        ]
        # Using MPP decode, RGA scaling, and MPP JPEG encoding...
        print(f"[FFmpeg] Detecting scene-change and extracting {image_width}x{image_height} thumbnails...")
        run_checked(command, cwd=temp_dir)

        metadata_file = temp_dir / "scene_times_raw.txt"
        # A clip may contain no detected scene cuts, so an absent/empty metadata file
        # is not fatal: the explicit initial-frame thumbnail can still be classified.
        metadata_text = (
            metadata_file.read_text(encoding="utf-8", errors="replace")
            if metadata_file.is_file()
            else ""
        )
        time_strings = re.findall(r"lavfi\.scd\.time=([0-9]+(?:\.[0-9]+)?)", metadata_text)
        timestamps = [Decimal(value) for value in time_strings]
        frames = sorted(temp_dir.glob("frame_*.jpg"))
        if len(frames) != len(timestamps):
            raise RuntimeError(
                f"Frame/timestamp mismatch: {len(frames)} images but {len(timestamps)} timestamps. "
                "No images were copied to the output directory."
            )

        names = [format_image_timestamp(value) for value in timestamps]
        if len(set(names)) != len(names):
            raise RuntimeError("Two or more scene timestamps round to the same millisecond; duplicate filenames were prevented.")

        # scdet has no preceding frame to compare against at the start of a clip.
        # Extract the first valid decoded frame separately so a recording that begins
        # on the presenter can still produce an image for classification. Use the same
        # MPP -> RGA -> hwmap -> yuv420p -> MPP-JPEG path as the tested scene frames.
        initial_frame_file = temp_dir / "initial_frame.jpg"
        initial_filter_graph = (
            f"scale_rkrga=w={image_width}:h={image_height}:format=yuv420p:"
            "force_yuv=8bit:force_chroma=420p,"
            "hwmap=mode=read,"
            "format=yuv420p,"
            f"setpts={setpts_expr},"
            "showinfo"
        )
        initial_frame_command = [
            FFMPEG_BINARY, "-hide_banner", "-nostdin", "-loglevel", "info",
            "-hwaccel", "rkmpp", "-hwaccel_output_format", "drm_prime",
            "-deint", "0",
            "-i", str(input_file),
            "-map", "0:v:0",
            "-vf", initial_filter_graph,
            "-frames:v", "1", "-an", "-fps_mode", "vfr",
            "-c:v", "mjpeg_rkmpp", "-qp_init", "80",
            "-f", "image2", initial_frame_file.name,
        ]
        initial_frame_result = run_checked(initial_frame_command, cwd=temp_dir)
        if not initial_frame_file.is_file() or initial_frame_file.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not create a valid thumbnail for the first video frame.")

        # Read the first frame's actual PTS after the same setpts transformation used
        # before scdet. This avoids guessing how FFmpeg normalized the transport-stream
        # and video-stream start times, and keeps the initial thumbnail on the same
        # timestamp timeline as the scene-change thumbnails.
        initial_frame_log = initial_frame_result.stderr + "\n" + initial_frame_result.stdout
        first_pts_match = re.search(
            r"\bpts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            initial_frame_log,
        )
        if first_pts_match is None:
            raise RuntimeError("Could not determine the initial frame timestamp from FFmpeg showinfo output.")
        initial_frame_time = Decimal(first_pts_match.group(1))
        initial_frame_name = format_image_timestamp(initial_frame_time)
        initial_frame_added = initial_frame_name not in names

        scenes_dir = output_dir / "scenes_all_other"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for source, name in zip(frames, names):
            shutil.move(str(source), str(scenes_dir / name))

        if initial_frame_added:
            shutil.move(str(initial_frame_file), str(scenes_dir / initial_frame_name))
        else:
            # A scene-change thumbnail already represents this millisecond; don't
            # create duplicate timestamps or classify the same instant twice.
            initial_frame_file.unlink(missing_ok=True)

    scenes_dir = output_dir / "scenes_all_other"
    image_paths = assign_scene_interval_filenames(scenes_dir, duration)
    print(f"[FFmpeg] Extracted {len(image_paths)} thumbnails with start/end times into: {scenes_dir}")
    return image_paths


def load_scene_image(path: Path, width: int, height: int) -> "np.ndarray":
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not load image '{path}': {exc}") from exc


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

    if class_count < 3:
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


def call_with_filtered_native_logs(function, suppressed_fragments: tuple[str, ...]):
    """Run a native RKNN call, hiding only explicitly identified harmless log lines."""
    sys.stdout.flush()
    sys.stderr.flush()
    result = None
    error: BaseException | None = None

    # RKNN logs are emitted by native code, so redirect OS file descriptors rather
    # than Python's sys.stdout/sys.stderr objects. Use a temporary file to avoid
    # filling a pipe while a large batch is processed.
    with tempfile.TemporaryFile(mode="w+b") as log_file:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            try:
                result = function()
            except BaseException as exc:
                error = exc
            finally:
                # Python's flush() does not flush the C stdio buffers used by
                # librknnrt. If those buffers are left pending, native messages can
                # be written only after fd 1/2 have been restored, bypassing the
                # filter and appearing at the very end of the program. Flush all C
                # stdio streams while they still point at the capture file.
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    import ctypes
                    libc = ctypes.CDLL(None)
                    libc.fflush.argtypes = [ctypes.c_void_p]
                    libc.fflush.restype = ctypes.c_int
                    libc.fflush(None)
                except (AttributeError, OSError):
                    # Keep output filtering best-effort if C stdio flushing is
                    # unavailable on an unusual Python runtime.
                    pass
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
        finally:
            os.close(saved_stdout)
            os.close(saved_stderr)

        log_file.seek(0)
        for raw_line in log_file:
            line = raw_line.decode("utf-8", errors="replace")
            if any(fragment in line for fragment in suppressed_fragments):
                continue
            # Keep every native diagnostic except the exact known harmless warnings.
            sys.stderr.write(line)
        sys.stderr.flush()

    if error is not None:
        raise error.with_traceback(error.__traceback__)
    return result


def classify_images(
    images: list[Path],
    model_parameters,
    target_class: str,
    threshold: float,
    batch_size: int,
    *,
    resnet_cpu: bool = False,
    feature_model_path: Path | None = None,
) -> tuple[list[int], list[float], list[str]]:
    """Classify images with RK3576 NPU by default, or TensorFlow CPU on request."""
    import numpy as np

    mean, scale, coef, intercept, class_names, width, height = model_parameters
    try:
        target_index = class_names.index(target_class)
    except ValueError as exc:
        raise RuntimeError(
            f"Target class '{target_class}' is not present in the classifier model."
        ) from exc

    if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
        print(
            f"[Warning] Classifier metadata uses {width}x{height}; extracted thumbnails "
            f"are configured for {IMAGE_WIDTH}x{IMAGE_HEIGHT}. Images will be resized."
        )
    print(f"[Model] Target class: {target_class}; probability threshold: {threshold:.3f}")

    target_probabilities: list[float] = []
    predicted_classes: list[str] = []

    def classify_features(features: "np.ndarray") -> "np.ndarray":
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise RuntimeError(f"Unexpected feature output shape: {features.shape}")
        if not np.isfinite(features).all():
            raise RuntimeError("Feature extractor returned NaN or infinite values.")
        logits = ((features - mean) / scale) @ coef.T + intercept
        # Numerically stable softmax, matching sklearn's multinomial predict_proba.
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Classifier produced NaN or infinite probabilities.")
        return probabilities

    if resnet_cpu:
        # Optional reference/fallback path: keep the original TensorFlow CPU inference.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50V2
            from tensorflow.keras.applications.resnet_v2 import preprocess_input
        except ImportError as exc:
            raise RuntimeError(
                "CPU classification was requested with --resnet-cpu, but TensorFlow is not "
                "available in this Python environment. Activate the TensorFlow environment first."
            ) from exc

        print("[Backend] TensorFlow ResNet50V2 on CPU (--resnet-cpu).")
        print("[Model] Loading frozen ResNet50V2 on CPU...")
        model = None
        try:
            model = ResNet50V2(
                include_top=False,
                weights="imagenet",
                pooling="avg",
                input_shape=(height, width, 3),
            )
            model.trainable = False

            for start in range(0, len(images), batch_size):
                batch_paths = images[start:start + batch_size]
                batch = np.stack(
                    [load_scene_image(image_path, width, height) for image_path in batch_paths],
                    axis=0,
                )
                features = model(preprocess_input(batch), training=False).numpy().astype(np.float32)
                probabilities = classify_features(features)
                target_probabilities.extend(probabilities[:, target_index].astype(float).tolist())
                predicted_classes.extend(
                    class_names[int(index)] for index in np.argmax(probabilities, axis=1)
                )
                print(
                    f"[Classify] Processed {min(start + len(batch_paths), len(images))}/{len(images)} images"
                )
        finally:
            if model is not None:
                del model
            tf.keras.backend.clear_session()
    else:
        # Default path: ResNet50V2 feature extraction is accelerated by the RK3576 NPU.
        if feature_model_path is None:
            raise RuntimeError("No RKNN feature model path was provided.")
        if not feature_model_path.is_file():
            raise RuntimeError(
                f"RKNN feature model not found: {feature_model_path}\n"
                "Expected resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn."
            )
        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            raise RuntimeError(
                f"The tested RKNN feature model has a fixed input size of "
                f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}, but classifier metadata specifies {width}x{height}. "
                "Use the matching resnet50v2_logistic_classifier.npz or select --resnet-cpu."
            )
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "RKNN NPU inference is the default, but rknnlite is not available in this Python "
                "environment. Activate the RKNN Lite environment or use --resnet-cpu."
            ) from exc

        print("[Backend] RK3576 NPU via RKNN Lite 2; CPU fallback is available with --resnet-cpu.")
        print(f"[Model] Loading RKNN feature model: {feature_model_path}")
        rknn = None
        try:
            rknn = RKNNLite()
            load_result = rknn.load_rknn(str(feature_model_path))
            if load_result != 0:
                raise RuntimeError(f"RKNNLite.load_rknn() failed with code {load_result}.")
            init_result = call_with_filtered_native_logs(
                rknn.init_runtime,
                (
                    "query RKNN_QUERY_INPUT_DYNAMIC_RANGE error",
                    "Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID",
                ),
            )
            if init_result != 0:
                raise RuntimeError(
                    f"RKNNLite.init_runtime() failed with code {init_result}. "
                    "Check that the RKNN Lite runtime and NPU driver are compatible."
                )

            for index, image_path in enumerate(images, start=1):
                # ResNet50V2 preprocess_input maps RGB [0, 255] to [-1, 1].
                # The exported RKNN graph has a fixed NCHW input [1, 3, 180, 320].
                image = load_scene_image(image_path, width, height)
                input_nhwc = image / np.float32(127.5) - np.float32(1.0)
                input_nchw = np.transpose(input_nhwc, (2, 0, 1))[np.newaxis, ...]
                input_nchw = np.ascontiguousarray(input_nchw, dtype=np.float32)
                try:
                    outputs = call_with_filtered_native_logs(
                        lambda: rknn.inference(inputs=[input_nchw], data_format=["nchw"]),
                        ("need NHWC data format, but NCHW set",),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"RKNN inference failed for image '{image_path.name}': {exc}"
                    ) from exc
                if not outputs or outputs[0] is None:
                    raise RuntimeError(
                        f"RKNN inference returned no feature tensor for '{image_path.name}'."
                    )
                features = np.asarray(outputs[0], dtype=np.float32).reshape(1, -1)
                probabilities = classify_features(features)
                target_probabilities.append(float(probabilities[0, target_index]))
                predicted_classes.append(class_names[int(np.argmax(probabilities[0]))])

                if index % batch_size == 0 or index == len(images):
                    print(f"[Classify] Processed {index}/{len(images)} images")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"RKNN Lite setup or inference failed: {exc}") from exc
        finally:
            if rknn is not None:
                try:
                    rknn.release()
                except Exception:
                    pass

    labels = [
        int(probability >= threshold and predicted == target_class)
        for probability, predicted in zip(target_probabilities, predicted_classes)
    ]
    return labels, target_probabilities, predicted_classes

def write_outputs(
    output_dir: Path,
    image_paths: list[Path],
    timestamps_ms: list[int],
    labels: list[int],
    probabilities: list[float],
    target_class: str,
    predicted_classes: list[str],
    candidate_threshold: float,
    threshold: float,
    csv_predict: bool,
) -> tuple[int, int]:
    import csv
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    validated_dir = output_dir / "scenes_validated"
    validated_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "scenes_to_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_flags = [
        int(candidate_threshold > 0 and candidate_threshold <= probability < threshold and not label)
        for label, probability in zip(labels, probabilities)
    ]
    records: list[tuple[str, str, str, float, str, int, int]] = []
    for path, timestamp_ms, label, probability, predicted, review_flag in zip(
        image_paths, timestamps_ms, labels, probabilities, predicted_classes, review_flags
    ):
        timestamp_text = format_segment_timestamp(timestamp_ms)
        probability_value = float(probability)
        records.append((
            path.name, timestamp_text, target_class, probability_value,
            predicted, label, review_flag,
        ))

    # CSV is optional; the formatted Excel report is always generated.
    if csv_predict:
        predictions_path = output_dir / "scene_predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Keep CSV headers machine-friendly for downstream data analysis.
            writer.writerow([
                "image", "timestamp", "target_class", "target_probability",
                "predicted_class", "validated", "review_candidate",
            ])
            for image_name, timestamp_text, target, probability, predicted, validated, review_flag in records:
                writer.writerow([
                    image_name, timestamp_text, target,
                    f"{probability:.6f}", predicted, validated, review_flag,
                ])

    # Create a separate, human-friendly Excel view while retaining the interoperable CSV.
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Scene predictions"
    headers = ["Image", "Timestamp", "Target", "Probability", "Predicted", "Validated", "Review"]
    worksheet.append(headers)
    for image_name, timestamp_text, target, probability, predicted, selected, review_flag in records:
        worksheet.append([
            image_name,
            timestamp_text,
            target,
            probability,
            predicted,
            "Yes" if selected else "No",
            "Yes" if review_flag else "No",
        ])

    # Distinct header, subtle cell separators, comfortable spacing for 12pt text, and no gridlines.
    header_fill = PatternFill("solid", fgColor="24476B")
    header_font = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=12, color="243447")
    separator = Side(style="thin", color="D8E1EB")
    header_separator = Side(style="thin", color="FFFFFF")
    body_border = Border(left=separator, right=separator, top=separator, bottom=separator)
    header_border = Border(
        left=header_separator, right=header_separator,
        top=header_separator, bottom=header_separator,
    )
    left_alignment = Alignment(horizontal="left", vertical="center", indent=1)
    center_alignment = Alignment(horizontal="center", vertical="center", indent=1)
    validated_fill = PatternFill("solid", fgColor="E2F0D9")
    validated_font = Font(name="Calibri", size=12, color="237245", bold=True)
    review_fill = PatternFill("solid", fgColor="FCE4D6")
    review_font = Font(name="Calibri", size=12, color="9C4A0F", bold=True)

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = header_border

    for row_idx in range(2, worksheet.max_row + 1):
        worksheet.row_dimensions[row_idx].height = 24
        for col_idx in range(1, worksheet.max_column + 1):
            cell = worksheet.cell(row=row_idx, column=col_idx)
            cell.font = body_font
            cell.alignment = left_alignment if col_idx == 1 else center_alignment
            cell.border = body_border
            if row_idx % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F5F8FC")
            # Apply actual cell fills for status markers; do not rely on conditional formatting.
            if col_idx == 6 and cell.value == "Yes":
                cell.fill = validated_fill
                cell.font = validated_font
            elif col_idx == 7 and cell.value == "Yes":
                cell.fill = review_fill
                cell.font = review_font
        # Store timestamps as text and probabilities as numeric values with fixed precision.
        worksheet.cell(row=row_idx, column=2).number_format = "@"
        worksheet.cell(row=row_idx, column=4).number_format = "0.000000"

    worksheet.row_dimensions[1].height = 32
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.sheet_view.showGridLines = False
    worksheet.sheet_view.zoomScale = 90
    worksheet.sheet_properties.pageSetUpPr.fitToPage = True
    worksheet.page_setup.fitToWidth = 1
    worksheet.page_setup.fitToHeight = 0
    worksheet.page_setup.orientation = "landscape"
    worksheet.print_title_rows = "1:1"

    # Fit widths to content with extra room for the filter dropdowns in the headers.
    # Explicit minimums keep short headers (Validated/Review) readable in Excel.
    minimum_widths = {1: 24, 2: 22, 3: 22, 4: 20, 5: 23, 6: 19, 7: 18}
    maximum_widths = {1: 64, 2: 24, 3: 26, 4: 22, 5: 28, 6: 21, 7: 20}
    for col_idx in range(1, worksheet.max_column + 1):
        values = [str(worksheet.cell(row=row_idx, column=col_idx).value or "")
                  for row_idx in range(1, worksheet.max_row + 1)]
        desired_width = max(len(value) for value in values) + (5 if col_idx == 1 else 6)
        minimum_width = max(len(headers[col_idx - 1]) + 5, minimum_widths[col_idx])
        worksheet.column_dimensions[get_column_letter(col_idx)].width = min(
            max(desired_width, minimum_width), maximum_widths[col_idx]
        )

    predictions_xlsx_path = output_dir / "scene_predictions.xlsx"
    try:
        workbook.save(predictions_xlsx_path)
    except OSError as exc:
        raise RuntimeError(f"Could not save Excel predictions report '{predictions_xlsx_path}': {exc}") from exc

    selected_count = sum(labels)
    review_count = sum(review_flags)

    for path, label, review_flag in zip(image_paths, labels, review_flags):
        if label == 1:
            shutil.move(str(path), str(validated_dir / path.name))
        elif review_flag:
            shutil.move(str(path), str(review_dir / path.name))

    return selected_count, review_count


def format_elapsed(seconds: float) -> str:
    """Format elapsed wall-clock time as HH:MM:SS.mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def main() -> int:
    total_started = time.perf_counter()
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Detect scene changes, classify presenter thumbnails, and organize scenes "
            "with their start and end times."
        ),
        add_help=False,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=100),
    )
    parser.add_argument("video", help="Input video file, for example news.ts")

    model_group = parser.add_argument_group("Model files")
    model_group.add_argument(
        "--classifier",
        type=Path,
        default=script_dir / ".." / "resnet" / "models" / "resnet50v2_logistic_classifier.npz",
        help=(
            "Multiclass logistic classifier parameters (default: "
            "../resnet/models/resnet50v2_logistic_classifier.npz beside this script)."
        ),
    )
    model_group.add_argument(
        "--extractor",
        type=Path,
        default=script_dir / ".." / "resnet" / "models" / "resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn",
        help=(
            "RKNN feature-extractor model for RK3576 NPU (default: "
            "../resnet/models/resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn beside this script)."
        ),
    )

    classification_group = parser.add_argument_group("Classification options")
    classification_group.add_argument(
        "--target-class",
        required=True,
        help="Class to detect; must match a class in the classifier model.",
    )
    classification_group.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Target-class probability threshold from 0 to 1 (default: 0.65).",
    )
    classification_group.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.0,
        help=(
            "Lower probability bound for images moved to scenes_to_review; 0 disables review-candidate "
            "classification (default: 0.0). Must not exceed --threshold."
        ),
    )

    report_group = parser.add_argument_group("Report options")
    report_group.add_argument(
        "--csv-predict",
        action="store_true",
        help="Also generate the machine-readable scene_predictions.csv report (disabled by default).",
    )

    video_group = parser.add_argument_group("Video and output options")
    video_group.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Working directory (default: <video_stem> beside the input video).",
    )
    video_group.add_argument(
        "--scene-threshold",
        type=float,
        default=10.0,
        help="FFmpeg scdet sensitivity threshold (default: 10).",
    )
    video_group.add_argument(
        "--width",
        type=int,
        default=IMAGE_WIDTH,
        help="Extracted thumbnail width in pixels (default: 320).",
    )
    video_group.add_argument(
        "--offset",
        type=Decimal,
        default=None,
        help="Manually override the timestamp offset in seconds; normally calculated with ffprobe.",
    )

    inference_group = parser.add_argument_group("Inference options")
    inference_group.add_argument(
        "--resnet-cpu",
        action="store_true",
        help="Use TensorFlow ResNet50V2 on CPU instead of the default RK3576 NPU backend.",
    )
    inference_group.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Images per TensorFlow CPU batch; on NPU, controls how often classification "
            "progress is printed (default: 16)."
        ),
    )

    help_group = parser.add_argument_group("Help")
    help_group.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )
    args = parser.parse_args()

    # Validate thresholds before checking files or beginning any processing.
    if not 0.0 < args.threshold < 1.0:
        print("[Error] --threshold must be between 0 and 1.", file=sys.stderr)
        return 2
    if not 0.0 <= args.candidate_threshold <= 1.0:
        print("[Error] --candidate-threshold must be between 0 and 1.", file=sys.stderr)
        return 2
    if args.candidate_threshold > args.threshold:
        print(
            "[Error] --candidate-threshold must not be greater than --threshold.",
            file=sys.stderr,
        )
        return 2

    input_file = Path(args.video).expanduser().resolve()
    output_dir = args.out.expanduser().resolve() if args.out else input_file.with_name(input_file.stem)
    classifier_path = args.classifier.expanduser().resolve()
    extractor_path = args.extractor.expanduser().resolve()

    if not input_file.is_file():
        print(f"[Error] Input video not found: {input_file}", file=sys.stderr)
        return 2
    if not input_file.suffix:
        print(
            f"[Error] The source video must have a filename extension so its source_link can match it: {input_file}",
            file=sys.stderr,
        )
        return 2
    if not classifier_path.is_file():
        print(f"[Error] Classifier model not found: {classifier_path}", file=sys.stderr)
        return 2
    if not args.resnet_cpu and not extractor_path.is_file():
        print(
            f"[Error] RKNN feature model not found: {extractor_path}\n"
            "Place resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn in the '../resnet/models' directory, "
            "specify --extractor, or use --resnet-cpu.",
            file=sys.stderr,
        )
        return 2
    try:
        import openpyxl  # noqa: F401 - required to generate the formatted Excel report
    except ImportError:
        print(
            "[Error] Excel report generation requires openpyxl. Install it in this Python environment with:\n"
            "python -m pip install openpyxl",
            file=sys.stderr,
        )
        return 2

    if not 0.0 <= args.scene_threshold <= 100.0:
        print("[Error] --scene-threshold must be between 0 and 100.", file=sys.stderr)
        return 2
    if args.width < 1 or args.batch_size < 1:
        print("[Error] --width and --batch-size must be positive integers.", file=sys.stderr)
        return 2
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

    try:
        model_parameters = load_multiclass_parameters(classifier_path)
        available_classes = model_parameters[4]
        if args.target_class not in available_classes:
            raise RuntimeError(
                f"Unknown target class '{args.target_class}'. "
                f"Available classes: {', '.join(available_classes)}"
            )
        print(f"\n[Model] Available classes: {', '.join(available_classes)}")
        extraction_started = time.perf_counter()
        image_paths = extract_scene_frames(
            input_file,
            output_dir,
            args.scene_threshold,
            args.width,
            args.offset,
        )
        extraction_elapsed = time.perf_counter() - extraction_started
        print(f"[Timing] Scene detection + thumbnail extraction: {format_elapsed(extraction_elapsed)}")

        timestamps_ms = [parse_timestamp_ms(path) for path in image_paths]
        if len(set(timestamps_ms)) != len(timestamps_ms):
            raise RuntimeError("Duplicate scene timestamps were found after extraction.")

        print("")

        classification_started = time.perf_counter()
        labels, probabilities, predicted_classes = classify_images(
            image_paths,
            model_parameters,
            args.target_class,
            args.threshold,
            args.batch_size,
            resnet_cpu=args.resnet_cpu,
            feature_model_path=extractor_path,
        )
        classification_elapsed = time.perf_counter() - classification_started
        backend_name = "CPU" if args.resnet_cpu else "RK3576 NPU"
        print(f"[Timing] ResNet50V2 classification ({backend_name}): {format_elapsed(classification_elapsed)}")

        selected_count, review_count = write_outputs(
            output_dir,
            image_paths,
            timestamps_ms,
            labels,
            probabilities,
            args.target_class,
            predicted_classes,
            args.candidate_threshold,
            args.threshold,
            args.csv_predict,
        )
        create_project_artifacts(output_dir, input_file)

        print(f"\n[Result] Validated as {args.target_class}: {selected_count}/{len(image_paths)} images.")
        print(f"[Result] Review candidates: {review_count} images.")
        print("[Done  ] Scene extraction and presenter classification completed.")
        print(f"[Timing] Total runtime: {format_elapsed(time.perf_counter() - total_started)}")
        return 0

    except (RuntimeError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        print(f"[Timing] Total runtime until error: {format_elapsed(time.perf_counter() - total_started)}")
        return 1
    except KeyboardInterrupt:
        print("[Error] Interrupted by user.", file=sys.stderr)
        print(f"[Timing] Total runtime until interruption: {format_elapsed(time.perf_counter() - total_started)}")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
