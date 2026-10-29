#!/usr/bin/env python3
"""Validate the RK3576 ResNet50V2 feature extractor on the target NanoPi."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np


FEATURE_COUNT = 2048
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
MIN_COSINE_SIMILARITY = 0.95


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILENAME = "resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn"
VALIDATION_FILENAME = "resnet50v2_rknn_validation.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "resnet" / "artifacts"
CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME
TARGET_MODEL_PATH = PROJECT_ROOT / "resnet" / "models" / MODEL_FILENAME
BACKUP_MODEL_PATH = TARGET_MODEL_PATH.with_name(TARGET_MODEL_PATH.name + ".bak")
VALIDATION_PATH = ARTIFACTS_DIR / VALIDATION_FILENAME


def call_with_filtered_native_logs(function, suppressed_fragments: tuple[str, ...]):
    """Run an RKNN call, suppressing only known harmless native log messages."""
    sys.stdout.flush()
    sys.stderr.flush()
    result = None
    error: BaseException | None = None

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
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    libc = ctypes.CDLL(None)
                    libc.fflush.argtypes = [ctypes.c_void_p]
                    libc.fflush.restype = ctypes.c_int
                    libc.fflush(None)
                except (AttributeError, OSError):
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
            sys.stderr.write(line)
        sys.stderr.flush()

    if error is not None:
        raise error.with_traceback(error.__traceback__)
    return result


def load_validation_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate the NCHW inputs and ONNX reference features."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"inputs_nchw", "reference_features", "sample_names", "input_layout"}
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(
                    "Validation archive is missing required fields: "
                    + ", ".join(sorted(missing))
                    + ". Regenerate it with make_features_nchw.py."
                )
            layout = str(archive["input_layout"].item())
            if layout != "NCHW":
                raise RuntimeError(f"Validation archive declares layout {layout!r}; expected 'NCHW'.")
            inputs = np.asarray(archive["inputs_nchw"], dtype=np.float32)
            references = np.asarray(archive["reference_features"], dtype=np.float32)
            names = archive["sample_names"].astype(str).tolist()
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Could not read validation archive '{path}': {exc}") from exc

    if inputs.ndim != 4 or inputs.shape[1:] != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(
            f"Unexpected input shape {inputs.shape}; expected (N, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )
    if references.shape != (len(inputs), FEATURE_COUNT):
        raise RuntimeError(
            f"Unexpected reference shape {references.shape}; expected ({len(inputs)}, {FEATURE_COUNT})."
        )
    if not len(inputs):
        raise RuntimeError("Validation archive contains no samples.")
    if len(names) != len(inputs):
        raise RuntimeError("Validation archive contains inconsistent sample names and inputs.")
    if not np.isfinite(inputs).all() or not np.isfinite(references).all():
        raise RuntimeError("Validation archive contains NaN or infinite values.")
    return inputs, references, names


def cosine_similarity(reference: np.ndarray, result: np.ndarray) -> float:
    ref64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    out64 = np.asarray(result, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(ref64) * np.linalg.norm(out64))
    if denominator == 0.0:
        return 1.0 if np.array_equal(ref64, out64) else 0.0
    return float(np.dot(ref64, out64) / denominator)


def format_duration(seconds: float) -> str:
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    centiseconds = int((seconds - whole_seconds) * 100)
    return f"{hours:02d}:{minutes:02d}:{remainder % 60:02d}.{centiseconds:02d}"

def promote_validated_model() -> None:
    """Promote the validated candidate and remove its temporary validation data."""
    if not TARGET_MODEL_PATH.parent.is_dir():
        raise RuntimeError(f"Model directory does not exist: {TARGET_MODEL_PATH.parent}")
    if not CANDIDATE_MODEL_PATH.is_file() or CANDIDATE_MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("The validated candidate model is no longer available for promotion.")

    backup_created = False
    if TARGET_MODEL_PATH.exists() and not (TARGET_MODEL_PATH.is_file() or TARGET_MODEL_PATH.is_symlink()):
        raise RuntimeError(f"Installed model path is not a file: {TARGET_MODEL_PATH}")
    if TARGET_MODEL_PATH.exists():
        if BACKUP_MODEL_PATH.exists() or BACKUP_MODEL_PATH.is_symlink():
            if BACKUP_MODEL_PATH.is_dir() and not BACKUP_MODEL_PATH.is_symlink():
                raise RuntimeError(f"Backup path exists as a directory: {BACKUP_MODEL_PATH}")
            BACKUP_MODEL_PATH.unlink()
        os.replace(TARGET_MODEL_PATH, BACKUP_MODEL_PATH)
        backup_created = True

    try:
        os.replace(CANDIDATE_MODEL_PATH, TARGET_MODEL_PATH)
    except OSError:
        # Restore the current model if promotion fails after it was backed up.
        if backup_created and BACKUP_MODEL_PATH.exists() and not TARGET_MODEL_PATH.exists():
            os.replace(BACKUP_MODEL_PATH, TARGET_MODEL_PATH)
        raise

    print(f"[Promote] Validated model installed at: {TARGET_MODEL_PATH}")
    if VALIDATION_PATH.exists():
        try:
            VALIDATION_PATH.unlink()
        except OSError as exc:
            print(
                f"[Warning] Model was promoted, but validation helper could not be removed: "
                f"{VALIDATION_PATH} ({exc})",
                file=sys.stderr,
            )
    print("[Done] Validation passed and model promotion completed.")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("=" * 68)
    print("RK3576 FEATURE EXTRACTOR - DEVICE VALIDATION")
    print("=" * 68)
    print(f"[Candidate ] {CANDIDATE_MODEL_PATH}")
    print(f"[Validation] {VALIDATION_PATH}")
    print(f"[Install to] {TARGET_MODEL_PATH}")

    candidate_exists = CANDIDATE_MODEL_PATH.is_file()
    validation_exists = VALIDATION_PATH.is_file()
    if not candidate_exists and not validation_exists:
        print("\n[Info] No candidate model or validation data found in artifacts.")
        print("[Done] Nothing to validate; the installed model was left unchanged.")
        return 0
    if not candidate_exists:
        if validation_exists:
            print("\n[Info] Validation data exists, but no candidate RKNN model is present.")
            print("[Done] Nothing to validate; no model was promoted and existing files were left unchanged.")
        else:
            print("\n[Info] No candidate RKNN model or validation data found in artifacts.")
            print("[Done] Nothing to validate; the installed model was left unchanged.")
        return 0
    if not validation_exists:
        raise FileNotFoundError(
            f"Validation archive not found: {VALIDATION_PATH}. "
            "The candidate will not be promoted without its matching validation data."
        )
    if CANDIDATE_MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(f"Candidate RKNN model is empty: {CANDIDATE_MODEL_PATH}")
    if not TARGET_MODEL_PATH.parent.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {TARGET_MODEL_PATH.parent}")

    print("\n[1/3] Checking validation data...", flush=True)
    inputs, references, sample_names = load_validation_data(VALIDATION_PATH)
    print(f"  Samples:        {len(inputs)}")
    print(f"  Input:          [1, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}] float32 NCHW")
    print(f"  Output:         [1, {FEATURE_COUNT}] float32 features")
    print(f"  Minimum cosine: {MIN_COSINE_SIMILARITY:.3f} per sample")

    try:
        # RKNN Lite 2.3.2 may emit this known deprecation warning with setuptools 80.x.
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "RKNN Lite 2 is not available in this environment. Activate the NanoPi RKNN Lite environment."
        ) from exc

    print("\n[2/3] Initializing RKNN Lite on the device...", flush=True)
    runtime = RKNNLite()
    started = time.perf_counter()
    try:
        ret = runtime.load_rknn(str(CANDIDATE_MODEL_PATH))
        if ret != 0:
            raise RuntimeError(f"Could not load the RKNN model (return code {ret}).")

        ret = call_with_filtered_native_logs(
            runtime.init_runtime,
            (
                "query RKNN_QUERY_INPUT_DYNAMIC_RANGE error",
                "Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID",
            ),
        )
        if ret not in (None, 0):
            raise RuntimeError(f"RKNN Lite runtime initialization failed (return code {ret}).")

        print("\n[3/3] Running NPU inference and comparing with ONNX references...", flush=True)
        predicted_features: list[np.ndarray] = []
        cosines: list[float] = []
        failed: list[str] = []

        for index, (sample, reference, name) in enumerate(
            zip(inputs, references, sample_names), start=1
        ):
            tensor = np.ascontiguousarray(sample[np.newaxis, ...], dtype=np.float32)
            outputs = call_with_filtered_native_logs(
                lambda tensor=tensor: runtime.inference(
                    inputs=[tensor], data_format=["nchw"]
                ),
                ("need NHWC data format, but NCHW set",),
            )
            if not outputs or outputs[0] is None:
                raise RuntimeError(f"No feature output returned for sample '{name}'.")
            result = np.asarray(outputs[0], dtype=np.float32).reshape(1, -1)
            if result.shape != (1, FEATURE_COUNT):
                raise RuntimeError(
                    f"Sample '{name}' returned shape {result.shape}; expected (1, {FEATURE_COUNT})."
                )
            if not np.isfinite(result).all():
                raise RuntimeError(f"Sample '{name}' returned NaN or infinite features.")

            predicted_features.append(result[0])
            cosine = cosine_similarity(reference, result)
            cosines.append(cosine)
            passed = cosine >= MIN_COSINE_SIMILARITY
            if not passed:
                failed.append(name)
            print(f"  {index:>2}/{len(inputs)} {name}: cosine={cosine:.6f} [{'PASS' if passed else 'FAIL'}]")

        predicted = np.stack(predicted_features).astype(np.float32, copy=False)
        difference = predicted.astype(np.float64) - references.astype(np.float64)
        mae = float(np.mean(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        max_abs = float(np.max(np.abs(difference)))
        mean_cosine = float(np.mean(cosines))
        min_cosine = float(np.min(cosines))
        elapsed = time.perf_counter() - started

        print("\n[Results]")
        print(f"  Mean absolute error:   {mae:.6f}")
        print(f"  RMSE:                  {rmse:.6f}")
        print(f"  Maximum absolute diff: {max_abs:.6f}")
        print(f"  Mean cosine:           {mean_cosine:.6f}")
        print(f"  Minimum cosine:        {min_cosine:.6f}")

        if failed:
            raise RuntimeError(
                f"Device validation failed for {len(failed)}/{len(inputs)} sample(s): "
                + ", ".join(failed)
                + f". Every sample must have cosine similarity >= {MIN_COSINE_SIMILARITY:.3f}."
            )
        print(f"\n[PASS] RK3576 NPU validation passed for all {len(inputs)} samples.")
        print(f"[TIME] Total runtime: {format_duration(elapsed)} (including initialization)\n")

    finally:
        try:
            runtime.release()
        except Exception:
            pass

    # Promotion is only reached after all samples pass and RKNN resources are released.
    promote_validated_model()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[FAIL] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
