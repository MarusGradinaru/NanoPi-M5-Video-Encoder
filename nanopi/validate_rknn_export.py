#!/usr/bin/env python3
"""Validate an RK3576 ResNet50V2 feature extractor and promote the selected batch variant."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from rknn_batch import model_filename_for_batch, validate_batch_size

FEATURE_COUNT = 2048
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
MIN_COSINE_SIMILARITY = 0.95

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "resnet" / "artifacts"
MODELS_DIR = PROJECT_ROOT / "resnet" / "models"
VALIDATION_STEM = "resnet50v2_rknn_validation"
ENGINE_PYTHON_DIR = Path(__file__).resolve().parent / "resnet_engine" / "python"


def parse_batch_size(value: str) -> int:
    """Argparse adapter for the project's supported fixed batch sizes."""
    try:
        batch_size = int(value)
        return validate_batch_size(batch_size)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def validation_filename_for_batch(batch_size: int) -> str:
    """Return the validation archive filename matching the RKNN batch variant."""
    validate_batch_size(batch_size)
    suffix = "" if batch_size == 1 else f"_batch{batch_size}"
    return f"{VALIDATION_STEM}{suffix}.npz"


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
                    + ". Regenerate it with make_onnx_nchw.py."
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
    minutes, seconds_remainder = divmod(remainder, 60)
    centiseconds = int((seconds - whole_seconds) * 100)
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:02d}.{centiseconds:02d}"


def promote_validated_model(
    candidate_path: Path,
    target_path: Path,
    validation_path: Path,
) -> None:
    """Promote the validated model and remove only its matching validation archive."""
    if not target_path.parent.is_dir():
        raise RuntimeError(f"Model directory does not exist: {target_path.parent}")
    if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
        raise RuntimeError("The validated candidate model is no longer available for promotion.")
    backup_path = target_path.with_name(target_path.name + ".bak")
    backup_created = False
    if target_path.exists() and not (target_path.is_file() or target_path.is_symlink()):
        raise RuntimeError(f"Installed model path is not a file: {target_path}")
    if target_path.exists():
        if backup_path.exists() or backup_path.is_symlink():
            if backup_path.is_dir() and not backup_path.is_symlink():
                raise RuntimeError(f"Backup path exists as a directory: {backup_path}")
            backup_path.unlink()
        os.replace(target_path, backup_path)
        backup_created = True
    try:
        os.replace(candidate_path, target_path)
    except OSError:
        if backup_created and backup_path.exists() and not target_path.exists():
            os.replace(backup_path, target_path)
        raise

    print(f"[Promote] Validated model installed at: {target_path}")
    if validation_path.exists():
        try:
            validation_path.unlink()
        except OSError as exc:
            print(
                f"[Warning] Model was promoted, but matching validation archive could not be removed: "
                f"{validation_path} ({exc})",
                file=sys.stderr,
            )
    print("[Done] Validation passed and model promotion completed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=parse_batch_size,
        default=4,
        help="Candidate RKNN model batch size: 1 or a positive even integer (default: 4).",
    )
    args = parser.parse_args(argv)

    model_filename = model_filename_for_batch(args.batch_size)
    validation_filename = validation_filename_for_batch(args.batch_size)
    candidate_model_path = ARTIFACTS_DIR / model_filename
    validation_path = ARTIFACTS_DIR / validation_filename
    target_model_path = MODELS_DIR / model_filename

    print("=" * 68)
    print("RK3576 FEATURE EXTRACTOR - DEVICE VALIDATION")
    print("=" * 68)
    print(f"[Batch size] {args.batch_size}")
    print(f"[Candidate ] {candidate_model_path}")
    print(f"[Validation] {validation_path}")
    print(f"[Install to] {target_model_path}")

    candidate_exists = candidate_model_path.is_file()
    validation_exists = validation_path.is_file()
    if not candidate_exists and not validation_exists:
        print("\n[Info] No candidate model or validation data found in artifacts.")
        print("[Done] Nothing to validate; the installed model was left unchanged.")
        return 0
    if not candidate_exists:
        raise FileNotFoundError(
            f"Selected candidate model not found: {candidate_model_path}. "
            f"Check --batch-size (requested: {args.batch_size}) and transfer the matching RKNN artifact."
        )
    if not validation_exists:
        raise FileNotFoundError(
            f"Validation archive not found: {validation_path}. "
            "The candidate will not be promoted without its matching validation data."
        )
    if candidate_model_path.stat().st_size == 0:
        raise RuntimeError(f"Candidate RKNN model is empty: {candidate_model_path}")
    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {MODELS_DIR}")

    print("\n[1/3] Checking validation data...", flush=True)
    inputs, references, sample_names = load_validation_data(validation_path)
    print(f"  Samples:        {len(inputs)}")
    print(f"  Input:          [{args.batch_size}, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}] float32 NCHW")
    print(f"  Output:         [{args.batch_size}, {FEATURE_COUNT}] float32 features")
    print(f"  Incomplete last batch: padded as needed; padding outputs are ignored")
    print(f"  Minimum cosine: {MIN_COSINE_SIMILARITY:.3f} per sample")

    # Import the extension that links to the shared libresnet_engine.so library.
    engine_python_path = str(ENGINE_PYTHON_DIR)
    if engine_python_path in sys.path:
        sys.path.remove(engine_python_path)
    sys.path.insert(0, engine_python_path)
    try:
        import resnet_engine
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the ResNetEngine Python extension. Build it on the NanoPi "
            "with: ./resnet_engine/src/build-resnet-engine-python-ext.sh"
        ) from exc

    print("\n[2/3] Initializing native ResNetEngine on the device...", flush=True)
    started = time.perf_counter()
    failed: list[str] = []
    sample_metrics: list[tuple[float, float, float, float]] = []
    engine = None
    try:
        engine = resnet_engine.ResNetEngine(str(candidate_model_path), args.batch_size, 2)
        info = engine.info
        if info.batch_size != args.batch_size:
            raise RuntimeError(
                f"Native engine loaded batch size {info.batch_size}, "
                f"but {args.batch_size} was requested."
            )
        if (info.input_width, info.input_height, info.input_channels) != (
            IMAGE_WIDTH, IMAGE_HEIGHT, 3
        ):
            raise RuntimeError(
                "Candidate model has incompatible input dimensions: "
                f"{info.input_width}x{info.input_height}x{info.input_channels}; "
                f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}x3."
            )
        if info.feature_count != FEATURE_COUNT:
            raise RuntimeError(
                f"Candidate model returns {info.feature_count} features; "
                f"expected {FEATURE_COUNT}."
            )

        print(f"  Native engine:  {info}")
        print(f"  Model:          {candidate_model_path}")
        print("\n[3/3] Running native Zero-Copy Multi-Batch inference and comparing with ONNX references...", flush=True)

        try:
            # The validation archive contains normalized float32 NCHW tensors.
            # The C++ engine converts these directly to native FLOAT16 NHWC,
            # pads the incomplete final batch, and discards padding outputs.
            results = np.asarray(engine.infer_preprocessed(inputs), dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Native RKNN inference failed: {exc}") from exc

        expected_shape = (len(inputs), FEATURE_COUNT)
        if results.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected native feature output shape {results.shape}; expected {expected_shape}."
            )
        if not np.isfinite(results).all():
            raise RuntimeError("Native RKNN output contains NaN or infinite values.")

        for index, (name, reference, result) in enumerate(
            zip(sample_names, references, results), start=1
        ):
            difference = result.astype(np.float64) - reference.astype(np.float64)
            mae = float(np.mean(np.abs(difference)))
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            max_abs = float(np.max(np.abs(difference)))
            cosine = cosine_similarity(reference, result)
            passed = cosine >= MIN_COSINE_SIMILARITY
            if not passed:
                failed.append(name)
            sample_metrics.append((mae, rmse, max_abs, cosine))
            print(
                f"  {index:>2}/{len(inputs)} {name}: "
                f"MAE={mae:.6f}, RMSE={rmse:.6f}, max_abs={max_abs:.6f}, "
                f"cosine={cosine:.6f} [{'PASS' if passed else 'FAIL'}]",
                flush=True,
            )

        elapsed = time.perf_counter() - started
        mean_mae = float(np.mean([row[0] for row in sample_metrics]))
        mean_rmse = float(np.mean([row[1] for row in sample_metrics]))
        maximum_absolute_diff = float(max(row[2] for row in sample_metrics))
        mean_cosine = float(np.mean([row[3] for row in sample_metrics]))
        minimum_cosine = float(min(row[3] for row in sample_metrics))
        print("\n[Results]")
        print(f"  Mean absolute error:   {mean_mae:.6f}")
        print(f"  RMSE:                  {mean_rmse:.6f}")
        print(f"  Maximum absolute diff: {maximum_absolute_diff:.6f}")
        print(f"  Mean cosine:           {mean_cosine:.6f}")
        print(f"  Minimum cosine:        {minimum_cosine:.6f}")
        if failed:
            raise RuntimeError(
                f"Device validation failed for {len(failed)}/{len(inputs)} sample(s): "
                + ", ".join(failed)
                + f". Every sample must have cosine similarity >= {MIN_COSINE_SIMILARITY:.3f}."
            )
        print(f"\n[PASS] RK3576 NPU validation passed for all {len(inputs)} samples.")
        print(f"[TIME] Total runtime: {format_duration(elapsed)} (including initialization)\n")
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass

    promote_validated_model(candidate_model_path, target_model_path, validation_path)
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
