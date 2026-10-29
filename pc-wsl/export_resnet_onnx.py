#!/usr/bin/env python3
"""Export frozen ResNet50V2 features to a fixed-shape NHWC ONNX graph."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Keep TensorFlow quiet on CPU-only WSL systems and limit worker thread pools.
# These defaults are applied before any TensorFlow import (only in child workers).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Use TensorFlow reference kernels without oneDNN fusion so the parity test
# compares a stable FP32 path against ONNX Runtime instead of fused CPU kernels.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
FEATURE_COUNT = 2048
OPSET = 13
# TensorFlow and ONNX Runtime may use different CPU kernels/operation ordering.
# Values below accept the observed small numerical drift while still catching
# structural, preprocessing, or major numerical errors.
MIN_COSINE_SIMILARITY = 0.9990
MAX_RELATIVE_RMSE = 0.05
SYNTHETIC_VALIDATION_SAMPLES = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SCRIPT_PATH = Path(__file__).resolve()
RESNET_ROOT = SCRIPT_PATH.parent.parent / "resnet"
DEFAULT_OUTPUT = RESNET_ROOT / "artifacts" / "resnet50v2_features.onnx"
DEFAULT_VALIDATION_DIR = RESNET_ROOT / "dataset"


def configure_tensorflow(tf) -> None:
    """Configure TensorFlow to use fixed CPU thread counts."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # In a fresh worker this should not occur, but do not hide real model errors.
        pass


def build_feature_model(tf):
    from tensorflow.keras.applications import ResNet50V2

    model = ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    model.trainable = False
    trainable_count = sum(int(tf.size(weight).numpy()) for weight in model.trainable_weights)
    if trainable_count != 0:
        raise RuntimeError(f"Expected a frozen backbone, found {trainable_count} trainable parameters.")
    if tuple(model.output_shape) != (None, FEATURE_COUNT):
        raise RuntimeError(f"Unexpected ResNet50V2 output shape: {model.output_shape}")
    return model


def make_synthetic_samples() -> tuple[list, list[str]]:
    """Create deterministic numerical stress tests when no image dataset is available."""
    import numpy as np

    rng = np.random.default_rng(20261020)
    samples = [
        rng.uniform(-1.0, 1.0, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
    ]
    names = ["synthetic-continuous-stress"]
    for index in range(SYNTHETIC_VALIDATION_SAMPLES - 1):
        pixels = rng.integers(
            0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8
        ).astype(np.float32)
        samples.append(pixels / np.float32(127.5) - np.float32(1.0))
        names.append(f"synthetic-rgb-{index + 1}")
    return samples, names


def discover_validation_images(validation_dir: Path, limit: int) -> list[Path]:
    """Pick a small, deterministic set of image files, preferring class diversity."""
    if not validation_dir.is_dir():
        return []

    class_dirs = sorted(
        (p for p in validation_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name.casefold(),
    )
    per_class: list[Path] = []
    all_images: list[Path] = []
    for class_dir in class_dirs:
        files = sorted(
            (p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.as_posix().casefold(),
        )
        if files:
            per_class.append(files[0])
            all_images.extend(files[1:])

    # Also support a flat directory without class subdirectories.
    if not class_dirs:
        all_images = sorted(
            (p for p in validation_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.as_posix().casefold(),
        )

    chosen = per_class[:limit]
    seen = set(chosen)
    for path in all_images:
        if len(chosen) >= limit:
            break
        if path not in seen:
            chosen.append(path)
            seen.add(path)
    return chosen


def load_validation_images(paths: list[Path]) -> tuple[list, list[str]]:
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    samples = []
    names = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
                pixels = np.asarray(image, dtype=np.float32)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"[Validation] Skipping unreadable image {path}: {exc}", file=sys.stderr)
            continue
        samples.append(pixels / np.float32(127.5) - np.float32(1.0))
        names.append(path.name)
    return samples, names


def worker_export(temp_onnx: Path) -> int:
    """Export the feature extractor to ONNX."""
    import tensorflow as tf
    import tf2onnx

    configure_tensorflow(tf)
    from tensorflow.keras import backend as keras_backend

    temp_onnx.parent.mkdir(parents=True, exist_ok=True)
    if temp_onnx.exists():
        temp_onnx.unlink()

    print("  Loading ImageNet ResNet50V2 weights...", flush=True)
    model = build_feature_model(tf)
    input_spec = tf.TensorSpec(
        shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=tf.float32,
        name="images",
    )

    @tf.function(input_signature=[input_spec], autograph=False)
    def inference_function(images):
        return model(images, training=False)

    print("  Converting model to ONNX (may take some time)...", flush=True)
    tf2onnx.convert.from_function(
        inference_function,
        input_signature=[input_spec],
        opset=OPSET,
        output_path=str(temp_onnx),
    )
    if not temp_onnx.is_file() or temp_onnx.stat().st_size == 0:
        raise RuntimeError("The ONNX exporter did not create a non-empty model file.")

    del model
    keras_backend.clear_session()
    print("  ONNX graph created.", flush=True)
    return 0


def worker_check_onnx(temp_onnx: Path) -> int:
    """Verify the ONNX graph structure."""
    import onnx

    onnx.checker.check_model(str(temp_onnx))
    print("  ONNX structure is valid.", flush=True)
    return 0


def worker_reference(reference_file: Path, validation_dir: Path | None) -> int:
    """Generate reference feature vectors using TensorFlow."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import backend as keras_backend

    configure_tensorflow(tf)

    samples, names = make_synthetic_samples()
    actual_validation_dir = validation_dir
    if actual_validation_dir is None and DEFAULT_VALIDATION_DIR.is_dir():
        actual_validation_dir = DEFAULT_VALIDATION_DIR

    if actual_validation_dir is not None:
        paths = discover_validation_images(actual_validation_dir, limit=3)
        if validation_dir is not None and not paths:
            raise RuntimeError(f"No readable image files found in validation directory: {validation_dir}")
        if paths:
            image_samples, image_names = load_validation_images(paths)
            samples.extend(image_samples)
            names.extend(f"dataset/{name}" for name in image_names)
            print(f"  Dataset images included: {len(image_samples)}", flush=True)
        elif validation_dir is None:
            print("  No dataset images found; using synthetic validation inputs.", flush=True)
    else:
        print("  Dataset directory not found; using synthetic validation inputs.", flush=True)

    model = build_feature_model(tf)
    features = np.empty((len(samples), FEATURE_COUNT), dtype=np.float32)
    sample_array = np.empty((len(samples), IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    print(f"  Computing reference outputs for {len(samples)} validation inputs...", flush=True)

    for index, sample in enumerate(samples):
        sample_array[index] = sample
        output = model(sample[np.newaxis, ...], training=False).numpy()
        if output.shape != (1, FEATURE_COUNT):
            raise RuntimeError(f"Unexpected TensorFlow output shape on sample {index + 1}: {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError(f"TensorFlow returned non-finite features for sample {index + 1}.")
        features[index] = output[0]

    reference_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(reference_file, samples=sample_array, tf_features=features, names=np.asarray(names, dtype=str))
    print("  Reference outputs generated.", flush=True)

    del model, sample_array, samples, features
    keras_backend.clear_session()
    return 0


def worker_validate(temp_onnx: Path, reference_file: Path) -> int:
    """Compare ONNX Runtime outputs with the TensorFlow reference outputs."""
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Disable graph rewrites during parity validation. This is a correctness
    # check, not a performance benchmark; avoid fusion changing FP32 rounding.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False

    with np.load(reference_file, allow_pickle=False) as reference:
        samples = reference["samples"]
        tf_features_all = reference["tf_features"]
        names = reference["names"].astype(str).tolist()

    print("  Running ONNX inference...", flush=True)
    session = ort.InferenceSession(
        str(temp_onnx),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_input = [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"Expected one ONNX input and output, found {len(inputs)} and {len(outputs)}.")
    if list(inputs[0].shape) != expected_input:
        raise RuntimeError(f"Unexpected ONNX input shape: {list(inputs[0].shape)}; expected {expected_input}.")

    if list(outputs[0].shape) != [1, FEATURE_COUNT]:
        raise RuntimeError(f"Unexpected ONNX output shape: {outputs[0].shape}; expected [1, {FEATURE_COUNT}].")

    metrics: list[tuple[float, float, float, float, float]] = []
    for index in range(len(samples)):
        sample = samples[index:index + 1]
        tf_features = tf_features_all[index].astype(np.float64)
        onnx_features = np.asarray(session.run(None, {inputs[0].name: sample})[0], dtype=np.float64)
        if onnx_features.shape != (1, FEATURE_COUNT):
            raise RuntimeError(f"Unexpected ONNX output shape on sample {index + 1}: {onnx_features.shape}")
        onnx_vector = onnx_features.reshape(-1)
        if not np.isfinite(onnx_vector).all() or not np.isfinite(tf_features).all():
            raise RuntimeError(f"TensorFlow or ONNX Runtime returned non-finite values on sample {index + 1}.")

        delta = onnx_vector - tf_features
        mae = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        max_abs = float(np.max(np.abs(delta)))
        reference_rms = float(np.sqrt(np.mean(np.square(tf_features))))
        relative_rmse = rmse / max(reference_rms, 1e-12)
        denominator = float(np.linalg.norm(tf_features) * np.linalg.norm(onnx_vector))
        cosine = float(np.dot(tf_features, onnx_vector) / denominator) if denominator > 0.0 else float(np.array_equal(tf_features, onnx_vector))
        metrics.append((mae, rmse, max_abs, cosine, relative_rmse))

    failed = [index for index, metric in enumerate(metrics) if metric[3] < MIN_COSINE_SIMILARITY or metric[4] > MAX_RELATIVE_RMSE]
    mean_mae = float(np.mean([metric[0] for metric in metrics]))
    mean_rmse = float(np.mean([metric[1] for metric in metrics]))
    max_abs = float(max(metric[2] for metric in metrics))
    min_cosine = float(min(metric[3] for metric in metrics))
    max_rel_rmse = float(max(metric[4] for metric in metrics))
    print(f"  Samples checked:             {len(metrics)}")
    print(f"  Mean absolute error (MAE):   {mean_mae:.8f}")
    print(f"  Root mean squared error:     {mean_rmse:.8f}")
    print(f"  Maximum absolute difference: {max_abs:.8f}")
    print(f"  Minimum cosine similarity:   {min_cosine:.7f}")
    print(f"  Maximum relative RMSE:       {max_rel_rmse:.3%}")

    if failed:
        print("  Samples outside acceptance limits:", file=sys.stderr)
        for index in failed:
            mae, rmse, max_abs_sample, cosine, relative_rmse = metrics[index]
            print(
                f"    {names[index]}: MAE={mae:.8f}, RMSE={rmse:.8f}, "
                f"max_abs={max_abs_sample:.8f}, relative_RMSE={relative_rmse:.3%}, "
                f"cosine={cosine:.7f}",
                file=sys.stderr,
            )
        raise RuntimeError(
            "TensorFlow and ONNX outputs did not meet the numerical acceptance criteria. "
            "The existing ONNX model was not changed."
        )
    print("  Numerical comparison passed.", flush=True)
    return 0


def run_worker(label: str, worker: str, temp_onnx: Path, reference_file: Path, validation_dir: Path | None) -> None:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--_worker", worker,
        "--_temp-onnx", str(temp_onnx),
        "--_reference-file", str(reference_file),
    ]
    if validation_dir is not None:
        command.extend(["--validation-dir", str(validation_dir)])
    print(f"\n{label}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")

    result = subprocess.run(command, env=env, check=False)
    if result.returncode != 0:
        if result.returncode < 0:
            detail = f"worker terminated by signal {-result.returncode}"
        else:
            detail = f"worker exited with code {result.returncode}"
        raise RuntimeError(f"{label} failed ({detail}).")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output ONNX path.")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Optional directory of real validation images; otherwise uses resnet/dataset/ under the project root when present.",
    )
    parser.add_argument("--_worker", choices=("export", "check", "reference", "validate"), help=argparse.SUPPRESS)
    parser.add_argument("--_temp-onnx", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_reference-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Internal worker modes are entered before the orchestration logic.
    if args._worker:
        try:
            if args._temp_onnx is None or args._reference_file is None:
                raise RuntimeError("Internal worker arguments are incomplete.")
            temp_onnx = args._temp_onnx.expanduser().resolve()
            reference_file = args._reference_file.expanduser().resolve()
            validation_dir = args.validation_dir.expanduser().resolve() if args.validation_dir else None
            if args._worker == "export":
                return worker_export(temp_onnx)
            if args._worker == "check":
                return worker_check_onnx(temp_onnx)
            if args._worker == "reference":
                return worker_reference(reference_file, validation_dir)
            return worker_validate(temp_onnx, reference_file)
        except KeyboardInterrupt:
            print("[FAIL] Interrupted by user.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_dir = args.validation_dir.expanduser().resolve() if args.validation_dir else None

    print("=" * 68)
    print("Step 2: EXPORT RESNET50V2 FEATURES -> ONNX (NHWC)")
    print("=" * 68)
    print(f"[Config] Input:        [1, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3] float32 NHWC")
    print("[Config] Input values: RGB normalized to [-1, 1]")
    print(f"[Config] Output:       [1, {FEATURE_COUNT}] float32 features")
    print(f"[Config] ONNX opset:   {OPSET}")
    print(f"[Output] {output_path}")

    try:
        # Temporary files are removed automatically; the existing output is replaced
        # only after all structural and numerical checks have passed.
        with tempfile.TemporaryDirectory(prefix=".resnet50v2_onnx_", dir=output_path.parent) as temp_dir:
            work_path = Path(temp_dir)
            temp_onnx = work_path / "resnet50v2_features.tmp.onnx"
            reference_file = work_path / "validation_reference.npz"

            run_worker("[1/4] Exporting ResNet50V2 to ONNX...", "export", temp_onnx, reference_file, None)
            run_worker("[2/4] Verifying ONNX model structure...", "check", temp_onnx, reference_file, None)
            run_worker("[3/4] Preparing validation outputs...", "reference", temp_onnx, reference_file, validation_dir)
            run_worker("[4/4] Validating ONNX outputs...", "validate", temp_onnx, reference_file, None)

            os.replace(temp_onnx, output_path)

        print("\n[PASS] ONNX model exported and validated.")
        print(f"[File] {output_path} ({output_path.stat().st_size:,} bytes)\n")
        return 0
    except KeyboardInterrupt:
        print("\n[FAIL] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        print("[Info] The existing ONNX model was not changed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
