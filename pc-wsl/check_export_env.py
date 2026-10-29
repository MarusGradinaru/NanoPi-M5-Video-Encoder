#!/usr/bin/env python3
"""Check the pinned WSL environment and ResNet50V2 release contract."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "resnet50v2_multiclass_logistic_classifier.npz"

EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "protobuf": "4.25.4",
    "tensorflow": "2.15.1",
    "tf2onnx": "1.17.0",
    "scikit-learn": "1.7.1",
    "onnx": "1.18.0",
    "onnxruntime": "1.22.1",
    "rknn-toolkit2": "2.3.2",
    "setuptools": "80.10.2",
}


def call_with_filtered_native_stderr(function, suppressed_fragments: tuple[str, ...]):
    """Run a function while hiding only explicitly identified harmless native stderr lines."""
    sys.stderr.flush()
    result = None
    error: BaseException | None = None

    # TensorFlow can write startup diagnostics directly through the native stderr
    # file descriptor, bypassing Python logging and TF_CPP_MIN_LOG_LEVEL. Capture
    # fd 2 temporarily so only the three known duplicate CUDA factory messages
    # are removed; any other warnings/errors are replayed unchanged.
    with tempfile.TemporaryFile(mode="w+b") as log_file:
        saved_stderr = os.dup(2)
        try:
            os.dup2(log_file.fileno(), 2)
            try:
                result = function()
            except BaseException as exc:
                error = exc
            finally:
                sys.stderr.flush()
                try:
                    libc = ctypes.CDLL(None)
                    libc.fflush.argtypes = [ctypes.c_void_p]
                    libc.fflush.restype = ctypes.c_int
                    libc.fflush(None)
                except (AttributeError, OSError):
                    pass
                os.dup2(saved_stderr, 2)
        finally:
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


TF_DUPLICATE_CUDA_LOGS = (
    "Unable to register cuDNN factory",
    "Unable to register cuFFT factory",
    "Unable to register cuBLAS factory",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required Python package is not installed: {name}") from exc


def check_versions() -> None:
    print("\n[1/4] Python and package versions")
    print(f"  Python: {platform.python_version()} ({sys.executable})")
    require(sys.version_info[:2] == (3, 11), "Python 3.11 is required for this pinned RKNN Toolkit2 wheel.")

    for package, expected in EXPECTED_VERSIONS.items():
        actual = package_version(package)
        print(f"  {package}: {actual}")
        require(actual == expected, f"Version mismatch for {package}: expected {expected}, found {actual}.")

    torch_version = package_version("torch")
    print(f"  torch: {torch_version}")
    require(torch_version.split("+", 1)[0] == "2.4.0", f"Expected torch 2.4.0 (CPU build), found {torch_version}.")
    try:
        import torch

        require(torch.version.cuda is None, f"Expected CPU-only PyTorch, but torch reports CUDA {torch.version.cuda}.")
    except ImportError as exc:
        raise RuntimeError("Required Python package is not installed: torch") from exc

    pillow_version = package_version("Pillow")
    print(f"  Pillow: {pillow_version}")

    print("\n  Running pip check...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout.strip()
    if output:
        for line in output.splitlines():
            print(f"    {line}")
    require(result.returncode == 0, "pip check reported dependency conflicts.")
    if not output:
        print("    No broken requirements found.")


def check_resnet() -> None:
    print("\n[2/4] Frozen ResNet50V2 feature extractor")
    import numpy as np

    def run_tensorflow_check() -> tuple[object, object, object, int, int]:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50V2

        model = ResNet50V2(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(180, 320, 3),
        )
        model.trainable = False
        sample = tf.zeros((1, 180, 320, 3), dtype=tf.float32)
        output = model(sample, training=False).numpy()
        trainable_parameters = sum(int(tf.size(weight).numpy()) for weight in model.trainable_weights)
        total_parameters = model.count_params()
        return tf, model, output, trainable_parameters, total_parameters

    tf, model, output, trainable_parameters, total_parameters = call_with_filtered_native_stderr(
        run_tensorflow_check,
        TF_DUPLICATE_CUDA_LOGS,
    )

    print("  Input shape:             (1, 180, 320, 3) NHWC")
    print(f"  Feature shape:           {tuple(output.shape)}")
    print(f"  Total parameters:        {total_parameters:,}")
    print(f"  Trainable parameters:    {trainable_parameters:,}")
    require(output.shape == (1, 2048), f"Unexpected ResNet feature shape: {output.shape}.")
    require(trainable_parameters == 0, "ResNet50V2 must be frozen; found trainable parameters.")
    require(np.isfinite(output).all(), "ResNet50V2 produced NaN or infinite values.")

    del model, output
    tf.keras.backend.clear_session()


def check_rknn_toolkit() -> None:
    print("\n[3/4] RKNN Toolkit2 initialization")
    # RKNN Toolkit2 2.3.2 uses pkg_resources, which emits this known deprecation
    # warning with setuptools 80.x. Suppress only that exact warning; other warnings
    # remain visible so actionable diagnostics are not hidden.
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
        module=r"rknn\.api\.rknn",
    )
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.release()
    print("  RKNN(verbose=False): initialized and released successfully.")


def check_classifier_contract() -> None:
    print("\n[4/4] Optional classifier compatibility check")
    if not CLASSIFIER_PATH.is_file():
        print(f"  [SKIP] Classifier not found: {CLASSIFIER_PATH}")
        print("         Export can continue; this check is only for integration compatibility.")
        return

    import numpy as np

    with np.load(CLASSIFIER_PATH, allow_pickle=False) as archive:
        required_keys = {
            "image_width",
            "image_height",
            "feature_count",
            "number_of_classes",
            "class_names",
            "scaler_mean",
            "scaler_scale",
            "classifier_coef",
            "classifier_intercept",
        }
        missing = sorted(required_keys.difference(archive.files))
        require(not missing, f"Classifier is missing required arrays: {', '.join(missing)}.")
        width = int(archive["image_width"].item())
        height = int(archive["image_height"].item())
        feature_count = int(archive["feature_count"].item())
        class_count = int(archive["number_of_classes"].item())
        class_names = archive["class_names"].astype(str).tolist()
        mean = np.asarray(archive["scaler_mean"])
        scale = np.asarray(archive["scaler_scale"])
        coef = np.asarray(archive["classifier_coef"])
        intercept = np.asarray(archive["classifier_intercept"])

    require((width, height) == (320, 180), f"Classifier expects {width}x{height}; expected 320x180.")
    require(feature_count == 2048, f"Classifier expects {feature_count} features; expected 2048.")
    require(class_count == len(class_names), "Classifier class count does not match class_names.")
    require(mean.shape == (2048,), f"Unexpected scaler_mean shape: {mean.shape}.")
    require(scale.shape == (2048,), f"Unexpected scaler_scale shape: {scale.shape}.")
    require(coef.shape == (class_count, 2048), f"Unexpected classifier_coef shape: {coef.shape}.")
    require(intercept.shape == (class_count,), f"Unexpected classifier_intercept shape: {intercept.shape}.")
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0), "Classifier scaler has invalid values.")

    print(f"  File:          {CLASSIFIER_PATH}")
    print(f"  Image size:    {width}x{height}")
    print(f"  Features:      {feature_count}")
    print(f"  Classes ({class_count}): {', '.join(class_names)}")
    print("  Classifier metadata matches the ResNet50V2 feature contract.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("=" * 68)
    print("Step 1: RKNN EXPORT ENVIRONMENT CHECK")
    print("=" * 68)
    check_versions()
    check_resnet()
    check_rknn_toolkit()
    check_classifier_contract()
    print("\n[PASS] Environment and ResNet50V2 checks passed.\n")
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
