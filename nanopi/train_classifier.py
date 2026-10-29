#!/usr/bin/env python3
"""Train a multiclass presenter classifier using frozen ResNet50V2 features."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_DIR = Path("../resnet/dataset")
ARTIFACTS_DIR = Path("../resnet/artifacts")
MODEL_DIR = Path("../resnet/models")
CACHE_FILE = ARTIFACTS_DIR / "resnet50v2_classifier_cache.npz"
REPORT_FILE = ARTIFACTS_DIR / "resnet50v2_classifier_training_report.txt"
FINAL_MODEL_FILE = MODEL_DIR / "resnet50v2_logistic_classifier.npz"
BACKUP_MODEL_FILE = FINAL_MODEL_FILE.with_name(FINAL_MODEL_FILE.name + ".bak")
FEATURE_MODEL_FILE = MODEL_DIR / "resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
FEATURE_COUNT = 2048
BATCH_SIZE = 16
RANDOM_STATE = 42


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
                # librknnrt. Flush them while fd 1/2 still point at the capture file.
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    import ctypes
                    libc = ctypes.CDLL(None)
                    libc.fflush.argtypes = [ctypes.c_void_p]
                    libc.fflush.restype = ctypes.c_int
                    libc.fflush(None)
                except (AttributeError, OSError):
                    # Keep output filtering best-effort if C stdio flushing is unavailable.
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


def discover_dataset(dataset_dir: Path) -> tuple[list[str], list[Path], np.ndarray]:
    """Discover classes from dataset subdirectories and return images and labels."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir.resolve()}")

    class_dirs = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not class_dirs:
        raise RuntimeError(
            f"No class directories found in {dataset_dir.resolve()}. "
            "Create one subdirectory per class and place its images inside."
        )

    class_names: list[str] = []
    images: list[Path] = []
    labels: list[int] = []

    for class_dir in class_dirs:
        class_images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        # Ignore empty directories so users may keep placeholders or future class folders.
        if not class_images:
            continue

        class_index = len(class_names)
        class_names.append(class_dir.name)
        images.extend(class_images)
        labels.extend([class_index] * len(class_images))
        print(f"[Dataset] {class_dir.name}: {len(class_images)} images")

        if len(class_images) < 2:
            raise RuntimeError(
                f"Class '{class_dir.name}' contains only {len(class_images)} image(s). "
                "At least two images per class are required; more diverse examples are strongly recommended."
            )

    if len(class_names) < 2:
        raise RuntimeError(
            f"At least two non-empty class directories are required in {dataset_dir.resolve()}."
        )

    labels_array = np.asarray(labels, dtype=np.int64)
    print(f"[Dataset] Classes: {len(class_names)}")
    print(f"[Dataset] Total: {len(images)} images")
    return class_names, images, labels_array


def load_feature_cache(
    cache_file: Path, expected_backend: str
) -> dict[str, tuple[int, int, np.ndarray]]:
    """Load cached feature vectors only when they match the selected extraction backend."""
    cache: dict[str, tuple[int, int, np.ndarray]] = {}
    if not cache_file.is_file():
        return cache

    try:
        with np.load(cache_file, allow_pickle=False) as data:
            # Cache files created before backend metadata was added came from TensorFlow/CPU.
            cached_backend = str(data["backend"].item()) if "backend" in data.files else "cpu"
            if cached_backend != expected_backend:
                print(
                    f"[Cache] Ignoring {cached_backend.upper()} cache because the selected "
                    f"backend is {expected_backend.upper()}."
                )
                return {}
            paths = data["paths"].astype(str)
            sizes = data["sizes"].astype(np.int64)
            mtimes = data["mtimes_ns"].astype(np.int64)
            features = data["features"].astype(np.float32)
            if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
                print("[Cache] Ignoring cache with an unexpected feature shape.")
                return {}
            if not (len(paths) == len(sizes) == len(mtimes) == len(features)):
                print("[Cache] Ignoring cache with inconsistent array lengths.")
                return {}
            for path, size, mtime, feature in zip(paths, sizes, mtimes, features):
                cache[path] = (int(size), int(mtime), feature)
        print(f"[Cache] Loaded {len(cache)} cached feature vectors.")
    except Exception as exc:
        print(f"[Cache] Could not read cache; rebuilding it. Reason: {exc}")
        return {}
    return cache


def prepare_image(path: Path) -> np.ndarray:
    """Load an image as RGB and resize it to the ResNet input dimensions."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not load image '{path}': {exc}") from exc


def extract_features(
    images: list[Path],
    dataset_dir: Path,
    cache_file: Path,
    backend: str,
    feature_model_file: Path,
) -> tuple[list[str], np.ndarray]:
    """Reuse unchanged cached vectors and extract new/changed features on the selected backend."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    old_cache = load_feature_cache(cache_file, backend)

    relative_paths: list[str] = []
    image_stats: list[tuple[int, int]] = []
    feature_by_path: dict[str, np.ndarray] = {}
    pending_paths: list[Path] = []

    for path in images:
        rel_path = path.relative_to(dataset_dir).as_posix()
        stat = path.stat()
        size, mtime_ns = int(stat.st_size), int(stat.st_mtime_ns)
        relative_paths.append(rel_path)
        image_stats.append((size, mtime_ns))
        cached = old_cache.get(rel_path)
        if cached is not None and cached[0] == size and cached[1] == mtime_ns:
            feature_by_path[rel_path] = cached[2]
        else:
            pending_paths.append(path)

    print(f"[Features] Reusing {len(images) - len(pending_paths)} cached vectors.")
    print(f"[Features] Extracting features for {len(pending_paths)} new or changed images.")

    if pending_paths and backend == "cpu":
        # TensorFlow is intentionally imported only for the explicit CPU fallback.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50V2
            from tensorflow.keras.applications.resnet_v2 import preprocess_input
        except ImportError as exc:
            raise RuntimeError(
                "CPU feature extraction requires TensorFlow. Install it in this environment "
                "or run without --use-cpu to use the RKNN NPU backend."
            ) from exc

        print("[Model] Loading frozen ResNet50V2 for CPU feature extraction...")
        model = None
        try:
            model = ResNet50V2(
                include_top=False,
                weights="imagenet",
                pooling="avg",
                input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            )
            model.trainable = False

            for start in range(0, len(pending_paths), BATCH_SIZE):
                batch_paths = pending_paths[start:start + BATCH_SIZE]
                batch = np.stack([prepare_image(path) for path in batch_paths], axis=0)
                batch = preprocess_input(batch)
                batch_features = model(batch, training=False).numpy().astype(np.float32)
                if batch_features.shape != (len(batch_paths), FEATURE_COUNT):
                    raise RuntimeError(f"Unexpected feature output shape: {batch_features.shape}")
                for path, vector in zip(batch_paths, batch_features):
                    rel_path = path.relative_to(dataset_dir).as_posix()
                    feature_by_path[rel_path] = vector
                done = min(start + len(batch_paths), len(pending_paths))
                print(f"[Features] Processed {done}/{len(pending_paths)} images")
        finally:
            if model is not None:
                del model
            tf.keras.backend.clear_session()

    elif pending_paths and backend == "rknn":
        # RKNN Lite is the default: TensorFlow is not required for the normal NanoPi path.
        if not feature_model_file.is_file():
            raise FileNotFoundError(
                f"RKNN feature model not found: {feature_model_file.resolve()}\n"
                "Expected ../resnet/models/resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn. "
                "Use --use-cpu only when TensorFlow is installed and CPU extraction is intended."
            )
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "RKNN Lite 2 is required for the default NPU backend but could not be imported. "
                "Activate the environment containing rknnlite, or pass --use-cpu if TensorFlow "
                "is installed."
            ) from exc

        print(f"[Model] Loading RKNN ResNet50V2 feature extractor: {feature_model_file}")
        rknn = RKNNLite()
        try:
            load_result = rknn.load_rknn(str(feature_model_file))
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
                    "Check RKNN Lite and NPU driver compatibility."
                )
            print("[RKNN] Runtime initialized on the RK3576 NPU.")

            for index, path in enumerate(pending_paths, start=1):
                # Match process_video.py and the validated RKNN feature test exactly:
                # RGB [0, 255] -> [-1, 1], then NHWC -> NCHW.
                image = prepare_image(path)
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
                        f"RKNN inference failed for image '{path.name}': {exc}"
                    ) from exc
                if not outputs or outputs[0] is None:
                    raise RuntimeError(
                        f"RKNN inference returned no feature tensor for '{path.name}'."
                    )
                vector = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
                if vector.size != FEATURE_COUNT:
                    raise RuntimeError(
                        f"Unexpected RKNN output size for '{path.name}': {vector.size}; "
                        f"expected {FEATURE_COUNT}."
                    )
                feature_by_path[path.relative_to(dataset_dir).as_posix()] = vector

                if index % BATCH_SIZE == 0 or index == len(pending_paths):
                    print(f"[Features] Processed {index}/{len(pending_paths)} images.")
        finally:
            try:
                rknn.release()
            except Exception:
                pass

    feature_matrix = np.stack([feature_by_path[path] for path in relative_paths]).astype(np.float32)
    sizes = np.asarray([item[0] for item in image_stats], dtype=np.int64)
    mtimes_ns = np.asarray([item[1] for item in image_stats], dtype=np.int64)

    # Tag the cache backend so TensorFlow and RKNN feature vectors are never mixed.
    np.savez_compressed(
        cache_file,
        paths=np.asarray(relative_paths, dtype=str),
        sizes=sizes,
        mtimes_ns=mtimes_ns,
        features=feature_matrix,
        backend=np.asarray(backend, dtype=str),
    )
    print(f"[Cache] Saved {len(relative_paths)} feature vectors to {cache_file} ({backend.upper()}).")
    return relative_paths, feature_matrix


def fit_classifier(features: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """Fit a balanced logistic-regression classifier for all discovered classes."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    classifier.fit(scaled_features, labels)
    return scaler, classifier


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS.ss."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:05.2f}"


def save_release_model(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    class_names: list[str],
) -> bool:
    """Save the release model, rotating any existing release model to .npz.bak."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = FINAL_MODEL_FILE.with_name(FINAL_MODEL_FILE.name + ".tmp")
    backup_created = False

    try:
        # Write the complete new model before moving or deleting any existing release files.
        with temporary_file.open("wb") as model_file:
            np.savez_compressed(
                model_file,
                scaler_mean=scaler.mean_.astype(np.float32),
                scaler_scale=scaler.scale_.astype(np.float32),
                classifier_coef=classifier.coef_.astype(np.float32),
                classifier_intercept=classifier.intercept_.astype(np.float32),
                class_names=np.asarray(class_names, dtype=str),
                classifier_classes=classifier.classes_.astype(np.int64),
                image_width=np.asarray(IMAGE_WIDTH, dtype=np.int32),
                image_height=np.asarray(IMAGE_HEIGHT, dtype=np.int32),
                feature_count=np.asarray(FEATURE_COUNT, dtype=np.int32),
                number_of_classes=np.asarray(len(class_names), dtype=np.int32),
            )

        # The previous backup is disposable; retain only the immediately previous release.
        if BACKUP_MODEL_FILE.exists():
            BACKUP_MODEL_FILE.unlink()

        if FINAL_MODEL_FILE.exists():
            FINAL_MODEL_FILE.replace(BACKUP_MODEL_FILE)
            backup_created = True

        try:
            temporary_file.replace(FINAL_MODEL_FILE)
        except Exception:
            # If promotion of the new model fails, restore the previous release if possible.
            if backup_created and BACKUP_MODEL_FILE.exists() and not FINAL_MODEL_FILE.exists():
                BACKUP_MODEL_FILE.replace(FINAL_MODEL_FILE)
            raise

        return backup_created
    finally:
        if temporary_file.exists():
            temporary_file.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the multiclass presenter classifier using the RK3576 NPU by default."
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Use TensorFlow/CPU for ResNet50V2 feature extraction instead of RKNN Lite/NPU.",
    )
    args = parser.parse_args(argv)
    backend = "cpu" if args.use_cpu else "rknn"

    run_started = time.perf_counter()
    dataset_dir = DATASET_DIR.resolve()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "[Backend] TensorFlow/CPU feature extraction selected (--use-cpu)."
        if backend == "cpu"
        else "[Backend] RK3576 NPU via RKNN Lite 2 (default)."
    )
    class_names, images, labels = discover_dataset(dataset_dir)
    feature_started = time.perf_counter()
    relative_paths, features = extract_features(
        images, dataset_dir, CACHE_FILE, backend, FEATURE_MODEL_FILE
    )
    feature_elapsed = time.perf_counter() - feature_started

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) != len(class_names) or np.min(counts) < 2:
        raise RuntimeError("Every discovered class must contain at least two usable images.")

    # This random split is a preliminary diagnostic. Related frames from the same
    # recording may appear in both sets, so also evaluate on completely unseen recordings.
    evaluation_started = time.perf_counter()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    validation_scaler, validation_classifier = fit_classifier(x_train, y_train)
    test_scaled = validation_scaler.transform(x_test)
    predictions = validation_classifier.predict(test_scaled)
    probabilities = validation_classifier.predict_proba(test_scaled)
    class_indices = np.arange(len(class_names), dtype=np.int64)
    evaluation_elapsed = time.perf_counter() - evaluation_started

    report = [
        "PRELIMINARY RANDOM-SPLIT MULTICLASS EVALUATION",
        "NOTE: This can be optimistic when related frames from the same recordings appear in both splits.",
        "Evaluate on completely separate recordings before relying on these results.",
        "",
        f"Feature extraction backend: {backend.upper()}",
        f"Dataset images: {len(relative_paths)}",
        f"Number of classes: {len(class_names)}",
        "Class counts:",
    ]
    report.extend(f"  {name}: {int(np.sum(labels == index))}" for index, name in enumerate(class_names))
    report.extend([
        f"Validation split accuracy: {accuracy_score(y_test, predictions):.4f}",
        "",
        classification_report(
            y_test,
            predictions,
            labels=class_indices,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "Confusion matrix (rows=true, columns=predicted; order follows the class list above):",
        str(confusion_matrix(y_test, predictions, labels=class_indices)),
        "",
        "Sample validation predictions (first 20):",
        "index,true_class,predicted_class,top_class_probability",
    ])
    for index, (true_label, pred_label, row) in enumerate(zip(y_test[:20], predictions[:20], probabilities[:20])):
        top_probability = float(np.max(row))
        report.append(
            f"{index},{class_names[int(true_label)]},{class_names[int(pred_label)]},{top_probability:.6f}"
        )

    # Fit the final deployable classifier using every labeled example in the dataset.
    final_fit_started = time.perf_counter()
    final_scaler, final_classifier = fit_classifier(features, labels)
    backup_created = save_release_model(final_scaler, final_classifier, class_names)
    final_fit_elapsed = time.perf_counter() - final_fit_started

    total_elapsed = time.perf_counter() - run_started
    report.extend([
        "",
        "RUNTIME",
        f"Feature extraction and cache: {format_duration(feature_elapsed)} ({feature_elapsed:.2f} seconds)",
        f"Validation and evaluation: {format_duration(evaluation_elapsed)} ({evaluation_elapsed:.2f} seconds)",
        f"Final classifier fit and export: {format_duration(final_fit_elapsed)} ({final_fit_elapsed:.2f} seconds)",
        f"Total training pipeline: {format_duration(total_elapsed)} ({total_elapsed:.2f} seconds)",
    ])
    report_text = "\n".join(report) + "\n"
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"[Model] Classifier model saved to: {FINAL_MODEL_FILE}")
    print("[Done ] Training, evaluation, and export completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[Error] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1)
