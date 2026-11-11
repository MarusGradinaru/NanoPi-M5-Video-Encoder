#!/usr/bin/env python3
"""Train a multiclass presenter classifier using frozen ResNet50V2 features."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rknn_batch import batch_size_from_model_path, model_filename_for_batch, validate_batch_size


DATASET_DIR = Path("../resnet/dataset")
ARTIFACTS_DIR = Path("../resnet/artifacts")
MODEL_DIR = Path("../resnet/models")
CACHE_FILE = ARTIFACTS_DIR / "resnet50v2_classifier_cache.npz"
REPORT_FILE = ARTIFACTS_DIR / "resnet50v2_classifier_training_report.txt"
FINAL_MODEL_FILE = MODEL_DIR / "resnet50v2_logistic_classifier.npz"
BACKUP_MODEL_FILE = FINAL_MODEL_FILE.with_name(FINAL_MODEL_FILE.name + ".bak")
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
FEATURE_COUNT = 2048
CPU_BATCH_SIZE = 16
NPU_INPUT_CHUNK_SIZE = 64
DEFAULT_RKNN_BATCH_SIZE = 4
NPU_CORE_COUNT = 2
RANDOM_STATE = 42
ENGINE_PYTHON_DIR = Path(__file__).resolve().parent / "resnet_engine" / "python"


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
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
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
    """Load a native-size JPEG as RGB for the explicit TensorFlow/CPU fallback."""
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(path) as image:
            if image.format != "JPEG":
                raise RuntimeError(f"Unsupported image format for '{path}': {image.format}")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise RuntimeError(
                    f"Image '{path}' has dimensions {image.width}x{image.height}; "
                    f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}. The ResNetEngine does not resize images."
                )
            return np.asarray(image.convert("RGB"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not load JPEG image '{path}': {exc}") from exc


def consume_engine_stream(
    stream,
    dataset_dir: Path,
    expected_paths: list[Path],
    feature_by_path: dict[str, np.ndarray],
    progress_state: dict[str, int],
    total_pending: int,
    stream_description: str,
) -> None:
    """Consume one native folder/file stream and store features by dataset-relative path."""
    expected_relative = {
        path.resolve().relative_to(dataset_dir).as_posix()
        for path in expected_paths
    }
    received_relative: set[str] = set()

    try:
        for chunk_paths, chunk_features in stream:
            chunk_features = np.asarray(chunk_features, dtype=np.float32)
            if chunk_features.ndim != 2 or chunk_features.shape != (
                len(chunk_paths), FEATURE_COUNT
            ):
                raise RuntimeError(
                    f"ResNetEngine returned feature shape {chunk_features.shape} for "
                    f"{len(chunk_paths)} image(s); expected ({len(chunk_paths)}, {FEATURE_COUNT})."
                )
            if not np.isfinite(chunk_features).all():
                raise RuntimeError("ResNetEngine returned NaN or infinite feature values.")

            for image_path, vector in zip(chunk_paths, chunk_features):
                relative_path = Path(image_path).resolve().relative_to(dataset_dir).as_posix()
                if relative_path not in expected_relative:
                    raise RuntimeError(
                        f"ResNetEngine returned an unexpected image path: {image_path}"
                    )
                if relative_path in received_relative:
                    raise RuntimeError(
                        f"ResNetEngine returned the same image more than once: {image_path}"
                    )
                received_relative.add(relative_path)
                feature_by_path[relative_path] = vector

            progress_state["processed"] += len(chunk_paths)
            print(
                f"[Features] Processed {progress_state['processed']}/{total_pending} images "
                f"({stream_description})."
            )

        stats = stream.stats
        if not stats.get("completed", False):
            raise RuntimeError(f"ResNetEngine did not complete {stream_description}.")
        if stats.get("images_failed", 0):
            failures = stats.get("failures", [])
            details = "; ".join(
                f"{item.get('path', '<unknown>')}: {item.get('reason', 'unknown error')}"
                for item in failures[:5]
            )
            more = "" if len(failures) <= 5 else f"; and {len(failures) - 5} more"
            raise RuntimeError(
                f"ResNetEngine failed to process {stats['images_failed']} JPEG(s) in "
                f"{stream_description}: {details}{more}"
            )
        if stats.get("images_processed", 0) != len(expected_paths):
            raise RuntimeError(
                f"ResNetEngine processed {stats.get('images_processed', 0)} image(s) in "
                f"{stream_description}; expected {len(expected_paths)}."
            )
        if received_relative != expected_relative:
            missing = sorted(expected_relative - received_relative)
            extra = sorted(received_relative - expected_relative)
            raise RuntimeError(
                f"ResNetEngine path mismatch in {stream_description}. "
                f"Missing: {missing[:5]}; unexpected: {extra[:5]}."
            )
    finally:
        stream.close()


def extract_features(
    images: list[Path],
    dataset_dir: Path,
    cache_file: Path,
    backend: str,
    feature_model_file: Path | None,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    """Reuse unchanged cached vectors and extract new/changed features on the selected backend."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_backend = "cpu" if backend == "cpu" else f"rknn_engine_mpp_rga_batch{batch_size}"
    old_cache = load_feature_cache(cache_file, cache_backend)

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

            for start in range(0, len(pending_paths), CPU_BATCH_SIZE):
                batch_paths = pending_paths[start:start + CPU_BATCH_SIZE]
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
        if feature_model_file is None or not feature_model_file.is_file():
            raise FileNotFoundError(
                f"RKNN feature model not found: {feature_model_file}\n"
                "Expected the matching ResNetEngine batch model in ../resnet/models."
            )
        model_batch_size = batch_size_from_model_path(feature_model_file)
        if model_batch_size != batch_size:
            raise RuntimeError(
                f"Selected model has fixed batch size {model_batch_size}, but "
                f"--batch-size is {batch_size}: {feature_model_file.name}."
            )

        engine_python_path = str(ENGINE_PYTHON_DIR)
        if engine_python_path in sys.path:
            sys.path.remove(engine_python_path)
        sys.path.insert(0, engine_python_path)
        try:
            import resnet_engine
        except ImportError as exc:
            raise RuntimeError(
                "The ResNetEngine Python extension could not be imported. Build it on the NanoPi "
                "with: ./resnet_engine/src/build-resnet-engine-python-ext.sh"
            ) from exc

        print(f"[Model] Loading RKNN ResNet50V2 feature extractor: {feature_model_file}")
        print(
            f"[Engine] ResNetEngine, batch size {batch_size}, "
            f"{NPU_CORE_COUNT} NPU cores; chunk size {NPU_INPUT_CHUNK_SIZE} images."
        )
        engine = None
        progress_state = {"processed": 0}
        try:
            engine = resnet_engine.ResNetEngine(
                str(feature_model_file), batch_size=batch_size, npu_core_count=NPU_CORE_COUNT
            )
            engine_info = engine.info
            if engine_info.batch_size != batch_size:
                raise RuntimeError(
                    f"ResNetEngine initialized with batch size {engine_info.batch_size}; "
                    f"expected {batch_size}."
                )
            if engine_info.feature_count != FEATURE_COUNT:
                raise RuntimeError(
                    f"ResNetEngine returned {engine_info.feature_count} features; "
                    f"expected {FEATURE_COUNT}."
                )
            print(f"[Engine] {engine_info}")

            # Keep the dataset's class-folder organization. If an entire class folder
            # needs fresh features, let C++ discover and process that folder directly.
            # When only a subset changed, use process_files() to preserve incremental
            # feature-cache reuse without re-running inference for unchanged JPEGs.
            all_paths_by_class: dict[Path, list[Path]] = {}
            pending_paths_by_class: dict[Path, list[Path]] = {}
            for path in images:
                relative = path.relative_to(dataset_dir)
                class_dir = dataset_dir / relative.parts[0]
                all_paths_by_class.setdefault(class_dir, []).append(path)
            for path in pending_paths:
                relative = path.relative_to(dataset_dir)
                class_dir = dataset_dir / relative.parts[0]
                pending_paths_by_class.setdefault(class_dir, []).append(path)

            for class_dir, class_pending in pending_paths_by_class.items():
                class_images = all_paths_by_class[class_dir]
                if len(class_pending) == len(class_images):
                    stream = engine.process_folder(
                        str(class_dir),
                        chunk_size=NPU_INPUT_CHUNK_SIZE,
                        recursive=True,
                    )
                    stream_description = f"folder {class_dir.name}"
                    expected_paths = class_images
                else:
                    stream = engine.process_files(
                        [str(path) for path in class_pending],
                        chunk_size=NPU_INPUT_CHUNK_SIZE,
                    )
                    stream_description = f"{class_dir.name} (changed/new files)"
                    expected_paths = class_pending

                consume_engine_stream(
                    stream=stream,
                    dataset_dir=dataset_dir,
                    expected_paths=expected_paths,
                    feature_by_path=feature_by_path,
                    progress_state=progress_state,
                    total_pending=len(pending_paths),
                    stream_description=stream_description,
                )
        finally:
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass

    feature_matrix = np.stack([feature_by_path[path] for path in relative_paths]).astype(np.float32)
    sizes = np.asarray([item[0] for item in image_stats], dtype=np.int64)
    mtimes_ns = np.asarray([item[1] for item in image_stats], dtype=np.int64)

    # Tag the cache by backend and fixed RKNN batch variant so incompatible vectors are never mixed.
    np.savez_compressed(
        cache_file,
        paths=np.asarray(relative_paths, dtype=str),
        sizes=sizes,
        mtimes_ns=mtimes_ns,
        features=feature_matrix,
        backend=np.asarray(cache_backend, dtype=str),
    )
    print(f"[Cache] Saved {len(relative_paths)} feature vectors to {cache_file} ({cache_backend.upper()}).")
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
        help="Use TensorFlow/CPU for ResNet50V2 feature extraction instead of the RK3576 NPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_RKNN_BATCH_SIZE,
        help="Fixed batch size of the RKNN feature-extractor model (default: 4; ignored with --use-cpu).",
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=None,
        help="RKNN feature-extractor model path (default: matching batch model in ../resnet/models).",
    )
    args = parser.parse_args(argv)
    backend = "cpu" if args.use_cpu else "rknn"
    batch_size = DEFAULT_RKNN_BATCH_SIZE
    feature_model_file: Path | None = None
    if backend == "rknn":
        try:
            batch_size = validate_batch_size(args.batch_size)
            if args.extractor is not None:
                feature_model_file = args.extractor.expanduser().resolve()
            else:
                feature_model_file = (MODEL_DIR / model_filename_for_batch(batch_size)).resolve()
            model_batch_size = batch_size_from_model_path(feature_model_file)
            if model_batch_size != batch_size:
                raise ValueError(
                    f"Selected model has fixed batch size {model_batch_size}, "
                    f"but --batch-size is {batch_size}: {feature_model_file.name}."
                )
        except ValueError as exc:
            raise SystemExit(f"[Error] {exc}") from exc

    run_started = time.perf_counter()
    dataset_dir = DATASET_DIR.resolve()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "[Backend] TensorFlow/CPU feature extraction selected (--use-cpu)."
        if backend == "cpu"
        else f"[Backend] RK3576 NPU via shared ResNetEngine library (batch {batch_size})."
    )
    if backend == "rknn":
        print(f"[Model] Feature extractor: {feature_model_file}")
    class_names, images, labels = discover_dataset(dataset_dir)
    feature_started = time.perf_counter()
    relative_paths, features = extract_features(
        images, dataset_dir, CACHE_FILE, backend, feature_model_file, batch_size
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
        f"Feature extraction backend: {'TensorFlow/CPU' if backend == 'cpu' else f'ResNetEngine/RKNN batch {batch_size}'}",
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
