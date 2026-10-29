#!/usr/bin/env python3
"""Convert the ResNet50V2 ONNX input from NHWC to NCHW and validate equivalence."""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from pathlib import Path

import numpy as np


RESNET_ROOT = Path(__file__).resolve().parent.parent / "resnet"
ARTIFACTS_DIR = RESNET_ROOT / "artifacts"
DEFAULT_INPUT = ARTIFACTS_DIR / "resnet50v2_features.onnx"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "resnet50v2_features_nchw.onnx"
DEFAULT_VALIDATION = ARTIFACTS_DIR / "resnet50v2_rknn_validation.npz"
DEFAULT_VALIDATION_DIR = RESNET_ROOT / "dataset"
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
FEATURE_COUNT = 2048
SYNTHETIC_SAMPLE_COUNT = 3
REAL_IMAGE_SAMPLE_COUNT = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fixed_shape(value_info) -> list[int | str | None]:
    shape: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def discover_validation_images(validation_dir: Path, limit: int) -> list[Path]:
    """Choose a small deterministic set of images, preferring different classes."""
    if not validation_dir.is_dir():
        return []

    class_dirs = sorted(
        (path for path in validation_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    chosen: list[Path] = []
    remaining: list[Path] = []

    for class_dir in class_dirs:
        files = sorted(
            (
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if files:
            chosen.append(files[0])
            remaining.extend(files[1:])

    if not class_dirs:
        remaining = sorted(
            (
                path for path in validation_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )

    seen = set(chosen)
    for path in remaining:
        if len(chosen) >= limit:
            break
        if path not in seen:
            chosen.append(path)
            seen.add(path)
    return chosen[:limit]


def make_synthetic_samples() -> tuple[list[np.ndarray], list[str]]:
    """Create deterministic inputs that exercise both continuous and RGB-like values."""
    rng = np.random.default_rng(20261020)
    samples = [
        rng.uniform(-1.0, 1.0, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
    ]
    names = ["synthetic-continuous"]

    for index in range(SYNTHETIC_SAMPLE_COUNT - 1):
        pixels = rng.integers(
            0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8
        ).astype(np.float32)
        samples.append(pixels / np.float32(127.5) - np.float32(1.0))
        names.append(f"synthetic-rgb-{index + 1}")
    return samples, names


def load_real_samples(paths: list[Path], validation_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load RGB images and apply the same resize and normalization used by training."""
    from PIL import Image, UnidentifiedImageError

    samples: list[np.ndarray] = []
    names: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
                pixels = np.asarray(image, dtype=np.float32)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"[Dataset] Skipping unreadable image '{path.name}': {exc}", file=sys.stderr)
            continue

        samples.append(pixels / np.float32(127.5) - np.float32(1.0))
        try:
            names.append(f"dataset/{path.relative_to(validation_dir).as_posix()}")
        except ValueError:
            names.append(f"dataset/{path.name}")
    return samples, names


def make_session_options(ort):
    """Configure predictable single-threaded execution for a small parity test."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return options


def run_outputs(session, samples: list[np.ndarray], input_layout: str) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    outputs: list[np.ndarray] = []
    for index, sample in enumerate(samples, start=1):
        tensor = sample[np.newaxis, ...]
        if input_layout == "NCHW":
            tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        result = session.run(None, {input_meta.name: tensor})[0]
        result = np.asarray(result, dtype=np.float32)
        if result.shape != (1, FEATURE_COUNT):
            raise RuntimeError(
                f"{input_layout} inference returned unexpected shape at sample {index}: {result.shape}."
            )
        if not np.isfinite(result).all():
            raise RuntimeError(f"{input_layout} inference returned non-finite values at sample {index}.")
        outputs.append(result.reshape(-1))
    return np.stack(outputs).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source NHWC ONNX model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output NCHW ONNX model.")
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION, help="Output validation archive.")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Optional image dataset for validation; defaults to resnet/dataset/ when present.",
    )
    args = parser.parse_args()

    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort

    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    validation_dir = args.validation_dir.expanduser().resolve() if args.validation_dir else DEFAULT_VALIDATION_DIR

    if not source_path.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {source_path}")
    if source_path == output_path:
        raise ValueError("Source and output ONNX paths must be different.")
    if validation_path in (source_path, output_path):
        raise ValueError("The validation archive must have a different path from both ONNX models.")
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)

    temp_onnx = output_path.with_name(output_path.stem + ".tmp.onnx")
    temp_validation = validation_path.with_name(validation_path.stem + ".tmp.npz")
    for path in (temp_onnx, temp_validation):
        if path.exists():
            path.unlink()

    print("=" * 68)
    print("Step 3: PREPARE NCHW ONNX FEATURE EXTRACTOR")
    print("=" * 68)
    print(f"[Source]     {source_path}")
    print(f"[Output]     {output_path}")
    print(f"[Validation] {validation_path}")

    try:
        print("\n[1/3] Checking source model...", flush=True)
        source_model = onnx.load(str(source_path))
        onnx.checker.check_model(source_model)
        graph = source_model.graph
        if len(graph.input) != 1 or len(graph.output) != 1:
            raise RuntimeError(
                f"Expected one input and one output; found {len(graph.input)} input(s) and {len(graph.output)} output(s)."
            )

        graph_input = graph.input[0]
        input_name = graph_input.name
        if fixed_shape(graph_input) != [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]:
            raise RuntimeError(
                f"Expected source input [1,{IMAGE_HEIGHT},{IMAGE_WIDTH},3] NHWC, found {fixed_shape(graph_input)}."
            )
        if graph_input.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError("Source input must be float32.")
        if fixed_shape(graph.output[0]) != [1, FEATURE_COUNT]:
            raise RuntimeError(
                f"Expected source output [1,{FEATURE_COUNT}], found {fixed_shape(graph.output[0])}."
            )
        if graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError("Source output must be float32.")

        print("  Input:  [1, 180, 320, 3] float32 NHWC")
        print("  Output: [1, 2048] float32 features")

        print("\n[2/3] Creating NCHW model...", flush=True)
        # Add a transpose at the input boundary only; keep all feature-extractor ops unchanged.
        internal_nhwc_name = f"{input_name}_internal_nhwc"
        for node in graph.node:
            for index, name in enumerate(node.input):
                if name == input_name:
                    node.input[index] = internal_nhwc_name

        transpose = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[internal_nhwc_name],
            perm=[0, 2, 3, 1],
            name="input_nchw_to_nhwc",
        )
        graph.node.insert(0, transpose)

        for dim, value in zip(
            graph_input.type.tensor_type.shape.dim,
            (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
        ):
            dim.ClearField("dim_param")
            dim.dim_value = value

        # The input-boundary transpose has a fully known shape, so shape inference is
        # unnecessary here. File-based checking avoids holding another large model copy.
        onnx.save(source_model, str(temp_onnx))
        del source_model, graph, graph_input, transpose
        gc.collect()
        onnx.checker.check_model(str(temp_onnx))

        validation_paths = discover_validation_images(validation_dir, REAL_IMAGE_SAMPLE_COUNT)
        synthetic_samples, sample_names = make_synthetic_samples()
        real_samples, real_names = load_real_samples(validation_paths, validation_dir) if validation_paths else ([], [])
        samples_nhwc = synthetic_samples + real_samples
        sample_names.extend(real_names)
        if not samples_nhwc:
            raise RuntimeError("No validation samples were available.")
        print(f"  Model layout: NCHW [1, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]")
        print(f"  Validation samples: {len(samples_nhwc)} ({len(synthetic_samples)} synthetic, {len(real_samples)} dataset)")

        session_options = make_session_options(ort)
        print("\n[3/3] Verifying numerical equivalence...", flush=True)
        print("  Running source model...", flush=True)
        original_session = ort.InferenceSession(
            str(source_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        source_input_shape = original_session.get_inputs()[0].shape
        source_output_shape = original_session.get_outputs()[0].shape
        if source_input_shape != [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]:
            raise RuntimeError(f"ONNX Runtime sees an unexpected source input shape: {source_input_shape}.")
        if source_output_shape != [1, FEATURE_COUNT]:
            raise RuntimeError(f"ONNX Runtime sees an unexpected source output shape: {source_output_shape}.")

        original_features = run_outputs(original_session, samples_nhwc, "NHWC")
        del original_session
        gc.collect()

        samples_nchw = [np.ascontiguousarray(np.transpose(sample, (2, 0, 1)), dtype=np.float32) for sample in samples_nhwc]
        print("  Running converted model...", flush=True)

        modified_session = ort.InferenceSession(
            str(temp_onnx), sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        modified_input_shape = modified_session.get_inputs()[0].shape
        modified_output_shape = modified_session.get_outputs()[0].shape
        if modified_input_shape != [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]:
            raise RuntimeError(f"ONNX Runtime sees an unexpected converted input shape: {modified_input_shape}.")
        if modified_output_shape != [1, FEATURE_COUNT]:
            raise RuntimeError(f"ONNX Runtime sees an unexpected converted output shape: {modified_output_shape}.")

        modified_features = run_outputs(modified_session, samples_nchw, "NCHW")
        del modified_session
        gc.collect()

        difference = modified_features.astype(np.float64) - original_features.astype(np.float64)
        mae = float(np.mean(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        max_abs = float(np.max(np.abs(difference)))
        if not np.allclose(original_features, modified_features, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                f"NHWC/NCHW numerical comparison failed (MAE={mae:.8f}, RMSE={rmse:.8f}, max_abs={max_abs:.8f})."
            )

        with temp_validation.open("wb") as stream:
            np.savez_compressed(
                stream,
                inputs_nchw=np.stack(samples_nchw).astype(np.float32),
                reference_features=original_features,
                sample_names=np.asarray(sample_names, dtype=str),
                input_layout=np.asarray("NCHW"),
                preprocessing=np.asarray("RGB float32 pixels resized to 320x180 and mapped to [-1, 1] before inference"),
                source_model=np.asarray(source_path.name),
            )

        os.replace(temp_onnx, output_path)
        os.replace(temp_validation, validation_path)
        print(f"  Samples checked:       {len(samples_nhwc)}")
        print(f"  Mean absolute error:   {mae:.8f}")
        print(f"  RMSE:                  {rmse:.8f}")
        print(f"  Maximum absolute diff: {max_abs:.8f}")
        print("\n[PASS] NCHW model validated; output features match the NHWC source.")
        print(f"[Model]      {output_path} ({output_path.stat().st_size:,} bytes)")
        print(f"[Validation] {validation_path} ({validation_path.stat().st_size:,} bytes)\n")
        return 0
    finally:
        for path in (temp_onnx, temp_validation):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[FAIL] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
