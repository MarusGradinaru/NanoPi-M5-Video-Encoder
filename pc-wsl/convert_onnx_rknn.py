#!/usr/bin/env python3
"""Convert a validated NCHW ONNX ResNet50V2 feature extractor to RK3576 RKNN."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np

RESNET_ROOT = Path(__file__).resolve().parent.parent / "resnet"
ARTIFACTS_DIR = RESNET_ROOT / "artifacts"
DEFAULT_INPUT = ARTIFACTS_DIR / "resnet50v2_features_nchw.onnx"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn"
DEFAULT_VALIDATION = ARTIFACTS_DIR / "resnet50v2_rknn_validation.npz"
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
FEATURE_COUNT = 2048
MIN_COSINE_SIMILARITY = 0.95


def parse_batch_size(value: str) -> int:
    """Accept batch size 1 or any positive even integer."""
    try:
        batch_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be an integer") from exc

    if batch_size < 1 or (batch_size != 1 and batch_size % 2 != 0):
        raise argparse.ArgumentTypeError(
            "batch size must be 1 or a positive even integer (2, 4, 6, ...)."
        )
    return batch_size


def output_path_for_batch(batch_size: int) -> Path:
    """Keep the historical filename for batch 1; distinguish larger batches."""
    if batch_size == 1:
        return DEFAULT_OUTPUT
    return DEFAULT_OUTPUT.with_name(
        f"{DEFAULT_OUTPUT.stem}_batch{batch_size}{DEFAULT_OUTPUT.suffix}"
    )


def validation_path_for_batch(batch_size: int) -> Path:
    """Keep the historical validation name for batch 1; distinguish larger batches."""
    if batch_size == 1:
        return DEFAULT_VALIDATION
    return DEFAULT_VALIDATION.with_name(
        f"{DEFAULT_VALIDATION.stem}_batch{batch_size}{DEFAULT_VALIDATION.suffix}"
    )


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


def load_validation_archive(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate the saved NCHW inputs and ONNX reference features."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"inputs_nchw", "reference_features", "sample_names", "input_layout"}
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(
                f"Validation archive is missing required fields: {', '.join(sorted(missing))}. "
                "Regenerate it by running make_features_nchw.py."
            )
        layout = str(archive["input_layout"].item())
        if layout != "NCHW":
            raise RuntimeError(f"Validation archive declares input layout {layout!r}; expected 'NCHW'.")
        inputs = np.asarray(archive["inputs_nchw"], dtype=np.float32)
        references = np.asarray(archive["reference_features"], dtype=np.float32)
        names = archive["sample_names"].astype(str).tolist()
    if inputs.ndim != 4 or inputs.shape[1:] != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(
            f"Unexpected validation input shape: {inputs.shape}; expected [N, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]."
        )
    if len(inputs) == 0:
        raise RuntimeError("Validation archive contains no samples.")
    if references.shape != (len(inputs), FEATURE_COUNT):
        raise RuntimeError(
            f"Unexpected reference feature shape: {references.shape}; expected ({len(inputs)}, {FEATURE_COUNT})."
        )
    if len(names) != len(inputs):
        raise RuntimeError("Validation archive contains a different number of sample names and input tensors.")
    if not np.isfinite(inputs).all() or not np.isfinite(references).all():
        raise RuntimeError("Validation archive contains NaN or infinite values.")
    return inputs, references, names


def cosine_similarity(reference: np.ndarray, result: np.ndarray) -> float:
    ref64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    out64 = np.asarray(result, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(ref64) * np.linalg.norm(out64))
    if denominator == 0.0:
        return float(np.array_equal(ref64, out64))
    return float(np.dot(ref64, out64) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Validated NCHW ONNX model.")
    parser.add_argument(
        "--batch-size",
        type=parse_batch_size,
        default=1,
        metavar="N",
        help="RKNN inference batch size: 1 or a positive even integer (default: 1).",
    )
    parser.add_argument("--platform", default="rk3576", choices=("rk3576",), help="Target Rockchip platform.")
    parser.add_argument(
        "--validation",
        type=Path,
        default=None,
        help="Override the validation archive path (default is selected by --batch-size).",
    )
    parser.add_argument(
        "--skip-simulator-test", action="store_true",
        help="Skip host simulator inference. Device validation is still required.",
    )
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = output_path_for_batch(args.batch_size).resolve()
    selected_validation = (
        args.validation
        if args.validation is not None
        else validation_path_for_batch(args.batch_size)
    )
    validation_path = selected_validation.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"NCHW ONNX model not found: {input_path}; run make_onnx_nchw.py first.")
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validation archive not found: {validation_path}; run make_onnx_nchw.py first."
        )
    if input_path == output_path:
        raise ValueError("Input ONNX and output RKNN paths must be different.")
    if output_path.suffix.lower() != ".rknn":
        raise ValueError(f"Output file must have the .rknn extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".tmp.rknn")
    if temp_path.exists():
        temp_path.unlink()
    try:
        import onnx

        print("=" * 68)
        print("Step 4: CONVERT ONNX FEATURE EXTRACTOR -> RKNN")
        print("=" * 68)
        print(f"[Input]       {input_path}")
        print(f"[Platform]    {args.platform}")
        print("[Precision]   FP16, without INT8 quantization")
        print(f"[Batch size]  {args.batch_size}")
        print(f"[Output]      {output_path}")
        print(f"[Validation]  {validation_path}")
        print("\n[1/4] Checking ONNX model and validation data...", flush=True)
        onnx_model = onnx.load(str(input_path))
        onnx.checker.check_model(onnx_model)
        if len(onnx_model.graph.input) != 1:
            raise RuntimeError(f"Expected one ONNX input, found {len(onnx_model.graph.input)}.")
        if len(onnx_model.graph.output) != 1:
            raise RuntimeError(f"Expected one ONNX output, found {len(onnx_model.graph.output)}.")
        if fixed_shape(onnx_model.graph.input[0]) != [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]:
            raise RuntimeError(
                f"ONNX input must be NCHW [1, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}], "
                f"found {fixed_shape(onnx_model.graph.input[0])}."
            )
        if onnx_model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError("ONNX input must be float32.")
        if fixed_shape(onnx_model.graph.output[0]) != [1, FEATURE_COUNT]:
            raise RuntimeError(
                f"ONNX output must be [1, {FEATURE_COUNT}], "
                f"found {fixed_shape(onnx_model.graph.output[0])}."
            )
        if onnx_model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError("ONNX output must be float32.")
        del onnx_model
        gc.collect()
        inputs, references, sample_names = load_validation_archive(validation_path)
        print(f"  ONNX input shape: [1, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}] float32 NCHW")
        print(f"  RKNN input shape: [{args.batch_size}, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}] float32 NCHW")
        print(f"  RKNN output shape: [{args.batch_size}, {FEATURE_COUNT}] float32")
        print(f"  Validation samples: {len(inputs)}")
        print(f"  Minimum cosine similarity: {MIN_COSINE_SIMILARITY:.3f}")
        print("\n[2/4] Configuring RKNN Toolkit2...", flush=True)
        # Suppress only RKNN's known pkg_resources deprecation warning.
        # Do not intercept stdout/stderr: RKNN's progress bars must remain untouched.
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\.",
            category=UserWarning,
            module=r"rknn\.api\.rknn",
        )
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        try:
            ret = rknn.config(
                target_platform=args.platform,
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[1.0, 1.0, 1.0]],
                optimization_level=2,
            )
            if ret not in (None, 0):
                raise RuntimeError(f"RKNN configuration failed with return code {ret}.")
            ret = rknn.load_onnx(model=str(input_path))
            if ret != 0:
                raise RuntimeError(f"Could not load ONNX model into RKNN Toolkit2 (code {ret}).")
            print("\n[3/4] Building and exporting FP16 RKNN model...", flush=True)
            if args.batch_size == 1:
                # Preserve the original build call and the historical batch-1 model.
                ret = rknn.build(do_quantization=False)
            else:
                ret = rknn.build(
                    do_quantization=False,
                    rknn_batch_size=args.batch_size,
                )
            if ret != 0:
                raise RuntimeError(f"RKNN model build failed with return code {ret}.")
            ret = rknn.export_rknn(str(temp_path))
            if ret != 0:
                raise RuntimeError(f"RKNN export failed with return code {ret}.")
            if not temp_path.is_file() or temp_path.stat().st_size == 0:
                raise RuntimeError("RKNN export did not produce a non-empty model file.")
            if args.skip_simulator_test:
                print("\n[4/4] Simulator test skipped by request.", flush=True)
            else:
                print("\n[4/4] Validating RKNN outputs in the host simulator...", flush=True)
                ret = rknn.init_runtime()
                if ret not in (None, 0):
                    raise RuntimeError(f"RKNN simulator initialization failed with return code {ret}.")
                results = []
                failures = []
                for start in range(0, len(inputs), args.batch_size):
                    batch_inputs = inputs[start:start + args.batch_size]
                    batch_count = len(batch_inputs)
                    batch_names = sample_names[start:start + batch_count]
                    batch_references = references[start:start + batch_count]

                    # RKNN models with a fixed batch dimension require a full batch.
                    # Pad only the final incomplete batch; its extra outputs are ignored.
                    if batch_count < args.batch_size:
                        padding_count = args.batch_size - batch_count
                        padding = np.repeat(batch_inputs[-1:], padding_count, axis=0)
                        batch_inputs = np.concatenate((batch_inputs, padding), axis=0)

                    tensor = np.ascontiguousarray(batch_inputs, dtype=np.float32)
                    output_list = rknn.inference(inputs=[tensor], data_format=["nchw"])
                    if not output_list or output_list[0] is None:
                        raise RuntimeError(
                            f"RKNN simulator returned no output for batch starting at sample {start + 1}."
                        )
                    batch_results = np.asarray(output_list[0], dtype=np.float32)
                    expected_output_size = args.batch_size * FEATURE_COUNT
                    if batch_results.size != expected_output_size:
                        raise RuntimeError(
                            f"Unexpected RKNN output shape for batch starting at sample {start + 1}: "
                            f"{batch_results.shape}; expected {expected_output_size} values "
                            f"([{args.batch_size}, {FEATURE_COUNT}])."
                        )
                    batch_results = batch_results.reshape(args.batch_size, FEATURE_COUNT)
                    if not np.isfinite(batch_results).all():
                        raise RuntimeError(
                            f"RKNN output contains NaN or infinite values in batch starting at sample {start + 1}."
                        )

                    for offset in range(batch_count):
                        index = start + offset
                        name = batch_names[offset]
                        reference = batch_references[offset]
                        result = batch_results[offset]
                        difference = result.astype(np.float64) - reference.astype(np.float64)
                        mae = float(np.mean(np.abs(difference)))
                        rmse = float(np.sqrt(np.mean(np.square(difference))))
                        max_abs = float(np.max(np.abs(difference)))
                        cosine = cosine_similarity(reference, result)
                        passed = cosine >= MIN_COSINE_SIMILARITY
                        if not passed:
                            failures.append(name)
                        results.append((mae, rmse, max_abs, cosine))
                        status = "PASS" if passed else "FAIL"
                        print(
                            f"  {index + 1:>2}/{len(inputs)} {name}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
                            f"max_abs={max_abs:.6f}, cosine={cosine:.6f} [{status}]",
                            flush=True,
                        )
                mean_mae = float(np.mean([row[0] for row in results]))
                mean_rmse = float(np.mean([row[1] for row in results]))
                max_abs_all = float(max(row[2] for row in results))
                min_cosine = float(min(row[3] for row in results))
                print(f"  Mean MAE:              {mean_mae:.6f}")
                print(f"  Mean RMSE:             {mean_rmse:.6f}")
                print(f"  Maximum absolute diff: {max_abs_all:.6f}")
                print(f"  Minimum cosine:        {min_cosine:.6f}")
                if failures:
                    raise RuntimeError(
                        "RKNN simulator output did not meet the similarity criterion for: "
                        + ", ".join(failures)
                    )
            os.replace(temp_path, output_path)
        finally:
            try:
                rknn.release()
            except Exception:
                pass
        if args.skip_simulator_test:
            print("\n[PASS] RKNN model built; simulator test was skipped.")
        else:
            print("\n[PASS] RKNN model built and all simulator checks passed.")
        print(f"[Model] {output_path} ({output_path.stat().st_size:,} bytes)")
        print("[Next] Validate this candidate on the NanoPi before using it in the application.\n")
        return 0
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[FAIL] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
