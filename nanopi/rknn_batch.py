#!/usr/bin/env python3
"""Helpers for selecting and handling fixed-batch RKNN feature-extractor models."""

from __future__ import annotations

import re
from pathlib import Path

MODEL_STEM = "resnet50v2_feature_extractor_rk3576_fp16_320x180"
_BATCH_SUFFIX = re.compile(r"_batch(\d+)$")


def validate_batch_size(batch_size: int) -> int:
    """Accept batch size 1 or any positive even integer."""
    if batch_size < 1 or (batch_size != 1 and batch_size % 2 != 0):
        raise ValueError(
            "RKNN batch size must be 1 or a positive even integer (2, 4, 6, ...)."
        )
    return batch_size


def batch_size_from_model_path(model_path: Path) -> int:
    """Infer the fixed batch size from the converter's model filename convention."""
    match = _BATCH_SUFFIX.search(model_path.stem)
    if match is None:
        return 1
    try:
        batch_size = int(match.group(1))
    except ValueError as exc:  # Defensive; the regular expression already restricts digits.
        raise ValueError(f"Invalid RKNN batch suffix in model filename: {model_path.name}") from exc
    try:
        return validate_batch_size(batch_size)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported RKNN batch size in model filename '{model_path.name}': {exc}"
        ) from exc


def model_filename_for_batch(batch_size: int) -> str:
    """Return the artifact filename used by pc-wsl/build.sh."""
    validate_batch_size(batch_size)
    suffix = "" if batch_size == 1 else f"_batch{batch_size}"
    return f"{MODEL_STEM}{suffix}.rknn"


def resolve_feature_model_path(
    models_dir: Path,
    explicit_path: Path | None = None,
    batch_size: int = 4,
) -> Path:
    """Use an explicit model, or resolve the model matching the requested batch size."""
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    return (models_dir / model_filename_for_batch(batch_size)).resolve()
