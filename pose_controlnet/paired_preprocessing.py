"""Shared, read-only geometric preprocessing for paired pose-control images.

Geometry is deliberately computed once from the RGB/control pair's common
source size, then applied to both images.  Physical paths must arrive through
``pose_controlnet.dataset_index`` manifest records; this module never searches
the dataset filesystem or infers paths from filenames.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from pose_controlnet.dataset_index import (
    EXPECTED_SPLIT_COUNTS,
    DatasetIndex,
    DatasetIndexError,
    ManifestRecord,
)


_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


REFERENCE_KREA_BUCKETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
    (704, 1472),
    (1472, 704),
)


class PairedPreprocessingError(ValueError):
    """Raised when a resolved image/control pair cannot share geometry."""


@dataclass(frozen=True)
class ResizeCropGeometry:
    """One deterministic resize-to-cover and center-crop operation."""

    source_size: tuple[int, int]
    bucket: tuple[int, int]
    scale: float
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessedPair:
    """A resolved manifest pair after its shared geometry has been applied."""

    record: ManifestRecord
    geometry: ResizeCropGeometry
    rgb: Image.Image
    control: Image.Image


def choose_bucket(
    source_size: tuple[int, int],
    buckets: Sequence[tuple[int, int]] = REFERENCE_KREA_BUCKETS,
) -> tuple[int, int]:
    """Choose the nearest bucket by absolute log aspect-ratio distance."""
    width, height = _validate_size(source_size, "source")
    if not buckets:
        raise PairedPreprocessingError("At least one bucket is required")
    aspect_ratio = width / height
    try:
        return min(
            buckets,
            key=lambda bucket: abs(math.log(aspect_ratio) - math.log(_aspect_ratio(bucket))),
        )
    except (TypeError, ValueError) as exc:
        raise PairedPreprocessingError("Buckets must contain positive (width, height) pairs") from exc


def resize_center_crop_geometry(
    source_size: tuple[int, int], bucket: tuple[int, int]
) -> ResizeCropGeometry:
    """Return the reference resize-to-cover + center-crop geometry.

    The resize dimensions use Python's ``round`` and the crop origin uses
    floor integer division, matching the selected Krea reference behavior.
    """
    source_width, source_height = _validate_size(source_size, "source")
    bucket_width, bucket_height = _validate_size(bucket, "bucket")
    scale = max(bucket_width / source_width, bucket_height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    if resized_width < bucket_width or resized_height < bucket_height:
        raise PairedPreprocessingError(
            f"Resize-to-cover failed for source {source_size} and bucket {bucket}"
        )
    left = (resized_width - bucket_width) // 2
    top = (resized_height - bucket_height) // 2
    return ResizeCropGeometry(
        source_size=(source_width, source_height),
        bucket=(bucket_width, bucket_height),
        scale=scale,
        resized_size=(resized_width, resized_height),
        crop_box=(left, top, left + bucket_width, top + bucket_height),
    )


def preprocess_pair(
    record: ManifestRecord,
    *,
    buckets: Sequence[tuple[int, int]] = REFERENCE_KREA_BUCKETS,
) -> PreprocessedPair:
    """Open one indexed pair, verify source dimensions, and apply one geometry."""
    try:
        with Image.open(record.rgb_path) as rgb_source, Image.open(record.control_path) as control_source:
            rgb_size = rgb_source.size
            control_size = control_source.size
            if rgb_size != control_size:
                raise PairedPreprocessingError(
                    f"Source dimensions disagree for stem {record.stem!r}: "
                    f"RGB {rgb_size}, control {control_size}"
                )
            geometry = resize_center_crop_geometry(rgb_size, choose_bucket(rgb_size, buckets))
            rgb = apply_resize_center_crop_geometry(rgb_source.convert("RGB"), geometry)
            control = apply_resize_center_crop_geometry(control_source.convert("RGB"), geometry)
    except PairedPreprocessingError:
        raise
    except (OSError, ValueError) as exc:
        raise PairedPreprocessingError(
            f"Unable to read resolved pair for stem {record.stem!r}: "
            f"RGB={record.rgb_path}, control={record.control_path}"
        ) from exc
    return PreprocessedPair(record=record, geometry=geometry, rgb=rgb, control=control)


def preprocess_pair_with_persisted_geometry(
    record: ManifestRecord, geometry: ResizeCropGeometry
) -> PreprocessedPair:
    """Apply an already-verified paired geometry without recomputing it.

    Evaluation uses this for native-capacity examples: the persisted shard
    geometry is authoritative, so it must not be reselected from the source
    aspect ratio or silently replaced by an alternate-resolution policy.
    """
    try:
        with Image.open(record.rgb_path) as rgb_source, Image.open(record.control_path) as control_source:
            if rgb_source.size != geometry.source_size or control_source.size != geometry.source_size:
                raise PairedPreprocessingError(
                    f"Persisted source geometry disagrees for stem {record.stem!r}: "
                    f"expected {geometry.source_size}, RGB {rgb_source.size}, control {control_source.size}"
                )
            rgb = apply_resize_center_crop_geometry(rgb_source.convert("RGB"), geometry)
            control = apply_resize_center_crop_geometry(control_source.convert("RGB"), geometry)
    except PairedPreprocessingError:
        raise
    except (OSError, ValueError) as exc:
        raise PairedPreprocessingError(
            f"Unable to read resolved pair for stem {record.stem!r}: "
            f"RGB={record.rgb_path}, control={record.control_path}"
        ) from exc
    return PreprocessedPair(record=record, geometry=geometry, rgb=rgb, control=control)


def inspect_resolved_samples(
    records: Iterable[ManifestRecord], limit: int
) -> list[dict[str, object]]:
    """Process up to ``limit`` resolved records and return serializable facts."""
    if limit < 1:
        raise PairedPreprocessingError("Inspection limit must be at least 1")
    reports = []
    for record in records:
        pair = preprocess_pair(record)
        reports.append(
            {
                "stem": record.stem,
                "source_dimensions": list(pair.geometry.source_size),
                "bucket": list(pair.geometry.bucket),
                "resize_dimensions": list(pair.geometry.resized_size),
                "crop_box": list(pair.geometry.crop_box),
                "output_dimensions": list(pair.rgb.size),
            }
        )
        if len(reports) == limit:
            break
    return reports


def apply_resize_center_crop_geometry(image: Image.Image, geometry: ResizeCropGeometry) -> Image.Image:
    """Apply a precomputed shared resize-to-cover / center-crop operation.

    Local inference uses this alongside :func:`resize_center_crop_geometry` so
    a user-supplied skeleton follows exactly the same image transform as a
    training-pair control image.
    """
    if image.size != geometry.source_size:
        raise PairedPreprocessingError(
            f"Image size {image.size} does not match shared source size {geometry.source_size}"
        )
    resized = image.resize(geometry.resized_size, _LANCZOS)
    output = resized.crop(geometry.crop_box)
    if output.size != geometry.bucket:
        raise PairedPreprocessingError(
            f"Output size {output.size} does not match bucket {geometry.bucket}"
        )
    return output


# Kept private-name compatible for existing read-only diagnostics; new callers
# use the public helper above.
_apply_geometry = apply_resize_center_crop_geometry


def _validate_size(size: tuple[int, int], label: str) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise PairedPreprocessingError(f"{label.capitalize()} size must be a (width, height) pair")
    width, height = size
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise PairedPreprocessingError(f"{label.capitalize()} size must contain positive integers: {size!r}")
    return width, height


def _aspect_ratio(size: tuple[int, int]) -> float:
    width, height = _validate_size(size, "bucket")
    return width / height


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only paired geometry inspection.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED_SPLIT_COUNTS), default="train")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    try:
        index = DatasetIndex.discover(args.dataset_root)
        root = index.dataset_root
        validation = index.validate_manifests(
            {
                "train": root / "manifests/train.jsonl",
                "val": root / "manifests/val.jsonl",
                "diagnostic_val": root / "manifests/diagnostic_val.jsonl",
            },
            expected_counts=EXPECTED_SPLIT_COUNTS,
            expected_total=17_416,
        )
        reports = inspect_resolved_samples(validation.records_by_split[args.split], args.limit)
    except (DatasetIndexError, PairedPreprocessingError) as exc:
        parser.error(str(exc))
    print(json.dumps({"split": args.split, "samples": reports, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
