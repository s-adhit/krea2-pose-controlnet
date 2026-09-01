"""Lightweight validation of persisted paired evaluation geometry."""
from __future__ import annotations

from typing import Any, Mapping

from pose_controlnet.paired_preprocessing import resize_center_crop_geometry


def persisted_scoring_geometry(sample: Mapping[str, Any], *, label: str = "Turbo") -> dict[str, list[int]]:
    """Return canonical shard geometry and reject stale persisted fields."""
    stem = sample.get("stem", "<unknown>")
    fields = ("source_size", "resized_size", "crop_box", "bucket")
    missing = [field for field in fields if sample.get(field) is None]
    if missing:
        raise ValueError(f"{label} scoring geometry for stem {stem!r} is missing persisted paired fields: {', '.join(missing)}")
    try:
        source_size = tuple(sample["source_size"])
        bucket = tuple(sample["bucket"])
        canonical = resize_center_crop_geometry(source_size, bucket)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} scoring geometry for stem {stem!r} is malformed") from exc
    geometry = {
        "source_size": list(canonical.source_size),
        "resized_size": list(canonical.resized_size),
        "crop_box": list(canonical.crop_box),
        "bucket": list(canonical.bucket),
    }
    persisted = {field: sample[field] for field in geometry}
    if persisted != geometry:
        raise ValueError(f"{label} scoring geometry for stem {stem!r} disagrees with canonical paired preprocessing")
    return geometry
