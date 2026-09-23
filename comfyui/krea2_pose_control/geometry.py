"""Shared release geometry helpers, copied from paired preprocessing semantics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from PIL import Image

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


@dataclass(frozen=True)
class Geometry:
    source_size: tuple[int, int]
    bucket: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]


def resize_center_crop_geometry(source_size: tuple[int, int], bucket: tuple[int, int]) -> Geometry:
    source_width, source_height = map(int, source_size)
    bucket_width, bucket_height = map(int, bucket)
    if min(source_width, source_height, bucket_width, bucket_height) <= 0:
        raise ValueError("image geometry must be positive")
    scale = max(bucket_width / source_width, bucket_height / source_height)
    resized = (round(source_width * scale), round(source_height * scale))
    left, top = (resized[0] - bucket_width) // 2, (resized[1] - bucket_height) // 2
    return Geometry((source_width, source_height), (bucket_width, bucket_height), resized,
                    (left, top, left + bucket_width, top + bucket_height))


def apply_geometry(image: Image.Image, geometry: Geometry) -> Image.Image:
    if image.size != geometry.source_size:
        raise ValueError("image does not match geometry source size")
    return image.resize(geometry.resized_size, _LANCZOS).crop(geometry.crop_box)
