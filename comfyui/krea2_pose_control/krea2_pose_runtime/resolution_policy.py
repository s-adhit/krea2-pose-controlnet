"""Shared, locked image-geometry policies for production-facing surfaces."""
from __future__ import annotations


NATIVE_RESOLUTION_POLICY = "native"
RESOLUTION_768_POLICY = "768"

# Quantised to the project's 64-pixel convention while keeping each bucket
# close to 768² pixels.  This is a production geometry policy, not an
# overfit-experiment implementation detail.
RESOLUTION_768_BUCKETS: tuple[tuple[int, int], ...] = (
    (768, 768),
    (704, 896), (896, 704),
    (640, 960), (960, 640),
    (576, 1024), (1024, 576),
    (512, 1152), (1152, 512),
)


def canonical_resolution_policy(value: str) -> str:
    """Normalize a supported shared resolution policy name."""
    value = str(value).strip().lower()
    if value == "current":
        value = NATIVE_RESOLUTION_POLICY
    if value not in (NATIVE_RESOLUTION_POLICY, RESOLUTION_768_POLICY):
        raise ValueError(f"Unknown resolution policy {value!r}; choose native or 768")
    return value


def buckets_for_resolution(value: str) -> tuple[tuple[int, int], ...] | None:
    """Return alternate buckets; native means use persisted paired geometry."""
    return None if canonical_resolution_policy(value) == NATIVE_RESOLUTION_POLICY else RESOLUTION_768_BUCKETS
