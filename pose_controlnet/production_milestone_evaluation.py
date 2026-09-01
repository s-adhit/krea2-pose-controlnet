"""Shared contracts for the dual-mode production Turbo milestone benchmark.

``native`` deliberately consumes the immutable diagnostic shard geometry.
``dynamic-768`` deliberately rebuilds geometry from the resolved source pair
with the same bucket selector and resize/crop implementation used by the
full-768 cache builder.  The two modes have separate artifact roots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from pose_controlnet.resolution_policy import RESOLUTION_768_BUCKETS
from pose_controlnet.evaluation_geometry import persisted_scoring_geometry
from pose_controlnet.paired_preprocessing import choose_bucket, resize_center_crop_geometry


# Historical parent-run defaults.  Callers may explicitly evaluate another
# exact local checkpoint series (for example the 3k -> 5k cooldown) without
# changing mode, geometry, or artifact-isolation rules.
PRODUCTION_MILESTONE_STEPS = (500, 1000, 1500, 2000, 2500, 3000)
EVALUATION_MODES = ("native", "dynamic-768")
SUMMARY_COLUMNS = ("checkpoint_step", "mode", "pck_005", "pck_010", "pck_020", "clip_mean_cosine_similarity")


class ProductionMilestoneEvaluationError(ValueError):
    """A dual-mode milestone evaluation violates its immutable contract."""


def normalize_modes(modes: Iterable[str] | None) -> tuple[str, ...]:
    """Validate requested modes and keep the caller's explicit ordering."""
    resolved = tuple(EVALUATION_MODES if modes is None else modes)
    if not resolved or len(resolved) != len(set(resolved)) or any(mode not in EVALUATION_MODES for mode in resolved):
        raise ProductionMilestoneEvaluationError(
            f"Evaluation modes must be a unique non-empty subset of {EVALUATION_MODES}"
        )
    return resolved


def mode_output_root(output_root: str | Path, step: int, mode: str) -> Path:
    """Return the only allowed root for one checkpoint/mode result."""
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ProductionMilestoneEvaluationError(f"Production milestone step must be a positive integer, got {step!r}")
    if mode not in EVALUATION_MODES:
        raise ProductionMilestoneEvaluationError(f"Unknown evaluation mode {mode!r}")
    return Path(output_root) / f"step_{step:06d}" / mode


def geometry_for_mode(*, mode: str, native_sample: Mapping[str, Any], source_size: tuple[int, int]) -> dict[str, list[int]]:
    """Return PCK geometry without ever substituting one mode for the other."""
    if mode == "native":
        # This is the locked historical path: validate and return the exact
        # paired geometry persisted alongside the native diagnostic latents.
        return persisted_scoring_geometry(native_sample, label="Turbo")
    if mode != "dynamic-768":
        raise ProductionMilestoneEvaluationError(f"Unknown evaluation mode {mode!r}")
    # These are the exact imports used by full_768_cache: no local bucket list
    # and no hand-written resize/crop arithmetic are permitted here.
    bucket = choose_bucket(source_size, RESOLUTION_768_BUCKETS)
    geometry = resize_center_crop_geometry(source_size, bucket)
    return {
        "source_size": list(geometry.source_size),
        "resized_size": list(geometry.resized_size),
        "crop_box": list(geometry.crop_box),
        "bucket": list(geometry.bucket),
    }


def mode_metadata(*, mode: str, stem: str, prompt: str, seed: int, geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Stable per-sample provenance, used to reject cross-mode reuse."""
    if mode not in EVALUATION_MODES:
        raise ProductionMilestoneEvaluationError(f"Unknown evaluation mode {mode!r}")
    required = ("source_size", "resized_size", "crop_box", "bucket")
    if any(key not in geometry for key in required):
        raise ProductionMilestoneEvaluationError("Evaluation geometry is incomplete")
    return {
        "mode": mode,
        "stem": stem,
        "prompt": prompt,
        "seed": int(seed),
        "source_size": list(geometry["source_size"]),
        "resized_size": list(geometry["resized_size"]),
        "crop_box": list(geometry["crop_box"]),
        "bucket": list(geometry["bucket"]),
    }


def assert_mode_metadata(metadata: Mapping[str, Any], *, mode: str, stem: str) -> None:
    """Fail closed if an artifact path contains another mode's provenance."""
    if metadata.get("mode") != mode or metadata.get("stem") != stem:
        raise ProductionMilestoneEvaluationError(
            f"Existing artifact metadata belongs to another mode or stem: expected {mode}/{stem}"
        )


def cross_checkpoint_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Make mode a required comparison dimension, never an implied default."""
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        step, mode = row.get("checkpoint_step"), row.get("mode")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0 or mode not in EVALUATION_MODES:
            raise ProductionMilestoneEvaluationError("Summary rows require a positive production milestone step and explicit mode")
        key = (step, mode)
        if key in seen:
            raise ProductionMilestoneEvaluationError(f"Duplicate cross-checkpoint summary row: {key}")
        seen.add(key)
        normalized.append(dict(row))
    normalized.sort(key=lambda row: (int(row["checkpoint_step"]), EVALUATION_MODES.index(str(row["mode"]))))
    return {"format_version": 1, "modes": list(EVALUATION_MODES), "checkpoints": normalized}
