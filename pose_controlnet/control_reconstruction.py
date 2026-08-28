"""Fail-closed raster reconstruction audit for authoritative pose targets."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from scipy.ndimage import distance_transform_edt

from pose_controlnet.pose_targets import PoseTargetError

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

# Calibrated from the deterministic full v3 baseline (16 samples from each
# available source): threshold-10 IoU 0.6380--0.8321, symmetric mean distance
# 0.0917--2.6909 px, and p95 1--3 px.  These are source-rerendered controls
# after the exact persisted PIL preprocessing, not fixed-stroke final renders.
PRIMARY_RECONSTRUCTION_CRITERIA = {
    "min_foreground_iou_at_10": 0.63,
    "max_symmetric_mean_distance": 2.75,
    "max_p95_foreground_distance": 3.0,
}


# OpenPose body-18 topology.  It matches the historic unified topology only
# when the source specification explicitly verifies that renderer provenance.
BODY_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 1), (0, 14), (0, 15), (14, 16), (15, 17))
COCO_TO_BODY18 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
# PoseBridge uses the standard OpenPose rainbow in limb order (RGB).
BODY_COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170))


def render_record(record: Mapping[str, Any], *, coordinate_space: str = "training") -> Image.Image:
    """Render a record with the verified historic renderer in an explicit frame.

    ``source`` consumes preserved raw authoritative coordinates.  This is the
    reconstruction path, including the reviewed off-canvas Human-Art values.
    ``training`` is retained solely as a useful direct final-frame raster
    diagnostic; it is not the primary reconstruction evidence after scaling.
    """
    renderer = record.get("renderer")
    if not isinstance(renderer, Mapping) or renderer.get("validated_historical_renderer") is not True:
        raise PoseTargetError(f"{record.get('stem')}: historical renderer has not been verified")
    if renderer.get("topology") != "openpose_body18":
        raise PoseTargetError(f"{record.get('stem')}: unsupported renderer topology")
    if coordinate_space == "source":
        width, height = map(int, record["source_size"])
        keypoint_field = "keypoints_source"
    elif coordinate_space == "training":
        width, height = map(int, record["bucket"])
        keypoint_field = "keypoints_training"
    else:
        raise PoseTargetError(f"{record.get('stem')}: unsupported reconstruction coordinate space {coordinate_space!r}")
    line_width = int(renderer.get("line_width", 3))
    endpoint_radius = int(renderer.get("endpoint_radius", 4))
    endpoint_rgb = tuple(renderer.get("endpoint_rgb", (255, 255, 255)))
    if line_width != 3 or endpoint_radius != 4 or endpoint_rgb != (255, 255, 255):
        raise PoseTargetError("Invalid historical renderer line parameters")
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for person in record["people"]:
        body = [[0.0, 0.0, 0.0] for _ in range(18)]
        for coco_index, body_index in enumerate(COCO_TO_BODY18):
            body[body_index] = person[keypoint_field][coco_index]
        left, right = body[5], body[2]
        if left[2] > 0 and right[2] > 0:
            body[1] = [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, min(left[2], right[2])]
        for limb_index, (first, second) in enumerate(BODY_LIMBS):
            if body[first][2] > 0 and body[second][2] > 0:
                first_xy = tuple(int(round(value)) for value in body[first][:2])
                second_xy = tuple(int(round(value)) for value in body[second][:2])
                draw.line((first_xy, second_xy), fill=BODY_COLORS[limb_index], width=3)
        # Endpoints are drawn after all limbs, matching PoseBridge's white
        # radius-four landmark circles.  The synthesized neck is intentionally
        # only here: no sidecar person receives it as a reward joint.
        for x, y, visibility in body:
            if visibility > 0:
                cx, cy = int(round(x)), int(round(y))
                draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 255))
    return canvas


def prepared_control_image(record: Mapping[str, Any], control_path: str | Path) -> Image.Image:
    """Put the stored source control into the exact persisted training frame."""
    expected = Image.open(control_path).convert("RGB")
    source_size = tuple(record.get("source_size", ()))
    resized_size = tuple(record.get("resized_size", ()))
    crop_box = tuple(record.get("crop_box", ()))
    if len(source_size) != 2 or len(resized_size) != 2 or len(crop_box) != 4:
        # Unit-test fixtures may already provide a final-frame control.
        return expected
    if expected.size != source_size:
        raise PoseTargetError(f"{record.get('stem')}: stored control size {expected.size} != source_size {source_size}")
    return expected.resize(resized_size, _LANCZOS).crop(crop_box)


def _foreground_metrics(expected: np.ndarray, reconstructed: np.ndarray, *, threshold: int) -> dict[str, float | int]:
    foreground_a = np.any(expected > threshold, axis=2)
    foreground_b = np.any(reconstructed > threshold, axis=2)
    union = int(np.logical_or(foreground_a, foreground_b).sum())
    intersection = int(np.logical_and(foreground_a, foreground_b).sum())
    if not foreground_a.any() and not foreground_b.any():
        symmetric_mean, p95 = 0.0, 0.0
    elif not foreground_a.any() or not foreground_b.any():
        symmetric_mean = p95 = float(max(foreground_a.shape))
    else:
        # EDT of an inverted foreground mask returns distance to the closest
        # foreground pixel.  Both directions make this symmetric and robust to
        # small raster-scale changes in a historically thin source stroke.
        distances = np.concatenate((distance_transform_edt(~foreground_b)[foreground_a], distance_transform_edt(~foreground_a)[foreground_b]))
        symmetric_mean, p95 = float(distances.mean()), float(np.percentile(distances, 95))
    return {
        "foreground_iou": 1.0 if union == 0 else intersection / union,
        "symmetric_foreground_mean_distance": symmetric_mean,
        "p95_foreground_distance": p95,
        "foreground_pixels_expected": int(foreground_a.sum()),
        "foreground_pixels_reconstructed": int(foreground_b.sum()),
    }


def compare_control(record: Mapping[str, Any], control_path: str | Path | Image.Image, *, threshold: int = 10) -> tuple[dict[str, Any], Image.Image, Image.Image, Image.Image]:
    """Compare stored control against a source-rerendered, identically prepared control.

    The returned direct-final metrics are retained as diagnostics only.  The
    source-rerendered metrics are the primary geometric evidence because the
    stored historical 3px raster is enlarged by the same PIL preprocessing.
    """
    expected = control_path.convert("RGB") if isinstance(control_path, Image.Image) else prepared_control_image(record, control_path)
    has_source_geometry = all(key in record for key in ("source_size", "resized_size", "crop_box"))
    source_reconstructed = render_record(record, coordinate_space="source") if has_source_geometry else render_record(record)
    reconstructed = source_reconstructed.resize(tuple(record["resized_size"]), _LANCZOS).crop(tuple(record["crop_box"])) if has_source_geometry else source_reconstructed
    if expected.size != reconstructed.size:
        raise PoseTargetError(f"{record.get('stem')}: stored control size {expected.size} != bucket {reconstructed.size}")
    a, b = np.asarray(expected), np.asarray(reconstructed)
    by_threshold = {str(value): _foreground_metrics(a, b, threshold=value) for value in (1, threshold, 32)}
    primary = by_threshold[str(threshold)]
    direct = np.asarray(render_record(record, coordinate_space="training"))
    direct_metrics = _foreground_metrics(a, direct, threshold=threshold)
    metrics = {
        "stem": record["stem"], "primary_comparison": "source_rerender_then_exact_pil_training_preprocessing",
        "foreground_metrics_by_threshold": by_threshold,
        "foreground_iou": primary["foreground_iou"],
        "symmetric_foreground_mean_distance": primary["symmetric_foreground_mean_distance"],
        "p95_foreground_distance": primary["p95_foreground_distance"],
        "mean_absolute_error": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()),
        "stored_foreground_pixels": primary["foreground_pixels_expected"], "reconstructed_foreground_pixels": primary["foreground_pixels_reconstructed"],
        "direct_final_vector_metrics": direct_metrics,
    }
    return metrics, expected, reconstructed, ImageChops.difference(expected, reconstructed)


def select_reconstruction_records(records: Iterable[Mapping[str, Any]], *, per_source: int) -> dict[str, list[Mapping[str, Any]]]:
    """Select only authoritative-target records; unavailable coverage is not a failure."""
    selected: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("pose_reward_available") is not True:
            continue
        source = str(record["source"])
        if len(selected[source]) < per_source:
            selected[source].append(record)
    return dict(selected)


def summarize_reconstruction(
    rows: Iterable[dict[str, Any]], *, min_foreground_iou: float = PRIMARY_RECONSTRUCTION_CRITERIA["min_foreground_iou_at_10"],
    max_mae: float | None = None, max_symmetric_mean_distance: float = PRIMARY_RECONSTRUCTION_CRITERIA["max_symmetric_mean_distance"],
    max_p95_foreground_distance: float = PRIMARY_RECONSTRUCTION_CRITERIA["max_p95_foreground_distance"],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    sources = {}
    for source, items in sorted(grouped.items()):
        failed = [item["stem"] for item in items if item["foreground_iou"] < min_foreground_iou or item["symmetric_foreground_mean_distance"] > max_symmetric_mean_distance or item["p95_foreground_distance"] > max_p95_foreground_distance or (max_mae is not None and item["mean_absolute_error"] > max_mae)]
        sources[source] = {
            "samples": len(items),
            "mean_foreground_iou_at_10": sum(item["foreground_iou"] for item in items) / len(items),
            "mean_symmetric_foreground_distance": sum(item["symmetric_foreground_mean_distance"] for item in items) / len(items),
            "max_p95_foreground_distance": max(item["p95_foreground_distance"] for item in items),
            "mean_absolute_error": sum(item["mean_absolute_error"] for item in items) / len(items),
            "failures": failed, "status": "PASS" if not failed else "FAIL",
        }
    return {"pass_criteria": {"primary_comparison": "source_rerender_then_exact_pil_training_preprocessing", "thresholds": [1, 10, 32], "min_foreground_iou_at_10": min_foreground_iou, "max_symmetric_foreground_mean_distance": max_symmetric_mean_distance, "max_p95_foreground_distance": max_p95_foreground_distance, "max_mean_absolute_error_diagnostic_only": max_mae}, "sources": sources, "status": "PASS" if all(item["status"] == "PASS" for item in sources.values()) else "FAIL"}
