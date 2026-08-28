"""Fail-closed raster reconstruction audit for authoritative pose targets."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from pose_controlnet.pose_targets import PoseTargetError

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


# OpenPose body-18 topology.  It matches the historic unified topology only
# when the source specification explicitly verifies that renderer provenance.
BODY_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 1), (0, 14), (0, 15), (14, 16), (15, 17))
COCO_TO_BODY18 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
# PoseBridge uses the standard OpenPose rainbow in limb order (RGB).
BODY_COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170))


def render_record(record: Mapping[str, Any]) -> Image.Image:
    """Render a record only with an explicitly verified historic renderer spec."""
    renderer = record.get("renderer")
    if not isinstance(renderer, Mapping) or renderer.get("validated_historical_renderer") is not True:
        raise PoseTargetError(f"{record.get('stem')}: historical renderer has not been verified")
    if renderer.get("topology") != "openpose_body18":
        raise PoseTargetError(f"{record.get('stem')}: unsupported renderer topology")
    width, height = map(int, record["bucket"])
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
            body[body_index] = person["keypoints_training"][coco_index]
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


def compare_control(record: Mapping[str, Any], control_path: str | Path | Image.Image, *, threshold: int = 10) -> tuple[dict[str, Any], Image.Image, Image.Image, Image.Image]:
    expected = control_path.convert("RGB") if isinstance(control_path, Image.Image) else prepared_control_image(record, control_path)
    reconstructed = render_record(record)
    if expected.size != reconstructed.size:
        raise PoseTargetError(f"{record.get('stem')}: stored control size {expected.size} != bucket {reconstructed.size}")
    a, b = np.asarray(expected), np.asarray(reconstructed)
    foreground_a, foreground_b = np.any(a > threshold, axis=2), np.any(b > threshold, axis=2)
    union = int(np.logical_or(foreground_a, foreground_b).sum()); intersection = int(np.logical_and(foreground_a, foreground_b).sum())
    metrics = {"stem": record["stem"], "foreground_iou": 1.0 if union == 0 else intersection / union, "mean_absolute_error": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), "stored_foreground_pixels": int(foreground_a.sum()), "reconstructed_foreground_pixels": int(foreground_b.sum())}
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


def summarize_reconstruction(rows: Iterable[dict[str, Any]], *, min_foreground_iou: float, max_mae: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    sources = {}
    for source, items in sorted(grouped.items()):
        failed = [item["stem"] for item in items if item["foreground_iou"] < min_foreground_iou or item["mean_absolute_error"] > max_mae]
        sources[source] = {"samples": len(items), "mean_foreground_iou": sum(item["foreground_iou"] for item in items) / len(items), "mean_absolute_error": sum(item["mean_absolute_error"] for item in items) / len(items), "failures": failed, "status": "PASS" if not failed else "FAIL"}
    return {"pass_criteria": {"min_foreground_iou": min_foreground_iou, "max_mean_absolute_error": max_mae}, "sources": sources, "status": "PASS" if all(item["status"] == "PASS" for item in sources.values()) else "FAIL"}
