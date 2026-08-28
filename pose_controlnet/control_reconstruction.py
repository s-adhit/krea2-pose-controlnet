"""Fail-closed raster reconstruction audit for authoritative pose targets."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from pose_controlnet.pose_targets import PoseTargetError


# OpenPose body-18 topology.  It matches the historic unified topology only
# when the source specification explicitly verifies that renderer provenance.
BODY18_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 1), (0, 14), (0, 15), (14, 16), (15, 17))
COCO_TO_BODY18 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)


def render_record(record: Mapping[str, Any]) -> Image.Image:
    """Render a record only with an explicitly verified historic renderer spec."""
    renderer = record.get("renderer")
    if not isinstance(renderer, Mapping) or renderer.get("validated_historical_renderer") is not True:
        raise PoseTargetError(f"{record.get('stem')}: historical renderer has not been verified")
    if renderer.get("topology") != "openpose_body18":
        raise PoseTargetError(f"{record.get('stem')}: unsupported renderer topology")
    width, height = map(int, record["bucket"])
    color = tuple(renderer.get("line_rgb", (255, 255, 255)))
    line_width = int(renderer.get("line_width", 4))
    if len(color) != 3 or line_width < 1:
        raise PoseTargetError("Invalid historical renderer line parameters")
    canvas = Image.new("RGB", (width, height), (0, 0, 0)); draw = ImageDraw.Draw(canvas)
    for person in record["people"]:
        body = [[0.0, 0.0, 0.0] for _ in range(18)]
        for coco_index, body_index in enumerate(COCO_TO_BODY18):
            body[body_index] = person["keypoints_training"][coco_index]
        left, right = body[5], body[2]
        if left[2] > 0 and right[2] > 0:
            body[1] = [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, min(left[2], right[2])]
        for first, second in BODY18_LIMBS:
            if body[first][2] > 0 and body[second][2] > 0:
                draw.line((body[first][0], body[first][1], body[second][0], body[second][1]), fill=color, width=line_width)
    return canvas


def compare_control(record: Mapping[str, Any], control_path: str | Path, *, threshold: int = 10) -> tuple[dict[str, Any], Image.Image, Image.Image, Image.Image]:
    expected = Image.open(control_path).convert("RGB")
    reconstructed = render_record(record)
    if expected.size != reconstructed.size:
        raise PoseTargetError(f"{record.get('stem')}: stored control size {expected.size} != bucket {reconstructed.size}")
    a, b = np.asarray(expected), np.asarray(reconstructed)
    foreground_a, foreground_b = np.any(a > threshold, axis=2), np.any(b > threshold, axis=2)
    union = int(np.logical_or(foreground_a, foreground_b).sum()); intersection = int(np.logical_and(foreground_a, foreground_b).sum())
    metrics = {"stem": record["stem"], "foreground_iou": 1.0 if union == 0 else intersection / union, "mean_absolute_error": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), "stored_foreground_pixels": int(foreground_a.sum()), "reconstructed_foreground_pixels": int(foreground_b.sum())}
    return metrics, expected, reconstructed, ImageChops.difference(expected, reconstructed)


def summarize_reconstruction(rows: Iterable[dict[str, Any]], *, min_foreground_iou: float, max_mae: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    sources = {}
    for source, items in sorted(grouped.items()):
        failed = [item["stem"] for item in items if item["foreground_iou"] < min_foreground_iou or item["mean_absolute_error"] > max_mae]
        sources[source] = {"samples": len(items), "mean_foreground_iou": sum(item["foreground_iou"] for item in items) / len(items), "mean_absolute_error": sum(item["mean_absolute_error"] for item in items) / len(items), "failures": failed, "status": "PASS" if not failed else "FAIL"}
    return {"pass_criteria": {"min_foreground_iou": min_foreground_iou, "max_mean_absolute_error": max_mae}, "sources": sources, "status": "PASS" if all(item["status"] == "PASS" for item in sources.values()) else "FAIL"}
