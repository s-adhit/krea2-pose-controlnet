"""Exact, release-scoped PoseBridge COCO-17 conditioning renderer.

The model was trained on this raster contract.  It intentionally does not
accept OpenPose's hand/face extensions or arbitrary generic pose maps.
"""
from __future__ import annotations

from collections.abc import Iterable

from PIL import Image, ImageDraw

COCO17_JOINT_COUNT = 17
# COCO index -> historic unified Body-18 index.  Index 1 is a synthesized neck.
COCO_TO_BODY18 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
BODY_LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
)
BODY_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
)
LINE_WIDTH = 3
ENDPOINT_RADIUS = 4


class PoseRenderError(ValueError):
    """Raised when a caller supplies something other than COCO-17 joints."""


def _body18(coco17: Iterable[Iterable[float]], confidence_threshold: float) -> list[list[float]]:
    points = [list(map(float, point)) for point in coco17]
    if len(points) != COCO17_JOINT_COUNT or any(len(point) != 3 for point in points):
        raise PoseRenderError("expected exactly 17 COCO [x, y, confidence] joints")
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise PoseRenderError("keypoint confidence threshold must be in [0, 1]")
    body = [[0.0, 0.0, 0.0] for _ in range(18)]
    for coco_index, body_index in enumerate(COCO_TO_BODY18):
        x, y, score = points[coco_index]
        # A missing/low-confidence joint remains absent; no location is inferred.
        body[body_index] = [x, y, score if score >= confidence_threshold else 0.0]
    left_shoulder, right_shoulder = body[5], body[2]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0:
        body[1] = [
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
            min(left_shoulder[2], right_shoulder[2]),
        ]
    return body


def render_coco17_people(size: tuple[int, int], people: Iterable[Iterable[Iterable[float]]], *,
                          keypoint_confidence_threshold: float = 0.5) -> Image.Image:
    """Render COCO-17 people at source geometry using the training contract."""
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise PoseRenderError("render size must be positive")
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for person in people:
        body = _body18(person, keypoint_confidence_threshold)
        for index, (first, second) in enumerate(BODY_LIMBS):
            if body[first][2] > 0 and body[second][2] > 0:
                start = tuple(int(round(value)) for value in body[first][:2])
                end = tuple(int(round(value)) for value in body[second][:2])
                draw.line((start, end), fill=BODY_COLORS[index], width=LINE_WIDTH)
        # Draw after all limbs: this is significant for exact source appearance.
        for x, y, confidence in body:
            if confidence > 0:
                cx, cy = int(round(x)), int(round(y))
                draw.ellipse((cx - ENDPOINT_RADIUS, cy - ENDPOINT_RADIUS,
                              cx + ENDPOINT_RADIUS, cy + ENDPOINT_RADIUS), fill=(255, 255, 255))
    return canvas
