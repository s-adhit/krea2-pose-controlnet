"""DWPose body-only normalization for the frozen Krea-2 conditioning contract.

This module deliberately consumes DWPose's structured body results, never its
OpenPose-style preview raster.  The output is exactly one COCO-17 record per
detected person, in the original input-image pixel coordinate system.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

# Index in DWPose's OpenPose Body-18 ``pose.body.keypoints`` for each required
# COCO-17 joint.  DWPose's index 1 is its own synthesized neck and is never
# consumed here: the frozen Krea renderer is solely responsible for neck
# synthesis from the two retained shoulders.
DWPOSE_BODY18_FOR_COCO17 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
DWPOSE_BODY18_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
)


class DWPoseNormalizationError(ValueError):
    """Raised when a DWPose body result cannot be safely normalized."""


def _point_values(point: Any) -> tuple[float, float, float] | None:
    if point is None:
        return None
    try:
        if hasattr(point, "x"):
            values = (point.x, point.y, point.score)
        else:
            values = (point[0], point[1], point[2])
        x, y, score = (float(value) for value in values)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise DWPoseNormalizationError("DWPose body joint must provide x, y, and confidence") from error
    if not all(math.isfinite(value) for value in (x, y, score)):
        raise DWPoseNormalizationError("DWPose body joint contains NaN or Inf")
    return x, y, score


def normalize_dwpose_body18(body_keypoints: Sequence[Any] | Iterable[Any], *,
                             keypoint_confidence: float) -> list[list[float]]:
    """Map one DWPose Body-18 person to the project COCO-17 representation.

    Low-confidence or absent joints become explicit absent COCO entries.  No
    coordinate is inferred, resized, transposed, or borrowed from another
    person.  Hands, dense face points, and DWPose's synthesized neck are not
    read at all.
    """
    if not 0 <= keypoint_confidence <= 1:
        raise DWPoseNormalizationError("keypoint confidence threshold must be in [0, 1]")
    body = list(body_keypoints)
    if len(body) < 18:
        raise DWPoseNormalizationError("DWPose must return its 18 body joints before normalization")
    coco17: list[list[float]] = []
    for body_index in DWPOSE_BODY18_FOR_COCO17:
        values = _point_values(body[body_index])
        if values is None:
            coco17.append([0.0, 0.0, 0.0])
            continue
        x, y, score = values
        # Negative confidences are not valid detections and must not reach the
        # frozen renderer.  Coordinates are otherwise source-canvas pixels.
        if score < keypoint_confidence or score <= 0:
            coco17.append([0.0, 0.0, 0.0])
        else:
            coco17.append([x, y, score])
    return coco17


def dwpose_person_confidence(coco17: Sequence[Sequence[float]]) -> float:
    """Mean confidence of the retained, physical COCO-17 body joints."""
    scores = [float(point[2]) for point in coco17 if float(point[2]) > 0]
    return 0.0 if not scores else sum(scores) / len(scores)
