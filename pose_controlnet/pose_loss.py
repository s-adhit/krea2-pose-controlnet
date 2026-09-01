"""Neutral loss-combination contract for production pose consistency."""
from __future__ import annotations

import math

import torch


def combine_flow_and_pose_loss(flow_loss: torch.Tensor, pose_loss: torch.Tensor | None,
                               active_count: int, lambda_pose: float) -> torch.Tensor:
    """Apply pose loss only when its differentiable graph was constructed.

    The exact-zero endpoint retains the active-reference integrity checks while
    contributing no pose gradient, which is required by finishing checkpoints.
    """
    if not math.isfinite(lambda_pose) or lambda_pose < 0:
        raise ValueError("lambda_pose must be finite and non-negative when pose reward is enabled")
    if active_count < 0:
        raise ValueError("active_count must be non-negative")
    if active_count == 0:
        if pose_loss is not None:
            raise ValueError("pose_loss must be absent when no samples are pose-active")
        return flow_loss
    if pose_loss is None:
        raise ValueError("pose_loss is required when samples are pose-active")
    if lambda_pose == 0:
        return flow_loss
    return flow_loss + float(lambda_pose) * pose_loss
