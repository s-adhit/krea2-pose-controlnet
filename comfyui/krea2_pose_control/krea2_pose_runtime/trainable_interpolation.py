"""Strict interpolation for Pose Control-LoRA trainable tensor states."""
from __future__ import annotations

from typing import Mapping

import torch


def validate_trainable_interpolation(parent: Mapping[str, object], finish: Mapping[str, object]) -> None:
    """Require two complete, compatible trainable-model state mappings."""
    parent_keys, finish_keys = set(parent), set(finish)
    if parent_keys != finish_keys:
        raise ValueError(
            "Trainable interpolation requires exact matching trainable keys: "
            f"missing={sorted(parent_keys - finish_keys)[:5]}, "
            f"unexpected={sorted(finish_keys - parent_keys)[:5]}"
        )
    for key in sorted(parent_keys):
        parent_tensor, finish_tensor = parent[key], finish[key]
        if not isinstance(parent_tensor, torch.Tensor) or not isinstance(finish_tensor, torch.Tensor):
            raise ValueError(f"Trainable interpolation requires tensors only: {key}")
        if parent_tensor.shape != finish_tensor.shape:
            raise ValueError(
                f"Trainable interpolation tensor shape mismatch for {key}: "
                f"{tuple(parent_tensor.shape)} != {tuple(finish_tensor.shape)}"
            )
        if not (parent_tensor.is_floating_point() and finish_tensor.is_floating_point()):
            raise ValueError(f"Trainable interpolation requires floating trainable tensor: {key}")


def interpolate_trainable_state(parent: Mapping[str, object], finish: Mapping[str, object], alpha: float) -> dict[str, torch.Tensor]:
    """Blend matching trainable tensors in FP32, restoring the parent dtype."""
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise ValueError(f"Interpolation alpha must be strictly between zero and one, got {alpha!r}")
    validate_trainable_interpolation(parent, finish)
    return {
        key: (
            parent[key].detach().to(device="cpu", dtype=torch.float32) * (1.0 - alpha)
            + finish[key].detach().to(device="cpu", dtype=torch.float32) * alpha
        ).to(dtype=parent[key].dtype)
        for key in sorted(parent)
    }
