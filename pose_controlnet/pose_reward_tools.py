"""Small, testable contracts for canonical pose-consistency tooling.

This module owns loss-combination guards used by production and preserved
historical experiments; the production implementation is in
``pose_controlnet.pose_consistency``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import torch

from pose_controlnet.pose_loss import combine_flow_and_pose_loss


GRAD_EPSILON = 1e-12


def select_trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return exactly the current optimizer boundary and reject frozen grads."""
    selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not selected:
        raise ValueError("model has no trainable parameters")
    selected_ids = {id(parameter) for _, parameter in selected}
    for _, parameter in model.named_parameters():
        if parameter.requires_grad != (id(parameter) in selected_ids):
            raise RuntimeError("inconsistent trainable parameter selection")
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError("frozen parameter unexpectedly has a gradient")
    return selected


def pose_active_mask(timesteps: torch.Tensor, pose_reward_available: torch.Tensor,
                     selected_timesteps: Sequence[float], *, tolerance: float = 1e-6) -> torch.Tensor:
    """Return per-sample eligibility without changing flow timestep sampling."""
    if timesteps.ndim != 1 or pose_reward_available.shape != timesteps.shape:
        raise ValueError("timesteps and pose_reward_available must be matching rank-1 tensors")
    selected = tuple(float(value) for value in selected_timesteps)
    if not selected or any(not 0.0 < value < 1.0 for value in selected):
        raise ValueError("selected pose timesteps must be non-empty values inside (0, 1)")
    if tolerance < 0:
        raise ValueError("timestep tolerance must be non-negative")
    matches = torch.zeros_like(timesteps, dtype=torch.bool)
    for value in selected:
        matches |= (timesteps - value).abs() <= tolerance
    return matches & pose_reward_available.to(device=timesteps.device, dtype=torch.bool)


def validate_smoke_invocation(*, lambda_pose: float | None, pose_timesteps: Sequence[float] | None,
                              run_name: str | None, checkpoint_dir: str | Path | None,
                              parent_checkpoint: str | Path, verify_parent: bool,
                              allow_existing_destination: bool = False) -> tuple[float, tuple[float, ...], Path]:
    """Require all scientific and destination choices at the command boundary."""
    if lambda_pose is None:
        raise ValueError("--lambda-pose is required")
    if not math.isfinite(lambda_pose) or lambda_pose <= 0:
        raise ValueError("--lambda-pose must be finite and positive")
    selected = tuple(float(value) for value in (pose_timesteps or ()))
    if not selected or any(not 0.0 < value < 1.0 for value in selected) or len(set(selected)) != len(selected):
        raise ValueError("--pose-timesteps must be a non-empty unique list inside (0, 1)")
    if not run_name or not run_name.strip():
        raise ValueError("--run-name is required")
    if checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required")
    parent = Path(parent_checkpoint)
    destination = Path(checkpoint_dir) / run_name
    if not allow_existing_destination and destination.resolve() == parent.parent.resolve():
        raise ValueError("smoke output must not share the parent checkpoint directory")
    if not allow_existing_destination and destination.exists():
        raise FileExistsError(f"Refusing to write smoke checkpoints into non-empty directory: {destination}")
    if verify_parent and not parent.is_file():
        raise FileNotFoundError(f"Required parent checkpoint is unavailable: {parent}")
    return float(lambda_pose), selected, destination


def gradient_interaction(flow_grads: Iterable[torch.Tensor | None], pose_grads: Iterable[torch.Tensor | None], *,
                         epsilon: float = GRAD_EPSILON) -> dict[str, float | None]:
    """Incrementally compute norms/dot/cosine without flattening large tensors."""
    flow_sq = pose_sq = dot = 0.0
    for flow, pose in zip(flow_grads, pose_grads):
        if flow is None and pose is None:
            continue
        if flow is None or pose is None or flow.shape != pose.shape:
            raise ValueError("matching flow/pose gradient tensors are required")
        flow32, pose32 = flow.detach().float(), pose.detach().float()
        if not torch.isfinite(flow32).all() or not torch.isfinite(pose32).all():
            raise FloatingPointError("non-finite parameter gradient")
        flow_sq += float(flow32.square().sum().item())
        pose_sq += float(pose32.square().sum().item())
        dot += float((flow32 * pose32).sum().item())
    flow_norm, pose_norm = math.sqrt(flow_sq), math.sqrt(pose_sq)
    ratio = pose_norm / flow_norm if flow_norm > epsilon else None
    cosine = dot / (flow_norm * pose_norm) if flow_norm > epsilon and pose_norm > epsilon else None
    return {"flow_grad_norm": flow_norm, "pose_grad_norm": pose_norm, "ratio": ratio, "dot": dot, "cosine": cosine}


def lambda_calibration(interaction: dict[str, float | None], targets: Sequence[float] = (.01, .05, .10, .20)) -> dict[str, float | None]:
    ratio = interaction["ratio"]
    return {f"lambda_{int(target * 100)}pct": target / ratio if ratio is not None and ratio > GRAD_EPSILON else None
            for target in targets}


def combined_gradient_diagnostics(interaction: dict[str, float | None], lambdas: dict[str, float | None]) -> dict[str, dict[str, float | None]]:
    flow_norm, pose_norm, dot = interaction["flow_grad_norm"], interaction["pose_grad_norm"], interaction["dot"]
    assert isinstance(flow_norm, float) and isinstance(pose_norm, float) and isinstance(dot, float)
    result: dict[str, dict[str, float | None]] = {}
    for name, value in lambdas.items():
        if value is None or flow_norm <= GRAD_EPSILON:
            result[name] = {"lambda": value, "total_over_flow": None, "cosine_total_flow": None}
            continue
        total_sq = flow_norm * flow_norm + 2 * value * dot + value * value * pose_norm * pose_norm
        total_norm = math.sqrt(max(total_sq, 0.0))
        result[name] = {"lambda": value, "total_over_flow": total_norm / flow_norm,
                        "cosine_total_flow": (flow_norm * flow_norm + value * dot) / (total_norm * flow_norm)
                        if total_norm > GRAD_EPSILON else None}
    return result
