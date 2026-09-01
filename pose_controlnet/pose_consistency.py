"""Canonical production pose-consistency loss and resumable diagnostics.

The production objective is flow-matching MSE plus the selected normalized
coordinate Huber auxiliary term.  This module deliberately owns the shared
implementation so production training never depends on an experimental script.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol

import torch
import torch.nn.functional as F
from einops import rearrange

from pose_controlnet.diffusion import (
    forward_pose_control, make_flow_pair, patchify_and_position,
    sample_controlled_pose_exposure_timestep,
)
from pose_controlnet.pose_critic import FixedBoxKeypointRCNNCritic, differentiable_pose_loss
from pose_controlnet.pose_loss import combine_flow_and_pose_loss
from pose_controlnet.vae_preprocessing import decode_normalized_latents_autograd, qwen_decoded_to_unit_rgb


def pose_active_window(timesteps: torch.Tensor, available: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Inclusive production pose eligibility; flow timestep sampling is unchanged."""
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError("pose timestep window must satisfy 0 < min <= max < 1")
    if timesteps.ndim != 1 or available.shape != timesteps.shape:
        raise ValueError("timestep and availability shapes must match")
    return available.to(device=timesteps.device, dtype=torch.bool) & (timesteps >= lower) & (timesteps <= upper)


def should_build_pose_graph(active_indices: torch.Tensor) -> bool:
    """Avoid VAE/critic work when no sample is pose-active."""
    return bool(active_indices.numel())


def _person_tensors(record: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes, targets, valid = [], [], []
    for person in record["people"]:
        xywh = person.get("bbox_training_xywh")
        joints = person.get("joint_provenance")
        if not isinstance(xywh, list) or len(xywh) != 4 or not isinstance(joints, list) or len(joints) != 17:
            raise ValueError(f"{record['stem']}: incomplete authoritative person geometry")
        x, y, width, height = map(float, xywh)
        if width <= 0 or height <= 0:
            continue
        boxes.append((x, y, x + width, y + height))
        targets.append([joint["training_coordinate"] for joint in joints])
        valid.append([bool(joint["reward_joint_valid"]) for joint in joints])
    if not boxes:
        raise ValueError(f"{record['stem']}: no positive-area authoritative fixed boxes")
    return (
        torch.tensor(boxes, device=device, dtype=torch.float32),
        torch.tensor(targets, device=device, dtype=torch.float32),
        torch.tensor(valid, device=device, dtype=torch.bool),
    )


def unpatchify_latent_tokens(tokens: torch.Tensor, latent_hw: tuple[int, int], patch: int) -> torch.Tensor:
    """Invert the project image-token layout without changing token order."""
    height, width = latent_hw
    if height % patch or width % patch:
        raise ValueError("latent spatial dimensions must be divisible by model patch size")
    expected_tokens = (height // patch) * (width // patch)
    if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
        raise ValueError("token count does not match requested latent spatial dimensions")
    channels = tokens.shape[2] // (patch * patch)
    if channels * patch * patch != tokens.shape[2]:
        raise ValueError("token feature width is not an integral patch layout")
    return rearrange(tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                     h=height // patch, w=width // patch, c=channels, ph=patch, pw=patch)


def production_pose_consistency_loss(
    model: torch.nn.Module, vae: Any, critic: FixedBoxKeypointRCNNCritic,
    batch: dict[str, Any], sidecar_by_stem: Mapping[str, dict[str, Any]],
    cfg: PoseConsistencyRuntimeConfig, device: torch.device, generator: torch.Generator,
    *, pose_loss_name: str, lambda_pose: float, timestep_min: float,
    timestep_max: float, forced_exposure_probability: float,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the canonical flow-MSE + normalized-coordinate-Huber objective.

    The timestep sampler, active-set handling, Huber target normalization, and
    diagnostics intentionally match the production checkpoint provenance.
    """
    clean = batch["latent"].to(device=device, dtype=torch.float32, non_blocking=True)
    control = batch["control"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    if clean.shape != control.shape or not torch.isfinite(clean).all() or not torch.isfinite(control).all():
        raise FloatingPointError("invalid paired latent batch")
    records = [sidecar_by_stem[stem] for stem in batch["stems"]]
    available = torch.tensor([bool(record.get("pose_reward_available", False)) for record in records], device=device)
    timestep, forced, natural_active, active = sample_controlled_pose_exposure_timestep(
        clean.shape[0], (clean.shape[-2] // model.config.patch) * (clean.shape[-1] // model.config.patch),
        cfg, device, generator, pose_reward_available=available,
        force_probability=forced_exposure_probability, final_timestep_min=timestep_min,
        final_timestep_max=timestep_max,
    )
    noise = torch.randn(clean.shape, device=device, dtype=torch.float32, generator=generator)
    noisy, target = make_flow_pair(clean, noise, timestep)
    context = batch["context"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    text_mask = batch["text_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    image_tokens, pos, mask = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
    control_tokens, _, _ = patchify_and_position(control, context.shape[1], model.config.patch, text_mask)
    target_tokens, _, _ = patchify_and_position(target, context.shape[1], model.config.patch, text_mask)
    velocity = forward_pose_control(model, image_tokens, control_tokens, context, timestep.to(torch.bfloat16), pos, mask,
                                    gradient_checkpointing_blocks=cfg.gradient_checkpointing_blocks)
    flow_loss = F.mse_loss(velocity.float(), target_tokens.float())
    if not torch.isfinite(flow_loss):
        raise FloatingPointError("non-finite flow-matching MSE")
    active_indices = active.nonzero(as_tuple=False).flatten()
    pose_loss: torch.Tensor | None = None
    if should_build_pose_graph(active_indices):
        x0_hat_tokens = image_tokens - timestep.view(-1, 1, 1).to(image_tokens) * velocity
        x0_hat = unpatchify_latent_tokens(x0_hat_tokens[active_indices], tuple(clean.shape[-2:]), model.config.patch)
        decoded = qwen_decoded_to_unit_rgb(decode_normalized_latents_autograd(vae, x0_hat)).float()
        boxes, targets, valid = zip(*(_person_tensors(records[index], device) for index in active_indices.tolist()))
        heatmaps = critic(decoded, list(boxes))
        pose_loss = differentiable_pose_loss(
            pose_loss_name, heatmaps.logits, torch.cat(targets), heatmaps.boxes_training, torch.cat(valid),
            temperature=1.0, gaussian_sigma=1.5,
        )
        if not torch.isfinite(pose_loss):
            raise FloatingPointError(f"non-finite pose loss: {pose_loss_name}")
    total = combine_flow_and_pose_loss(flow_loss, pose_loss, int(active_indices.numel()), lambda_pose)
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite total loss")
    if not collect_diagnostics:
        return total, {"pose_active_count_tensor": active.sum(), "pose_eligible_count_tensor": available.sum()}
    return total, {
        "flow_loss": float(flow_loss.item()), "pose_loss": float(pose_loss.item()) if pose_loss is not None else None,
        "total_loss": float(total.item()), "pose_active_fraction": float(active.float().mean().item()),
        "pose_active_count": int(active_indices.numel()), "pose_eligible_count": int(available.sum().item()),
        "pose_forced_count": int(forced.sum().item()), "pose_natural_active_count": int(natural_active.sum().item()),
        "timesteps": [float(value) for value in timestep.detach().cpu()],
        "active_timesteps": [float(value) for value in timestep[active].detach().cpu()],
    }


def aggregate_step_diagnostics(microbatch_diagnostics: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Report optimizer-step diagnostics across every accumulation microbatch."""
    if not microbatch_diagnostics:
        raise ValueError("Cannot aggregate an empty optimizer step")
    flow_losses = [float(item["flow_loss"]) for item in microbatch_diagnostics]
    total_losses = [float(item["total_loss"]) for item in microbatch_diagnostics]
    timesteps = [float(value) for item in microbatch_diagnostics for value in item["timesteps"]]
    active_counts = [int(item["pose_active_count"]) for item in microbatch_diagnostics]
    eligible_counts = [int(item["pose_eligible_count"]) for item in microbatch_diagnostics]
    forced_counts = [int(item.get("pose_forced_count", 0)) for item in microbatch_diagnostics]
    natural_active_counts = [int(item.get("pose_natural_active_count", item["pose_active_count"])) for item in microbatch_diagnostics]
    pose_losses = [float(item["pose_loss"]) for item in microbatch_diagnostics if item["pose_loss"] is not None]
    if not timesteps:
        raise ValueError("Optimizer-step diagnostics contain no timesteps")
    active_samples = sum(active_counts)
    forced_samples, natural_active_samples, eligible_samples = sum(forced_counts), sum(natural_active_counts), sum(eligible_counts)
    if active_samples != forced_samples + natural_active_samples:
        raise ValueError("Pose-exposure diagnostics double-count or omit active samples")
    active_timesteps = [float(value) for item in microbatch_diagnostics for value in item.get("active_timesteps", [])]
    return {
        "flow_loss": sum(flow_losses) / len(flow_losses), "total_loss": sum(total_losses) / len(total_losses),
        "pose_loss": sum(pose_losses) / len(pose_losses) if pose_losses else None,
        "pose_active_count": active_samples, "pose_active_fraction": active_samples / len(timesteps),
        "pose_active_samples_step": active_samples,
        "pose_active_microbatches_step": sum(count > 0 for count in active_counts),
        "pose_eligible_samples_step": eligible_samples, "pose_forced_samples_step": forced_samples,
        "pose_natural_active_samples_step": natural_active_samples,
        "pose_forced_fraction_of_eligible_step": forced_samples / eligible_samples if eligible_samples else 0.0,
        "pose_total_active_fraction_of_eligible_step": active_samples / eligible_samples if eligible_samples else 0.0,
        "pose_loss_mean_active": sum(pose_losses) / len(pose_losses) if pose_losses else None,
        "pose_loss_max_active": max(pose_losses) if pose_losses else None,
        "flow_loss_mean_step": sum(flow_losses) / len(flow_losses),
        "total_loss_mean_step": sum(total_losses) / len(total_losses),
        "timestep_min_step": min(timesteps), "timestep_max_step": max(timesteps),
        "timestep_mean_step": sum(timesteps) / len(timesteps),
        "active_timestep_min_step": min(active_timesteps) if active_timesteps else None,
        "active_timestep_mean_step": sum(active_timesteps) / len(active_timesteps) if active_timesteps else None,
        "active_timestep_max_step": max(active_timesteps) if active_timesteps else None,
    }


def update_cumulative_counters(counters: Mapping[str, int], metrics: Mapping[str, Any]) -> dict[str, int]:
    """Carry pose-exposure evidence across atomically resumable checkpoints."""
    updated = {key: int(value) for key, value in counters.items()}
    for counter, metric in (("eligible_samples_seen", "pose_eligible_samples_step"),
                            ("forced_samples", "pose_forced_samples_step"),
                            ("naturally_active_samples", "pose_natural_active_samples_step"),
                            ("total_active_samples", "pose_active_samples_step")):
        updated[counter] += int(metrics[metric])
    if updated["total_active_samples"] != updated["forced_samples"] + updated["naturally_active_samples"]:
        raise AssertionError("Cumulative pose exposure counters double-count activity")
    return updated
class PoseConsistencyRuntimeConfig(Protocol):
    """Only the established runtime fields consumed by this loss path."""
    gradient_checkpointing_blocks: int
    mu_x1: float
    mu_y1: float
    mu_x2: float
    mu_y2: float
    timestep_aux_prob: float
    timestep_aux_min: float
    timestep_aux_max: float

