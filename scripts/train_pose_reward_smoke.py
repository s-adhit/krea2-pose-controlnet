"""Explicit, isolated Gate-E pose-reward continuation smoke trainer.

Normal ``train.py`` remains flow-only.  This command is blocked unless an
operator provides the parent, lambda, timestep window, and isolated run name.
It samples production flow timesteps unchanged and only adds the explicitly
selected differentiable pose loss to eligible samples within that supplied
window.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

import train
from pose_controlnet.checkpointing import HFTrainingCheckpointMirror, load_training_state, save_training_state
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.diffusion import (
    CONTROLLED_POSE_EXPOSURE_POLICY_VERSION, forward_pose_control, make_flow_pair,
    patchify_and_position, sample_controlled_pose_exposure_timestep,
)
from pose_controlnet.keypoint_critic import (
    POSE_LOSS_NAMES,
    FixedBoxKeypointRCNNCritic,
    differentiable_pose_loss,
)
from pose_controlnet.keypoint_critic_audit import assert_frozen_no_parameter_grad
from pose_controlnet.model import audit_control_model, build_pose_model, trainable_params, trainable_state_dict
from pose_controlnet.pose_reward_tools import combine_flow_and_pose_loss, validate_smoke_invocation
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.seed import set_seed
from pose_controlnet.vae_preprocessing import decode_normalized_latents_autograd, load_krea_vae, qwen_decoded_to_unit_rgb
from pose_controlnet.wandb_logging import OptionalWandbMirror
from scripts.audit_keypoint_critic import _person_tensors
from scripts.audit_keypoint_critic_timestep import _sha256


GATE_E_METADATA_KEY = "gate_e"
GATE_E_METADATA_FORMAT = 2
_GATE_E_CRITICAL_CONFIG_FIELDS = (
    "raw_ckpt", "shard_dir", "rank", "alpha", "lr", "microbatch_size",
    "gradient_accumulation_steps", "warmup_steps", "max_grad_norm",
    "caption_dropout", "control_dropout", "compile", "fused_adamw", "gradient_checkpointing",
    "gradient_checkpointing_blocks", "mu_x1", "mu_y1", "mu_x2", "mu_y2",
    "timestep_aux_prob", "timestep_aux_min", "timestep_aux_max", "seed",
    "run_name", "max_steps", "save_every", "hf_repo_id", "hf_mirror_every_steps",
    "source_checkpoint", "source_step", "target_step", "control_input_lr",
    "control_input_lr_multiplier", "required_checkpoint_steps",
)


def pose_active_window(timesteps: torch.Tensor, available: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Inclusive per-sample eligibility; the production sampler is not modified."""
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError("pose timestep window must satisfy 0 < min <= max < 1")
    if timesteps.ndim != 1 or available.shape != timesteps.shape:
        raise ValueError("timestep and availability shapes must match")
    return available.to(device=timesteps.device, dtype=torch.bool) & (timesteps >= lower) & (timesteps <= upper)


def should_build_pose_graph(active_indices: torch.Tensor) -> bool:
    """Keep decoder/critic work completely absent when no sample is active."""
    return bool(active_indices.numel())


def load_gate_e_microbatch(data: PreparedLatentShardDataset, indices: list[int]) -> dict[str, Any]:
    """Load one planned microbatch with the production shard-dataset API."""
    items = [data[index] for index in indices]
    batch = collate(items)
    batch["stems"] = [item["stem"] for item in items]
    return batch


def _validate_parent(parent_path: Path, expected_sha256: str | None) -> dict[str, Any]:
    if not parent_path.is_file(): raise FileNotFoundError(f"Parent checkpoint is unavailable: {parent_path}")
    state = load_training_state(parent_path)
    global_step = state["global_step"]
    if global_step < 1:
        raise ValueError(f"Pose-reward parent must have a positive global_step, got {global_step}")
    actual_sha256 = _sha256(parent_path) if expected_sha256 else None
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError("Parent checkpoint SHA256 mismatch")
    return state


def resolve_target_global_step(loaded_global_step: int, *, target_global_step: int | None,
                               max_steps: int | None) -> int:
    """Resolve an explicit final step; legacy --max-steps means added updates."""
    if (target_global_step is None) == (max_steps is None):
        raise ValueError("provide exactly one of --target-global-step or --max-steps")
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        target_global_step = loaded_global_step + max_steps
    assert target_global_step is not None
    if target_global_step <= loaded_global_step:
        raise ValueError("--target-global-step must be strictly greater than the loaded checkpoint global_step")
    return target_global_step


def checkpoint_publication_steps(loaded_global_step: int, target_global_step: int,
                                 save_every: int) -> tuple[int, ...]:
    """Exact new local publication points, including an off-cadence final step."""
    if loaded_global_step < 0 or target_global_step <= loaded_global_step or save_every < 1:
        raise ValueError("invalid checkpoint publication range")
    first = ((loaded_global_step // save_every) + 1) * save_every
    return tuple(sorted(set(range(first, target_global_step + 1, save_every)) | {target_global_step}))


def _critical_train_config(cfg: train.TrainConfig) -> dict[str, Any]:
    values = asdict(cfg)
    return {field: values[field] for field in _GATE_E_CRITICAL_CONFIG_FIELDS}


def _immutable_parent_identity(parent_path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the first immutable parent identity across controlled resumes."""
    if GATE_E_METADATA_KEY not in state:
        return {"global_step": int(state["global_step"]), "sha256": _sha256(parent_path), "filename": parent_path.name}
    metadata = state.get(GATE_E_METADATA_KEY)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("immutable_parent"), dict):
        raise ValueError("Pose-reward resume checkpoint lacks immutable step-1500 parent provenance")
    identity = dict(metadata["immutable_parent"])
    if (not isinstance(identity.get("global_step"), int) or identity["global_step"] < 1
            or not isinstance(identity.get("sha256"), str) or len(identity["sha256"]) != 64
            or not isinstance(identity.get("filename"), str) or not identity["filename"]):
        raise ValueError("Pose-reward resume checkpoint has malformed immutable parent provenance")
    return identity


def _gate_e_metadata(cfg: train.TrainConfig, *, pose_loss: str, lambda_pose: float, timestep_min: float,
                     timestep_max: float, forced_exposure_probability: float,
                     hf_subdir: str, immutable_parent: Mapping[str, Any],
                     cumulative_counters: Mapping[str, int], model_state: Mapping[str, Any],
                     wandb_run_id: str | None = None) -> dict[str, Any]:
    metadata = {
        "format": GATE_E_METADATA_FORMAT,
        "pose_loss": pose_loss,
        "temperature": 1.0,
        "lambda_pose": lambda_pose,
        "pose_timestep_window": [timestep_min, timestep_max],
        "forced_exposure_probability": forced_exposure_probability,
        "forced_sampler_policy": CONTROLLED_POSE_EXPOSURE_POLICY_VERSION,
        "immutable_parent": dict(immutable_parent),
        "hf_subdir": hf_subdir,
        "cumulative_counters": {key: int(value) for key, value in cumulative_counters.items()},
        "critical_train_config": _critical_train_config(cfg),
        "trainable_state_names": sorted(model_state),
    }
    if wandb_run_id:
        metadata["wandb_run_id"] = wandb_run_id
    return metadata


def gate_e_wandb_run_id(state: Mapping[str, Any]) -> str | None:
    """Read optional W&B continuity state without making local resume depend on it."""
    metadata = state.get(GATE_E_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        return None
    run_id = metadata.get("wandb_run_id")
    return run_id.strip() if isinstance(run_id, str) and run_id.strip() else None


def gate_e_wandb_config(*, cfg: train.TrainConfig, immutable_parent: Mapping[str, Any],
                        pose_loss: str, lambda_pose: float, timestep_min: float, timestep_max: float,
                        forced_exposure_probability: float, target_global_step: int,
                        hf_subdir: str, sidecar_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """The non-secret static experiment record mirrored once to W&B."""
    return {
        "experiment": "gate_e_pose_reward_smoke",
        "parent_checkpoint": dict(immutable_parent),
        "model_base": "Krea-2 Raw",
        "raw_checkpoint": cfg.raw_ckpt,
        "pose_loss": pose_loss,
        "lambda_pose": lambda_pose,
        "pose_timestep_window": [timestep_min, timestep_max],
        "forced_pose_exposure_probability": forced_exposure_probability,
        "forced_sampler_policy": CONTROLLED_POSE_EXPOSURE_POLICY_VERSION,
        "microbatch_size": cfg.microbatch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": cfg.microbatch_size * cfg.gradient_accumulation_steps,
        "target_global_step": target_global_step,
        "save_every_steps": cfg.save_every,
        "hf_repo_id": cfg.hf_repo_id,
        "hf_subdir": hf_subdir,
        "canonical_sidecar_records_sha256": sidecar_metadata.get("records_sha256"),
    }


def gate_e_wandb_step_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten existing cumulative counters for W&B without changing JSONL payloads."""
    payload = dict(metrics)
    counters = payload.get("pose_cumulative_counters")
    if isinstance(counters, Mapping):
        payload.update({
            "cumulative_eligible": int(counters["eligible_samples_seen"]),
            "cumulative_forced": int(counters["forced_samples"]),
            "cumulative_natural_active": int(counters["naturally_active_samples"]),
            "cumulative_total_active": int(counters["total_active_samples"]),
        })
    return payload


def parse_wandb_tags(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tags = [tag.strip() for tag in value.split(",") if tag.strip()]
    return tags or None


def validate_gate_e_resume_checkpoint(checkpoint: Path, state: dict[str, Any], *, cfg: train.TrainConfig,
                                      pose_loss: str, lambda_pose: float, timestep_min: float, timestep_max: float,
                                      forced_exposure_probability: float, hf_subdir: str,
                                      immutable_parent: Mapping[str, Any]) -> dict[str, int]:
    """Fail closed unless a checkpoint proves this exact controlled branch."""
    if GATE_E_METADATA_KEY not in state:
        if (immutable_parent.get("global_step") != state["global_step"]
                or immutable_parent.get("filename") != checkpoint.name):
            raise ValueError("Gate-E resume metadata is malformed or missing")
        return _empty_cumulative_counters()
    metadata = state.get(GATE_E_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("Gate-E resume metadata is malformed")
    if metadata.get("format") != GATE_E_METADATA_FORMAT:
        raise ValueError("Gate-E resume metadata format is unsupported")
    expected = _gate_e_metadata(cfg, pose_loss=pose_loss, lambda_pose=lambda_pose, timestep_min=timestep_min,
                                timestep_max=timestep_max,
                                forced_exposure_probability=forced_exposure_probability,
                                hf_subdir=hf_subdir, immutable_parent=immutable_parent,
                                cumulative_counters=metadata.get("cumulative_counters", {}),
                                model_state=state["model"])
    for key in ("pose_loss", "temperature", "lambda_pose", "pose_timestep_window",
                "forced_exposure_probability", "forced_sampler_policy", "immutable_parent", "hf_subdir",
                "critical_train_config", "trainable_state_names"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Gate-E resume configuration mismatch for {key}")
    counters = metadata.get("cumulative_counters")
    if not isinstance(counters, dict) or set(counters) != set(_empty_cumulative_counters()):
        raise ValueError("Gate-E resume cumulative exposure counters are malformed")
    if any(not isinstance(value, int) or value < 0 for value in counters.values()):
        raise ValueError("Gate-E resume cumulative exposure counters are invalid")
    if counters["total_active_samples"] != counters["naturally_active_samples"] + counters["forced_samples"]:
        raise ValueError("Gate-E resume cumulative exposure counters double-count activity")
    return {key: int(value) for key, value in counters.items()}


def _empty_cumulative_counters() -> dict[str, int]:
    return {"eligible_samples_seen": 0, "forced_samples": 0,
            "naturally_active_samples": 0, "total_active_samples": 0}


def _validate_hf_branch_args(*, hf_repo_id: str, hf_subdir: str, run_name: str,
                             save_every: int, mirror_every_steps: int) -> str:
    normalized = hf_subdir.strip("/")
    if not hf_repo_id.strip():
        raise ValueError("--hf-repo-id is required for this recoverable experiment")
    if not normalized or ".." in Path(normalized).parts:
        raise ValueError("--hf-subdir must be a non-empty isolated relative namespace")
    if normalized != run_name:
        raise ValueError("--hf-subdir must exactly match --run-name for isolated local/remote provenance")
    if mirror_every_steps != save_every:
        raise ValueError("--hf-mirror-every-steps must equal --save-every so every published checkpoint is mirrored")
    return normalized


def validate_gate_e_destination(parent_path: Path, destination: Path, *, loaded_global_step: int,
                                target_global_step: int) -> None:
    """Keep a new run isolated and a continuation anchored to its own run root."""
    target_path = destination / f"step_{target_global_step:06d}.pt"
    if target_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing target checkpoint: {target_path}")
    if parent_path.parent.resolve() != destination.resolve():
        if destination.resolve() == parent_path.parent.resolve():
            raise ValueError("smoke output must not share the canonical parent checkpoint directory")
        if destination.exists():
            raise FileExistsError(f"Refusing to start Gate-E in an existing directory: {destination}")
        return
    if not destination.is_dir() or destination.resolve() != parent_path.parent.resolve():
        raise ValueError("Gate-E continuation must use the existing parent checkpoint run directory")


@dataclass(frozen=True)
class GateERunSetup:
    """Validated non-GPU setup with checkpoint location and contents kept distinct."""

    parent_path: Path
    parent_state: dict[str, Any]
    destination: Path
    cfg: train.TrainConfig
    target_global_step: int
    hf_subdir: str
    immutable_parent: dict[str, Any]
    cumulative_counters: dict[str, int]


def prepare_gate_e_run_setup(*, parent_path: Path, expected_parent_sha256: str | None,
                             raw_ckpt: str, latent_root: str, checkpoint_dir: str,
                             run_name: str, microbatch_size: int,
                             gradient_accumulation_steps: int, save_every: int,
                             hf_repo_id: str, hf_subdir: str,
                             hf_mirror_every_steps: int, pose_loss: str, lambda_pose: float,
                             timestep_min: float, timestep_max: float,
                             forced_exposure_probability: float,
                             target_global_step: int | None,
                             max_steps: int | None, destination: Path) -> GateERunSetup:
    """Validate new-start or controlled-resume setup without model or network work."""
    if pose_loss not in POSE_LOSS_NAMES:
        raise ValueError(f"unsupported --pose-loss {pose_loss!r}; expected one of {POSE_LOSS_NAMES}")
    parent_state = _validate_parent(parent_path, expected_parent_sha256)
    resolved_target_global_step = resolve_target_global_step(
        parent_state["global_step"], target_global_step=target_global_step, max_steps=max_steps,
    )
    parent_cfg = train.train_config_from_checkpoint_values(parent_state["config"])
    cfg = replace(
        parent_cfg, raw_ckpt=raw_ckpt, shard_dir=latent_root, ckpt_dir=checkpoint_dir,
        run_name=run_name, max_steps=resolved_target_global_step,
        allow_extended_training=True, microbatch_size=microbatch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, save_every=save_every,
        val_every=10**9, metrics_jsonl_path=str(destination / "metrics.jsonl"),
        hf_repo_id=hf_repo_id, hf_mirror_every_steps=hf_mirror_every_steps,
    )
    immutable_parent = _immutable_parent_identity(parent_path, parent_state)
    cumulative_counters = validate_gate_e_resume_checkpoint(
        parent_path, parent_state, cfg=cfg, pose_loss=pose_loss, lambda_pose=lambda_pose,
        timestep_min=timestep_min, timestep_max=timestep_max,
        forced_exposure_probability=forced_exposure_probability, hf_subdir=hf_subdir,
        immutable_parent=immutable_parent,
    )
    validate_gate_e_destination(
        parent_path, destination, loaded_global_step=parent_state["global_step"],
        target_global_step=resolved_target_global_step,
    )
    return GateERunSetup(
        parent_path=parent_path, parent_state=parent_state, destination=destination,
        cfg=cfg, target_global_step=resolved_target_global_step, hf_subdir=hf_subdir,
        immutable_parent=immutable_parent, cumulative_counters=cumulative_counters,
    )


def _pose_smoke_loss(model: torch.nn.Module, vae: Any, critic: FixedBoxKeypointRCNNCritic, batch: dict[str, Any],
                     sidecar_by_stem: dict[str, dict[str, Any]], cfg: train.TrainConfig, device: torch.device,
                     generator: torch.Generator, *, pose_loss_name: str, lambda_pose: float, timestep_min: float,
                     timestep_max: float, forced_exposure_probability: float,
                     collect_diagnostics: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
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
    context, text_mask = batch["context"].to(device=device, dtype=torch.bfloat16, non_blocking=True), batch["text_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    image_tokens, pos, mask = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
    control_tokens, _, _ = patchify_and_position(control, context.shape[1], model.config.patch, text_mask)
    target_tokens, _, _ = patchify_and_position(target, context.shape[1], model.config.patch, text_mask)
    velocity = forward_pose_control(model, image_tokens, control_tokens, context, timestep.to(torch.bfloat16), pos, mask,
                                    gradient_checkpointing_blocks=cfg.gradient_checkpointing_blocks)
    flow_loss = F.mse_loss(velocity.float(), target_tokens.float())
    if not torch.isfinite(flow_loss): raise FloatingPointError("non-finite flow-matching MSE")
    active_indices = active.nonzero(as_tuple=False).flatten()
    pose_loss: torch.Tensor | None = None
    if should_build_pose_graph(active_indices):
        x0_hat_tokens = image_tokens - timestep.view(-1, 1, 1).to(image_tokens) * velocity
        from scripts.audit_keypoint_critic_timestep import unpatchify_latent_tokens
        x0_hat = unpatchify_latent_tokens(x0_hat_tokens[active_indices], tuple(clean.shape[-2:]), model.config.patch)
        decoded = qwen_decoded_to_unit_rgb(decode_normalized_latents_autograd(vae, x0_hat)).float()
        boxes, targets, valid = zip(*(_person_tensors(records[index], device) for index in active_indices.tolist()))
        heatmaps = critic(decoded, list(boxes))
        pose_loss_value = differentiable_pose_loss(
            pose_loss_name, heatmaps.logits, torch.cat(targets), heatmaps.boxes_training, torch.cat(valid),
            temperature=1.0, gaussian_sigma=1.5,
        )
        if not torch.isfinite(pose_loss_value): raise FloatingPointError(f"non-finite pose loss: {pose_loss_name}")
        pose_loss = pose_loss_value
    total = combine_flow_and_pose_loss(flow_loss, pose_loss, int(active_indices.numel()), lambda_pose)
    if not torch.isfinite(total): raise FloatingPointError("non-finite total loss")
    if not collect_diagnostics:
        # Keep only device-side counters for a timing harness.  The loss graph,
        # sampler, active-set selection, and gradients are exactly unchanged;
        # this avoids serializing diagnostic scalars on every timed microbatch.
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
    natural_active_counts = [int(item.get("pose_natural_active_count", item["pose_active_count"]))
                             for item in microbatch_diagnostics]
    pose_losses = [float(item["pose_loss"]) for item in microbatch_diagnostics if item["pose_loss"] is not None]
    if not timesteps:
        raise ValueError("Optimizer-step diagnostics contain no timesteps")
    active_samples = sum(active_counts)
    forced_samples, natural_active_samples, eligible_samples = sum(forced_counts), sum(natural_active_counts), sum(eligible_counts)
    if active_samples != forced_samples + natural_active_samples:
        raise ValueError("Pose-exposure diagnostics double-count or omit active samples")
    active_timesteps = [float(value) for item in microbatch_diagnostics for value in item.get("active_timesteps", [])]
    return {
        "flow_loss": sum(flow_losses) / len(flow_losses),
        "total_loss": sum(total_losses) / len(total_losses),
        "pose_loss": sum(pose_losses) / len(pose_losses) if pose_losses else None,
        "pose_active_count": active_samples,
        "pose_active_fraction": active_samples / len(timesteps),
        "pose_active_samples_step": active_samples,
        "pose_active_microbatches_step": sum(count > 0 for count in active_counts),
        "pose_eligible_samples_step": eligible_samples,
        "pose_forced_samples_step": forced_samples,
        "pose_natural_active_samples_step": natural_active_samples,
        "pose_forced_fraction_of_eligible_step": forced_samples / eligible_samples if eligible_samples else 0.0,
        "pose_total_active_fraction_of_eligible_step": active_samples / eligible_samples if eligible_samples else 0.0,
        "pose_loss_mean_active": sum(pose_losses) / len(pose_losses) if pose_losses else None,
        "pose_loss_max_active": max(pose_losses) if pose_losses else None,
        "flow_loss_mean_step": sum(flow_losses) / len(flow_losses),
        "total_loss_mean_step": sum(total_losses) / len(total_losses),
        "timestep_min_step": min(timesteps),
        "timestep_max_step": max(timesteps),
        "timestep_mean_step": sum(timesteps) / len(timesteps),
        "active_timestep_min_step": min(active_timesteps) if active_timesteps else None,
        "active_timestep_mean_step": sum(active_timesteps) / len(active_timesteps) if active_timesteps else None,
        "active_timestep_max_step": max(active_timesteps) if active_timesteps else None,
    }


def update_cumulative_counters(counters: Mapping[str, int], metrics: Mapping[str, Any]) -> dict[str, int]:
    """Carry branch exposure evidence across atomically resumable checkpoints."""
    updated = {key: int(value) for key, value in counters.items()}
    for counter, metric in (("eligible_samples_seen", "pose_eligible_samples_step"),
                            ("forced_samples", "pose_forced_samples_step"),
                            ("naturally_active_samples", "pose_natural_active_samples_step"),
                            ("total_active_samples", "pose_active_samples_step")):
        updated[counter] += int(metrics[metric])
    if updated["total_active_samples"] != updated["forced_samples"] + updated["naturally_active_samples"]:
        raise AssertionError("Cumulative pose exposure counters double-count activity")
    return updated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", default=None,
                        help="optional SHA256 for the explicitly supplied immutable parent")
    parser.add_argument("--raw-ckpt", required=True); parser.add_argument("--latent-root", required=True); parser.add_argument("--text-conditioning-root", required=True); parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", required=True); parser.add_argument("--run-name", required=True); parser.add_argument("--lambda-pose", type=float, required=True)
    parser.add_argument("--pose-loss", choices=POSE_LOSS_NAMES, default="gaussian_heatmap_kl",
                        help="differentiable fixed-box reward (default: gaussian_heatmap_kl)")
    parser.add_argument("--pose-timestep-min", type=float, required=True); parser.add_argument("--pose-timestep-max", type=float, required=True)
    parser.add_argument("--forced-pose-exposure-probability", type=float, required=True,
                        help="per-eligible-sample probability of a final-window forced timestep")
    parser.add_argument("--hf-repo-id", required=True, help="private repository for this isolated branch")
    parser.add_argument("--hf-subdir", required=True, help="isolated remote branch namespace; must match --run-name")
    parser.add_argument("--hf-mirror-every-steps", type=int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-global-step", type=int,
                        help="exclusive final optimizer step; required form for an explicit continuation")
    target.add_argument("--max-steps", type=int,
                        help="legacy compatibility: additional optimizer steps after the loaded checkpoint")
    parser.add_argument("--save-every", type=int, required=True); parser.add_argument("--microbatch-size", type=int, required=True); parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default=None,
                        help="optional W&B project; omit to keep W&B fully disabled")
    parser.add_argument("--wandb-entity", default=None, help="optional W&B entity")
    parser.add_argument("--wandb-run-name", default=None, help="optional W&B display name; defaults to --run-name")
    parser.add_argument("--wandb-group", default=None, help="optional W&B group")
    parser.add_argument("--wandb-tags", default=None, help="optional comma-separated W&B tags")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.save_every < 1 or args.microbatch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("save-every, microbatch-size, and gradient-accumulation-steps must be positive")
    if not args.wandb_project and any((args.wandb_entity, args.wandb_run_name, args.wandb_group, args.wandb_tags)):
        parser.error("--wandb-entity, --wandb-run-name, --wandb-group, and --wandb-tags require --wandb-project")
    try: lambda_pose, _, destination = validate_smoke_invocation(lambda_pose=args.lambda_pose, pose_timesteps=(args.pose_timestep_min, args.pose_timestep_max), run_name=args.run_name, checkpoint_dir=args.checkpoint_dir, parent_checkpoint=args.parent_checkpoint, verify_parent=True, allow_existing_destination=True)
    except (ValueError, FileExistsError, FileNotFoundError) as error: parser.error(str(error))
    try:
        hf_subdir = _validate_hf_branch_args(hf_repo_id=args.hf_repo_id, hf_subdir=args.hf_subdir,
                                             run_name=args.run_name, save_every=args.save_every,
                                             mirror_every_steps=args.hf_mirror_every_steps)
        if not 0.0 <= args.forced_pose_exposure_probability <= 1.0:
            raise ValueError("--forced-pose-exposure-probability must be in [0, 1]")
    except ValueError as error:
        parser.error(str(error))
    try:
        setup = prepare_gate_e_run_setup(
            parent_path=args.parent_checkpoint, expected_parent_sha256=args.expected_parent_sha256,
            raw_ckpt=args.raw_ckpt, latent_root=args.latent_root, checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name, microbatch_size=args.microbatch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps, save_every=args.save_every,
            hf_repo_id=args.hf_repo_id, hf_subdir=hf_subdir,
            hf_mirror_every_steps=args.hf_mirror_every_steps, pose_loss=args.pose_loss, lambda_pose=lambda_pose,
            timestep_min=args.pose_timestep_min, timestep_max=args.pose_timestep_max,
            forced_exposure_probability=args.forced_pose_exposure_probability,
            target_global_step=args.target_global_step, max_steps=args.max_steps,
            destination=destination,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    parent_path = setup.parent_path
    parent_state = setup.parent_state
    destination = setup.destination
    cfg = setup.cfg
    target_global_step = setup.target_global_step
    immutable_parent = setup.immutable_parent
    cumulative_counters = setup.cumulative_counters
    if not torch.cuda.is_available() or args.device != "cuda": raise RuntimeError("Run Gate-E only from the GH200 host shell with CUDA visible")
    set_seed(cfg.seed); device = torch.device("cuda")
    sidecar_metadata, records = load_sidecar(args.sidecar); by_stem = {record["stem"]: record for record in records}
    data = PreparedLatentShardDataset(cfg.shard_dir, "train", text_conditioning_root=args.text_conditioning_root)
    if not set(item[3] for item in data.records) <= set(by_stem): raise ValueError("latent shards contain stems absent from immutable sidecar")
    plan = train.DeterministicBucketBatches(data.records, cfg.microbatch_size, cfg.seed)
    model = build_pose_model(cfg.raw_ckpt, 64, 64, "cuda"); audit_control_model(model, rank=64); train.load_trainable_state_dict(model, parent_state["model"]); model.train()
    optimizer = train.build_optimizer(model, cfg); scheduler = train.OptimizerStepWarmup(optimizer, cfg.warmup_steps)
    global_step, epoch, batch_position, restored_generator = train.restore_full_training_state(model, optimizer, scheduler, parent_state)
    vae = load_krea_vae(device); critic = FixedBoxKeypointRCNNCritic().to(device).eval(); assert_frozen_no_parameter_grad(vae, critic)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + global_step)
    if restored_generator is not None: generator.set_state(restored_generator)
    resumed_wandb_run_id = gate_e_wandb_run_id(parent_state)
    if not args.wandb_project and resumed_wandb_run_id:
        print("[wandb] disabled because --wandb-project was omitted; preserving checkpoint run id for later recovery", flush=True)
    wandb_mirror = OptionalWandbMirror(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or args.run_name,
        group=args.wandb_group,
        tags=parse_wandb_tags(args.wandb_tags),
        resume_run_id=resumed_wandb_run_id,
        config=gate_e_wandb_config(
            cfg=cfg, immutable_parent=immutable_parent, pose_loss=args.pose_loss, lambda_pose=lambda_pose,
            timestep_min=args.pose_timestep_min, timestep_max=args.pose_timestep_max,
            forced_exposure_probability=args.forced_pose_exposure_probability,
            target_global_step=target_global_step, hf_subdir=hf_subdir,
            sidecar_metadata=sidecar_metadata,
        ),
    )
    wandb_run_id = wandb_mirror.run_id
    if wandb_mirror.enabled:
        print(f"[wandb] mirroring to project={args.wandb_project} run_id={wandb_run_id}", flush=True)
    stopped = False
    def stop_handler(signum, _frame):
        nonlocal stopped; stopped = True; print(f"received signal {signum}; checkpointing at boundary", flush=True)
    signal.signal(signal.SIGINT, stop_handler); signal.signal(signal.SIGTERM, stop_handler)
    if parent_path.parent.resolve() != destination.resolve():
        destination.mkdir(parents=True, exist_ok=False)
    metadata_path = destination / "experiment_metadata.json"
    def write_experiment_metadata() -> None:
        metadata = _gate_e_metadata(
            cfg, pose_loss=args.pose_loss, lambda_pose=lambda_pose, timestep_min=args.pose_timestep_min,
            timestep_max=args.pose_timestep_max,
            forced_exposure_probability=args.forced_pose_exposure_probability,
            hf_subdir=hf_subdir, immutable_parent=immutable_parent,
            cumulative_counters=cumulative_counters, model_state=trainable_state_dict(model),
            wandb_run_id=wandb_run_id,
        )
        metadata_path.write_text(json.dumps({"gate_e": metadata, "config": asdict(cfg)}, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    write_experiment_metadata()

    def mirror_result(success: bool, step: int | None, error: str | None, reason: str | None) -> None:
        if success and step is not None:
            print(f"[hf] mirrored {args.hf_repo_id}/{hf_subdir}/full/step_{step:06d}.pt", flush=True)
        elif not success:
            print(f"[hf] mirror failure reason={reason} error={error}", flush=True)

    publication_steps = checkpoint_publication_steps(global_step, target_global_step, args.save_every)
    mirror = HFTrainingCheckpointMirror(repo_id=args.hf_repo_id, run_name=hf_subdir,
                                        protected_milestone_steps=publication_steps,
                                        prune_local_after_success=False, on_result=mirror_result)
    mirror.start()
    metrics_path = destination / "metrics.jsonl"; optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < cfg.max_steps and not stopped:
            batches = plan.for_epoch(epoch)
            if batch_position >= len(batches): epoch, batch_position = epoch + 1, 0; continue
            started = time.monotonic(); microbatch_diagnostics: list[Mapping[str, Any]] = []
            for accumulation_index in range(cfg.gradient_accumulation_steps):
                if batch_position >= len(batches): epoch, batch_position = epoch + 1, 0; batches = plan.for_epoch(epoch)
                batch = load_gate_e_microbatch(data, batches[batch_position]); batch_position += 1
                train.apply_cached_caption_dropout(batch, data.text_conditioning.unconditional, cfg.caption_dropout, cfg.seed, global_step * cfg.gradient_accumulation_steps + accumulation_index)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, diagnostics = _pose_smoke_loss(
                        model, vae, critic, batch, by_stem, cfg, device, generator,
                        pose_loss_name=args.pose_loss,
                        lambda_pose=lambda_pose, timestep_min=args.pose_timestep_min,
                        timestep_max=args.pose_timestep_max,
                        forced_exposure_probability=args.forced_pose_exposure_probability,
                    )
                microbatch_diagnostics.append(diagnostics)
                (loss / cfg.gradient_accumulation_steps).backward()
            grad_norm = train.optimizer_update(optimizer, scheduler, trainable_params(model), cfg.max_grad_norm); global_step += 1
            assert_frozen_no_parameter_grad(vae, critic)
            metrics = {"global_step": global_step, "lambda_pose": lambda_pose,
                       "pose_timestep_window": [args.pose_timestep_min, args.pose_timestep_max],
                       "forced_pose_exposure_probability": args.forced_pose_exposure_probability,
                       "global_grad_norm": grad_norm, "sec_per_step": time.monotonic() - started,
                       **aggregate_step_diagnostics(microbatch_diagnostics)}
            cumulative_counters = update_cumulative_counters(cumulative_counters, metrics)
            metrics["pose_cumulative_counters"] = dict(cumulative_counters)
            with metrics_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
            wandb_mirror.log(gate_e_wandb_step_metrics(metrics), step=global_step)
            if global_step % cfg.save_every == 0 or global_step == cfg.max_steps or stopped:
                model_state = trainable_state_dict(model)
                path = save_training_state(destination / f"step_{global_step:06d}.pt", {
                    "model": model_state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "global_step": global_step, "epoch": epoch, "batch_position": batch_position,
                    "rng": train._capture_rng(), "flow_generator_state": generator.get_state(), "config": asdict(cfg),
                    GATE_E_METADATA_KEY: _gate_e_metadata(
                        cfg, pose_loss=args.pose_loss, lambda_pose=lambda_pose, timestep_min=args.pose_timestep_min,
                        timestep_max=args.pose_timestep_max,
                        forced_exposure_probability=args.forced_pose_exposure_probability,
                        hf_subdir=hf_subdir, immutable_parent=immutable_parent,
                        cumulative_counters=cumulative_counters, model_state=model_state,
                        wandb_run_id=wandb_run_id,
                    ),
                }, overwrite=False)
                if not mirror.submit(path, reason="step"):
                    raise RuntimeError(f"Could not queue local checkpoint for HF mirroring: {path}")
    finally:
        mirror.stop()
        wandb_mirror.close()
        write_experiment_metadata()
        assert_frozen_no_parameter_grad(vae, critic)
    if mirror.last_error is not None:
        raise RuntimeError(f"One or more HF checkpoint uploads failed; local checkpoints remain intact: {mirror.last_error}")
    artifact_prefix = f"{hf_subdir}/"
    if not mirror.upload_artifact(metrics_path, path_in_repo=artifact_prefix + "metrics.jsonl"):
        raise RuntimeError("HF upload failed for final metrics.jsonl; use the retry command without retraining")
    if not mirror.upload_artifact(metadata_path, path_in_repo=artifact_prefix + "experiment_metadata.json"):
        raise RuntimeError("HF upload failed for final experiment metadata; use the retry command without retraining")
    print(f"[hf] mirrored metadata under {args.hf_repo_id}/{hf_subdir}/", flush=True)


if __name__ == "__main__": main()
