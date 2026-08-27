"""Evaluation-only Krea-2 Turbo sampling contract for Pose-ControlNet.

This deliberately has no training-loop, optimizer, or gradient APIs.  It is a
small adaptation of the official Krea sampler: the image tokens remain noisy,
the clean pose tokens are concatenated at every denoising forward, and Turbo's
constant shift is passed explicitly rather than inferred from resolution.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch
from einops import rearrange

from pose_controlnet.diffusion import forward_pose_control, patchify_and_position
from pose_controlnet.evaluation import ordered_checkpoints
from pose_controlnet.model import trainable_state_dict


TURBO_STEPS = 8
TURBO_CFG = 0.0
TURBO_MU = 1.15
TURBO_SIGMA = 1.0
TURBO_CHECKPOINT_STEPS = (800, 1500)
CANONICAL_EVALUATION_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500")


def turbo_schedule(*, image_sequence_length: int, steps: int = TURBO_STEPS,
                   mu: float = TURBO_MU) -> list[float]:
    """Exact ``krea-ai/krea-2/sampling.py:timesteps`` with Turbo's pinned mu.

    ``image_sequence_length`` is accepted only to make resolution invariance
    explicit and testable.  The official formula uses it only if ``mu`` is
    omitted; Turbo never omits it.
    """
    if image_sequence_length < 1:
        raise ValueError("image_sequence_length must be positive")
    if steps != TURBO_STEPS:
        raise ValueError(f"Krea-2 Turbo evaluation requires exactly {TURBO_STEPS} steps, got {steps}")
    if mu != TURBO_MU:
        raise ValueError(f"Krea-2 Turbo evaluation requires mu={TURBO_MU}, got {mu}")
    # Keep torch's default float32 construction: this deliberately matches the
    # authoritative upstream ``torch.linspace(1, 0, steps + 1)`` byte for byte.
    ts = torch.linspace(1, 0, steps + 1)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** TURBO_SIGMA)
    return shifted.tolist()


def turbo_metadata() -> dict[str, Any]:
    return {
        "model": "Krea-2 Turbo",
        "steps": TURBO_STEPS,
        "cfg": TURBO_CFG,
        "mu": TURBO_MU,
        "mu_resolution_dependent": False,
        "schedule_source": "https://github.com/krea-ai/krea-2/blob/main/sampling.py",
    }


def assert_turbo_output_isolated(output_dir: str | Path) -> Path:
    """Reject the immutable canonical RAW evaluation namespace and descendants."""
    output = Path(output_dir).resolve()
    canonical = CANONICAL_EVALUATION_ROOT.resolve()
    if output == canonical or canonical in output.parents:
        raise ValueError(f"Turbo evaluation output must not collide with canonical evaluation path: {canonical}")
    return output


def exact_turbo_checkpoints(*, checkpoint_dir: str | Path, hf_repo_id: str,
                            hf_recovery_dir: str | Path | None = None) -> list[tuple[int, Path]]:
    """Resolve only complete, checksum-validated exact 800/1500 archive states."""
    resolved = ordered_checkpoints(
        checkpoint_dir, steps=TURBO_CHECKPOINT_STEPS,
        later_checkpoint_dir=checkpoint_dir, archive_checkpoint_dir=checkpoint_dir,
        hf_repo_id=hf_repo_id, hf_recovery_dir=hf_recovery_dir,
    )
    if tuple(step for step, _ in resolved) != TURBO_CHECKPOINT_STEPS or any(path is None for _, path in resolved):
        raise AssertionError("Turbo benchmark must resolve exactly checkpoints 800 and 1500")
    return [(step, path) for step, path in resolved if path is not None]


@torch.inference_mode()
def sample_turbo_pose_image(model: Any, vae_decode_fn, sample: dict[str, Any], device: torch.device,
                            seed: int, *, steps: int = TURBO_STEPS,
                            guidance: float = TURBO_CFG, mu: float = TURBO_MU):
    """Sample a control-conditioned Turbo image with genuine CFG disablement.

    The implementation follows the official Euler loop and CFG branch.  With
    ``guidance == 0`` it intentionally does not obtain unconditional text
    conditioning, positions, masks, or a second model forward.
    """
    if guidance != TURBO_CFG:
        raise ValueError(f"Turbo benchmark requires CFG={TURBO_CFG}, got {guidance}")
    if "context" not in sample or "mask" not in sample:
        raise ValueError("Turbo sampling requires cached conditional text conditioning")
    latent = sample["latent"][None].to(device)
    control_latent = sample["control"][None].to(device=device, dtype=torch.bfloat16)
    if not torch.isfinite(control_latent).all() or control_latent.abs().max().item() == 0.0:
        raise ValueError("Turbo sampling requires finite, non-empty pose control latents")
    noise = torch.randn(latent.shape, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device=device).manual_seed(seed))
    text = sample["context"][None].to(device=device, dtype=torch.bfloat16)
    text_mask = sample["mask"][None].to(device=device, dtype=torch.bool)
    patch = model.config.patch
    image, pos, mask = patchify_and_position(noise, text.shape[1], patch, text_mask)
    control, _, _ = patchify_and_position(control_latent, text.shape[1], patch, text_mask)
    schedule = turbo_schedule(image_sequence_length=image.shape[1], steps=steps, mu=mu)
    for current, previous in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full((1,), current, dtype=image.dtype, device=device)
        # Control remains on this only (conditional) forward at every step.
        velocity = forward_pose_control(model, image, control, text, timestep, pos, mask,
                                        gradient_checkpointing_blocks=0)
        image = image + (previous - current) * velocity
    height, width = latent.shape[-2:]
    decoded_latent = rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                               ph=patch, pw=patch, h=height // patch, w=width // patch)
    pixels = vae_decode_fn(decoded_latent.to(torch.bfloat16))
    return ((pixels.clamp(-1, 1) * 0.5 + 0.5) * 255.0)[0].permute(1, 2, 0).float().cpu().byte().numpy()


def assert_exact_diagnostic_stems(manifest_stems: Iterable[str], dataset_stems: Iterable[str]) -> tuple[str, ...]:
    """Require all and only the immutable 24-record diagnostic manifest order."""
    stems = tuple(manifest_stems)
    if len(stems) != 24 or len(stems) != len(set(stems)):
        raise ValueError("Turbo benchmark requires exactly the complete 24-sample diagnostic manifest")
    if set(stems) != set(dataset_stems):
        raise ValueError("Prepared diagnostic dataset membership differs from immutable manifest")
    return stems


def raw_to_turbo_control_compatibility(model: Any, training_state: dict[str, Any]) -> dict[str, Any]:
    """Establish the only safe Raw-trained-control -> Turbo loading contract.

    The base checkpoint has already passed the shared official MMDiT structural
    inspection.  This validates the full training checkpoint schema's recorded
    RAW provenance and requires the control/LoRA tensor key set and shapes to
    exactly equal the surgically-expanded Turbo model before the caller loads
    any weights.
    """
    config = training_state.get("config")
    raw_checkpoint = config.get("raw_ckpt") if isinstance(config, dict) else None
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError("Training checkpoint lacks recorded Krea-2 Raw provenance")
    saved = training_state.get("model")
    if not isinstance(saved, dict):
        raise ValueError("Training checkpoint has no trainable control/LoRA state")
    expected = trainable_state_dict(model)
    if set(saved) != set(expected):
        raise ValueError("Raw-trained control state does not match Turbo control/LoRA key contract")
    mismatched = [name for name in expected if tuple(saved[name].shape) != tuple(expected[name].shape)]
    if mismatched:
        raise ValueError(f"Raw-trained control state does not match Turbo tensor shapes: {mismatched[:5]}")
    return {"raw_training_checkpoint": raw_checkpoint, "shared_official_mmdit_config": True,
            "control_lora_key_count": len(expected), "shape_mismatches": 0}
