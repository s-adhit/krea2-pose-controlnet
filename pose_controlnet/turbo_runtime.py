"""Locked Krea-2 Turbo sampling and Raw-control compatibility runtime."""
from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange

from pose_controlnet.diffusion import forward_pose_control, patchify_and_position
from pose_controlnet.model import trainable_state_dict


TURBO_STEPS = 8
TURBO_CFG = 0.0
TURBO_MU = 1.15
TURBO_SIGMA = 1.0


def turbo_schedule(*, image_sequence_length: int, steps: int = TURBO_STEPS,
                   mu: float = TURBO_MU) -> list[float]:
    """Exact upstream Krea Turbo schedule with its pinned constant shift."""
    if image_sequence_length < 1:
        raise ValueError("image_sequence_length must be positive")
    if steps != TURBO_STEPS:
        raise ValueError(f"Krea-2 Turbo evaluation requires exactly {TURBO_STEPS} steps, got {steps}")
    if mu != TURBO_MU:
        raise ValueError(f"Krea-2 Turbo evaluation requires mu={TURBO_MU}, got {mu}")
    ts = torch.linspace(1, 0, steps + 1)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** TURBO_SIGMA)
    return shifted.tolist()


def turbo_metadata() -> dict[str, Any]:
    return {
        "model": "Krea-2 Turbo", "steps": TURBO_STEPS, "cfg": TURBO_CFG,
        "mu": TURBO_MU, "mu_resolution_dependent": False,
        "schedule_source": "https://github.com/krea-ai/krea-2/blob/main/sampling.py",
    }


def scale_turbo_control_latent(control_latent: torch.Tensor, control_scale: float = 1.0) -> torch.Tensor:
    """Apply the inference-only control scale without changing the identity path."""
    if not isinstance(control_scale, (float, int)) or isinstance(control_scale, bool):
        raise TypeError("control_scale must be a finite numeric value")
    control_scale = float(control_scale)
    if not math.isfinite(control_scale) or control_scale <= 0.0:
        raise ValueError("control_scale must be finite and positive")
    return control_latent if control_scale == 1.0 else control_latent * control_scale


@torch.inference_mode()
def sample_turbo_pose_image(model: Any, vae_decode_fn, sample: dict[str, Any], device: torch.device,
                            seed: int, *, steps: int = TURBO_STEPS,
                            guidance: float = TURBO_CFG, mu: float = TURBO_MU,
                            control_scale: float = 1.0):
    """Sample a control-conditioned Turbo image with the locked CFG-0 loop."""
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
    control, _, _ = patchify_and_position(
        scale_turbo_control_latent(control_latent, control_scale), text.shape[1], patch, text_mask
    )
    schedule = turbo_schedule(image_sequence_length=image.shape[1], steps=steps, mu=mu)
    for current, previous in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full((1,), current, dtype=image.dtype, device=device)
        velocity = forward_pose_control(model, image, control, text, timestep, pos, mask,
                                        gradient_checkpointing_blocks=0)
        image = image + (previous - current) * velocity
    height, width = latent.shape[-2:]
    decoded_latent = rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                               ph=patch, pw=patch, h=height // patch, w=width // patch)
    pixels = vae_decode_fn(decoded_latent.to(torch.bfloat16))
    return ((pixels.clamp(-1, 1) * 0.5 + 0.5) * 255.0)[0].permute(1, 2, 0).float().cpu().byte().numpy()


def raw_to_turbo_control_compatibility(model: Any, training_state: dict[str, Any]) -> dict[str, Any]:
    """Validate the Raw-trained trainable state against the expanded Turbo model."""
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
