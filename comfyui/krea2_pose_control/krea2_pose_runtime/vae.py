"""Qwen-Image VAE boundary for the frozen ComfyUI inference path.

Qwen's image VAE uses a one-frame video representation at its public
interface (``B×C×1×H×W``).  The Krea MMDiT does not: normalized diffusion
latents are always ``B×C×H×W``.  Keep the temporal conversion here so no
diffusion or sampler code has to guess how to handle Qwen tensors.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from PIL import Image


def load_krea_vae(device: torch.device | str, dtype: torch.dtype = torch.bfloat16) -> Any:
    try:
        import diffusers
        from diffusers import AutoencoderKLQwenImage
    except ImportError as error:
        raise RuntimeError("diffusers with AutoencoderKLQwenImage is required") from error
    if not hasattr(diffusers, "AutoencoderKLQwenImage"):
        raise RuntimeError(f"diffusers {diffusers.__version__} lacks AutoencoderKLQwenImage")
    return AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae", torch_dtype=dtype).to(device).eval().requires_grad_(False)


def _statistics(vae: Any, device: torch.device | str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std, channels = vae.config.latents_mean, vae.config.latents_std, vae.config.z_dim
    if len(mean) != channels or len(std) != channels:
        raise RuntimeError("Qwen VAE latent statistics do not match z_dim")
    return (torch.tensor(mean, device=device, dtype=dtype).view(1, channels, 1, 1, 1),
            torch.tensor(std, device=device, dtype=dtype).view(1, channels, 1, 1, 1))


def _remove_singleton_temporal_dimension(latents: torch.Tensor, *, source: str) -> torch.Tensor:
    """Convert Qwen's ``B×C×1×H×W`` output to Krea's ``B×C×H×W`` layout."""
    if latents.ndim != 5:
        raise ValueError(f"{source} must be B×C×1×H×W, got {tuple(latents.shape)}")
    if latents.shape[2] != 1:
        raise ValueError(
            f"{source} temporal dimension must be exactly 1 for a still image, got {tuple(latents.shape)}"
        )
    return latents.squeeze(2).contiguous()


def _require_qwen_image_latents(vae: Any, latents: torch.Tensor) -> None:
    channels = getattr(vae.config, "z_dim", None)
    if latents.ndim != 5 or latents.shape[1] != channels:
        raise ValueError(
            "Qwen VAE encoded latents must be B×C×1×H×W "
            f"with C={channels}, got {tuple(latents.shape)}"
        )
    if latents.shape[2] != 1:
        raise ValueError(
            "Qwen VAE encoded latent temporal dimension must be exactly 1 for a still image, "
            f"got {tuple(latents.shape)}"
        )
    if not torch.isfinite(latents).all():
        raise ValueError("Qwen VAE encoded latents contain NaN or Inf")


def _require_normalized_diffusion_latents(vae: Any, latents: torch.Tensor) -> None:
    channels = getattr(vae.config, "z_dim", None)
    if latents.ndim != 4 or latents.shape[1] != channels:
        raise ValueError(
            "normalized Krea diffusion latents must be B×C×H×W "
            f"with C={channels}, got {tuple(latents.shape)}"
        )
    if not torch.isfinite(latents).all():
        raise ValueError("normalized Krea diffusion latents contain NaN or Inf")


def pil_to_qwen_vae_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(127.5).sub_(1).unsqueeze(0).unsqueeze(2)


@torch.inference_mode()
def encode_preprocessed_image(vae: Any, image: Image.Image, *, device: torch.device | str,
                              generator: torch.Generator) -> torch.Tensor:
    pixels = pil_to_qwen_vae_tensor(image).to(device=device, dtype=torch.bfloat16)
    raw = vae.encode(pixels).latent_dist.sample(generator=generator)
    _require_qwen_image_latents(vae, raw)
    mean, std = _statistics(vae, raw.device, raw.dtype)
    latent = (raw - mean) / std
    latent = _remove_singleton_temporal_dimension(latent, source="Qwen VAE encoded latents")
    if not torch.isfinite(latent).all() or not bool(latent.abs().max() > 0):
        raise RuntimeError("pose VAE encoding is empty or non-finite")
    return latent


@torch.inference_mode()
def decode_normalized_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    _require_normalized_diffusion_latents(vae, latents)
    mean, std = _statistics(vae, latents.device, latents.dtype)
    decoded = vae.decode(latents.unsqueeze(2) * std + mean).sample
    if decoded.ndim != 5 or decoded.shape[1] != 3:
        raise RuntimeError(f"Qwen VAE decode must produce B×3×1×H×W, got {tuple(decoded.shape)}")
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Qwen VAE decode produced NaN or Inf")
    return _remove_singleton_temporal_dimension(decoded, source="Qwen VAE decoded image")
