"""Project-owned Qwen/Krea VAE encoding for paired RGB and pose controls.

Krea-2 Raw uses the Qwen-Image VAE, whose image interface is a one-frame
video tensor (``B, C, T, H, W``).  This module deliberately consumes only
already-resolved ``ManifestRecord`` objects and ``preprocess_pair`` output;
dataset resolution and paired geometry remain owned by their existing modules.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image

from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
from pose_controlnet.paired_preprocessing import PreprocessedPair, preprocess_pair


KREA_VAE_REPO_ID = "Qwen/Qwen-Image"
KREA_VAE_SUBFOLDER = "vae"
KREA_VAE_CLASS_NAME = "AutoencoderKLQwenImage"
KREA_VAE_LATENT_CHANNELS = 16
KREA_VAE_SPATIAL_COMPRESSION = 8


class VAEPreprocessingError(ValueError):
    """Raised when the VAE contract or an encoded paired sample is invalid."""


class _LatentDistribution(Protocol):
    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor: ...


class _EncodedOutput(Protocol):
    latent_dist: _LatentDistribution


class _VAE(Protocol):
    config: Any

    def encode(self, x: torch.Tensor) -> _EncodedOutput: ...

    def decode(self, z: torch.Tensor) -> Any: ...


@dataclass(frozen=True)
class EncodedPair:
    """Normalized clean VAE latents, ready for downstream channel concatenation."""

    pair: PreprocessedPair
    latent: torch.Tensor
    control: torch.Tensor


def load_krea_vae(device: torch.device | str, dtype: torch.dtype = torch.bfloat16) -> Any:
    """Load only Krea-2 Raw's Qwen-Image VAE component in inference mode."""
    try:
        import diffusers
        from diffusers import AutoencoderKLQwenImage
    except ImportError as exc:  # pragma: no cover - exercised on deployment setup
        raise VAEPreprocessingError(
            "diffusers with AutoencoderKLQwenImage is required for the Krea VAE"
        ) from exc
    if not hasattr(diffusers, KREA_VAE_CLASS_NAME):
        raise VAEPreprocessingError(
            f"Installed diffusers {diffusers.__version__} lacks {KREA_VAE_CLASS_NAME}"
        )
    vae = AutoencoderKLQwenImage.from_pretrained(
        KREA_VAE_REPO_ID, subfolder=KREA_VAE_SUBFOLDER, torch_dtype=dtype
    )
    return vae.to(device).eval().requires_grad_(False)


def pil_to_qwen_vae_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a preprocessed RGB PIL image to a CPU ``1×3×1×H×W`` tensor."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise VAEPreprocessingError(f"Expected RGB image, got array shape {pixels.shape}")
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)
    return tensor.unsqueeze(0).unsqueeze(2)


def normalize_qwen_latents(latents: torch.Tensor, vae: _VAE) -> torch.Tensor:
    """Apply Qwen-Image's per-channel latent standardization exactly."""
    _validate_raw_latents(latents, vae)
    mean, std = _latent_statistics(vae, latents.device, latents.dtype)
    normalized = (latents - mean) / std
    if not torch.isfinite(normalized).all():
        raise VAEPreprocessingError("Normalized VAE latents contain NaN or Inf")
    return normalized


@torch.inference_mode()
def decode_normalized_latents(vae: _VAE, latents: torch.Tensor) -> torch.Tensor:
    """Inference-only wrapper around the project's normalized Qwen VAE decode."""
    return _decode_normalized_latents(vae, latents)


def decode_normalized_latents_autograd(vae: _VAE, latents: torch.Tensor) -> torch.Tensor:
    """Decode normalized image latents while preserving their autograd graph.

    This is intentionally narrow: production evaluation continues to use the
    inference-mode wrapper above, while audit code can prove gradients through
    the exact same denormalize/decode convention.  The VAE must be frozen so
    the graph terminates at ``latents``, not at VAE parameters.
    """
    parameters = getattr(vae, "parameters", None)
    if callable(parameters) and any(parameter.requires_grad for parameter in parameters()):
        raise VAEPreprocessingError("Autograd VAE decode requires frozen VAE parameters")
    decoded = _decode_normalized_latents(vae, latents)
    if latents.requires_grad and (not decoded.requires_grad or decoded.grad_fn is None):
        raise VAEPreprocessingError("Autograd VAE decode detached the latent graph")
    return decoded


def qwen_decoded_to_unit_rgb(decoded: torch.Tensor) -> torch.Tensor:
    """Map Qwen VAE output to the [0, 1] RGB convention used by the critic.

    This is the tensor equivalent of project evaluation's
    ``(pixels.clamp(-1, 1) * 0.5 + 0.5)`` conversion.  It deliberately keeps
    clamp and scaling in the graph for the VAE/critic audit.
    """
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise VAEPreprocessingError(f"Decoded image must be B×3×H×W, got {tuple(decoded.shape)}")
    if not torch.isfinite(decoded).all():
        raise VAEPreprocessingError("Qwen VAE decoded image contains NaN or Inf")
    return decoded.clamp(-1.0, 1.0).mul(0.5).add(0.5)


def _decode_normalized_latents(vae: _VAE, latents: torch.Tensor) -> torch.Tensor:
    """Shared implementation for inference and gradient-preserving decode."""
    if latents.ndim != 4 or latents.shape[1] != getattr(vae.config, "z_dim", None):
        raise VAEPreprocessingError(f"Normalized image latents must be B×C×H×W, got {tuple(latents.shape)}")
    mean, std = _latent_statistics(vae, latents.device, latents.dtype)
    raw = latents.unsqueeze(2) * std + mean
    decoded = vae.decode(raw).sample
    if decoded.ndim != 5 or decoded.shape[2] != 1 or not torch.isfinite(decoded).all():
        raise VAEPreprocessingError("Qwen VAE decode produced invalid image tensor")
    return decoded.squeeze(2)


@torch.inference_mode()
def encode_preprocessed_image(
    vae: _VAE,
    image: Image.Image,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Encode one already geometry-normalized RGB image to a clean latent.

    This is the single-image counterpart to :func:`encode_preprocessed_pair`.
    It intentionally keeps the Qwen tensor conversion, posterior sampling, and
    latent normalization identical to shard preparation, which lets local
    inference encode a pose control without fabricating an unused RGB image.
    """
    pixels = pil_to_qwen_vae_tensor(image).to(device=device, dtype=dtype)
    raw = vae.encode(pixels).latent_dist.sample(generator=generator)
    return _squeeze_image_time(normalize_qwen_latents(raw, vae))


@torch.inference_mode()
def encode_preprocessed_pair(
    vae: _VAE,
    pair: PreprocessedPair,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> EncodedPair:
    """Encode a shared-geometry RGB/control pair and remove its singleton time axis.

    The VAE's posterior is sampled, matching Qwen-Image's ControlNet pipeline.
    The returned tensors have ``C×H×W`` layout and retain the VAE compute dtype;
    shard serialization can explicitly cast them to float32 without changing the
    normalization convention.
    """
    latent = encode_preprocessed_image(vae, pair.rgb, device=device, dtype=dtype, generator=generator)
    control = encode_preprocessed_image(vae, pair.control, device=device, dtype=dtype, generator=generator)
    if latent.shape != control.shape:
        raise VAEPreprocessingError(
            f"RGB/control latent shapes differ: {tuple(latent.shape)} vs {tuple(control.shape)}"
        )
    if control.abs().max().item() == 0.0:
        raise VAEPreprocessingError("Control latent has no measurable nonzero signal")
    return EncodedPair(pair=pair, latent=latent, control=control)


def select_representative_records(
    records: tuple[ManifestRecord, ...], scan_limit: int = 256
) -> dict[str, ManifestRecord]:
    """Pick at most three records by source aspect ratio without preprocessing them.

    The bounded scan is deliberately for smoke-test selection only; it never
    encodes candidates and it is not a dataset-validation substitute.
    """
    if scan_limit < 1:
        raise VAEPreprocessingError("scan_limit must be at least one")
    selected: dict[str, ManifestRecord] = {}
    for record in records[:scan_limit]:
        with Image.open(record.rgb_path) as image:
            width, height = image.size
        category: Literal["square", "portrait", "landscape"]
        ratio = width / height
        if 0.9 <= ratio <= 1.1:
            category = "square"
        elif ratio < 1.0:
            category = "portrait"
        else:
            category = "landscape"
        selected.setdefault(category, record)
        if len(selected) == 3:
            break
    return selected


def tensor_report(tensor: torch.Tensor) -> dict[str, object]:
    """Return concise, JSON-safe diagnostics for one latent tensor."""
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "rms": float(values.square().mean().sqrt().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def _latent_statistics(vae: _VAE, device: torch.device | str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean_values = getattr(vae.config, "latents_mean", None)
    std_values = getattr(vae.config, "latents_std", None)
    channels = getattr(vae.config, "z_dim", None)
    if not isinstance(channels, int) or channels < 1:
        raise VAEPreprocessingError("VAE config has no positive z_dim")
    if not isinstance(mean_values, (list, tuple)) or not isinstance(std_values, (list, tuple)):
        raise VAEPreprocessingError("VAE config must provide latents_mean and latents_std")
    if len(mean_values) != channels or len(std_values) != channels:
        raise VAEPreprocessingError("VAE latent statistics must match z_dim")
    mean = torch.tensor(mean_values, device=device, dtype=dtype).view(1, channels, 1, 1, 1)
    std = torch.tensor(std_values, device=device, dtype=dtype).view(1, channels, 1, 1, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise VAEPreprocessingError("VAE latent statistics must be finite with positive standard deviations")
    return mean, std


def _validate_raw_latents(latents: torch.Tensor, vae: _VAE) -> None:
    if latents.ndim != 5:
        raise VAEPreprocessingError(f"Qwen VAE latents must be B×C×T×H×W, got {tuple(latents.shape)}")
    if latents.shape[1] != getattr(vae.config, "z_dim", None):
        raise VAEPreprocessingError("Latent channel count does not match VAE z_dim")
    if not torch.isfinite(latents).all():
        raise VAEPreprocessingError("Raw VAE latents contain NaN or Inf")


def _squeeze_image_time(latents: torch.Tensor) -> torch.Tensor:
    if latents.shape[0] != 1 or latents.shape[2] != 1:
        raise VAEPreprocessingError(
            f"One image must encode to 1×C×1×H×W, got {tuple(latents.shape)}"
        )
    return latents.squeeze(0).squeeze(1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny real-data paired Qwen/Krea VAE smoke verification.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-limit", type=int, default=256)
    args = parser.parse_args()

    index = DatasetIndex.discover(args.dataset_root)
    manifest = index.dataset_root / "manifests" / f"{args.split}.jsonl"
    records = index.validate_manifests({args.split: manifest}).records_by_split[args.split]
    selected = select_representative_records(records, args.scan_limit)
    vae = load_krea_vae(args.device)
    reports = []
    for category, record in selected.items():
        encoded = encode_preprocessed_pair(vae, preprocess_pair(record), device=args.device)
        reports.append({
            "category": category,
            "stem": record.stem,
            "bucket": list(encoded.pair.geometry.bucket),
            "rgb": tensor_report(encoded.latent),
            "control": tensor_report(encoded.control),
            "shape_match": encoded.latent.shape == encoded.control.shape,
            "control_nonzero": bool(encoded.control.abs().max().item() > 0.0),
        })
    print(json.dumps({"vae": f"{KREA_VAE_REPO_ID}/{KREA_VAE_SUBFOLDER}", "samples": reports, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
