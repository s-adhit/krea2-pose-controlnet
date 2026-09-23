"""Thin ComfyUI-facing adapter over the canonical frozen Turbo sampling path."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors import safe_open

from ..geometry import apply_geometry, resize_center_crop_geometry
from .model import build_turbo_pose_model, load_trainable_state_dict, trainable_state_dict
from .release_artifact import load_release_artifact
from .text import PoseTextConditioner
from .turbo_runtime import TURBO_CFG, TURBO_MU, TURBO_STEPS, raw_to_turbo_control_compatibility, sample_turbo_pose_image
from .vae import decode_normalized_latents, encode_preprocessed_image, load_krea_vae

RELEASE_FILENAME = "krea2-pose-control-mix025.safetensors"
RELEASE_SHA256 = "6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1"
RELEASE_TENSOR_COUNT = 450
RELEASE_PARAMETER_COUNT = 215_488_512


class ReleaseValidationError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_release_artifact(value: str | Path) -> Path:
    """Resolve a regular local path or ``hf://org/repo/path/to/file`` once."""
    text = str(value).strip()
    if not text:
        raise FileNotFoundError("release artifact path is required")
    if not text.startswith("hf://"):
        path = Path(text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"release artifact not found: {path}")
        return path
    parts = text.removeprefix("hf://").split("/")
    if len(parts) < 3:
        raise ValueError("HF artifact path must be hf://<owner>/<repo>/<filename>")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for hf:// release artifacts") from error
    return Path(hf_hub_download(repo_id="/".join(parts[:2]), filename="/".join(parts[2:]), repo_type="model"))


def validate_release_artifact(path: str | Path, *, inspect_tensors: bool = True) -> Path:
    artifact = Path(path)
    if sha256_file(artifact) != RELEASE_SHA256:
        raise ReleaseValidationError(f"release SHA-256 mismatch: expected {RELEASE_SHA256}")
    if inspect_tensors:
        with safe_open(artifact, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            count = sum(math.prod(handle.get_slice(name).get_shape()) for name in names)
        if len(names) != RELEASE_TENSOR_COUNT or count != RELEASE_PARAMETER_COUNT:
            raise ReleaseValidationError(
                f"release tensor contract mismatch: tensors={len(names)}, parameters={count}"
            )
    return artifact


def _validate_dimensions(size: tuple[int, int]) -> None:
    if min(size) <= 0 or size[0] % 16 or size[1] % 16:
        raise ValueError("generation width and height must be positive and divisible by 16")


@dataclass
class Runtime:
    model: Any
    vae: Any
    conditioner: Any
    device: torch.device


def load_runtime(*, turbo_path: str | Path, release_artifact: str | Path,
                 base_raw_path: str | Path = "", device: str = "cuda") -> Runtime:
    """Load only frozen release tensors into the official Turbo architecture.

    The optional Raw path is checked for user-error visibility; Turbo is the
    actual inference base.  No NFS checkpoint or historical endpoint is used.
    """
    turbo = Path(turbo_path).expanduser()
    if not turbo.is_file():
        raise FileNotFoundError(f"Krea-2 Turbo checkpoint not found: {turbo}")
    if base_raw_path and not Path(base_raw_path).expanduser().is_file():
        raise FileNotFoundError(f"Krea-2 Raw checkpoint not found: {base_raw_path}")
    artifact_path = validate_release_artifact(resolve_release_artifact(release_artifact))
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Krea-2 Turbo generation")
    artifact = load_release_artifact(artifact_path)
    model = build_turbo_pose_model(str(turbo), rank=64, alpha=64, device=device).eval()
    raw_to_turbo_control_compatibility(model, artifact)
    expected = trainable_state_dict(model)
    if len(expected) != RELEASE_TENSOR_COUNT or sum(value.numel() for value in expected.values()) != RELEASE_PARAMETER_COUNT:
        raise RuntimeError("runtime trainable tensor contract differs from frozen release")
    load_trainable_state_dict(model, artifact["model"])
    return Runtime(model=model, vae=load_krea_vae(device),
                   conditioner=PoseTextConditioner(device=device), device=torch.device(device))


def generate(runtime: Runtime, *, prompt: str, pose_image: Image.Image, seed: int,
             control_scale: float = 1.0, width: int | None = None, height: int | None = None,
             steps: int = TURBO_STEPS, mu: float = TURBO_MU) -> Image.Image:
    """Canonical CFG-0, 8-step, fixed-mu Turbo generation; no Style-LoRA path."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if steps != TURBO_STEPS or mu != TURBO_MU:
        raise ValueError(f"frozen release requires steps={TURBO_STEPS}, mu={TURBO_MU}")
    if not isinstance(seed, int) or seed < 0 or seed >= 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    source = pose_image.convert("RGB")
    target = source.size if width is None and height is None else (int(width or 0), int(height or 0))
    _validate_dimensions(target)
    prepared = apply_geometry(source, resize_center_crop_geometry(source.size, target))
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    control = encode_preprocessed_image(runtime.vae, prepared, device=runtime.device, generator=generator)
    context, mask = runtime.conditioner([prompt])
    sample = {"latent": torch.zeros_like(control), "control": control, "context": context[0], "mask": mask[0]}
    pixels = sample_turbo_pose_image(runtime.model, lambda z: decode_normalized_latents(runtime.vae, z), sample,
                                     runtime.device, seed, steps=steps, guidance=TURBO_CFG, mu=mu,
                                     control_scale=float(control_scale))
    return Image.fromarray(pixels)
