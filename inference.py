"""Canonical local Krea-2 Turbo Pose Control-LoRA inference entrypoint.

This wrapper deliberately composes the project-owned Turbo sampler, VAE,
online text conditioner, ControlInputLayer model construction, and strict
trainable-state loader.  It contains no model or sampling reimplementation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
from pose_controlnet.overfit_capacity import RESOLUTION_768_BUCKETS
from pose_controlnet.paired_preprocessing import (
    apply_resize_center_crop_geometry,
    choose_bucket,
    resize_center_crop_geometry,
)
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.turbo_evaluation import (
    TURBO_CFG,
    TURBO_MU,
    TURBO_STEPS,
    raw_to_turbo_control_compatibility,
    sample_turbo_pose_image,
    scale_turbo_control_latent,
    turbo_metadata,
)
from pose_controlnet.vae_preprocessing import (
    decode_normalized_latents,
    encode_preprocessed_image,
    load_krea_vae,
)


DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768
DEFAULT_SEED = 42
POSE_CHECKPOINT_CANDIDATES = (
    "parent-4000",
    "finish-control-a4300",
    "finish-anneal-b4200",
)


class InferenceError(ValueError):
    """Raised when local inference inputs or artifacts violate this contract."""


@dataclass(frozen=True)
class PoseInferenceRequest:
    """Inputs for one canonical local pose-conditioned generation."""

    turbo_checkpoint: Path
    pose_lora_checkpoint: Path
    prompt: str
    pose_image: Path
    output: Path
    seed: int = DEFAULT_SEED
    width: int | None = DEFAULT_WIDTH
    height: int | None = DEFAULT_HEIGHT
    dynamic_768_bucket: bool = False
    steps: int = TURBO_STEPS
    cfg: float = TURBO_CFG
    mu: float = TURBO_MU
    control_scale: float = 1.0
    device: str = "cuda"


@dataclass(frozen=True)
class PreparedPoseControl:
    """A pose image transformed with the project shared geometry contract."""

    image: Image.Image
    mode: str
    geometry: dict[str, list[int]]


@dataclass(frozen=True)
class InferenceRuntime:
    """Loaded project components; injectable for wrappers and CPU tests."""

    model: Any
    vae: Any
    conditioner: Any
    device: torch.device
    checkpoint_step: int | None


@dataclass(frozen=True)
class PoseInferenceResult:
    """Locations and reproducibility facts produced by :func:`generate_pose`."""

    output: Path
    metadata: Path
    checkpoint_step: int | None
    geometry: dict[str, list[int]]


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--turbo-ckpt", type=Path, required=True, help="Krea-2 Turbo safetensors checkpoint")
    command.add_argument("--pose-lora-ckpt", "--control-ckpt", dest="pose_lora_ckpt", type=Path, required=True,
                         help="full pose-control training checkpoint (explicit candidate selection)")
    command.add_argument("--prompt", required=True)
    command.add_argument("--pose-image", type=Path, required=True, help="rendered pose skeleton image")
    command.add_argument("--output", type=Path, required=True, help="generated .png/.jpg image path")
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--width", type=int, default=None)
    command.add_argument("--height", type=int, default=None)
    command.add_argument("--dynamic-768-bucket", action="store_true",
                         help="choose the shared production 768 bucket from the pose image aspect ratio")
    command.add_argument("--steps", type=int, default=TURBO_STEPS)
    command.add_argument("--cfg", type=float, default=TURBO_CFG)
    command.add_argument("--mu", type=float, default=TURBO_MU)
    command.add_argument("--control-scale", type=float, default=1.0)
    command.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return command


def request_from_args(args: argparse.Namespace) -> PoseInferenceRequest:
    if args.dynamic_768_bucket and (args.width is not None or args.height is not None):
        raise InferenceError("--dynamic-768-bucket cannot be combined with --width or --height")
    if (args.width is None) != (args.height is None):
        raise InferenceError("--width and --height must be supplied together")
    return PoseInferenceRequest(
        turbo_checkpoint=args.turbo_ckpt,
        pose_lora_checkpoint=args.pose_lora_ckpt,
        prompt=args.prompt,
        pose_image=args.pose_image,
        output=args.output,
        seed=args.seed,
        width=None if args.dynamic_768_bucket else (DEFAULT_WIDTH if args.width is None else args.width),
        height=None if args.dynamic_768_bucket else (DEFAULT_HEIGHT if args.height is None else args.height),
        dynamic_768_bucket=args.dynamic_768_bucket,
        steps=args.steps,
        cfg=args.cfg,
        mu=args.mu,
        control_scale=args.control_scale,
        device=args.device,
    )


def _validate_request(request: PoseInferenceRequest) -> None:
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise InferenceError("prompt must be a non-empty string")
    if not isinstance(request.seed, int) or isinstance(request.seed, bool) or not 0 <= request.seed < 2**63:
        raise InferenceError("seed must be an integer in [0, 2**63)")
    if request.dynamic_768_bucket:
        if request.width is not None or request.height is not None:
            # The CLI normalizes its implicit dimensions before construction;
            # callable users select dynamic mode with explicit None values.
            raise InferenceError("dynamic_768_bucket requires width and height to be None")
    else:
        if request.width is None or request.height is None:
            raise InferenceError("fixed geometry requires both width and height")
        _validate_dimensions(request.width, request.height)
    if request.steps != TURBO_STEPS or request.cfg != TURBO_CFG or request.mu != TURBO_MU:
        raise InferenceError(
            f"canonical Turbo inference is locked to steps={TURBO_STEPS}, cfg={TURBO_CFG}, mu={TURBO_MU}"
        )
    try:
        scale_turbo_control_latent(torch.zeros(1), request.control_scale)
    except (TypeError, ValueError) as exc:
        raise InferenceError(f"invalid control_scale: {exc}") from exc
    if request.output.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise InferenceError("output must have a supported image suffix: .png, .jpg, .jpeg, or .webp")


def _validate_dimensions(width: int, height: int) -> None:
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
        raise InferenceError("width and height must be integers")
    if width <= 0 or height <= 0:
        raise InferenceError("width and height must be positive")
    if width % 16 or height % 16:
        raise InferenceError("width and height must be divisible by 16 for VAE/model token alignment")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing or not a file: {path}")


def load_pose_image(path: Path) -> Image.Image:
    _require_file(path, "pose image")
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise InferenceError(f"Malformed pose image: {path}") from exc


def prepare_pose_control(request: PoseInferenceRequest) -> PreparedPoseControl:
    """Apply the same resize-to-cover/center-crop helper used by training."""
    source = load_pose_image(request.pose_image)
    if request.dynamic_768_bucket:
        bucket = choose_bucket(source.size, RESOLUTION_768_BUCKETS)
        mode = "production-dynamic-768"
    else:
        assert request.width is not None and request.height is not None
        _validate_dimensions(request.width, request.height)
        bucket = (request.width, request.height)
        mode = "explicit"
    geometry = resize_center_crop_geometry(source.size, bucket)
    image = apply_resize_center_crop_geometry(source, geometry)
    return PreparedPoseControl(
        image=image,
        mode=mode,
        geometry={
            "source_size": list(geometry.source_size),
            "resized_size": list(geometry.resized_size),
            "crop_box": list(geometry.crop_box),
            "bucket": list(geometry.bucket),
        },
    )


def seed_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """Return the explicit generator used for posterior sampling of control."""
    return torch.Generator(device=device).manual_seed(seed)


def load_inference_runtime(request: PoseInferenceRequest) -> InferenceRuntime:
    """Load Turbo, the project control/LoRA state, VAE, and online text model."""
    _require_file(request.turbo_checkpoint, "Turbo checkpoint")
    _require_file(request.pose_lora_checkpoint, "pose control checkpoint")
    device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Krea-2 Turbo inference; run from the GH200 host shell")
    try:
        state = load_training_state(request.pose_lora_checkpoint)
    except (ValueError, KeyError, TypeError) as exc:
        raise InferenceError(f"Pose control checkpoint metadata is incompatible: {request.pose_lora_checkpoint}") from exc
    step = state.get("global_step")
    if not isinstance(step, int):
        raise InferenceError("Pose control checkpoint metadata lacks integer global_step")
    try:
        model = build_turbo_pose_model(str(request.turbo_checkpoint), 64, 64, str(device)).eval()
        raw_to_turbo_control_compatibility(model, state)
        load_trainable_state_dict(model, state["model"])
    except (KeyError, RuntimeError, ValueError, AssertionError) as exc:
        raise InferenceError(f"Pose control checkpoint is incompatible with Krea-2 Turbo: {request.pose_lora_checkpoint}") from exc
    return InferenceRuntime(
        model=model,
        vae=load_krea_vae(device),
        conditioner=PoseTextConditioner(device=str(device), dtype=torch.bfloat16),
        device=device,
        checkpoint_step=step,
    )


def _control_latent(runtime: InferenceRuntime, control: PreparedPoseControl, seed: int) -> torch.Tensor:
    latent = encode_preprocessed_image(
        runtime.vae,
        control.image,
        device=runtime.device,
        generator=seed_generator(seed, runtime.device),
    )
    if not torch.isfinite(latent).all() or latent.abs().max().item() == 0.0:
        raise InferenceError("Pose VAE encoding produced an empty or non-finite control latent")
    return latent


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def metadata_path_for(output: Path) -> Path:
    return output.with_suffix(".json")


def build_metadata(request: PoseInferenceRequest, control: PreparedPoseControl,
                   checkpoint_step: int | None) -> dict[str, Any]:
    return {
        "format_version": 1,
        "mode": "turbo-pose-control",
        "turbo": turbo_metadata(),
        "prompt": request.prompt,
        "seed": request.seed,
        "width": control.geometry["bucket"][0],
        "height": control.geometry["bucket"][1],
        "steps": request.steps,
        "cfg": request.cfg,
        "mu": request.mu,
        "control_scale": float(request.control_scale),
        "turbo_checkpoint": str(request.turbo_checkpoint.resolve()),
        "pose_lora_checkpoint": str(request.pose_lora_checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "pose_image": str(request.pose_image.resolve()),
        "geometry_mode": control.mode,
        "geometry": control.geometry,
        "output_path": str(request.output.resolve()),
    }


def generate_pose(request: PoseInferenceRequest, *, runtime: InferenceRuntime | None = None) -> PoseInferenceResult:
    """Generate one image with the canonical locked Turbo pose-control recipe."""
    _validate_request(request)
    _require_file(request.turbo_checkpoint, "Turbo checkpoint")
    _require_file(request.pose_lora_checkpoint, "pose control checkpoint")
    control = prepare_pose_control(request)
    loaded = runtime or load_inference_runtime(request)
    control_latent = _control_latent(loaded, control, request.seed)
    context, mask = loaded.conditioner([request.prompt])
    sample = {
        "latent": torch.zeros_like(control_latent),
        "control": control_latent,
        "context": context[0],
        "mask": mask[0],
    }
    pixels = sample_turbo_pose_image(
        loaded.model,
        lambda latent: decode_normalized_latents(loaded.vae, latent),
        sample,
        loaded.device,
        request.seed,
        steps=request.steps,
        guidance=request.cfg,
        mu=request.mu,
        control_scale=request.control_scale,
    )
    request.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(request.output)
    metadata = _write_json(metadata_path_for(request.output), build_metadata(request, control, loaded.checkpoint_step))
    return PoseInferenceResult(
        output=request.output,
        metadata=metadata,
        checkpoint_step=loaded.checkpoint_step,
        geometry=control.geometry,
    )


def main(argv: Sequence[str] | None = None) -> PoseInferenceResult:
    request = request_from_args(parser().parse_args(argv))
    result = generate_pose(request)
    print(json.dumps({"output": str(result.output), "metadata": str(result.metadata)}, sort_keys=True))
    return result


if __name__ == "__main__":  # pragma: no cover - CLI exercised through main
    main()
