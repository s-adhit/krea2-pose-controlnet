"""Canonical local Krea-2 Turbo Pose Control-LoRA inference entrypoint.

This wrapper deliberately composes the project-owned Turbo sampler, VAE,
online text conditioner, ControlInputLayer model construction, and strict
trainable-state loader.  It contains no model or sampling reimplementation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
from pose_controlnet.resolution_policy import RESOLUTION_768_BUCKETS
from pose_controlnet.paired_preprocessing import (
    apply_resize_center_crop_geometry,
    choose_bucket,
    resize_center_crop_geometry,
)
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.turbo_runtime import (
    TURBO_CFG,
    TURBO_MU,
    TURBO_STEPS,
    raw_to_turbo_control_compatibility,
    sample_turbo_pose_image,
    scale_turbo_control_latent,
    turbo_metadata,
)
from pose_controlnet.style_lora import (
    STYLE_LORA_SPECS,
    StyleLoRAAdapter,
    StyleLoRAAudit,
    applied_style_lora,
    audit_style_lora,
    sha256 as sha256_file,
)
from pose_controlnet.trainable_interpolation import interpolate_trainable_state
from pose_controlnet.vae_preprocessing import (
    decode_normalized_latents,
    encode_preprocessed_image,
    load_krea_vae,
)


DEFAULT_SEED = 42
RELEASE_CONTRACT_PATH = Path(__file__).resolve().parent / "docs/evaluation/release/final_release_v1.json"
RELEASE_CONTRACT_SHA256 = "9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b"
CANONICAL_CANDIDATE = "mix-025"
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
DYNAMIC_768_GEOMETRY = "dynamic_768_bucket"

# Kept as a direct mirror of the frozen JSON to make the release path useful
# without parsing the contract at import time.  ``load_release_contract``
# verifies the artifact and the resolver binds execution to these values.
CANONICAL_ENDPOINTS = {
    "parent-4000": {
        "path": "/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt",
        "sha256": "0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd",
        "step": 4000,
    },
    "finish-control-a4300": {
        "path": "/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-finish-control-4000-to4500/step_004300.pt",
        "sha256": "17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff",
        "step": 4300,
    },
}
STYLE_DEFAULT_STRENGTHS = {"darkbrush": 0.75, "rainywindow": 0.50, "retroanime": 0.50, "realism": 0.25}


class InferenceError(ValueError):
    """Raised when local inference inputs or artifacts violate this contract."""


@dataclass(frozen=True)
class PoseInferenceRequest:
    """Inputs for one canonical local pose-conditioned generation."""

    turbo_checkpoint: Path
    pose_lora_checkpoint: Path | None
    prompt: str
    pose_image: Path
    output: Path
    seed: int = DEFAULT_SEED
    width: int | None = None
    height: int | None = None
    dynamic_768_bucket: bool = False
    steps: int = TURBO_STEPS
    cfg: float = TURBO_CFG
    mu: float = TURBO_MU
    control_scale: float = 1.0
    device: str = "cuda"
    candidate: str = CANONICAL_CANDIDATE
    parent_checkpoint: Path | None = None
    finish_checkpoint: Path | None = None
    geometry_mode: str = NATIVE_GEOMETRY
    style_lora_paths: tuple[Path, ...] = ()
    style_name: str | None = None
    style_strength: float | None = None


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
    candidate: "ResolvedPoseCandidate | None" = None
    style_adapter: StyleLoRAAdapter | None = None
    style_audit: StyleLoRAAudit | None = None


@dataclass(frozen=True)
class ResolvedPoseCandidate:
    """Validated trainable state and immutable provenance for one runtime."""

    candidate_id: str
    trainable_state: Mapping[str, torch.Tensor]
    compatibility_state: Mapping[str, Any]
    checkpoint_step: int | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ResolvedStyleLoRA:
    audit: StyleLoRAAudit
    strength: float


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
    command.add_argument("--candidate", choices=(CANONICAL_CANDIDATE,), default=CANONICAL_CANDIDATE,
                         help="frozen release candidate (default: mix-025)")
    command.add_argument("--parent-ckpt", type=Path, help="override path for the pinned parent-4000 endpoint")
    command.add_argument("--finish-ckpt", type=Path, help="override path for the pinned finish-control-a4300 endpoint")
    command.add_argument("--pose-lora-ckpt", "--control-ckpt", dest="pose_lora_ckpt", type=Path,
                         help="historical explicit full pose-control checkpoint (not the canonical mix)")
    command.add_argument("--prompt", required=True)
    command.add_argument("--pose-image", type=Path, required=True, help="rendered pose skeleton image")
    command.add_argument("--output", type=Path, required=True, help="generated .png/.jpg image path")
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--width", type=int, default=None)
    command.add_argument("--height", type=int, default=None)
    command.add_argument("--dynamic-768-bucket", action="store_true",
                         help="explicit opt-in: choose the shared 768 bucket from the pose image aspect ratio")
    command.add_argument("--steps", type=int, default=TURBO_STEPS)
    command.add_argument("--cfg", type=float, default=TURBO_CFG)
    command.add_argument("--mu", type=float, default=TURBO_MU)
    command.add_argument("--control-scale", type=float, default=1.0)
    command.add_argument("--style-lora", type=Path, action="append", default=[],
                         help="one Style-LoRA safetensors path; custom paths require --style-name")
    command.add_argument("--style-name", choices=tuple(STYLE_LORA_SPECS),
                         help="use one frozen project-known Style-LoRA default, or declare a custom path's mapping")
    command.add_argument("--style-strength", type=float,
                         help="Style-LoRA strength; named styles use their frozen default when omitted")
    command.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return command


def request_from_args(args: argparse.Namespace) -> PoseInferenceRequest:
    if args.dynamic_768_bucket and (args.width is not None or args.height is not None):
        raise InferenceError("--dynamic-768-bucket cannot be combined with --width or --height")
    if (args.width is None) != (args.height is None):
        raise InferenceError("--width and --height must be supplied together")
    if args.pose_lora_ckpt is not None and (args.parent_ckpt is not None or args.finish_ckpt is not None):
        raise InferenceError("--pose-lora-ckpt cannot be combined with --parent-ckpt or --finish-ckpt")
    return PoseInferenceRequest(
        turbo_checkpoint=args.turbo_ckpt,
        pose_lora_checkpoint=args.pose_lora_ckpt,
        prompt=args.prompt,
        pose_image=args.pose_image,
        output=args.output,
        seed=args.seed,
        width=None if args.dynamic_768_bucket else args.width,
        height=None if args.dynamic_768_bucket else args.height,
        dynamic_768_bucket=args.dynamic_768_bucket,
        steps=args.steps,
        cfg=args.cfg,
        mu=args.mu,
        control_scale=args.control_scale,
        device=args.device,
        candidate=args.candidate,
        parent_checkpoint=args.parent_ckpt,
        finish_checkpoint=args.finish_ckpt,
        geometry_mode=DYNAMIC_768_GEOMETRY if args.dynamic_768_bucket else NATIVE_GEOMETRY,
        style_lora_paths=tuple(args.style_lora),
        style_name=args.style_name,
        style_strength=args.style_strength,
    )


def _validate_request(request: PoseInferenceRequest) -> None:
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise InferenceError("prompt must be a non-empty string")
    if not isinstance(request.seed, int) or isinstance(request.seed, bool) or not 0 <= request.seed < 2**63:
        raise InferenceError("seed must be an integer in [0, 2**63)")
    if request.dynamic_768_bucket and request.geometry_mode != DYNAMIC_768_GEOMETRY:
        raise InferenceError("dynamic_768_bucket conflicts with geometry_mode")
    if request.geometry_mode not in (NATIVE_GEOMETRY, DYNAMIC_768_GEOMETRY, "explicit"):
        raise InferenceError(f"invalid geometry mode: {request.geometry_mode}")
    if request.dynamic_768_bucket or request.geometry_mode == DYNAMIC_768_GEOMETRY:
        if request.width is not None or request.height is not None:
            raise InferenceError("dynamic_768_bucket requires width and height to be None")
    else:
        if (request.width is None) != (request.height is None):
            raise InferenceError("native/explicit geometry requires both width and height when either is supplied")
        if request.width is not None and request.height is not None:
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
    if request.pose_lora_checkpoint is not None and (request.parent_checkpoint is not None or request.finish_checkpoint is not None):
        raise InferenceError("explicit pose checkpoint cannot be combined with canonical endpoints")
    if request.candidate != CANONICAL_CANDIDATE:
        raise InferenceError(f"unsupported candidate: {request.candidate}")
    if len(request.style_lora_paths) > 1:
        raise InferenceError("exactly zero or one Style-LoRA is supported; multiple Style-LoRAs are not supported")
    if request.style_strength is not None and (not isinstance(request.style_strength, (int, float))
                                               or isinstance(request.style_strength, bool)
                                               or not math.isfinite(float(request.style_strength))
                                               or request.style_strength < 0):
        raise InferenceError("style strength must be finite and non-negative")
    if request.style_strength is not None and not request.style_lora_paths and request.style_name is None:
        raise InferenceError("--style-strength requires --style-lora or --style-name")


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


def load_release_contract() -> dict[str, Any]:
    """Read the immutable v1 contract and reject any byte or schema drift."""
    _require_file(RELEASE_CONTRACT_PATH, "frozen release contract")
    if sha256_file(RELEASE_CONTRACT_PATH) != RELEASE_CONTRACT_SHA256:
        raise InferenceError(f"Frozen release contract SHA-256 mismatch: {RELEASE_CONTRACT_PATH}")
    try:
        contract = json.loads(RELEASE_CONTRACT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Frozen release contract is invalid JSON: {RELEASE_CONTRACT_PATH}") from exc
    candidate = contract.get("candidate") if isinstance(contract, dict) else None
    interpolation = candidate.get("interpolation") if isinstance(candidate, dict) else None
    if (not isinstance(interpolation, dict) or candidate.get("id") != CANONICAL_CANDIDATE
            or interpolation.get("alpha") != 0.25
            or interpolation.get("formula") != "(1 - alpha) * parent-4000 + alpha * finish-control-a4300"):
        raise InferenceError("Frozen release contract lacks the canonical mix-025 interpolation")
    endpoints = {row.get("id"): row for row in interpolation.get("endpoints", []) if isinstance(row, dict)}
    for endpoint_id, frozen in CANONICAL_ENDPOINTS.items():
        observed = endpoints.get(endpoint_id)
        if not isinstance(observed, dict) or observed.get("path") != frozen["path"] or observed.get("sha256") != frozen["sha256"]:
            raise InferenceError(f"Frozen release contract endpoint mismatch: {endpoint_id}")
    return contract


def _endpoint_records(request: PoseInferenceRequest) -> list[dict[str, Any]]:
    overrides = (request.parent_checkpoint, request.finish_checkpoint)
    if any(path is not None for path in overrides) and not all(path is not None for path in overrides):
        raise InferenceError("canonical mix-025 requires both --parent-ckpt and --finish-ckpt when overriding endpoints")
    endpoints = []
    for endpoint_id, override in zip(("parent-4000", "finish-control-a4300"), overrides):
        frozen = CANONICAL_ENDPOINTS[endpoint_id]
        endpoints.append({"id": endpoint_id, "path": Path(override or frozen["path"]),
                          "sha256": frozen["sha256"], "step": frozen["step"]})
    return endpoints


def _validate_endpoint(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(endpoint["path"])
    _require_file(path, f"canonical {endpoint['id']} endpoint checkpoint")
    observed = sha256_file(path)
    if observed != endpoint["sha256"]:
        raise InferenceError(f"Canonical {endpoint['id']} endpoint SHA-256 mismatch: {path}")
    try:
        state = load_training_state(path)
    except (ValueError, KeyError, TypeError) as exc:
        raise InferenceError(f"Canonical {endpoint['id']} endpoint is not a valid full training checkpoint: {path}") from exc
    if state.get("global_step") != endpoint["step"]:
        raise InferenceError(f"Canonical {endpoint['id']} endpoint embedded step does not match its frozen contract: {path}")
    if not isinstance(state.get("model"), Mapping):
        raise InferenceError(f"Canonical {endpoint['id']} endpoint lacks trainable model tensors: {path}")
    return state


def resolve_pose_candidate(request: PoseInferenceRequest) -> ResolvedPoseCandidate:
    """Resolve either the pinned mix-025 blend or one historical checkpoint."""
    if request.pose_lora_checkpoint is not None:
        _require_file(request.pose_lora_checkpoint, "pose control checkpoint")
        try:
            state = load_training_state(request.pose_lora_checkpoint)
        except (ValueError, KeyError, TypeError) as exc:
            raise InferenceError(f"Pose control checkpoint metadata is incompatible: {request.pose_lora_checkpoint}") from exc
        step = state.get("global_step")
        if not isinstance(step, int) or not isinstance(state.get("model"), Mapping):
            raise InferenceError("Pose control checkpoint lacks a usable trainable model state")
        return ResolvedPoseCandidate(
            candidate_id="explicit-checkpoint", trainable_state=state["model"], compatibility_state=state,
            checkpoint_step=step,
            provenance={"candidate_id": "explicit-checkpoint", "pose_lora_checkpoint": str(request.pose_lora_checkpoint.resolve())},
        )

    contract = load_release_contract()
    endpoints = _endpoint_records(request)
    states = [_validate_endpoint(endpoint) for endpoint in endpoints]
    raw_checkpoints = [state.get("config", {}).get("raw_ckpt") for state in states]
    if not all(isinstance(raw, str) and raw for raw in raw_checkpoints) or raw_checkpoints[0] != raw_checkpoints[1]:
        raise InferenceError("Canonical mix-025 endpoints do not share exact Krea-2 Raw provenance")
    try:
        trainable = interpolate_trainable_state(states[0]["model"], states[1]["model"], 0.25)
    except ValueError as exc:
        raise InferenceError(f"Canonical mix-025 trainable tensor interpolation is incompatible: {exc}") from exc
    endpoint_provenance = [
        {"id": endpoint["id"], "path": str(Path(endpoint["path"]).resolve()),
         "sha256": endpoint["sha256"], "step": endpoint["step"]}
        for endpoint in endpoints
    ]
    interpolation = {
        "candidate_id": CANONICAL_CANDIDATE, "alpha": 0.25,
        "formula": "(1 - alpha) * parent-4000 + alpha * finish-control-a4300",
        "tensor_scope": "state['model'] trainable control/LoRA tensors only", "compute_dtype": "float32",
        "endpoints": endpoint_provenance,
    }
    return ResolvedPoseCandidate(
        candidate_id=CANONICAL_CANDIDATE, trainable_state=trainable,
        compatibility_state={"config": {"raw_ckpt": raw_checkpoints[0]}, "model": trainable}, checkpoint_step=None,
        provenance={"release_contract": {"id": contract["release_id"], "path": str(RELEASE_CONTRACT_PATH),
                                           "sha256": RELEASE_CONTRACT_SHA256},
                    "candidate_id": CANONICAL_CANDIDATE, "candidate_kind": "trainable_tensor_interpolation",
                    "interpolation": interpolation},
    )


def resolve_style_lora(request: PoseInferenceRequest) -> ResolvedStyleLoRA | None:
    """Audit exactly one requested adapter before it can reach the model."""
    if not request.style_lora_paths and request.style_name is None:
        return None
    style_name = request.style_name
    supplied = request.style_lora_paths[0] if request.style_lora_paths else None
    if style_name is None:
        for name, spec in STYLE_LORA_SPECS.items():
            if supplied == Path(spec["path"]):
                style_name = name
                break
        if style_name is None:
            raise InferenceError("A custom --style-lora requires --style-name; no namespace is guessed")
    path = supplied or Path(STYLE_LORA_SPECS[style_name]["path"])
    _require_file(path, "Style-LoRA")
    # A named default remains hash-pinned.  An explicitly supplied path is
    # structurally audited against the declared known namespace and recorded
    # verbatim; it is never silently substituted with a default.
    expected_digest = None if supplied is not None else STYLE_LORA_SPECS[style_name]["sha256"]
    audit = audit_style_lora(style_name, expected_path=path, expected_sha256=expected_digest or sha256_file(path))
    if not audit.supported:
        raise InferenceError(f"Style-LoRA is incompatible: {path}: {audit.errors}")
    strength = STYLE_DEFAULT_STRENGTHS[style_name] if request.style_strength is None else float(request.style_strength)
    return ResolvedStyleLoRA(audit=audit, strength=strength)


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
    if request.dynamic_768_bucket or request.geometry_mode == DYNAMIC_768_GEOMETRY:
        bucket = choose_bucket(source.size, RESOLUTION_768_BUCKETS)
        mode = DYNAMIC_768_GEOMETRY
    elif request.width is None or request.height is None:
        # Native public inference keeps the supplied pose canvas exactly; it
        # never substitutes a square or dynamic-768 bucket behind the user's
        # back.  The model/VAE alignment requirement remains explicit.
        bucket = source.size
        _validate_dimensions(*bucket)
        mode = NATIVE_GEOMETRY
    else:
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


def load_inference_runtime(request: PoseInferenceRequest, *, candidate: ResolvedPoseCandidate | None = None,
                           style: ResolvedStyleLoRA | None = None) -> InferenceRuntime:
    """Load Turbo, the project control/LoRA state, VAE, and online text model."""
    _require_file(request.turbo_checkpoint, "Turbo checkpoint")
    device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Krea-2 Turbo inference; run from the GH200 host shell")
    resolved = candidate or resolve_pose_candidate(request)
    resolved_style = style or resolve_style_lora(request)
    try:
        model = build_turbo_pose_model(str(request.turbo_checkpoint), 64, 64, str(device)).eval()
        raw_to_turbo_control_compatibility(model, dict(resolved.compatibility_state))
        load_trainable_state_dict(model, resolved.trainable_state)
    except (KeyError, RuntimeError, ValueError, AssertionError) as exc:
        raise InferenceError("Pose control candidate is incompatible with Krea-2 Turbo") from exc
    adapter = None
    if resolved_style is not None and resolved_style.strength != 0.0:
        try:
            adapter = StyleLoRAAdapter.load(resolved_style.audit, device=device)
        except (ValueError, RuntimeError) as exc:
            raise InferenceError(f"Style-LoRA could not be loaded safely: {resolved_style.audit.path}") from exc
    return InferenceRuntime(
        model=model,
        vae=load_krea_vae(device),
        conditioner=PoseTextConditioner(device=str(device), dtype=torch.bfloat16),
        device=device,
        checkpoint_step=resolved.checkpoint_step,
        candidate=resolved,
        style_adapter=adapter,
        style_audit=None if resolved_style is None else resolved_style.audit,
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
                   checkpoint_step: int | None, candidate: ResolvedPoseCandidate | None = None,
                   style: ResolvedStyleLoRA | None = None) -> dict[str, Any]:
    candidate_metadata = candidate.provenance if candidate is not None else {
        "candidate_id": "explicit-checkpoint" if request.pose_lora_checkpoint is not None else request.candidate,
    }
    style_metadata = {
        "name": None if style is None else style.audit.style_id,
        "path": None if style is None else str(Path(style.audit.path).resolve()),
        "sha256": None if style is None else style.audit.sha256,
        "strength": 0.0 if style is None else style.strength,
        "trigger_phrase": None,
        "effective_prompt": request.prompt,
    }
    release = candidate_metadata.get("release_contract")
    if release is None:
        contract = load_release_contract()
        release = {"id": contract["release_id"], "path": str(RELEASE_CONTRACT_PATH),
                   "sha256": RELEASE_CONTRACT_SHA256}
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
        "pose_lora_checkpoint": (None if request.pose_lora_checkpoint is None
                                  else str(request.pose_lora_checkpoint.resolve())),
        "checkpoint_step": checkpoint_step,
        "pose_image": str(request.pose_image.resolve()),
        "pose_image_sha256": sha256_file(request.pose_image),
        "geometry_mode": control.mode,
        "geometry": control.geometry,
        "output_dimensions": control.geometry["bucket"],
        "release": release,
        "candidate": candidate_metadata,
        "style_lora": style_metadata,
        "output_path": str(request.output.resolve()),
    }


def generate_pose(request: PoseInferenceRequest, *, runtime: InferenceRuntime | None = None) -> PoseInferenceResult:
    """Generate one image with the canonical locked Turbo pose-control recipe."""
    _validate_request(request)
    _require_file(request.turbo_checkpoint, "Turbo checkpoint")
    resolved_candidate = runtime.candidate if runtime is not None else resolve_pose_candidate(request)
    resolved_style = resolve_style_lora(request)
    control = prepare_pose_control(request)
    loaded = runtime or load_inference_runtime(request, candidate=resolved_candidate, style=resolved_style)
    control_latent = _control_latent(loaded, control, request.seed)
    context, mask = loaded.conditioner([request.prompt])
    sample = {
        "latent": torch.zeros_like(control_latent),
        "control": control_latent,
        "context": context[0],
        "mask": mask[0],
    }
    def sample_pixels():
        return sample_turbo_pose_image(
            loaded.model, lambda latent: decode_normalized_latents(loaded.vae, latent), sample, loaded.device,
            request.seed, steps=request.steps, guidance=request.cfg, mu=request.mu,
            control_scale=request.control_scale,
        )
    if loaded.style_adapter is not None:
        # The context manager removes every hook even if sampling fails; style
        # tensors are never merged into the Pose Control-LoRA state.
        with applied_style_lora(loaded.model, loaded.style_adapter, resolved_style.strength if resolved_style else 0.0):
            pixels = sample_pixels()
    else:
        pixels = sample_pixels()
    request.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(request.output)
    metadata = _write_json(metadata_path_for(request.output), build_metadata(
        request, control, loaded.checkpoint_step, loaded.candidate or resolved_candidate, resolved_style,
    ))
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
