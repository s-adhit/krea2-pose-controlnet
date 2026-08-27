"""Evaluation-only Krea-2 Turbo sampling contract for Pose-ControlNet.

This deliberately has no training-loop, optimizer, or gradient APIs.  It is a
small adaptation of the official Krea sampler: the image tokens remain noisy,
the clean pose tokens are concatenated at every denoising forward, and Turbo's
constant shift is passed explicitly rather than inferred from resolution.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from einops import rearrange

from pose_controlnet.diffusion import forward_pose_control, patchify_and_position
from pose_controlnet.evaluation import ordered_checkpoints
from pose_controlnet.checkpointing import (
    load_training_state,
    validated_hf_checkpoint_for_step,
    validated_local_checkpoint_for_hf_step,
)
from pose_controlnet.model import trainable_state_dict
from pose_controlnet.paired_preprocessing import resize_center_crop_geometry


TURBO_STEPS = 8
TURBO_CFG = 0.0
TURBO_MU = 1.15
TURBO_SIGMA = 1.0
# The original two checkpoints remain the backwards-compatible default for
# callers which do not pass ``--steps``.  Reports always use the full ordered
# comparison set once every result is available.
TURBO_CHECKPOINT_STEPS = (800, 900, 1200, 1500)
DEFAULT_TURBO_CHECKPOINT_STEPS = (800, 1500)
CANONICAL_EVALUATION_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500")
ORIGINAL_TURBO_EVALUATION_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0")
LR5E5_TURBO_EVALUATION_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0-lr5e5")
LR5E5_CHECKPOINT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500")
LR5E5_HF_RUN_NAME = "pose-learning-900-lr5e5-to1500"
LR5E5_HF_REPO_ID = "adhit-420/Krea-2-PoseControl-LoRA-checkpoints"
LR5E5_TURBO_CHECKPOINT_STEPS = (1000, 1100, 1200, 1300, 1400, 1500)
TIMESTEP_TURBO_EVALUATION_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0-timestep-lowmid20")
TIMESTEP_CHECKPOINT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-1500-timestep-lowmid20-to1800")
TIMESTEP_HF_RUN_NAME = "pose-learning-1500-timestep-lowmid20-to1800"
TIMESTEP_HF_REPO_ID = "adhit-420/Krea-2-PoseControl-LoRA-checkpoints"
TIMESTEP_TURBO_CHECKPOINT_STEPS = (1600, 1700, 1800)
CONTROL_SCALE_TURBO_EVALUATION_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/turbo-control-scale-step1500")
CONTROL_SCALE_VALUES = (0.75, 1.0, 1.25, 1.5, 2.0)


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


def turbo_scoring_geometry(sample: Mapping[str, Any]) -> dict[str, list[int]]:
    """Return canonical persisted paired geometry for one Turbo PCK sample.

    This consumes the source dimensions and bucket recorded in the prepared
    shard, never generated-image pixels.  Recomputing through the shared
    paired-preprocessing helper both preserves the canonical contract and
    detects a malformed/stale shard geometry before PCK is run.
    """
    stem = sample.get("stem", "<unknown>")
    fields = ("source_size", "resized_size", "crop_box", "bucket")
    missing = [field for field in fields if sample.get(field) is None]
    if missing:
        raise ValueError(
            f"Turbo scoring geometry for stem {stem!r} is missing persisted paired fields: {', '.join(missing)}"
        )
    try:
        source_size = tuple(sample["source_size"])
        bucket = tuple(sample["bucket"])
        canonical = resize_center_crop_geometry(source_size, bucket)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Turbo scoring geometry for stem {stem!r} is malformed") from exc
    geometry = {
        "source_size": list(canonical.source_size),
        "resized_size": list(canonical.resized_size),
        "crop_box": list(canonical.crop_box),
    }
    persisted = {field: sample[field] for field in geometry}
    if persisted != geometry:
        raise ValueError(f"Turbo scoring geometry for stem {stem!r} disagrees with canonical paired preprocessing")
    return geometry


def assert_turbo_output_isolated(output_dir: str | Path) -> Path:
    """Reject the immutable canonical RAW evaluation namespace and descendants."""
    output = Path(output_dir).resolve()
    canonical = CANONICAL_EVALUATION_ROOT.resolve()
    if output == canonical or canonical in output.parents:
        raise ValueError(f"Turbo evaluation output must not collide with canonical evaluation path: {canonical}")
    return output


def exact_turbo_checkpoints(*, checkpoint_dir: str | Path, hf_repo_id: str,
                            hf_recovery_dir: str | Path | None = None,
                            steps: Iterable[int] = DEFAULT_TURBO_CHECKPOINT_STEPS) -> list[tuple[int, Path]]:
    """Resolve only complete, checksum-validated requested Turbo archive states."""
    requested = tuple(steps)
    if not requested or len(requested) != len(set(requested)) or any(step not in TURBO_CHECKPOINT_STEPS for step in requested):
        raise ValueError(f"Turbo checkpoint steps must be a unique non-empty subset of {TURBO_CHECKPOINT_STEPS}")
    resolved = ordered_checkpoints(
        checkpoint_dir, steps=requested,
        later_checkpoint_dir=checkpoint_dir, archive_checkpoint_dir=checkpoint_dir,
        hf_repo_id=hf_repo_id, hf_recovery_dir=hf_recovery_dir,
    )
    if tuple(step for step, _ in resolved) != requested or any(path is None for _, path in resolved):
        raise AssertionError(f"Turbo benchmark must resolve exactly requested checkpoints {requested}")
    return [(step, path) for step, path in resolved if path is not None]


def assert_lr5e5_turbo_output_isolated(output_dir: str | Path) -> Path:
    """Reject both historical Turbo namespaces for the LR-continuation branch."""
    output = assert_turbo_output_isolated(output_dir)
    original = ORIGINAL_TURBO_EVALUATION_ROOT.resolve()
    if output == original or original in output.parents:
        raise ValueError(f"LR=5e-5 Turbo output must not collide with original Turbo results: {original}")
    return output


def assert_timestep_turbo_output_isolated(output_dir: str | Path) -> Path:
    """Keep the timestep-exposure Turbo branch separate from both predecessors."""
    output = assert_lr5e5_turbo_output_isolated(output_dir)
    lr5e5 = LR5E5_TURBO_EVALUATION_ROOT.resolve()
    if output == lr5e5 or lr5e5 in output.parents:
        raise ValueError(f"Timestep Turbo output must not collide with LR=5e-5 Turbo results: {lr5e5}")
    return output


def assert_control_scale_turbo_output_isolated(output_dir: str | Path) -> Path:
    """Keep the inference-only control-scale sweep outside every prior tree."""
    output = assert_timestep_turbo_output_isolated(output_dir)
    timestep = TIMESTEP_TURBO_EVALUATION_ROOT.resolve()
    if output == timestep or timestep in output.parents:
        raise ValueError(f"Control-scale Turbo output must not collide with timestep Turbo results: {timestep}")
    return output


def scale_turbo_control_latent(control_latent: torch.Tensor, control_scale: float = 1.0) -> torch.Tensor:
    """Apply the inference-only control scale without touching other inputs.

    The identity case deliberately returns the exact existing tensor, preserving
    the established scale-1.0 sampling path byte-for-byte through patchification.
    """
    if not isinstance(control_scale, (float, int)) or isinstance(control_scale, bool):
        raise TypeError("control_scale must be a finite numeric value")
    control_scale = float(control_scale)
    if not math.isfinite(control_scale) or control_scale <= 0.0:
        raise ValueError("control_scale must be finite and positive")
    if control_scale == 1.0:
        return control_latent
    return control_latent * control_scale


def exact_lr5e5_turbo_checkpoints(*, checkpoint_dir: str | Path, hf_repo_id: str,
                                   hf_recovery_dir: str | Path | None = None,
                                   steps: Iterable[int] = LR5E5_TURBO_CHECKPOINT_STEPS) -> list[tuple[int, Path]]:
    """Resolve only marker-backed exact checkpoints from the LR=5e-5 HF branch.

    A local file alone is deliberately insufficient: every selected state is
    fetched through ``validated_hf_checkpoint_for_step`` from the sole branch
    namespace, which checks the exact completion marker, SHA-256, full state
    deserialization/schema, and embedded ``global_step``.  It therefore cannot
    substitute the original 1500 run, a timed mirror, nearest, or latest state.
    """
    requested = tuple(steps)
    if requested != LR5E5_TURBO_CHECKPOINT_STEPS:
        raise ValueError(f"LR=5e-5 Turbo evaluation requires exactly {LR5E5_TURBO_CHECKPOINT_STEPS}, got {requested}")
    if Path(checkpoint_dir).resolve() != LR5E5_CHECKPOINT_ROOT.resolve():
        raise ValueError(f"LR=5e-5 Turbo evaluation requires checkpoint root {LR5E5_CHECKPOINT_ROOT}")
    if hf_repo_id != LR5E5_HF_REPO_ID:
        raise ValueError(f"LR=5e-5 Turbo evaluation requires HF repo {LR5E5_HF_REPO_ID}")
    recovery_root = Path(hf_recovery_dir) if hf_recovery_dir is not None else LR5E5_CHECKPOINT_ROOT / "hf-recovery-turbo"
    resolved: list[tuple[int, Path]] = []
    for step in requested:
        checkpoint = validated_hf_checkpoint_for_step(
            repo_id=LR5E5_HF_REPO_ID, run_name=LR5E5_HF_RUN_NAME, step=step,
            download_dir=recovery_root / LR5E5_HF_RUN_NAME,
        )
        if checkpoint is None:
            remote = f"{LR5E5_HF_RUN_NAME}/full/step_{step:06d}.pt"
            raise FileNotFoundError(f"Required exact completion-marked LR=5e-5 checkpoint is unavailable: {remote}")
        state = load_training_state(checkpoint)
        if state["global_step"] != step:
            raise ValueError(f"LR=5e-5 checkpoint filename/embedded step mismatch: {checkpoint} has {state['global_step']}")
        resolved.append((step, checkpoint))
    return resolved


def exact_lr5e5_step1500_local_checkpoint(*, checkpoint_dir: str | Path, hf_repo_id: str,
                                            marker_download_dir: str | Path) -> Path:
    """Return only the completed local LR-only step-1500 source checkpoint.

    The local state remains the sole checkpoint payload.  Its matching HF
    completion marker is consulted only to verify the requested branch/step
    identity and checksum; there is deliberately no remote checkpoint fallback.
    """
    if Path(checkpoint_dir).resolve() != LR5E5_CHECKPOINT_ROOT.resolve():
        raise ValueError(f"LR-only step-1500 diagnostics require checkpoint root {LR5E5_CHECKPOINT_ROOT}")
    if hf_repo_id != LR5E5_HF_REPO_ID:
        raise ValueError(f"LR-only step-1500 diagnostics require HF repo {LR5E5_HF_REPO_ID}")
    checkpoint = validated_local_checkpoint_for_hf_step(
        checkpoint=LR5E5_CHECKPOINT_ROOT / "step_001500.pt",
        repo_id=LR5E5_HF_REPO_ID,
        run_name=LR5E5_HF_RUN_NAME,
        step=1500,
        marker_download_dir=Path(marker_download_dir),
    )
    if checkpoint is None:
        remote = f"{LR5E5_HF_RUN_NAME}/full/step_001500.pt"
        raise FileNotFoundError(f"Required local LR-only step-1500 checkpoint failed exact HF marker validation: {remote}")
    state = load_training_state(checkpoint)
    if state["global_step"] != 1500:
        raise ValueError(f"LR-only source checkpoint filename/embedded step mismatch: {checkpoint} has {state['global_step']}")
    return checkpoint


def exact_timestep_turbo_checkpoints(*, checkpoint_dir: str | Path, hf_repo_id: str,
                                     hf_recovery_dir: str | Path | None = None,
                                     steps: Iterable[int] = TIMESTEP_TURBO_CHECKPOINT_STEPS) -> list[tuple[int, Path]]:
    """Resolve only the three completion-marked timestep-exposure states.

    Steps 1600 and 1700 are obtained only through the existing exact-step HF
    validator.  Local step 1800 is accepted only after its exact HF completion
    marker validates its checksum, complete training-state schema, and
    embedded ``global_step``. It cannot select a nearest/latest file, a timed
    mirror, the original branch, or the LR-only continuation.
    """
    requested = tuple(steps)
    if requested != TIMESTEP_TURBO_CHECKPOINT_STEPS:
        raise ValueError(
            f"Timestep Turbo evaluation requires exactly {TIMESTEP_TURBO_CHECKPOINT_STEPS}, got {requested}"
        )
    if Path(checkpoint_dir).resolve() != TIMESTEP_CHECKPOINT_ROOT.resolve():
        raise ValueError(f"Timestep Turbo evaluation requires checkpoint root {TIMESTEP_CHECKPOINT_ROOT}")
    if hf_repo_id != TIMESTEP_HF_REPO_ID:
        raise ValueError(f"Timestep Turbo evaluation requires HF repo {TIMESTEP_HF_REPO_ID}")
    recovery_root = Path(hf_recovery_dir) if hf_recovery_dir is not None else TIMESTEP_CHECKPOINT_ROOT / "hf-recovery-turbo"
    resolved: list[tuple[int, Path]] = []
    for step in requested:
        if step == 1800:
            checkpoint = validated_local_checkpoint_for_hf_step(
                checkpoint=TIMESTEP_CHECKPOINT_ROOT / "step_001800.pt",
                repo_id=TIMESTEP_HF_REPO_ID, run_name=TIMESTEP_HF_RUN_NAME, step=step,
                marker_download_dir=recovery_root / TIMESTEP_HF_RUN_NAME,
            )
        else:
            checkpoint = validated_hf_checkpoint_for_step(
                repo_id=TIMESTEP_HF_REPO_ID, run_name=TIMESTEP_HF_RUN_NAME, step=step,
                download_dir=recovery_root / TIMESTEP_HF_RUN_NAME,
            )
        if checkpoint is None:
            remote = f"{TIMESTEP_HF_RUN_NAME}/full/step_{step:06d}.pt"
            raise FileNotFoundError(f"Required exact completion-marked timestep checkpoint is unavailable: {remote}")
        state = load_training_state(checkpoint)
        if state["global_step"] != step:
            raise ValueError(f"Timestep checkpoint filename/embedded step mismatch: {checkpoint} has {state['global_step']}")
        resolved.append((step, checkpoint))
    return resolved


def assert_turbo_diagnostic_contract(spec: Mapping[str, Any], original_spec: Mapping[str, Any], *, branch_name: str) -> None:
    """Require exactly the established Turbo diagnostic inputs and seeds."""
    required = ("stems", "per_stem_seeds", "sample_identities")
    if spec.get("kind") != "turbo_fixed_pose" or original_spec.get("kind") != "turbo_fixed_pose":
        raise ValueError(f"{branch_name} Turbo evaluation requires the established turbo_fixed_pose diagnostic spec")
    if spec.get("seed") != 420200 or original_spec.get("seed") != 420200:
        raise ValueError(f"{branch_name} Turbo evaluation requires the immutable diagnostic seed 420200")
    if any(spec.get(key) != original_spec.get(key) for key in required):
        raise ValueError(f"{branch_name} Turbo diagnostic stems, inputs, or per-stem seeds differ from original Turbo evaluation")
    if spec.get("turbo") != turbo_metadata() or original_spec.get("turbo") != turbo_metadata():
        raise ValueError(f"{branch_name} Turbo contract differs from the established 8-step CFG-0 mu=1.15 contract")


def assert_lr5e5_diagnostic_contract(spec: Mapping[str, Any], original_spec: Mapping[str, Any]) -> None:
    """Backward-compatible name for the LR-only continuation contract."""
    assert_turbo_diagnostic_contract(spec, original_spec, branch_name="LR=5e-5")


@torch.inference_mode()
def sample_turbo_pose_image(model: Any, vae_decode_fn, sample: dict[str, Any], device: torch.device,
                            seed: int, *, steps: int = TURBO_STEPS,
                            guidance: float = TURBO_CFG, mu: float = TURBO_MU,
                            control_scale: float = 1.0):
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
    control, _, _ = patchify_and_position(
        scale_turbo_control_latent(control_latent, control_scale), text.shape[1], patch, text_mask
    )
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
