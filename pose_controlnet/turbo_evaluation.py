"""Evaluation-only Krea-2 Turbo sampling contract for Pose-ControlNet.

This deliberately has no training-loop, optimizer, or gradient APIs.  It is a
small adaptation of the official Krea sampler: the image tokens remain noisy,
the clean pose tokens are concatenated at every denoising forward, and Turbo's
constant shift is passed explicitly rather than inferred from resolution.
"""
from __future__ import annotations

import math
import hashlib
import json
import re
from dataclasses import dataclass
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
from pose_controlnet.evaluation_geometry import persisted_scoring_geometry


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


@dataclass(frozen=True)
class TurboExperiment:
    """Machine-readable, experiment-owned inputs for the generic evaluator."""

    experiment_name: str
    checkpoint_root: Path
    hf_repo_id: str
    hf_namespace: str
    output_root: Path
    steps: tuple[int, ...] | None
    checkpoint_validation: Mapping[str, Any]
    baseline: Mapping[str, Any] | None
    labels: Mapping[str, Any]
    training_metadata: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    paths: Mapping[str, Any]

    @property
    def hf_run_name(self) -> str:
        return self.hf_namespace.removesuffix("/").removesuffix("/full")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def controlled_branch_metadata(checkpoints: Iterable[tuple[int, Path]]) -> dict[str, Any]:
    """Read provenance from controlled checkpoints instead of a copied spec.

    Static experiment-defining values must agree across every requested
    checkpoint.  Counters and checkpoint checksums are deliberately retained
    per checkpoint because they are expected to progress during training.
    """
    static_keys = (
        "pose_loss", "temperature", "lambda_pose", "pose_timestep_window",
        "forced_exposure_probability", "forced_sampler_policy", "immutable_parent",
        "hf_subdir",
    )
    expected_static: dict[str, Any] | None = None
    expected_config: dict[str, Any] | None = None
    per_checkpoint: dict[str, dict[str, Any]] = {}
    for step, checkpoint in checkpoints:
        state = load_training_state(checkpoint)
        if state.get("global_step") != step:
            raise ValueError(f"Checkpoint filename/embedded step mismatch: {checkpoint} has {state.get('global_step')}")
        gate = state.get("gate_e")
        config = state.get("config")
        if not isinstance(gate, Mapping) or not isinstance(config, Mapping):
            raise ValueError(f"Controlled checkpoint lacks gate_e/config metadata: {checkpoint}")
        static = {key: gate.get(key) for key in static_keys}
        if any(value is None for value in static.values()):
            raise ValueError(f"Controlled checkpoint has incomplete gate_e metadata: {checkpoint}")
        if expected_static is None:
            expected_static = static
        elif static != expected_static:
            raise ValueError(f"Controlled branch metadata is inconsistent at step {step}")
        config_static = {
            "microbatch_size": config.get("microbatch_size"),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
            "save_every_steps": config.get("save_every"),
            "target_global_step": config.get("max_steps"),
        }
        if expected_config is None:
            expected_config = config_static
        elif config_static != expected_config:
            raise ValueError(f"Controlled branch runtime metadata is inconsistent at step {step}")
        counters = gate.get("cumulative_counters")
        if not isinstance(counters, Mapping):
            raise ValueError(f"Controlled checkpoint lacks cumulative exposure counters: {checkpoint}")
        per_checkpoint[str(step)] = {
            "checkpoint_sha256": _sha256(checkpoint),
            "cumulative_pose_counters": {str(key): int(value) for key, value in counters.items()},
            **config_static,
        }
    if expected_static is None:
        raise ValueError("At least one controlled checkpoint is required for metadata extraction")
    first = next(iter(per_checkpoint.values()))
    return {
        "pose_loss": expected_static["pose_loss"],
        "pose_loss_temperature": expected_static["temperature"],
        "lambda_pose": expected_static["lambda_pose"],
        "pose_timestep_window": expected_static["pose_timestep_window"],
        "forced_pose_exposure_probability": expected_static["forced_exposure_probability"],
        "forced_sampler_policy": expected_static["forced_sampler_policy"],
        "parent_checkpoint": expected_static["immutable_parent"],
        "hf_subdir": expected_static["hf_subdir"],
        "microbatch_size": first["microbatch_size"],
        "gradient_accumulation_steps": first["gradient_accumulation_steps"],
        "per_checkpoint": per_checkpoint,
    }


def validate_controlled_experiment_metadata(checkpoint_root: str | Path,
                                            metadata: Mapping[str, Any]) -> None:
    """Fail closed if the optional branch-side metadata disagrees with checkpoints."""
    path = Path(checkpoint_root) / "experiment_metadata.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Controlled experiment metadata is invalid JSON: {path}") from exc
    gate = payload.get("gate_e") if isinstance(payload, Mapping) else None
    if not isinstance(gate, Mapping):
        raise ValueError(f"Controlled experiment metadata lacks gate_e: {path}")
    comparisons = {
        "pose_loss": metadata["pose_loss"], "lambda_pose": metadata["lambda_pose"],
        "pose_timestep_window": metadata["pose_timestep_window"],
        "forced_exposure_probability": metadata["forced_pose_exposure_probability"],
        "forced_sampler_policy": metadata["forced_sampler_policy"],
        "immutable_parent": metadata["parent_checkpoint"], "hf_subdir": metadata["hf_subdir"],
    }
    if any(gate.get(key) != value for key, value in comparisons.items()):
        raise ValueError(f"Controlled experiment metadata disagrees with requested checkpoints: {path}")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Turbo experiment spec requires non-empty string {key!r}")
    return value


def load_turbo_experiment_spec(path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> TurboExperiment:
    """Load an evaluator spec; only locations/labels/metadata are configurable.

    Metric and sampler semantics are intentionally validated separately against
    the centrally-owned Turbo contract below.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Turbo experiment spec is missing: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Turbo experiment spec is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Turbo experiment spec must be a JSON object")
    return turbo_experiment_from_payload(payload, overrides=overrides)


def turbo_experiment_from_payload(payload: Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None) -> TurboExperiment:
    """Validate a resolved CLI payload with the same contract as JSON specs."""
    merged = dict(payload)
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value
    experiment_name = _required_string(merged, "experiment_name")
    namespace = _required_string(merged, "hf_namespace").rstrip("/") + "/"
    if not re.fullmatch(r"[^/]+/full/", namespace):
        raise ValueError("hf_namespace must be exactly '<experiment-name>/full/'")
    if namespace != f"{experiment_name}/full/":
        raise ValueError("hf_namespace must match experiment_name exactly")
    diagnostics = merged.get("diagnostics", {})
    paths = merged.get("paths", {})
    if not isinstance(diagnostics, dict) or not isinstance(paths, dict):
        raise ValueError("Turbo experiment diagnostics and paths must be JSON objects")
    expected_contract = {**turbo_metadata(), "control_scale": 1.0}
    if merged.get("turbo_contract") != expected_contract:
        raise ValueError("Turbo experiment spec must use the established 8-step CFG-0 mu=1.15 control-scale-1.0 contract")
    diagnostic_count = diagnostics.get("expected_count")
    if not isinstance(diagnostic_count, int) or diagnostic_count < 1:
        raise ValueError("Turbo experiment diagnostics.expected_count must be a positive integer")
    _required_string(diagnostics, "canonical_manifest")
    checkpoint_root = Path(_required_string(merged, "checkpoint_root"))
    output_root = Path(_required_string(merged, "output_root"))
    if output_root.resolve() == checkpoint_root.resolve() or checkpoint_root.resolve() in output_root.resolve().parents:
        raise ValueError("Turbo evaluation output_root must not be inside checkpoint_root")
    raw_steps = merged.get("steps")
    if raw_steps is not None and (not isinstance(raw_steps, list) or any(isinstance(step, bool) or not isinstance(step, int) for step in raw_steps)):
        raise ValueError("Turbo experiment spec steps must be a list of integer exact checkpoints")
    spec_steps = normalize_turbo_steps(raw_steps) if raw_steps is not None else None
    validation = merged.get("checkpoint_validation", {"mode": "hf_completion_marker"})
    if not isinstance(validation, dict):
        raise ValueError("Turbo experiment checkpoint_validation must be an object")
    mode = validation.get("mode", "hf_completion_marker")
    if mode not in {"hf_completion_marker", "direct_local"}:
        raise ValueError("Turbo experiment checkpoint_validation.mode must be 'hf_completion_marker' or 'direct_local'")
    expected_sha256 = validation.get("expected_sha256", {})
    if not isinstance(expected_sha256, dict) or any(
        not isinstance(step, str) or not step.isdigit() or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for step, digest in expected_sha256.items()
    ):
        raise ValueError("Turbo experiment checkpoint_validation.expected_sha256 must map decimal steps to lowercase SHA-256 digests")
    if spec_steps is not None and any(int(step) not in spec_steps for step in expected_sha256):
        raise ValueError("Turbo experiment checkpoint_validation.expected_sha256 contains a step outside the configured steps")
    if "baseline" in merged and merged["baseline"] is not None and not isinstance(merged["baseline"], dict):
        raise ValueError("Turbo experiment baseline must be an object when configured")
    hf_repo_id = merged.get("hf_repo_id", "")
    if not isinstance(hf_repo_id, str):
        raise ValueError("Turbo experiment hf_repo_id must be a string")
    if mode != "direct_local" and not hf_repo_id.strip():
        raise ValueError("Turbo experiment requires hf_repo_id for marker-backed validation")
    return TurboExperiment(
        experiment_name=experiment_name,
        checkpoint_root=checkpoint_root,
        hf_repo_id=hf_repo_id.strip(),
        hf_namespace=namespace,
        output_root=output_root,
        steps=spec_steps,
        checkpoint_validation=validation,
        baseline=merged.get("baseline") if isinstance(merged.get("baseline"), dict) else None,
        labels=merged.get("labels", {}) if isinstance(merged.get("labels", {}), dict) else {},
        training_metadata=merged.get("training_metadata", {}) if isinstance(merged.get("training_metadata", {}), dict) else {},
        diagnostics=diagnostics,
        paths=paths,
    )


def normalize_turbo_steps(steps: Iterable[int]) -> tuple[int, ...]:
    """Validate an explicit exact-step request without an experiment allowlist."""
    result = tuple(steps)
    if not result or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in result):
        raise ValueError("Turbo checkpoint steps must be non-empty non-negative integers")
    if len(result) != len(set(result)):
        raise ValueError("Turbo checkpoint steps must be unique; duplicate requested steps are rejected")
    return result


def discover_turbo_checkpoint_steps(checkpoint_root: str | Path) -> tuple[int, ...]:
    """Discover only direct exact-step checkpoint files under the configured root."""
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Configured checkpoint root is missing: {root}")
    found: list[int] = []
    for candidate in root.iterdir():
        match = re.fullmatch(r"step_(\d{6})\.pt", candidate.name)
        if match and candidate.is_file():
            step = int(match.group(1))
            state = load_training_state(candidate)
            if state["global_step"] != step:
                raise ValueError(f"Checkpoint filename/embedded step mismatch: {candidate} has {state['global_step']}")
            found.append(step)
    if not found:
        raise FileNotFoundError(f"No valid step_XXXXXX.pt files found in configured checkpoint root: {root}")
    return tuple(sorted(found))


def exact_local_turbo_checkpoints(*, checkpoint_root: str | Path, hf_repo_id: str, hf_namespace: str,
                                  marker_download_dir: str | Path, steps: Iterable[int]) -> list[tuple[int, Path]]:
    """Strictly validate exact local checkpoints against their matching HF markers.

    This never downloads a checkpoint payload and has no nearest/latest or
    alternate-namespace fallback.  The checkpoint must exist at the exact
    direct-child filename before marker/SHA/schema/global-step validation.
    """
    root = Path(checkpoint_root)
    requested = normalize_turbo_steps(steps)
    if not root.is_dir():
        raise FileNotFoundError(f"Configured checkpoint root is missing: {root}")
    run_name = hf_namespace.rstrip("/").removesuffix("/full")
    if not run_name or hf_namespace.rstrip("/") != f"{run_name}/full":
        raise ValueError("hf_namespace must be exactly '<experiment-name>/full/'")
    resolved: list[tuple[int, Path]] = []
    for step in requested:
        checkpoint = root / f"step_{step:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required exact local checkpoint is missing: {checkpoint}")
        validated = validated_local_checkpoint_for_hf_step(
            checkpoint=checkpoint, repo_id=hf_repo_id, run_name=run_name, step=step,
            marker_download_dir=Path(marker_download_dir) / run_name,
        )
        if validated is None:
            remote = f"{hf_namespace}step_{step:06d}.pt.complete.json"
            raise FileNotFoundError(f"Required exact HF completion marker/SHA/schema validation failed: {remote}")
        state = load_training_state(validated)
        if state["global_step"] != step:
            raise ValueError(f"Checkpoint filename/embedded step mismatch: {validated} has {state['global_step']}")
        resolved.append((step, validated))
    return resolved


def exact_direct_local_turbo_checkpoints(*, checkpoint_root: str | Path,
                                         steps: Iterable[int],
                                         expected_sha256: Mapping[str, str] | None = None) -> list[tuple[int, Path]]:
    """Resolve exact local checkpoint files without an HF-marker fallback.

    This is for bounded local smoke branches whose checkpoints were never
    mirrored.  It still rejects absent files, filename/embedded-step mismatch,
    and verifies every SHA-256 explicitly supplied by the experiment spec.
    """
    root = Path(checkpoint_root)
    requested = normalize_turbo_steps(steps)
    if not root.is_dir():
        raise FileNotFoundError(f"Configured checkpoint root is missing: {root}")
    expected = dict(expected_sha256 or {})
    resolved: list[tuple[int, Path]] = []
    for step in requested:
        checkpoint = root / f"step_{step:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required exact local checkpoint is missing: {checkpoint}")
        digest = expected.get(str(step))
        if digest is not None:
            hasher = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            actual = hasher.hexdigest()
            if actual != digest:
                raise ValueError(f"Required exact local checkpoint SHA-256 mismatch: {checkpoint}")
        state = load_training_state(checkpoint)
        if state["global_step"] != step:
            raise ValueError(f"Checkpoint filename/embedded step mismatch: {checkpoint} has {state['global_step']}")
        resolved.append((step, checkpoint))
    return resolved


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
    return persisted_scoring_geometry(sample, label="Turbo")


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


def assert_exact_diagnostic_stems(manifest_stems: Iterable[str], dataset_stems: Iterable[str], *,
                                  expected_count: int = 24) -> tuple[str, ...]:
    """Require all and only the configured immutable diagnostic manifest order."""
    stems = tuple(manifest_stems)
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("Turbo diagnostic expected_count must be a positive integer")
    if len(stems) != expected_count or len(stems) != len(set(stems)):
        raise ValueError(f"Turbo benchmark requires exactly the complete {expected_count}-sample diagnostic manifest")
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
