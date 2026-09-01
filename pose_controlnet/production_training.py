"""Locked full-dataset launcher mechanics for Krea-2 Pose Control-LoRA.

This module is deliberately separate from the bounded Gate-F ``train.py``
entry point.  It owns only the full 768 cached-latent recipe and has no
generation or evaluation path.  W&B and Hugging Face are explicit optional,
failure-isolated observability services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from math import cos, pi
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

import train
from pose_controlnet.checkpointing import (
    HFTrainingCheckpointMirror,
    load_training_state,
    newest_valid_local_checkpoint,
    save_training_state,
)
from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.full_768_cache import FULL_TRAIN_COUNT, verify_full_768_cache
from pose_controlnet.keypoint_critic import FixedBoxKeypointRCNNCritic
from pose_controlnet.keypoint_critic_audit import assert_frozen_no_parameter_grad
from pose_controlnet.model import audit_control_model, build_pose_model, trainable_params, trainable_state_dict
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.seed import set_seed
from pose_controlnet.throughput_benchmark import (
    LOCKED_EFFECTIVE_BATCH, LOCKED_LAMBDA_POSE, LOCKED_POSE_LOSS, LOCKED_POSE_WINDOW,
    LOCKED_RESOLUTION_POLICY,
)
from pose_controlnet.vae_preprocessing import load_krea_vae
from pose_controlnet.wandb_logging import DEFAULT_WANDB_PROJECT, OptionalWandbMirror
from scripts.train_pose_reward_smoke import _pose_smoke_loss, aggregate_step_diagnostics, update_cumulative_counters


DEFAULT_DATASET_ROOT = "/lambda/nfs/adhit/krea2-pose/posebridge_hf"
DEFAULT_TRAIN_MANIFEST = "/home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl"
DEFAULT_LATENT_ROOT = "/lambda/nfs/adhit/krea2-pose/posebridge_latents_768"
DEFAULT_TEXT_CONDITIONING_ROOT = "/lambda/nfs/adhit/krea2-pose/text_conditioning"
DEFAULT_POSE_SIDECAR = "/lambda/nfs/adhit/krea2-pose/pose_targets_v3_768"
DEFAULT_RAW_CKPT = "/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors"
DEFAULT_CHECKPOINT_DIR = "/lambda/nfs/adhit/krea2-pose/checkpoints"

PRODUCTION_METADATA_KEY = "production_pose_control"
PRODUCTION_METADATA_FORMAT = 1
POSE_CUMULATIVE_COUNTER_KEYS = (
    "eligible_samples_seen", "forced_samples", "naturally_active_samples", "total_active_samples",
)


@dataclass(frozen=True)
class ProductionRecipe:
    """The benchmark-selected runtime and locked scientific recipe."""

    rank: int = 64
    alpha: int = 64
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    control_dropout: float = 0.0
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    gradient_checkpointing_blocks: int = 0
    compile: bool = False
    fused_adamw: bool = False
    data_loader_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    pose_loss: str = LOCKED_POSE_LOSS
    lambda_pose: float = LOCKED_LAMBDA_POSE
    pose_timestep_min: float = LOCKED_POSE_WINDOW[0]
    pose_timestep_max: float = LOCKED_POSE_WINDOW[1]
    forced_pose_exposure_probability: float = 0.0
    resolution_policy: str = LOCKED_RESOLUTION_POLICY
    seed: int = 42
    caption_dropout: float = 0.10
    mu_x1: float = 256.0
    mu_y1: float = 0.5
    mu_x2: float = 6400.0
    mu_y2: float = 1.15
    timestep_aux_prob: float = 0.0
    timestep_aux_min: float = 0.0
    timestep_aux_max: float = 1.0

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        expected = ProductionRecipe()
        locked = (
            "rank", "alpha", "learning_rate", "betas", "weight_decay", "warmup_steps", "max_grad_norm", "control_dropout",
            "microbatch_size", "gradient_accumulation_steps", "gradient_checkpointing_blocks", "compile",
            "fused_adamw", "data_loader_workers", "persistent_workers", "pin_memory", "prefetch_factor",
            "pose_loss", "lambda_pose", "pose_timestep_min", "pose_timestep_max",
            "forced_pose_exposure_probability", "resolution_policy", "seed", "caption_dropout", "mu_x1", "mu_y1",
            "mu_x2", "mu_y2", "timestep_aux_prob", "timestep_aux_min", "timestep_aux_max",
        )
        changed = {name: {"requested": getattr(self, name), "locked": getattr(expected, name)}
                   for name in locked if getattr(self, name) != getattr(expected, name)}
        if changed:
            raise ValueError(f"Production recipe is locked; refusing overrides: {changed}")
        if self.effective_batch_size != LOCKED_EFFECTIVE_BATCH:
            raise ValueError(f"effective batch must be {LOCKED_EFFECTIVE_BATCH}")

    def scientific_identity(self) -> dict[str, Any]:
        result = asdict(self)
        result["effective_batch_size"] = self.effective_batch_size
        return result


@dataclass(frozen=True)
class CooldownContinuation:
    """An explicit scientific continuation, distinct from an exact resume."""

    parent_checkpoint: Path
    parent_step: int
    schedule: str
    start_lr: float
    final_lr: float
    continuation_steps: int

    @property
    def end_step(self) -> int:
        return self.parent_step + self.continuation_steps

    def metadata(self, *, parent_run_name: str, parent_sha256: str) -> dict[str, Any]:
        return {
            "kind": "scientific_continuation",
            "exact_resume": False,
            "parent_checkpoint": str(self.parent_checkpoint.resolve()),
            "parent_checkpoint_sha256": parent_sha256,
            "parent_run_name": parent_run_name,
            "parent_global_step": self.parent_step,
            "scheduler": self.schedule,
            "start_lr": self.start_lr,
            "final_lr": self.final_lr,
            "continuation_steps": self.continuation_steps,
            "end_global_step": self.end_step,
        }


class CosineContinuationScheduler:
    """Cosine LR over global updates ``parent_step + 1`` through ``end_step``.

    The first continuation optimizer update uses ``start_lr``.  The final
    update uses ``final_lr`` exactly, so a 3000 -> 5000 continuation has 2000
    updates indexed 0..1999 and progress ``index / 1999``.
    """

    name = "CosineContinuation"

    def __init__(self, optimizer: torch.optim.Optimizer, *, parent_step: int,
                 continuation_steps: int, start_lr: float, final_lr: float) -> None:
        if parent_step < 0 or continuation_steps < 2:
            raise ValueError("Cosine continuation requires a non-negative parent step and at least two updates")
        if start_lr <= 0 or final_lr <= 0 or final_lr > start_lr:
            raise ValueError("Cosine continuation learning rates must satisfy 0 < final <= start")
        self.optimizer = optimizer
        self.parent_step = parent_step
        self.continuation_steps = continuation_steps
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.step_count = parent_step
        self._apply_for_global_update(parent_step + 1)

    @property
    def end_step(self) -> int:
        return self.parent_step + self.continuation_steps

    def _learning_rate(self, global_update: int) -> float:
        index = global_update - self.parent_step - 1
        if not 0 <= index < self.continuation_steps:
            raise ValueError(f"Global update {global_update} is outside the configured continuation interval")
        fraction = index / (self.continuation_steps - 1)
        cosine_scale = 0.5 * (1.0 + cos(pi * fraction))
        return self.final_lr + (self.start_lr - self.final_lr) * cosine_scale

    def _apply_for_global_update(self, global_update: int) -> None:
        learning_rate = self._learning_rate(global_update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    @property
    def current_update_learning_rates(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        if self.step_count < self.end_step:
            self._apply_for_global_update(self.step_count + 1)

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "step_count": self.step_count, "parent_step": self.parent_step,
                "continuation_steps": self.continuation_steps, "start_lr": self.start_lr, "final_lr": self.final_lr}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict() | {"step_count": int(state.get("step_count", -1))}
        if {key: state.get(key) for key in expected if key != "step_count"} != {
            key: expected[key] for key in expected if key != "step_count"
        }:
            raise ValueError("Checkpoint cosine continuation schedule differs from current configuration")
        step_count = int(state["step_count"])
        if not self.parent_step <= step_count <= self.end_step:
            raise ValueError("Checkpoint cosine continuation step is outside the configured interval")
        self.step_count = step_count
        if step_count < self.end_step:
            self._apply_for_global_update(step_count + 1)
        else:
            self._apply_for_global_update(self.end_step)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--latent-root", default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--text-conditioning-root", default=DEFAULT_TEXT_CONDITIONING_ROOT)
    parser.add_argument("--pose-sidecar", default=DEFAULT_POSE_SIDECAR)
    parser.add_argument("--raw-ckpt", default=DEFAULT_RAW_CKPT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--diagnostics-every", type=int, default=50)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto' (local run directory only)")
    parser.add_argument("--continue-from", type=Path, default=None,
                        help="explicit parent checkpoint for a new scientific cooldown continuation")
    parser.add_argument("--continue-from-step", type=int, default=3000,
                        help="required parent global step for --continue-from (default: 3000)")
    parser.add_argument("--lr-schedule", choices=("cosine",), default=None,
                        help="replacement scheduler for --continue-from")
    parser.add_argument("--lr-final", type=float, default=None,
                        help="final learning rate for the explicit continuation scheduler")
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", dest="wandb", action="store_true",
                             help="enable best-effort W&B metric mirroring")
    wandb_group.add_argument("--no-wandb", dest="wandb", action="store_false",
                             help="disable W&B entirely (the default)")
    parser.set_defaults(wandb=False)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None,
                        help="optional W&B entity to pass explicitly to wandb.init")
    parser.add_argument("--wandb-name", default=None,
                        help="optional W&B display name; defaults to --run-name")
    parser.add_argument("--hf-repo-id", default="",
                        help="optional private HF model repo for completed checkpoint milestones")
    parser.add_argument("--hf-mirror-every-steps", type=int, default=0,
                        help="mirror only completed local checkpoints at this cadence; 0 disables mirroring")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--gradient-checkpointing-blocks", type=int, default=0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-loader-workers", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    return parser


def recipe_from_args(args: argparse.Namespace) -> ProductionRecipe:
    recipe = ProductionRecipe(
        microbatch_size=args.microbatch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing_blocks=args.gradient_checkpointing_blocks, compile=args.compile,
        fused_adamw=args.fused_adamw, data_loader_workers=args.data_loader_workers,
        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    recipe.validate()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.save_every < 1 or args.diagnostics_every < 1:
        raise ValueError("--save-every and --diagnostics-every must be positive")
    if args.hf_mirror_every_steps < 0:
        raise ValueError("--hf-mirror-every-steps must be non-negative")
    if args.hf_mirror_every_steps:
        if not args.hf_repo_id.strip():
            raise ValueError("--hf-mirror-every-steps requires --hf-repo-id")
        if args.hf_mirror_every_steps % args.save_every:
            raise ValueError("--hf-mirror-every-steps must be divisible by --save-every")
    return recipe


def cooldown_continuation_from_args(args: argparse.Namespace, recipe: ProductionRecipe) -> CooldownContinuation | None:
    """Parse the intentionally narrow 3k -> 5k cooldown contract.

    No ordinary resume enters this path: a parent checkpoint and every
    scheduler override must be named explicitly.
    """
    supplied = (args.continue_from is not None, args.lr_schedule is not None, args.lr_final is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("--continue-from, --lr-schedule, and --lr-final must be supplied together")
    if not any(supplied):
        return None
    if args.continue_from_step != 3000:
        raise ValueError("Cooldown continuation requires --continue-from-step 3000")
    if args.lr_schedule != "cosine" or args.lr_final != 1e-5:
        raise ValueError("Cooldown continuation requires --lr-schedule cosine --lr-final 1e-5")
    continuation = CooldownContinuation(
        parent_checkpoint=args.continue_from, parent_step=args.continue_from_step,
        schedule=args.lr_schedule, start_lr=recipe.learning_rate, final_lr=args.lr_final,
        continuation_steps=args.max_steps - args.continue_from_step,
    )
    if args.max_steps != 5000 or continuation.continuation_steps != 2000:
        raise ValueError("Cooldown continuation requires --max-steps 5000 (exactly 2000 additional optimizer steps)")
    if args.save_every != 250:
        raise ValueError("Cooldown continuation requires --save-every 250")
    return continuation


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable production identity file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Malformed production identity file: {path}")
    return value


def artifact_identity(*, dataset_root: str, train_manifest: str, latent_root: str,
                      text_conditioning_root: str, pose_sidecar: str, raw_ckpt: str,
                      verified_cache: Mapping[str, Any]) -> dict[str, Any]:
    """Capture content identities after the full verifier has accepted artifacts."""
    cache_root = Path(latent_root).resolve()
    cache_identity = _load_json(cache_root / "train_manifest_identity.json")
    for key in ("manifest_records_sha256", "ordered_stems_sha256", "ordered_stems"):
        if key not in cache_identity:
            raise ValueError(f"Verified cache identity lacks {key}")
    sidecar_metadata, _ = load_sidecar(pose_sidecar)
    if sidecar_metadata.get("records_sha256") != verified_cache.get("pose_records_sha256"):
        raise ValueError("Verified pose sidecar identity changed during launcher setup")
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_manifest": {"path": str(Path(train_manifest).resolve()), "raw_sha256": _sha256(train_manifest),
                           "records_sha256": cache_identity["manifest_records_sha256"],
                           "ordered_stems_sha256": cache_identity["ordered_stems_sha256"]},
        "latent_cache": {"path": str(cache_root), "cache_contract_sha256": verified_cache["cache_contract_sha256"],
                         "metadata_sha256": _sha256(cache_root / "shards.json")},
        "pose_sidecar": {"path": str(Path(pose_sidecar).resolve()), "records_sha256": verified_cache["pose_records_sha256"],
                         "metadata_sha256": _sha256(Path(pose_sidecar) / "metadata.json")},
        "text_conditioning": {"path": str(Path(text_conditioning_root).resolve()),
                               "metadata_sha256": _sha256(Path(text_conditioning_root) / "text_conditioning.json"),
                               "unconditional_sha256": _sha256(Path(text_conditioning_root) / "unconditional.pt")},
        "raw_checkpoint": {"path": str(Path(raw_ckpt).resolve()), "sha256": _sha256(raw_ckpt)},
    }


def verify_production_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Run the authoritative no-network full-cache/sidecar verifier before CUDA work."""
    verified = verify_full_768_cache(dataset_root=args.dataset_root, cache_root=args.latent_root,
                                     pose_sidecar=args.pose_sidecar, train_manifest=args.train_manifest)
    if verified.get("cache_samples") != FULL_TRAIN_COUNT or verified.get("resolution_policy") != "posebridge_full_train_768_v1":
        raise ValueError("Full cache verifier did not prove the locked 16,503-sample 768 artifact")
    return artifact_identity(dataset_root=args.dataset_root, train_manifest=args.train_manifest,
                             latent_root=args.latent_root, text_conditioning_root=args.text_conditioning_root,
                             pose_sidecar=args.pose_sidecar, raw_ckpt=args.raw_ckpt, verified_cache=verified)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(*, args: argparse.Namespace, recipe: ProductionRecipe, identities: Mapping[str, Any],
                 current_step: int, wandb_run_id: str | None = None,
                 continuation_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scheduler = ({"name": "OptimizerStepWarmup", "warmup_steps": 200} if continuation_metadata is None else {
        "name": CosineContinuationScheduler.name,
        "parent_step": continuation_metadata["parent_global_step"],
        "continuation_steps": continuation_metadata["continuation_steps"],
        "start_lr": continuation_metadata["start_lr"], "final_lr": continuation_metadata["final_lr"],
    })
    metadata = {
        "format": PRODUCTION_METADATA_FORMAT, "run_name": args.run_name, "current_step": current_step,
        "max_steps": args.max_steps, "scientific_recipe": recipe.scientific_identity(),
        "artifact_identity": dict(identities), "optimizer": {"name": "AdamW", "lr": 1e-4,
        "betas": [0.9, 0.99], "weight_decay": 0.0}, "scheduler": scheduler,
        "loader": {"workers": 4, "persistent_workers": True, "pin_memory": True, "prefetch_factor": 4},
        "observability": {
            "wandb": {"enabled": bool(args.wandb), "project": args.wandb_project,
                      "entity": args.wandb_entity, "name": args.wandb_name or args.run_name,
                      "run_id": wandb_run_id},
            "hf": {"repo_id": args.hf_repo_id, "mirror_every_steps": args.hf_mirror_every_steps},
        },
        "code_git_revision": _git_revision(),
    }
    if continuation_metadata is not None:
        metadata["continuation"] = dict(continuation_metadata)
    return metadata


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True); raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def empty_pose_cumulative_counters() -> dict[str, int]:
    return {key: 0 for key in POSE_CUMULATIVE_COUNTER_KEYS}


def validate_pose_cumulative_counters(counters: Mapping[str, Any]) -> dict[str, int]:
    if set(counters) != set(POSE_CUMULATIVE_COUNTER_KEYS):
        raise ValueError("Production pose cumulative counters are malformed")
    result = {key: counters[key] for key in POSE_CUMULATIVE_COUNTER_KEYS}
    if any(not isinstance(value, int) or value < 0 for value in result.values()):
        raise ValueError("Production pose cumulative counters are invalid")
    if result["total_active_samples"] != result["naturally_active_samples"] + result["forced_samples"]:
        raise ValueError("Production pose cumulative counters double-count activity")
    return result


def _legacy_counters_from_metrics(metrics_path: Path, global_step: int) -> dict[str, int] | None:
    """Recover old local-only smoke checkpoints that predate counter persistence."""
    if not metrics_path.is_file():
        return None
    try:
        for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines()):
            metric = json.loads(line)
            if metric.get("global_step") == global_step and isinstance(metric.get("pose_cumulative_counters"), dict):
                return validate_pose_cumulative_counters(metric["pose_cumulative_counters"])
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def pose_cumulative_counters_from_checkpoint(state: Mapping[str, Any], *, metrics_path: Path | None = None) -> dict[str, int]:
    """Restore persisted counters, with a precise legacy JSONL recovery path."""
    metadata = state.get(PRODUCTION_METADATA_KEY)
    counters = metadata.get("pose_cumulative_counters") if isinstance(metadata, Mapping) else None
    if isinstance(counters, Mapping):
        return validate_pose_cumulative_counters(counters)
    legacy = _legacy_counters_from_metrics(metrics_path, int(state["global_step"])) if metrics_path else None
    return legacy if legacy is not None else empty_pose_cumulative_counters()


def production_hf_milestone_steps(*, max_steps: int, mirror_every_steps: int, start_step: int = 0) -> tuple[int, ...]:
    if mirror_every_steps <= 0:
        return ()
    return tuple(step for step in range(mirror_every_steps, max_steps + 1, mirror_every_steps) if step > start_step)


def production_wandb_mirror(*, args: argparse.Namespace, recipe: ProductionRecipe,
                            identities: Mapping[str, Any], resume_state: Mapping[str, Any] | None = None,
                            wandb_module: Any | None = None,
                            continuation_metadata: Mapping[str, Any] | None = None) -> OptionalWandbMirror:
    """Create the remote-only mirror; disabled production runs import nothing network-related."""
    metadata = resume_state.get(PRODUCTION_METADATA_KEY) if isinstance(resume_state, Mapping) else None
    observed = metadata.get("observability") if isinstance(metadata, Mapping) else None
    wandb_state = observed.get("wandb") if isinstance(observed, Mapping) else None
    is_matching_continuation = (continuation_metadata is not None
                                and isinstance(metadata, Mapping)
                                and metadata.get("continuation") == dict(continuation_metadata))
    # A parent checkpoint must never reuse its W&B id.  A later exact resume
    # of this newly-created continuation may reuse only its own saved id.
    resume_run_id = (wandb_state.get("run_id") if isinstance(wandb_state, Mapping)
                     and (continuation_metadata is None or is_matching_continuation) else None)
    if not isinstance(resume_run_id, str) or not resume_run_id:
        resume_run_id = None
    return OptionalWandbMirror(
        project=args.wandb_project if args.wandb else None,
        entity=args.wandb_entity if args.wandb else None,
        run_name=args.wandb_name or args.run_name,
        config=run_metadata(args=args, recipe=recipe, identities=identities, current_step=0,
                            continuation_metadata=continuation_metadata),
        resume_run_id=resume_run_id,
        wandb_module=wandb_module,
    )


def _production_checkpoint_metadata(*, args: argparse.Namespace, recipe: ProductionRecipe,
                                    identities: Mapping[str, Any], current_step: int,
                                    cumulative_counters: Mapping[str, Any] | None = None,
                                    wandb_run_id: str | None = None,
                                    continuation_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    counters = validate_pose_cumulative_counters(cumulative_counters or empty_pose_cumulative_counters())
    return run_metadata(args=args, recipe=recipe, identities=identities, current_step=current_step,
                        wandb_run_id=wandb_run_id, continuation_metadata=continuation_metadata) | {
        "gradient_accumulation_position": 0,
        "pose_cumulative_counters": counters,
    }


def validate_resume_identity(state: Mapping[str, Any], *, args: argparse.Namespace,
                             recipe: ProductionRecipe, identities: Mapping[str, Any],
                             continuation_metadata: Mapping[str, Any] | None = None) -> None:
    metadata = state.get(PRODUCTION_METADATA_KEY)
    if not isinstance(metadata, dict) or metadata.get("format") != PRODUCTION_METADATA_FORMAT:
        raise ValueError("Resume checkpoint lacks production scientific identity metadata")
    expected = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities,
                                               current_step=int(state["global_step"]),
                                               continuation_metadata=continuation_metadata)
    for key in ("run_name", "scientific_recipe", "artifact_identity", "optimizer", "scheduler", "loader",
                "gradient_accumulation_position"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Unsafe production resume refused: {key} identity differs")
    if continuation_metadata is not None and metadata.get("continuation") != dict(continuation_metadata):
        raise ValueError("Unsafe production resume refused: continuation provenance differs")
    if state.get("global_step") != metadata.get("current_step"):
        raise ValueError("Unsafe production resume refused: checkpoint step and metadata disagree")
    expected_position = {"epoch": state.get("epoch"), "batch_position": state.get("batch_position"),
                         "sample_position": int(state.get("batch_position", -1)) * recipe.microbatch_size}
    if metadata.get("data_position") != expected_position:
        raise ValueError("Unsafe production resume refused: epoch/batch/sample position differs")
    if int(state["global_step"]) > args.max_steps:
        raise ValueError("Unsafe production resume refused: checkpoint exceeds requested --max-steps")


def build_train_config(args: argparse.Namespace, recipe: ProductionRecipe) -> TrainConfig:
    return TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root, ckpt_dir=args.checkpoint_dir,
                       run_name=args.run_name, rank=recipe.rank, alpha=recipe.alpha, lr=recipe.learning_rate,
                       microbatch_size=recipe.microbatch_size,
                       gradient_accumulation_steps=recipe.gradient_accumulation_steps,
                       max_steps=args.max_steps, allow_extended_training=True, warmup_steps=recipe.warmup_steps,
                       max_grad_norm=recipe.max_grad_norm, caption_dropout=recipe.caption_dropout,
                       compile=recipe.compile, fused_adamw=recipe.fused_adamw,
                       gradient_checkpointing=False, gradient_checkpointing_blocks=0,
                       save_every=args.save_every, diagnostics_every=args.diagnostics_every,
                       val_every=10 ** 9, validation_batches=0, wandb_enabled=args.wandb,
                       wandb_entity=args.wandb_entity, wandb_project=args.wandb_project,
                       metrics_jsonl_path=str(Path(args.checkpoint_dir) / args.run_name / "metrics.jsonl"),
                       hf_repo_id=args.hf_repo_id, hf_mirror_every_steps=args.hf_mirror_every_steps)


def _collate_with_stems(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate(items); batch["stems"] = [item["stem"] for item in items]
    return batch


def validate_full_768_dataset(data: PreparedLatentShardDataset) -> None:
    if len(data) != FULL_TRAIN_COUNT:
        raise ValueError(f"Expected {FULL_TRAIN_COUNT} full-train cached records, got {len(data)}")
    if data.text_conditioning is None:
        raise ValueError("Production launcher requires persistent cached text conditioning")
    allowed = {(96, 96), (112, 88), (88, 112), (120, 80), (80, 120), (128, 72), (72, 128), (144, 64), (64, 144)}
    unexpected = sorted({shape for _, _, shape, _ in data.records} - allowed)
    if unexpected:
        raise ValueError(f"Cached latent root is not the locked 768 bucket policy: {unexpected[:8]}")


def load_and_validate_pose_records(sidecar: str, data: PreparedLatentShardDataset) -> dict[str, dict[str, Any]]:
    _, records = load_sidecar(sidecar); by_stem = {str(record["stem"]): record for record in records}
    if {stem for _, _, _, stem in data.records} != set(by_stem):
        raise ValueError("pose sidecar membership must exactly equal the full training split")
    for _, _, latent_shape, stem in data.records:
        bucket = by_stem[stem].get("bucket")
        if not isinstance(bucket, list) or tuple(reversed(bucket)) != tuple(value * 8 for value in latent_shape):
            raise ValueError(f"{stem}: immutable pose-sidecar geometry differs from cached latent")
    return by_stem


def planned_microbatches(records: list[tuple[str, int, tuple[int, int], str]], *, recipe: ProductionRecipe,
                         epoch: int, batch_position: int, count: int) -> list[list[int]]:
    """Reconstruct the benchmarked epoch/bucket ordering from a resume position."""
    plan = train.DeterministicBucketBatches(records, recipe.microbatch_size, recipe.seed)
    result: list[list[int]] = []
    while len(result) < count:
        batches = plan.for_epoch(epoch)
        if batch_position >= len(batches):
            epoch, batch_position = epoch + 1, 0
            continue
        take = min(count - len(result), len(batches) - batch_position)
        result.extend(batches[batch_position:batch_position + take])
        batch_position += take
    return result


def advance_position(plan: train.DeterministicBucketBatches, epoch: int, batch_position: int) -> tuple[int, int]:
    batch_position += 1
    if batch_position >= len(plan.for_epoch(epoch)):
        return epoch + 1, 0
    return epoch, batch_position


def production_loader(data: PreparedLatentShardDataset, groups: list[list[int]], recipe: ProductionRecipe) -> DataLoader:
    return DataLoader(data, batch_sampler=groups, collate_fn=_collate_with_stems,
                      num_workers=recipe.data_loader_workers, persistent_workers=recipe.persistent_workers,
                      pin_memory=recipe.pin_memory, prefetch_factor=recipe.prefetch_factor)


def checkpoint_state(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler: Any, global_step: int, epoch: int, batch_position: int,
                     generator: torch.Generator, cfg: TrainConfig, args: argparse.Namespace,
                     recipe: ProductionRecipe, identities: Mapping[str, Any],
                     cumulative_counters: Mapping[str, Any] | None = None,
                     wandb_run_id: str | None = None,
                     continuation_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities,
                                               current_step=global_step, cumulative_counters=cumulative_counters,
                                               wandb_run_id=wandb_run_id, continuation_metadata=continuation_metadata)
    metadata["data_position"] = {"epoch": epoch, "batch_position": batch_position,
                                 "sample_position": batch_position * recipe.microbatch_size}
    return {"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "global_step": global_step, "epoch": epoch, "batch_position": batch_position,
            "rng": train._capture_rng(), "flow_generator_state": generator.get_state(), "config": asdict(cfg),
            PRODUCTION_METADATA_KEY: metadata}


def resolve_resume(path: str | None, run_dir: Path) -> Path | None:
    if path is None:
        return None
    if path == "auto":
        found = newest_valid_local_checkpoint(run_dir)
        if found is None:
            raise FileNotFoundError("--resume auto found no valid local production checkpoint")
        return found
    return Path(path)


def validate_cooldown_parent_identity(state: Mapping[str, Any], *, continuation: CooldownContinuation,
                                      recipe: ProductionRecipe, identities: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the locked 3k parent recipe before changing its scheduler."""
    metadata = state.get(PRODUCTION_METADATA_KEY)
    if not isinstance(metadata, dict) or metadata.get("format") != PRODUCTION_METADATA_FORMAT:
        raise ValueError("Cooldown parent lacks production scientific identity metadata")
    if int(state.get("global_step", -1)) != continuation.parent_step or metadata.get("current_step") != continuation.parent_step:
        raise ValueError(f"Cooldown parent must be exactly global step {continuation.parent_step}")
    expected = _production_checkpoint_metadata(args=argparse.Namespace(
        run_name=metadata.get("run_name"), max_steps=metadata.get("max_steps"), wandb=False,
        wandb_project=None, wandb_entity=None, wandb_name=None, hf_repo_id="", hf_mirror_every_steps=0,
    ), recipe=recipe, identities=identities, current_step=continuation.parent_step)
    for key in ("scientific_recipe", "artifact_identity", "optimizer", "scheduler", "loader",
                "gradient_accumulation_position"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Cooldown parent refused: immutable science identity {key} differs")
    if metadata.get("continuation") is not None:
        raise ValueError("Cooldown parent must be an original production checkpoint, not another continuation")
    expected_position = {"epoch": state.get("epoch"), "batch_position": state.get("batch_position"),
                         "sample_position": int(state.get("batch_position", -1)) * recipe.microbatch_size}
    if metadata.get("data_position") != expected_position:
        raise ValueError("Cooldown parent refused: deterministic data position differs")
    for key in ("model", "optimizer", "rng", "flow_generator_state"):
        if key not in state:
            raise ValueError(f"Cooldown parent lacks required resumable state: {key}")
    return metadata


def restore_cooldown_continuation_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                        state: Mapping[str, Any]) -> tuple[int, int, int, object | None]:
    """Restore weights, Adam moments, RNG and data position while replacing only LR policy."""
    train.load_trainable_state_dict(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    groups = optimizer.param_groups
    if len(groups) != 1 or tuple(groups[0].get("betas", ())) != (0.9, 0.99) or groups[0].get("weight_decay") != 0.0:
        raise ValueError("Cooldown parent AdamW hyperparameters differ from the locked recipe")
    if not optimizer.state:
        raise ValueError("Cooldown parent AdamW moments are absent; refusing to reset optimizer state")
    train._restore_rng(state["rng"])
    return int(state["global_step"]), int(state["epoch"]), int(state["batch_position"]), state.get("flow_generator_state")


def run(args: argparse.Namespace, *, verifier: Callable[[argparse.Namespace], dict[str, Any]] = verify_production_artifacts) -> None:
    recipe = recipe_from_args(args)
    continuation = cooldown_continuation_from_args(args, recipe)
    # Intentionally first: no model, VAE, optimizer, or CUDA allocation happens
    # until the full immutable cache and pose-sidecar contract passes.
    identities = verifier(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Run production training only from the GH200 host shell with CUDA visible")
    cfg = build_train_config(args, recipe); run_dir = Path(args.checkpoint_dir) / args.run_name
    resume_path = resolve_resume(args.resume, run_dir)
    if resume_path is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing fresh production run in nonempty directory: {run_dir}")
    parent_state: dict[str, Any] | None = None
    continuation_metadata: dict[str, Any] | None = None
    if continuation is not None:
        parent_path = continuation.parent_checkpoint.resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(f"Cooldown parent checkpoint is missing: {parent_path}")
        parent_state = load_training_state(parent_path)
        parent_metadata = validate_cooldown_parent_identity(parent_state, continuation=continuation,
                                                            recipe=recipe, identities=identities)
        continuation_metadata = continuation.metadata(parent_run_name=str(parent_metadata["run_name"]),
                                                      parent_sha256=_sha256(parent_path))
    set_seed(recipe.seed); device = torch.device("cuda")
    data = PreparedLatentShardDataset(args.latent_root, "train", text_conditioning_root=args.text_conditioning_root)
    validate_full_768_dataset(data); pose_records = load_and_validate_pose_records(args.pose_sidecar, data)
    model = build_pose_model(cfg.raw_ckpt, recipe.rank, recipe.alpha, "cuda")
    audit_control_model(model, rank=recipe.rank); train.configure_runtime(model, compile_enabled=False); model.train()
    optimizer = train.build_optimizer(model, cfg)
    scheduler: Any = (CosineContinuationScheduler(
        optimizer, parent_step=continuation.parent_step, continuation_steps=continuation.continuation_steps,
        start_lr=continuation.start_lr, final_lr=continuation.final_lr,
    ) if continuation is not None else train.OptimizerStepWarmup(optimizer, recipe.warmup_steps))
    vae = load_krea_vae(device); critic = FixedBoxKeypointRCNNCritic().to(device).eval(); assert_frozen_no_parameter_grad(vae, critic)
    global_step = epoch = batch_position = 0; generator = torch.Generator(device=device).manual_seed(recipe.seed)
    resume_state: dict[str, Any] | None = None
    if resume_path is not None:
        resume_state = load_training_state(resume_path)
        validate_resume_identity(resume_state, args=args, recipe=recipe, identities=identities,
                                 continuation_metadata=continuation_metadata)
        global_step, epoch, batch_position, generator_state = train.restore_full_training_state(model, optimizer, scheduler, resume_state)
        if generator_state is None: raise ValueError("Production checkpoint lacks flow/timestep generator state")
        generator.set_state(generator_state)
    elif continuation is not None:
        assert parent_state is not None
        resume_state = parent_state
        global_step, epoch, batch_position, generator_state = restore_cooldown_continuation_state(model, optimizer, parent_state)
        if generator_state is None: raise ValueError("Production checkpoint lacks flow/timestep generator state")
        generator.set_state(generator_state)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    counters = (pose_cumulative_counters_from_checkpoint(resume_state, metrics_path=metrics_path)
                if resume_state is not None else empty_pose_cumulative_counters())
    wandb = production_wandb_mirror(args=args, recipe=recipe, identities=identities, resume_state=resume_state,
                                    continuation_metadata=continuation_metadata)
    def report_mirror_result(success: bool, step: int | None, error: str | None, reason: str | None) -> None:
        if success:
            print(f"HF checkpoint mirrored: step={step} reason={reason}", flush=True)
        else:
            print(f"[hf] warning: mirror failed for step={step} reason={reason}; "
                  f"local checkpoint remains authoritative: {error}", flush=True)
    mirror = HFTrainingCheckpointMirror(
        repo_id=args.hf_repo_id, run_name=args.run_name, interval_seconds=float("inf"),
        protected_milestone_steps=production_hf_milestone_steps(
            max_steps=args.max_steps, mirror_every_steps=args.hf_mirror_every_steps,
            start_step=continuation.parent_step if continuation is not None else 0,
        ),
        on_result=report_mirror_result,
    )
    mirror.start()
    _atomic_json(run_dir / "run_metadata.json", run_metadata(
        args=args, recipe=recipe, identities=identities, current_step=global_step, wandb_run_id=wandb.run_id,
        continuation_metadata=continuation_metadata,
    ))
    plan = train.DeterministicBucketBatches(data.records, recipe.microbatch_size, recipe.seed)
    remaining_groups = (cfg.max_steps - global_step) * recipe.gradient_accumulation_steps
    iterator = (iter(production_loader(data, planned_microbatches(data.records, recipe=recipe, epoch=epoch,
                                         batch_position=batch_position, count=remaining_groups), recipe))
                if remaining_groups else iter(()))
    optimizer.zero_grad(set_to_none=True); stopped = False
    def stop_handler(signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True; print(f"received signal {signum}; checkpointing at optimizer boundary", flush=True)
    signal.signal(signal.SIGINT, stop_handler); signal.signal(signal.SIGTERM, stop_handler)
    try:
        while global_step < cfg.max_steps and not stopped:
            start_epoch, start_position = epoch, batch_position
            diagnostics: list[Mapping[str, Any]] = []; started = time.monotonic(); interrupted = False
            for accumulation_index in range(recipe.gradient_accumulation_steps):
                if stopped:
                    interrupted = True; break
                batch = next(iterator); epoch, batch_position = advance_position(plan, epoch, batch_position)
                train.apply_cached_caption_dropout(batch, data.text_conditioning.unconditional, cfg.caption_dropout, cfg.seed,
                                                   global_step * recipe.gradient_accumulation_steps + accumulation_index)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, diagnostic = _pose_smoke_loss(model, vae, critic, batch, pose_records, cfg, device, generator,
                        pose_loss_name=recipe.pose_loss, lambda_pose=recipe.lambda_pose,
                        timestep_min=recipe.pose_timestep_min, timestep_max=recipe.pose_timestep_max,
                        forced_exposure_probability=recipe.forced_pose_exposure_probability)
                diagnostics.append(diagnostic); (loss / recipe.gradient_accumulation_steps).backward()
            if interrupted:
                optimizer.zero_grad(set_to_none=True); epoch, batch_position = start_epoch, start_position
                break
            diagnostics_due = (global_step + 1) % cfg.diagnostics_every == 0
            control_norms = lora_norms = None
            def capture_diagnostics() -> None:
                nonlocal control_norms, lora_norms
                control_norms, lora_norms = train._diagnostic_grad_norms(model)
            learning_rate = scheduler.current_update_learning_rates[0]
            grad_norm = train.optimizer_update(optimizer, scheduler, trainable_params(model), cfg.max_grad_norm,
                                               before_step=capture_diagnostics if diagnostics_due else None); global_step += 1
            assert_frozen_no_parameter_grad(vae, critic)
            metrics = {"global_step": global_step, "learning_rate": learning_rate,
                       "global_grad_norm": grad_norm, "sec_per_step": time.monotonic() - started,
                       **aggregate_step_diagnostics(diagnostics)}
            counters = update_cumulative_counters(counters, metrics); metrics["pose_cumulative_counters"] = counters
            if diagnostics_due:
                metrics["control_input_grad_norms"] = control_norms
                metrics["lora_grad_norms"] = lora_norms
            with metrics_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            wandb.log(metrics, step=global_step)
            if global_step % cfg.save_every == 0 or global_step == cfg.max_steps or stopped:
                path = save_training_state(run_dir / f"step_{global_step:06d}.pt", checkpoint_state(
                    model=model, optimizer=optimizer, scheduler=scheduler, global_step=global_step, epoch=epoch,
                    batch_position=batch_position, generator=generator, cfg=cfg, args=args, recipe=recipe,
                    identities=identities, cumulative_counters=counters, wandb_run_id=wandb.run_id,
                    continuation_metadata=continuation_metadata), overwrite=False)
                _atomic_json(run_dir / "run_metadata.json", run_metadata(
                    args=args, recipe=recipe, identities=identities, current_step=global_step, wandb_run_id=wandb.run_id,
                    continuation_metadata=continuation_metadata,
                ))
                print(f"checkpoint saved: {path}", flush=True)
                if train.step_mirror_requested(global_step, args.hf_mirror_every_steps):
                    mirror.submit(path, reason="step")
    finally:
        if stopped:
            path = run_dir / f"step_{global_step:06d}.pt"
            if not path.exists():
                save_training_state(path, checkpoint_state(model=model, optimizer=optimizer, scheduler=scheduler,
                    global_step=global_step, epoch=epoch, batch_position=batch_position, generator=generator, cfg=cfg,
                    args=args, recipe=recipe, identities=identities, cumulative_counters=counters,
                    wandb_run_id=wandb.run_id, continuation_metadata=continuation_metadata), overwrite=False)
        mirror.stop(drain=True, timeout=None)
        wandb.close()
        assert_frozen_no_parameter_grad(vae, critic)
