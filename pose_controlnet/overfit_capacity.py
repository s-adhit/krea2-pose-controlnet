"""Fail-closed contracts for 32-sample pose-control capacity experiments.

This module is intentionally small and model-agnostic.  The trainer imports
the existing fresh Raw model, flow-MSE, optimizer, and checkpoint primitives;
these helpers only restrict their inputs to an immutable selected subset.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from torch.utils.data import Dataset

from pose_controlnet.pose_targets import source_for_stem
from pose_controlnet.resolution_policy import (
    NATIVE_RESOLUTION_POLICY,
    RESOLUTION_768_BUCKETS,
    RESOLUTION_768_POLICY,
    buckets_for_resolution,
    canonical_resolution_policy,
)


OVERFIT_SAMPLE_COUNT = 32
OVERFIT_SEED = 42
OVERFIT_STEPS = (0, 50, 100, 200, 300, 400, 500)
# Step zero is the fresh-model evaluation reference.  The remaining values are
# the only training checkpoints that the capacity experiment may create.
OVERFIT_CHECKPOINT_STEPS = OVERFIT_STEPS[1:]
OVERFIT_MAX_STEPS = 500
OVERFIT_MICROBATCH = 1
OVERFIT_ACCUMULATION = 8
OVERFIT_LR = 1e-4
OVERFIT_WARMUP = 0
OVERFIT_MANIFEST_ROOT = Path("configs/overfit_capacity/manifests")
# The definitive 768 Mixed-32 branch is deliberately constrained to the
# audited coordinate loss.  Historical KL experiments keep their own scripts;
# they cannot be recreated through this capacity runner.
POSE_LOSS_CHOICES = ("none", "normalized_coordinate_huber")
OVERFIT_EXPERIMENTS = {
    "overfit32-coco-r64-mse": "coco",
    "overfit32-humanart-painting-r64-mse": "humanart_painting",
    "overfit32-humanart-real-r64-mse": "humanart_real_human",
    "overfit32-humanart-sculpture-r64-mse": "humanart_sculpture",
    "overfit32-danbooru-r64-mse": "danbooru",
    "overfit32-mixed-r64-mse": "mixed",
}
MIXED_COMPOSITION = {
    "coco": 6,
    "humanart_painting": 7,
    "humanart_real_human": 7,
    "humanart_sculpture": 6,
    "danbooru": 6,
}


@dataclass(frozen=True)
class OverfitExperiment:
    name: str
    source: str
    manifest: Path


@dataclass(frozen=True)
class CapacityScientificConfig:
    """The only mutable axes of an otherwise fixed capacity experiment."""

    base_experiment: str
    resolution: str = NATIVE_RESOLUTION_POLICY
    pose_loss: str = "none"
    lambda_pose: float = 0.0
    forced_pose_exposure_probability: float = 0.0
    pose_timestep_min: float | None = None
    pose_timestep_max: float | None = None

    @property
    def experiment_name(self) -> str:
        return capacity_experiment_name(
            self.base_experiment, self.resolution, self.pose_loss, self.lambda_pose,
        )


def capacity_experiment_name(base_experiment: str, resolution: str, pose_loss: str, lambda_pose: float) -> str:
    """Construct a collision-resistant, scientifically legible run namespace."""
    base = {"mixed32": "overfit32-mixed-r64", "overfit32-mixed-r64-mse": "overfit32-mixed-r64"}.get(base_experiment)
    if base is None:
        raise ValueError("Only the immutable mixed32 capacity manifest is supported by the dynamic runner")
    policy = canonical_resolution_policy(resolution)
    if pose_loss == "none":
        return f"{base}-mse-res{policy}"
    token = {"normalized_coordinate_huber": "coord"}.get(pose_loss)
    if token is None:
        raise ValueError(f"Unsupported pose loss {pose_loss!r}")
    # ``repr(float)`` round-trips rather than rounding the selected calibrated
    # value (``2.5e-5`` must not silently become ``2e-5`` in provenance).
    # Keep the exponent sign: ``1e-5`` is materially different from ``1e5``.
    lambda_token = repr(float(lambda_pose)).replace("e-0", "e-")
    return f"{base}-{token}-l{lambda_token}-res{policy}"


def validate_capacity_scientific_config(config: CapacityScientificConfig) -> CapacityScientificConfig:
    resolution = canonical_resolution_policy(config.resolution)
    if config.base_experiment not in ("mixed32", "overfit32-mixed-r64-mse"):
        raise ValueError("Dynamic capacity experiments must reuse the immutable Mixed-32 manifest")
    if config.pose_loss not in POSE_LOSS_CHOICES:
        raise ValueError(f"Unsupported --pose-loss {config.pose_loss!r}")
    if not math.isfinite(config.lambda_pose) or config.lambda_pose < 0:
        raise ValueError("--lambda-pose must be finite and non-negative")
    if config.pose_loss == "none":
        if config.lambda_pose != 0:
            raise ValueError("--lambda-pose must be 0 when --pose-loss=none")
        if config.forced_pose_exposure_probability != 0 or config.pose_timestep_min is not None or config.pose_timestep_max is not None:
            raise ValueError("pose exposure/window must be absent when --pose-loss=none")
    else:
        if config.lambda_pose <= 0:
            raise ValueError("--lambda-pose must be positive when a pose loss is selected")
        if not 0 <= config.forced_pose_exposure_probability <= 1:
            raise ValueError("--forced-pose-exposure-probability must be in [0, 1]")
        lo, hi = config.pose_timestep_min, config.pose_timestep_max
        if lo is None or hi is None or not 0 < lo <= hi < 1:
            raise ValueError("pose loss requires a timestep window satisfying 0 < min <= max < 1")
    return CapacityScientificConfig(
        base_experiment=config.base_experiment, resolution=resolution, pose_loss=config.pose_loss,
        lambda_pose=float(config.lambda_pose),
        forced_pose_exposure_probability=float(config.forced_pose_exposure_probability),
        pose_timestep_min=config.pose_timestep_min, pose_timestep_max=config.pose_timestep_max,
    )


def experiment(name: str, *, manifest_root: Path = OVERFIT_MANIFEST_ROOT) -> OverfitExperiment:
    try:
        source = OVERFIT_EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown capacity experiment {name!r}; choose one of {sorted(OVERFIT_EXPERIMENTS)}") from exc
    return OverfitExperiment(name=name, source=source, manifest=manifest_root / f"{name}.jsonl")


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Required immutable capacity manifest is missing: {path}") from None
    if any(not isinstance(row, dict) or not isinstance(row.get("file_name"), str) or not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise ValueError(f"Capacity manifest has malformed or blank-caption records: {path}")
    return rows


def manifest_stems(path: Path) -> tuple[str, ...]:
    stems = tuple(Path(row["file_name"]).stem for row in _manifest_rows(path))
    if len(stems) != OVERFIT_SAMPLE_COUNT or len(stems) != len(set(stems)):
        raise ValueError(f"Capacity manifest must contain exactly {OVERFIT_SAMPLE_COUNT} unique samples: {path}")
    return stems


def validate_manifest(name: str, *, manifest_root: Path = OVERFIT_MANIFEST_ROOT,
                      train_manifest: Path = Path("data/manifests/train.jsonl")) -> tuple[str, ...]:
    spec = experiment(name, manifest_root=manifest_root)
    stems = manifest_stems(spec.manifest)
    train_stems = {Path(json.loads(line)["file_name"]).stem for line in train_manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    outside = sorted(set(stems) - train_stems)
    if outside:
        raise ValueError(f"Capacity manifest contains samples outside immutable train membership: {outside[:3]}")
    observed = {source_for_stem(stem) for stem in stems}
    if spec.source == "mixed":
        composition = {source: sum(source_for_stem(stem) == source for stem in stems) for source in MIXED_COMPOSITION}
        if composition != MIXED_COMPOSITION:
            raise ValueError(f"Mixed capacity manifest composition must be {MIXED_COMPOSITION}, got {composition}")
    elif observed != {spec.source}:
        raise ValueError(f"{name} has wrong domain membership: expected {spec.source}, got {sorted(observed)}")
    return stems


def validate_all_manifests(*, manifest_root: Path = OVERFIT_MANIFEST_ROOT,
                           train_manifest: Path = Path("data/manifests/train.jsonl")) -> dict[str, tuple[str, ...]]:
    values = {name: validate_manifest(name, manifest_root=manifest_root, train_manifest=train_manifest) for name in OVERFIT_EXPERIMENTS}
    mixed = set(values["overfit32-mixed-r64-mse"])
    for name, source in OVERFIT_EXPERIMENTS.items():
        if source != "mixed":
            expected = sum(source_for_stem(stem) == source for stem in mixed)
            if expected and not {stem for stem in mixed if source_for_stem(stem) == source} <= set(values[name]):
                raise ValueError(f"Mixed manifest samples for {source} must be reused from {name}")
    return values


class SelectedLatentShardDataset(Dataset):
    """A read-only exact subset of a prepared train dataset, indexed by stem."""
    def __init__(self, base: Dataset, stems: Iterable[str]) -> None:
        self.base = base
        self.stems = tuple(stems)
        if len(self.stems) != OVERFIT_SAMPLE_COUNT or len(set(self.stems)) != OVERFIT_SAMPLE_COUNT:
            raise ValueError("Selected capacity dataset must have exactly 32 unique manifest stems")
        records = getattr(base, "records", None)
        if not isinstance(records, list):
            raise TypeError("Selected capacity dataset requires PreparedLatentShardDataset records")
        by_stem = {record[3]: index for index, record in enumerate(records)}
        if len(by_stem) != len(records):
            raise ValueError("Prepared dataset contains duplicate stems")
        missing = sorted(set(self.stems) - set(by_stem))
        if missing:
            raise ValueError(f"Selected capacity samples do not all exist in prepared train shards: {missing[:3]}")
        self.indices = tuple(by_stem[stem] for stem in self.stems)
        self.records = [records[index] for index in self.indices]
        if {record[3] for record in self.records} != set(self.stems):
            raise AssertionError("Selected capacity dataset escaped its manifest")
        self.text_conditioning = getattr(base, "text_conditioning", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.indices):
            raise IndexError(index)
        item = self.base[self.indices[index]]
        if item.get("stem") != self.stems[index]:
            raise AssertionError("Prepared dataset returned a sample outside the selected manifest")
        return item


class SelectedDeterministicBatches:
    """Cycles every selected sample only; deterministic shuffling is epoch seeded."""
    def __init__(self, dataset: SelectedLatentShardDataset, microbatch_size: int, seed: int = OVERFIT_SEED) -> None:
        if microbatch_size != OVERFIT_MICROBATCH:
            raise ValueError("Capacity harness is intentionally fixed to microbatch size 1")
        self.dataset, self.microbatch_size, self.seed = dataset, microbatch_size, seed

    def for_epoch(self, epoch: int) -> list[list[int]]:
        indices = list(range(len(self.dataset)))
        random.Random(self.seed + epoch).shuffle(indices)
        batches = [[index] for index in indices]
        observed = {self.dataset[index]["stem"] for batch in batches for index in batch}
        if observed != set(self.dataset.stems) or len(batches) != OVERFIT_SAMPLE_COUNT:
            raise AssertionError("Capacity batch plan is not exactly the selected 32-sample set")
        return batches


def assert_fresh_initialization(*, resume: str | None = None, trained_state: object | None = None) -> None:
    if resume is not None or trained_state is not None:
        raise ValueError("Capacity experiments forbid resume or any trained Pose LoRA state")


def assert_overfit_contract(*, rank: int, warmup_steps: int, max_steps: int, lr: float,
                            microbatch_size: int, accumulation_steps: int, pose_reward_enabled: bool = False,
                            critic_enabled: bool = False, objective: str = "flow_mse",
                            checkpoint_steps: tuple[int, ...] = OVERFIT_CHECKPOINT_STEPS,
                            scientific_config: CapacityScientificConfig | None = None) -> None:
    if rank != 64 or warmup_steps != OVERFIT_WARMUP or max_steps != OVERFIT_MAX_STEPS:
        raise ValueError("Capacity contract requires fresh rank-64, zero warmup, and max_steps=500")
    if lr != OVERFIT_LR or microbatch_size != OVERFIT_MICROBATCH or accumulation_steps != OVERFIT_ACCUMULATION:
        raise ValueError("Capacity contract has a non-authoritative LR or effective-batch setting")
    if scientific_config is None:
        if pose_reward_enabled or critic_enabled or objective != "flow_mse":
            raise ValueError("Capacity contract permits flow-matching MSE only unless an explicit scientific config is supplied")
    else:
        scientific_config = validate_capacity_scientific_config(scientific_config)
        pose_enabled = scientific_config.pose_loss != "none"
        if pose_reward_enabled != pose_enabled or critic_enabled != pose_enabled:
            raise ValueError("Capacity pose-loss flags disagree with the persisted scientific configuration")
        expected_objective = "flow_mse_plus_pose" if pose_enabled else "flow_mse"
        if objective != expected_objective:
            raise ValueError("Capacity objective disagrees with the persisted scientific configuration")
    if tuple(checkpoint_steps) != OVERFIT_CHECKPOINT_STEPS:
        raise ValueError(f"Capacity checkpoint schedule must be exactly {OVERFIT_CHECKPOINT_STEPS}")


def parameter_audit() -> dict[str, Any]:
    """Exact static audit from the authoritative Krea-2 MMDiT dimensions."""
    per_target = {
        "attn.wq": 786_432, "attn.wk": 491_520, "attn.wv": 491_520,
        "attn.wo": 786_432, "attn.gate": 786_432,
        "mlp.gate": 1_441_792, "mlp.up": 1_441_792, "mlp.down": 1_441_792,
    }
    lora = sum(per_target.values()) * 28
    control = 6_144 * (16 * 2**2 * 2 + 1)
    trainable = lora + control
    base = 12_820_073_036
    original_first = 399_360
    total = base - original_first + trainable
    return {"base_parameter_count": base, "total_model_parameter_count": total,
            "trainable_parameter_count": trainable, "trainable_percentage": trainable / total * 100,
            "control_input_parameter_count": control, "lora_parameter_count": lora,
            "lora_target_modules": 224, "lora_parameters_per_target_per_block": per_target,
            "trainable_module_contract": ["first.weight", "first.bias", "28 x 8 LoRA targets x {A,B}"],
            "rank": 64, "alpha": 64}


def per_step_exposures(step: int) -> dict[str, float | int]:
    if step not in OVERFIT_STEPS:
        raise ValueError(f"Unexpected capacity checkpoint step {step}")
    presentations = step * OVERFIT_MICROBATCH * OVERFIT_ACCUMULATION
    return {"optimizer_step": step, "cumulative_sample_presentations": presentations,
            "dataset_equivalent_passes": presentations / OVERFIT_SAMPLE_COUNT}


def is_overfit_checkpoint_step(step: int) -> bool:
    """Return whether an optimizer boundary is an authoritative save milestone."""
    return step in OVERFIT_CHECKPOINT_STEPS


def should_continue_overfit(step: int) -> bool:
    """Capacity runs have one fixed terminal optimizer boundary: step 500."""
    return step < OVERFIT_MAX_STEPS


def deterministic_seed(stem: str) -> int:
    return int.from_bytes(hashlib.sha256(f"overfit32-v1:{OVERFIT_SEED}:{stem}".encode()).digest()[:8], "big") % (2**63 - 1)
