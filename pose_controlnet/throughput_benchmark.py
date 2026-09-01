"""Small, dependency-free contracts shared by GH200 throughput utilities.

Nothing here builds a model, touches a checkpoint, or mutates data.  Keeping
the benchmark recipe and runtime estimates separate makes their invariants
cheap to exercise on CPU/no-network CI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


LOCKED_TRAINING_SAMPLES = 16_503
LOCKED_EFFECTIVE_BATCH = 32
LOCKED_RESOLUTION_POLICY = "768"
LOCKED_POSE_LOSS = "normalized_coordinate_huber"
LOCKED_LAMBDA_POSE = 0.04
LOCKED_POSE_WINDOW = (0.10, 0.20)
RUNTIME_STEPS = (1000, 1500, 2000, 2500, 3000, 5000)


@dataclass(frozen=True)
class ThroughputBenchmarkRecipe:
    """One axis-isolated production benchmark invocation."""

    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    gradient_checkpointing_blocks: int = 0
    fused_adamw: bool = False
    compile: bool = False
    data_loader_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    prefetch_factor: int | None = None
    warmup_steps: int = 10
    timed_steps: int = 20
    resolution_policy: str = LOCKED_RESOLUTION_POLICY
    objective: str = "candidate"

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps

    def validate(self, *, expected_effective_batch: int = LOCKED_EFFECTIVE_BATCH) -> None:
        if self.microbatch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("microbatch_size and gradient_accumulation_steps must be positive")
        if self.effective_batch_size != expected_effective_batch:
            raise ValueError(
                f"effective batch must remain {expected_effective_batch}, got {self.effective_batch_size}"
            )
        if not 0 <= self.gradient_checkpointing_blocks <= 28:
            raise ValueError("gradient_checkpointing_blocks must be in [0, 28]")
        if self.data_loader_workers < 0:
            raise ValueError("data_loader_workers must be non-negative")
        if self.persistent_workers and self.data_loader_workers == 0:
            raise ValueError("persistent_workers requires data_loader_workers > 0")
        if self.prefetch_factor is not None and (self.data_loader_workers == 0 or self.prefetch_factor < 1):
            raise ValueError("prefetch_factor requires data_loader_workers > 0 and must be positive")
        if self.warmup_steps < 1 or self.timed_steps < 1:
            raise ValueError("warmup_steps and timed_steps must be positive")
        if self.resolution_policy != LOCKED_RESOLUTION_POLICY:
            raise ValueError(f"benchmark resolution policy must be {LOCKED_RESOLUTION_POLICY}")
        if self.objective not in ("candidate", "flow_only"):
            raise ValueError("objective must be 'candidate' or 'flow_only'")

    def asdict(self) -> dict[str, object]:
        return asdict(self) | {"effective_batch_size": self.effective_batch_size}


def projected_runtime(*, seconds_per_optimizer_step: float, effective_batch_size: int,
                      training_samples: int = LOCKED_TRAINING_SAMPLES,
                      steps: Iterable[int] = RUNTIME_STEPS) -> list[dict[str, float | int]]:
    """Return wall-clock and dataset-equivalent-pass estimates from one measurement."""
    if seconds_per_optimizer_step <= 0:
        raise ValueError("seconds_per_optimizer_step must be positive")
    if effective_batch_size < 1 or training_samples < 1:
        raise ValueError("effective_batch_size and training_samples must be positive")
    rows = []
    for count in steps:
        if not isinstance(count, int) or count < 1:
            raise ValueError("steps must contain positive integers")
        samples = count * effective_batch_size
        seconds = count * seconds_per_optimizer_step
        rows.append({
            "optimizer_steps": count,
            "sample_presentations": samples,
            "dataset_equivalent_passes": samples / training_samples,
            "wall_seconds": seconds,
            "wall_hours": seconds / 3600,
        })
    return rows


def required_benchmark_fields() -> frozenset[str]:
    """Fields that make a saved benchmark result comparable and auditable."""
    return frozenset({
        "recipe", "trainable_parameter_names", "trainable_parameter_count",
        "forward_seconds_mean", "backward_seconds_mean", "optimizer_seconds_mean",
        "optimizer_step_seconds_mean", "samples_per_second", "effective_samples_per_second",
        "data_wait_seconds_mean", "cuda_allocated_bytes", "cuda_peak_allocated_bytes",
        "pose_active_fraction", "pose_active_microbatch_fraction", "runtime_projection",
    })


def validate_benchmark_result(result: dict[str, object]) -> None:
    missing = required_benchmark_fields() - set(result)
    if missing:
        raise ValueError(f"benchmark result is missing fields: {sorted(missing)}")
    recipe = result["recipe"]
    if not isinstance(recipe, dict):
        raise ValueError("benchmark recipe must be a dictionary")
    ThroughputBenchmarkRecipe(**{key: recipe[key] for key in ThroughputBenchmarkRecipe.__dataclass_fields__}).validate()
