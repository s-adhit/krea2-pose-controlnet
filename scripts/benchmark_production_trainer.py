"""Bounded GH200 benchmark of the real cached-latent training path.

This command never saves checkpoints, runs generation/evaluation, or updates
the source data.  It measures only warmup and timed optimizer steps against
the full prepared train split.  Run one axis per invocation and save each JSON
result; use ``summarize_production_benchmark.py`` only after all rows exist.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

import train
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.keypoint_critic import FixedBoxKeypointRCNNCritic
from pose_controlnet.keypoint_critic_audit import assert_frozen_no_parameter_grad
from pose_controlnet.full_768_cache import verify_full_768_cache
from pose_controlnet.model import audit_control_model, build_pose_model, trainable_params
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.seed import set_seed
from pose_controlnet.throughput_benchmark import (
    LOCKED_EFFECTIVE_BATCH, LOCKED_LAMBDA_POSE, LOCKED_POSE_LOSS,
    LOCKED_POSE_WINDOW, LOCKED_TRAINING_SAMPLES, ThroughputBenchmarkRecipe,
    projected_runtime, validate_benchmark_result,
)
from pose_controlnet.vae_preprocessing import load_krea_vae
from scripts.train_pose_reward_smoke import _pose_smoke_loss


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--raw-ckpt", required=True)
    value.add_argument("--latent-root", required=True, help="Verified full 768 latent shard root (read-only).")
    value.add_argument("--dataset-root", required=True, help="Read-only PoseBridge snapshot used to validate immutable train identity.")
    value.add_argument("--train-manifest", default=None,
                       help="Authoritative project full-train manifest; defaults to checked-in data/manifests/train.jsonl.")
    value.add_argument("--text-conditioning-root", required=True, help="Verified cached text conditioning root (read-only).")
    value.add_argument("--pose-sidecar", required=True, help="Immutable full-train 768 pose-target sidecar directory.")
    value.add_argument("--output-json", required=True, type=Path, help="New JSON result path; never a checkpoint path.")
    value.add_argument("--microbatch-size", type=int, default=1)
    value.add_argument("--gradient-accumulation-steps", type=int, default=32)
    value.add_argument("--gradient-checkpointing-blocks", type=int, default=0)
    value.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False)
    value.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    value.add_argument("--data-loader-workers", type=int, default=0)
    value.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    value.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    value.add_argument("--prefetch-factor", type=int, default=None)
    value.add_argument("--warmup-steps", type=int, default=10)
    value.add_argument("--timed-steps", type=int, default=20)
    value.add_argument("--objective", choices=("candidate", "flow_only"), default="candidate",
                       help="flow_only isolates the exact incremental coordinate-pose cost.")
    value.add_argument("--label", required=True, help="Axis-isolated result label, e.g. baseline-gc28.")
    return value


def recipe_from_args(args: argparse.Namespace) -> ThroughputBenchmarkRecipe:
    recipe = ThroughputBenchmarkRecipe(
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing_blocks=args.gradient_checkpointing_blocks,
        fused_adamw=args.fused_adamw, compile=args.compile,
        data_loader_workers=args.data_loader_workers,
        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor, warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps, objective=args.objective,
    )
    recipe.validate()
    return recipe


def _allowed_768_latent_shapes() -> frozenset[tuple[int, int]]:
    # The locked policy's (width, height) buckets divided by VAE factor 8.
    return frozenset({(96, 96), (112, 88), (88, 112), (120, 80), (80, 120),
                      (128, 72), (72, 128), (144, 64), (64, 144)})


def validate_full_768_dataset(data: PreparedLatentShardDataset) -> None:
    if len(data) != LOCKED_TRAINING_SAMPLES:
        raise ValueError(f"Expected full train split of {LOCKED_TRAINING_SAMPLES}, got {len(data)}")
    unexpected = sorted({shape for _, _, shape, _ in data.records} - _allowed_768_latent_shapes())
    if unexpected:
        raise ValueError(f"latent root is not the locked 768 bucket policy; unexpected latent HxW: {unexpected[:8]}")
    if data.text_conditioning is None:
        raise ValueError("benchmark requires the production cached text-conditioning path")


def _collate_with_stems(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate(items)
    batch["stems"] = [item["stem"] for item in items]
    return batch


def planned_microbatches(data: PreparedLatentShardDataset, recipe: ThroughputBenchmarkRecipe) -> list[list[int]]:
    plan = train.DeterministicBucketBatches(data.records, recipe.microbatch_size, seed=42)
    needed = (recipe.warmup_steps + recipe.timed_steps) * recipe.gradient_accumulation_steps
    groups: list[list[int]] = []
    epoch = 0
    while len(groups) < needed:
        groups.extend(plan.for_epoch(epoch)); epoch += 1
    return groups[:needed]


def batch_iterator(data: PreparedLatentShardDataset, groups: list[list[int]], recipe: ThroughputBenchmarkRecipe) -> Iterator[dict[str, Any]]:
    if recipe.data_loader_workers == 0:
        return iter(_collate_with_stems([data[index] for index in group]) for group in groups)
    kwargs: dict[str, Any] = {
        "batch_sampler": groups, "collate_fn": _collate_with_stems,
        "num_workers": recipe.data_loader_workers, "pin_memory": recipe.pin_memory,
        "persistent_workers": recipe.persistent_workers,
    }
    if recipe.prefetch_factor is not None:
        kwargs["prefetch_factor"] = recipe.prefetch_factor
    return iter(DataLoader(data, **kwargs))


def load_and_validate_pose_records(sidecar: str, data: PreparedLatentShardDataset) -> dict[str, dict[str, Any]]:
    _, records = load_sidecar(sidecar)
    by_stem = {str(record["stem"]): record for record in records}
    data_stems = {stem for _, _, _, stem in data.records}
    if data_stems != set(by_stem):
        raise ValueError("pose sidecar membership must exactly equal the full training split")
    for _, _, latent_shape, stem in data.records:
        record = by_stem[stem]
        bucket = record.get("bucket")
        if not isinstance(bucket, list) or len(bucket) != 2 or tuple(reversed(bucket)) != tuple(value * 8 for value in latent_shape):
            raise ValueError(f"{stem}: immutable pose sidecar geometry does not match the supplied 768 latent cache")
    return by_stem


def _events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def run_step(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: train.OptimizerStepWarmup, data_iter: Iterator[dict[str, Any]],
             data: PreparedLatentShardDataset, cfg: train.TrainConfig, recipe: ThroughputBenchmarkRecipe,
             generator: torch.Generator, device: torch.device, pose_records: dict[str, dict[str, Any]],
             vae: Any | None, critic: FixedBoxKeypointRCNNCritic | None, global_microbatch: int) -> dict[str, float]:
    wall_start = time.perf_counter(); data_wait = 0.0
    forward_ms = backward_ms = optimizer_ms = 0.0
    active = eligible = 0
    for accumulation_index in range(recipe.gradient_accumulation_steps):
        loading_started = time.perf_counter(); batch = next(data_iter); data_wait += time.perf_counter() - loading_started
        train.apply_cached_caption_dropout(
            batch, data.text_conditioning.unconditional, cfg.caption_dropout, cfg.seed,
            global_microbatch + accumulation_index,
        )
        forward_start, forward_end = _events(); backward_start, backward_end = _events()
        forward_start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if recipe.objective == "candidate":
                assert vae is not None and critic is not None
                loss, diag = _pose_smoke_loss(
                    model, vae, critic, batch, pose_records, cfg, device, generator,
                    pose_loss_name=LOCKED_POSE_LOSS, lambda_pose=LOCKED_LAMBDA_POSE,
                    timestep_min=LOCKED_POSE_WINDOW[0], timestep_max=LOCKED_POSE_WINDOW[1],
                    forced_exposure_probability=0.0, collect_diagnostics=False,
                )
                active += int(diag["pose_active_count_tensor"].item())
                eligible += int(diag["pose_eligible_count_tensor"].item())
            else:
                loss, _ = train._flow_loss(
                    model, None, batch, cfg, device, generator,
                    gradient_checkpointing_blocks=cfg.gradient_checkpointing_blocks,
                    collect_diagnostics=False,
                )
        forward_end.record(); backward_start.record(); (loss / recipe.gradient_accumulation_steps).backward(); backward_end.record()
        backward_end.synchronize()
        forward_ms += forward_start.elapsed_time(forward_end); backward_ms += backward_start.elapsed_time(backward_end)
    optimizer_start, optimizer_end = _events(); optimizer_start.record()
    train.optimizer_update(optimizer, scheduler, trainable_params(model), cfg.max_grad_norm)
    optimizer_end.record(); optimizer_end.synchronize(); optimizer_ms = optimizer_start.elapsed_time(optimizer_end)
    if critic is not None:
        assert_frozen_no_parameter_grad(vae, critic)
    return {
        "forward_seconds": forward_ms / 1000, "backward_seconds": backward_ms / 1000,
        "optimizer_seconds": optimizer_ms / 1000, "optimizer_step_seconds": time.perf_counter() - wall_start,
        "data_wait_seconds": data_wait, "pose_active_count": float(active), "pose_eligible_count": float(eligible),
        "pose_microbatch_count": float(recipe.gradient_accumulation_steps),
    }


def mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def main() -> None:
    args = parser().parse_args(); recipe = recipe_from_args(args)
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result: {args.output_json}")
    # This must happen before model construction: a benchmark is invalid unless
    # both full-data artifacts pass the same verifier exposed to operators.
    verify_full_768_cache(dataset_root=args.dataset_root, cache_root=args.latent_root,
                          pose_sidecar=args.pose_sidecar,
                          **({"train_manifest": args.train_manifest} if args.train_manifest else {}))
    if not torch.cuda.is_available():
        raise RuntimeError("Run the production throughput benchmark only from the GH200 host shell")
    set_seed(42); device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device)
    data = PreparedLatentShardDataset(args.latent_root, "train", text_conditioning_root=args.text_conditioning_root)
    validate_full_768_dataset(data); pose_records = load_and_validate_pose_records(args.pose_sidecar, data)
    groups = planned_microbatches(data, recipe); data_iter = batch_iterator(data, groups, recipe)
    cfg = train.TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root, rank=64, alpha=64,
                            microbatch_size=recipe.microbatch_size, gradient_accumulation_steps=recipe.gradient_accumulation_steps,
                            max_steps=recipe.warmup_steps + recipe.timed_steps, warmup_steps=200,
                            compile=recipe.compile, fused_adamw=recipe.fused_adamw,
                            gradient_checkpointing=recipe.gradient_checkpointing_blocks > 0,
                            gradient_checkpointing_blocks=recipe.gradient_checkpointing_blocks,
                            wandb_enabled=False)
    build_started = time.perf_counter(); model = build_pose_model(cfg.raw_ckpt, cfg.rank, cfg.alpha, "cuda")
    audit = audit_control_model(model, rank=64); train.configure_runtime(model, compile_enabled=cfg.compile); model.train()
    optimizer = train.build_optimizer(model, cfg); scheduler = train.OptimizerStepWarmup(optimizer, cfg.warmup_steps)
    vae: Any | None = None; critic: FixedBoxKeypointRCNNCritic | None = None
    if recipe.objective == "candidate":
        vae = load_krea_vae(device); critic = FixedBoxKeypointRCNNCritic().to(device).eval(); assert_frozen_no_parameter_grad(vae, critic)
    setup_seconds = time.perf_counter() - build_started
    generator = torch.Generator(device=device).manual_seed(42); optimizer.zero_grad(set_to_none=True)
    for index in range(recipe.warmup_steps):
        run_step(model=model, optimizer=optimizer, scheduler=scheduler, data_iter=data_iter, data=data, cfg=cfg,
                 recipe=recipe, generator=generator, device=device, pose_records=pose_records, vae=vae, critic=critic,
                 global_microbatch=index * recipe.gradient_accumulation_steps)
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    rows = [run_step(model=model, optimizer=optimizer, scheduler=scheduler, data_iter=data_iter, data=data, cfg=cfg,
                     recipe=recipe, generator=generator, device=device, pose_records=pose_records, vae=vae, critic=critic,
                     global_microbatch=(recipe.warmup_steps + index) * recipe.gradient_accumulation_steps)
            for index in range(recipe.timed_steps)]
    step_seconds = mean(rows, "optimizer_step_seconds"); active = sum(row["pose_active_count"] for row in rows)
    result: dict[str, object] = {
        "format_version": 1, "label": args.label, "recipe": recipe.asdict(), "setup_seconds_including_compile": setup_seconds,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "trainable_parameter_count": audit["trainable_parameters"], "frozen_parameter_count": audit["frozen_parameters"],
        "forward_seconds_mean": mean(rows, "forward_seconds"), "backward_seconds_mean": mean(rows, "backward_seconds"),
        "optimizer_seconds_mean": mean(rows, "optimizer_seconds"), "optimizer_step_seconds_mean": step_seconds,
        "data_wait_seconds_mean": mean(rows, "data_wait_seconds"),
        "samples_per_second": recipe.microbatch_size * recipe.gradient_accumulation_steps / step_seconds,
        "effective_samples_per_second": LOCKED_EFFECTIVE_BATCH / step_seconds,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device), "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "pose_active_fraction": active / (recipe.timed_steps * LOCKED_EFFECTIVE_BATCH),
        "pose_active_microbatch_fraction": sum(row["pose_active_count"] > 0 for row in rows) / (recipe.timed_steps * recipe.gradient_accumulation_steps),
        "pose_eligible_fraction": sum(row["pose_eligible_count"] for row in rows) / (recipe.timed_steps * LOCKED_EFFECTIVE_BATCH),
        "runtime_projection": projected_runtime(seconds_per_optimizer_step=step_seconds, effective_batch_size=LOCKED_EFFECTIVE_BATCH),
        "gpu_utilization_guidance": "In a second terminal during the timed window run: nvidia-smi dmon -s pucvmt -d 1; record average GPU util, memory util, power, clocks, and any utilization troughs alongside this JSON.",
        "no_checkpoints_written": True,
    }
    validate_benchmark_result(result); args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
