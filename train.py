"""Bounded Gate-F trainer for Krea-2 Raw skeleton-control LoRA.

This entry point intentionally requires ``--max-steps`` and accepts at most a
100-step smoke.  It is not a production-run launcher.
"""
from __future__ import annotations

import argparse
import math
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from pose_controlnet.checkpointing import (
    HFTrainingCheckpointMirror, load_training_state, resolve_auto_resume,
    save_training_state,
)
from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position, sample_flow_timestep
from pose_controlnet.model import audit_control_model, build_pose_model, load_trainable_state_dict, trainable_params, trainable_state_dict
from pose_controlnet.seed import set_seed
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.wandb_logging import TrainingTelemetry


def effective_batch_size(microbatch_size: int, gradient_accumulation_steps: int, world_size: int = 1) -> int:
    if min(microbatch_size, gradient_accumulation_steps, world_size) < 1:
        raise ValueError("microbatch size, accumulation steps, and world size must be positive")
    return microbatch_size * gradient_accumulation_steps * world_size


class OptimizerStepWarmup:
    """Linear warmup whose counter advances only after an optimizer update."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int) -> None:
        self.optimizer, self.warmup_steps = optimizer, warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self._apply_for_update(1)

    def _apply_for_update(self, update_number: int) -> None:
        scale = min(1.0, update_number / self.warmup_steps) if self.warmup_steps else 1.0
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def step(self) -> None:
        self.step_count += 1
        self._apply_for_update(self.step_count + 1)

    @property
    def current_update_learning_rates(self) -> list[float]:
        """The rates installed for the optimizer update about to occur."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state: dict) -> None:
        if state["warmup_steps"] != self.warmup_steps:
            raise ValueError("Checkpoint warmup schedule differs from current configuration")
        self.step_count, self.base_lrs = int(state["step_count"]), list(state["base_lrs"])
        self._apply_for_update(self.step_count + 1)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    audit_control_model(model, rank=cfg.rank)
    params = trainable_params(model)
    if not params or any(not parameter.requires_grad for parameter in params):
        raise AssertionError("Optimizer parameter selection includes frozen or no tensors")
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    frozen_ids = {id(parameter) for parameter in model.parameters() if not parameter.requires_grad}
    if optimizer_ids != expected_ids or optimizer_ids & frozen_ids:
        raise AssertionError("Optimizer must contain exactly ControlInput and intended LoRA tensors")
    return optimizer


def optimizer_update(optimizer: torch.optim.Optimizer, scheduler: OptimizerStepWarmup,
                     parameters: list[torch.nn.Parameter], max_grad_norm: float,
                     before_step: Callable[[], None] | None = None) -> float:
    """The sole optimizer boundary: clip, update, schedule, then clear grads."""
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("Non-finite global gradient norm")
    if before_step is not None:
        before_step()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


class DeterministicBucketBatches:
    """Epoch-seeded, bucket-homogeneous batches reconstructible from position."""
    def __init__(self, records: list[tuple[str, int, tuple[int, int]]], microbatch_size: int, seed: int) -> None:
        self.records, self.microbatch_size, self.seed = records, microbatch_size, seed

    def for_epoch(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        by_bucket: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(self.records):
            bucket = record[2]
            by_bucket.setdefault(bucket, []).append(index)
        batches: list[list[int]] = []
        for indices in by_bucket.values():
            rng.shuffle(indices)
            batches.extend(indices[offset:offset + self.microbatch_size]
                           for offset in range(0, len(indices) - self.microbatch_size + 1, self.microbatch_size))
        rng.shuffle(batches)
        if not batches:
            raise ValueError("No full microbatches available from latent shards")
        return batches


def apply_caption_dropout(prompts: list[str], probability: float, seed: int, microbatch_index: int) -> list[str]:
    rng = random.Random(seed + 1_000_003 * microbatch_index)
    return ["" if rng.random() < probability else prompt for prompt in prompts]


def apply_cached_caption_dropout(batch: dict, unconditional: dict[str, torch.Tensor], probability: float, seed: int, microbatch_index: int) -> None:
    """Seeded 10% dropout selects cached unconditional text; it never alters archives."""
    rng = random.Random(seed + 1_000_003 * microbatch_index)
    entries = []
    for index in range(batch["context"].shape[0]):
        length = int(batch["text_mask"][index].sum().item())
        entries.append(unconditional if rng.random() < probability else {"context": batch["context"][index, :length], "mask": batch["text_mask"][index, :length]})
    max_length = max(entry["context"].shape[0] for entry in entries)
    batch["context"] = torch.stack([F.pad(entry["context"], (0, 0, 0, 0, 0, max_length - entry["context"].shape[0])) for entry in entries])
    batch["text_mask"] = torch.stack([F.pad(entry["mask"], (0, max_length - entry["mask"].shape[0])) for entry in entries])


def _capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def configure_runtime(model: torch.nn.Module, *, compile_enabled: bool) -> None:
    """Apply opt-in compilation only to a rank-stable text-projection boundary.

    RMSNorm is deliberately never compiled directly: it is shared by text MLPs
    (rank 3) and attention Q/K tensors (rank 4).  The text MLP always receives
    ``[batch, text_length, text_features]`` in this training entry point.
    """
    if compile_enabled:
        model.txtmlp.forward = torch.compile(model.txtmlp.forward, dynamic=True)


def _flow_loss(model, conditioner, batch: dict, cfg: TrainConfig, device: torch.device, generator: torch.Generator, *, gradient_checkpointing_blocks: int) -> tuple[torch.Tensor, dict]:
    clean = batch["latent"].to(device=device, dtype=torch.float32, non_blocking=True)
    control = batch["control"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
    if clean.shape != control.shape or not torch.isfinite(clean).all() or not torch.isfinite(control).all():
        raise FloatingPointError("Invalid paired latent batch")
    timestep = sample_flow_timestep(clean.shape[0], (clean.shape[-2] // model.config.patch) * (clean.shape[-1] // model.config.patch), cfg, device, generator)
    noise = torch.randn(clean.shape, device=device, dtype=torch.float32, generator=generator)
    noisy, target = make_flow_pair(clean, noise, timestep)
    if "context" in batch:
        context, text_mask = batch["context"].to(device=device, dtype=torch.bfloat16, non_blocking=True), batch["text_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    else:
        if conditioner is None: raise RuntimeError("Cached conditioning is required but absent")
        context, text_mask = conditioner(batch["prompts"])
    image_tokens, pos, mask = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
    control_tokens, _, _ = patchify_and_position(control, context.shape[1], model.config.patch, text_mask)
    target_tokens, _, _ = patchify_and_position(target, context.shape[1], model.config.patch, text_mask)
    prediction = forward_pose_control(model, image_tokens, control_tokens, context, timestep.to(torch.bfloat16), pos, mask,
                                      gradient_checkpointing_blocks=gradient_checkpointing_blocks)
    loss = F.mse_loss(prediction.float(), target_tokens.float())
    if not torch.isfinite(loss): raise FloatingPointError("Non-finite flow-matching MSE")
    diagnostics = {"control_latent_rms": control.float().square().mean().sqrt().item(), "control_latent_std": control.float().std(unbiased=False).item()}
    return loss, diagnostics


def validate_flow_loss(model, conditioner, batches: Iterable[dict], cfg: TrainConfig, device: torch.device, generator: torch.Generator) -> float:
    was_training = model.training; model.eval(); losses = []
    try:
        with torch.inference_mode():
            for batch in batches:
                loss, _ = _flow_loss(model, conditioner, batch, cfg, device, generator, gradient_checkpointing_blocks=0)
                losses.append(loss.item())
    finally:
        model.train(was_training)
    if not losses: raise ValueError("Validation received no batches")
    return sum(losses) / len(losses)


def _diagnostic_grad_norms(model: torch.nn.Module) -> tuple[dict[str, float], dict[str, float]]:
    control = {"full": float(model.first.weight.grad.float().norm()) if model.first.weight.grad is not None else 0.0,
               "control_half": float(model.first.weight.grad[:, model.first.weight.shape[1] // 2:].float().norm()) if model.first.weight.grad is not None else 0.0}
    lora = {}
    for name, parameter in model.named_parameters():
        if (name.endswith(".A") or name.endswith(".B")) and parameter.grad is not None:
            lora[name] = float(parameter.grad.float().norm())
            if len(lora) == 2: break
    return control, lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors")
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning",
                        help="Complete persistent Qwen conditioning root; pass --online-text-conditioning only for diagnostics")
    parser.add_argument("--online-text-conditioning", action="store_true", help="Diagnostic fallback that loads Qwen; never production mode")
    parser.add_argument("--checkpoint-dir", default="/lambda/nfs/adhit/krea2-pose/checkpoints")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--diagnostics-every", type=int, default=10)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="opt in to compiling the rank-stable text projection")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                        help="legacy shorthand: checkpoint all 28 main transformer blocks")
    parser.add_argument("--gradient-checkpointing-blocks", type=int, default=None, metavar="N",
                        help="checkpoint the first N of 28 main transformer blocks (0 disables; overrides legacy flag)")
    parser.add_argument("--resume", help="checkpoint path or 'auto' (newest valid local, then HF fallback)")
    parser.add_argument("--hf-repo-id", default="", help="private HF model repo for full checkpoint mirroring")
    parser.add_argument("--hf-mirror-every-seconds", type=float, default=3600,
                        help="wall-clock full-checkpoint mirror cadence (default: 3600)")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 100: parser.error("Gate-F entry point permits only explicit bounded 1..100-step runs")
    if args.microbatch_size < 1 or args.gradient_accumulation_steps < 1: parser.error("batch settings must be positive")
    if args.gradient_checkpointing_blocks is not None and not 0 <= args.gradient_checkpointing_blocks <= 28:
        parser.error("--gradient-checkpointing-blocks must be in [0, 28]")
    if args.hf_mirror_every_seconds < 0: parser.error("--hf-mirror-every-seconds must be non-negative")
    return args


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    gradient_checkpointing_blocks = (
        args.gradient_checkpointing_blocks
        if args.gradient_checkpointing_blocks is not None
        else (28 if args.gradient_checkpointing else 0)
    )
    return TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root, ckpt_dir=args.checkpoint_dir, run_name=args.run_name,
                       max_steps=args.max_steps, microbatch_size=args.microbatch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
                       max_grad_norm=args.max_grad_norm, validation_batches=args.validation_batches, val_every=args.val_every, save_every=args.save_every, diagnostics_every=args.diagnostics_every,
                       compile=args.compile, gradient_checkpointing=gradient_checkpointing_blocks > 0,
                       gradient_checkpointing_blocks=gradient_checkpointing_blocks,
                       wandb_enabled=not args.no_wandb, wandb_mode=args.wandb_mode,
                       metrics_jsonl_path=str(Path(args.checkpoint_dir) / args.run_name / "metrics.jsonl"),
                       hf_repo_id=args.hf_repo_id, hf_push_every_seconds=args.hf_mirror_every_seconds)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Run Gate-F smoke from the GH200 host shell with CUDA visible")
    cfg = config_from_args(args)
    set_seed(cfg.seed); device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats()
    print(f"effective_batch={effective_batch_size(cfg.microbatch_size, cfg.gradient_accumulation_steps)} (microbatch × accumulation × world_size=1)", flush=True)
    print(f"runtime: compile={cfg.compile} gradient_checkpointing_blocks={cfg.gradient_checkpointing_blocks}", flush=True)
    cached_text = not args.online_text_conditioning
    train_data, val_data = PreparedLatentShardDataset(cfg.shard_dir, "train", text_conditioning_root=args.text_conditioning_root if cached_text else None), PreparedLatentShardDataset(cfg.shard_dir, "val", text_conditioning_root=args.text_conditioning_root if cached_text else None)
    train_plan, val_plan = DeterministicBucketBatches(train_data.records, cfg.microbatch_size, cfg.seed), DeterministicBucketBatches(val_data.records, cfg.microbatch_size, cfg.seed + 17)
    model = build_pose_model(cfg.raw_ckpt, cfg.rank, cfg.alpha, "cuda")
    configure_runtime(model, compile_enabled=cfg.compile)
    model.train()
    conditioner = None if cached_text else PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    print(f"text_conditioning={'cached' if cached_text else 'online'} text_encoder_loaded={conditioner is not None}", flush=True)
    optimizer, scheduler = build_optimizer(model, cfg), None
    scheduler = OptimizerStepWarmup(optimizer, cfg.warmup_steps)
    global_step = epoch = batch_position = 0
    resume_generator_state = None
    if args.resume:
        resume_path = (resolve_auto_resume(checkpoint_dir=cfg.ckpt_dir, run_name=cfg.run_name,
                                           repo_id=cfg.hf_repo_id,
                                           remote_download_dir=Path(cfg.ckpt_dir) / cfg.run_name / "hf_recovery")
                       if args.resume == "auto" else Path(args.resume))
        if resume_path is None: raise FileNotFoundError("--resume auto found no valid local or HF full checkpoint")
        state = load_training_state(resume_path); load_trainable_state_dict(model, state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); global_step, epoch, batch_position = state["global_step"], state["epoch"], state["batch_position"]; _restore_rng(state["rng"])
        print(f"[resume] loaded validated full checkpoint {resume_path} at optimizer step {global_step} "
              f"(epoch={epoch}, batch_position={batch_position})", flush=True)
        resume_generator_state = state.get("flow_generator_state")
    telemetry = TrainingTelemetry(cfg, cfg.run_name)
    mirror = HFTrainingCheckpointMirror(repo_id=cfg.hf_repo_id, run_name=cfg.run_name,
                                        interval_seconds=cfg.hf_push_every_seconds, telemetry=telemetry)
    mirror.start()
    stopped = False
    def stop_handler(signum, _frame):
        nonlocal stopped
        stopped = True
        print(f"received signal {signum}; stopping after current optimizer boundary", flush=True)
    signal.signal(signal.SIGINT, stop_handler); signal.signal(signal.SIGTERM, stop_handler)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + global_step)
    if resume_generator_state is not None:
        generator.set_state(resume_generator_state)
    optimizer.zero_grad(set_to_none=True)
    last_checkpoint_time = time.monotonic()
    try:
        while global_step < cfg.max_steps and not stopped:
            batches = train_plan.for_epoch(epoch)
            if batch_position >= len(batches): epoch, batch_position = epoch + 1, 0; continue
            start = time.monotonic(); last_diag = None
            for accumulation_index in range(cfg.gradient_accumulation_steps):
                if batch_position >= len(batches):
                    epoch, batch_position = epoch + 1, 0
                    batches = train_plan.for_epoch(epoch)
                batch = collate([train_data[index] for index in batches[batch_position]]); batch_position += 1
                dropout_index = global_step * cfg.gradient_accumulation_steps + accumulation_index
                if cached_text:
                    apply_cached_caption_dropout(batch, train_data.text_conditioning.unconditional, cfg.caption_dropout, cfg.seed, dropout_index)
                else:
                    batch["prompts"] = apply_caption_dropout(batch["prompts"], cfg.caption_dropout, cfg.seed, dropout_index)
                with torch.autocast("cuda", dtype=torch.bfloat16): loss, last_diag = _flow_loss(model, conditioner, batch, cfg, device, generator, gradient_checkpointing_blocks=cfg.gradient_checkpointing_blocks)
                (loss / cfg.gradient_accumulation_steps).backward()
            diagnostics_due = last_diag is not None and (global_step + 1) % cfg.diagnostics_every == 0
            control_norms = lora_norms = None
            def capture_diagnostics() -> None:
                nonlocal control_norms, lora_norms
                control_norms, lora_norms = _diagnostic_grad_norms(model)
            learning_rate_used = scheduler.current_update_learning_rates[0]
            grad_norm = optimizer_update(
                optimizer, scheduler, trainable_params(model), cfg.max_grad_norm,
                before_step=capture_diagnostics if diagnostics_due else None,
            )
            global_step += 1
            elapsed = time.monotonic() - start; samples = effective_batch_size(cfg.microbatch_size, cfg.gradient_accumulation_steps)
            telemetry.log_train(loss=float(loss.item()), learning_rate=learning_rate_used, global_grad_norm=grad_norm, sec_per_step=elapsed, samples_per_second=samples / elapsed, step=global_step)
            telemetry.log_cuda_memory(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated(), step=global_step)
            if diagnostics_due:
                telemetry.log_control_diagnostics(**last_diag, control_input_grad_norms=control_norms, lora_grad_norms=lora_norms, step=global_step)
            if global_step % cfg.val_every == 0 or global_step == cfg.max_steps:
                val_batches = (collate([val_data[index] for index in group]) for group in val_plan.for_epoch(0)[:cfg.validation_batches])
                telemetry.log_validation_flow_loss(validate_flow_loss(model, conditioner, val_batches, cfg, device, generator), step=global_step)
            checkpoint_due = time.monotonic() - last_checkpoint_time >= cfg.checkpoint_every_seconds
            if global_step % cfg.save_every == 0 or global_step == cfg.max_steps or stopped or checkpoint_due:
                path = Path(cfg.ckpt_dir) / cfg.run_name / f"step_{global_step:06d}.pt"
                save_training_state(path, {"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": global_step, "epoch": epoch, "batch_position": batch_position, "rng": _capture_rng(), "flow_generator_state": generator.get_state(), "config": asdict(cfg)})
                last_checkpoint_time = time.monotonic()
                telemetry.log_checkpoint(checkpoint_step=global_step, checkpoint_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), step=global_step)
                mirror.maybe_submit(path)
    finally:
        mirror.stop()
        telemetry.close()


if __name__ == "__main__": main()
