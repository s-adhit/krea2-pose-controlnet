"""Neutral mechanics shared by production training and historical trainers."""
from __future__ import annotations

import inspect
import math
import random
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from pose_controlnet.model import audit_control_model, load_trainable_state_dict, trainable_params


class OptimizerConfig(Protocol):
    rank: int
    lr: float
    fused_adamw: bool


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
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state: dict) -> None:
        if state["warmup_steps"] != self.warmup_steps:
            raise ValueError("Checkpoint warmup schedule differs from current configuration")
        self.step_count, self.base_lrs = int(state["step_count"]), list(state["base_lrs"])
        self._apply_for_update(self.step_count + 1)


def _adamw_kwargs(cfg: OptimizerConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {"betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.0}
    if cfg.fused_adamw:
        if "fused" not in inspect.signature(torch.optim.AdamW).parameters:
            raise RuntimeError("This PyTorch build does not expose fused AdamW")
        if not torch.cuda.is_available():
            raise RuntimeError("fused AdamW is a CUDA-only production benchmark option")
        kwargs["fused"] = True
    return kwargs


def build_production_optimizer(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.AdamW:
    """Build the locked AdamW optimizer over exactly the trainable boundary."""
    audit_control_model(model, rank=cfg.rank)
    parameters = trainable_params(model)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise AssertionError("Optimizer parameter selection includes frozen or no tensors")
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, **_adamw_kwargs(cfg))
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    frozen_ids = {id(parameter) for parameter in model.parameters() if not parameter.requires_grad}
    if optimizer_ids != expected_ids or optimizer_ids & frozen_ids:
        raise AssertionError("Optimizer must contain exactly ControlInput and intended LoRA tensors")
    return optimizer


class DeterministicBucketBatches:
    """Epoch-seeded, bucket-homogeneous batches reconstructible from position."""
    def __init__(self, records: list[tuple[str, int, tuple[int, int]]], microbatch_size: int, seed: int) -> None:
        self.records, self.microbatch_size, self.seed = records, microbatch_size, seed

    def for_epoch(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        by_bucket: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(self.records):
            by_bucket.setdefault(record[2], []).append(index)
        batches: list[list[int]] = []
        for indices in by_bucket.values():
            rng.shuffle(indices)
            batches.extend(indices[offset:offset + self.microbatch_size]
                           for offset in range(0, len(indices) - self.microbatch_size + 1, self.microbatch_size))
        rng.shuffle(batches)
        if not batches:
            raise ValueError("No full microbatches available from latent shards")
        return batches


def apply_cached_caption_dropout(batch: dict, unconditional: dict[str, torch.Tensor], probability: float,
                                 seed: int, microbatch_index: int) -> None:
    rng = random.Random(seed + 1_000_003 * microbatch_index)
    entries = []
    for index in range(batch["context"].shape[0]):
        length = int(batch["text_mask"][index].sum().item())
        entries.append(unconditional if rng.random() < probability else {
            "context": batch["context"][index, :length], "mask": batch["text_mask"][index, :length],
        })
    max_length = max(entry["context"].shape[0] for entry in entries)
    batch["context"] = torch.stack([F.pad(entry["context"], (0, 0, 0, 0, 0, max_length - entry["context"].shape[0])) for entry in entries])
    batch["text_mask"] = torch.stack([F.pad(entry["mask"], (0, max_length - entry["mask"].shape[0])) for entry in entries])


def capture_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def assert_frozen_no_parameter_grad(*modules: torch.nn.Module) -> None:
    """Reject trainable or accumulated gradients across a frozen boundary."""
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise RuntimeError("frozen boundary parameter is unexpectedly trainable")
            if parameter.grad is not None:
                raise RuntimeError("frozen boundary parameter unexpectedly received a gradient")


def configure_runtime(model: torch.nn.Module, *, compile_enabled: bool) -> None:
    if compile_enabled:
        model.txtmlp.forward = torch.compile(model.txtmlp.forward, dynamic=True)


def diagnostic_grad_norms(model: torch.nn.Module) -> tuple[dict[str, float], dict[str, float]]:
    control = {"full": float(model.first.weight.grad.float().norm()) if model.first.weight.grad is not None else 0.0,
               "control_half": float(model.first.weight.grad[:, model.first.weight.shape[1] // 2:].float().norm()) if model.first.weight.grad is not None else 0.0}
    lora: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if (name.endswith(".A") or name.endswith(".B")) and parameter.grad is not None:
            lora[name] = float(parameter.grad.float().norm())
            if len(lora) == 2:
                break
    return control, lora


def optimizer_update(optimizer: torch.optim.Optimizer, scheduler: object, parameters: list[torch.nn.Parameter],
                     max_grad_norm: float, before_step: Callable[[], None] | None = None) -> float:
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("Non-finite global gradient norm")
    if before_step is not None:
        before_step()
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return grad_norm


def step_mirror_requested(global_step: int, every_steps: int) -> bool:
    return every_steps > 0 and global_step > 0 and global_step % every_steps == 0


def restore_full_training_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: object,
                                state: dict) -> tuple[int, int, int, object | None]:
    load_trainable_state_dict(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])
    return state["global_step"], state["epoch"], state["batch_position"], state.get("flow_generator_state")
