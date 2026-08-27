"""Bounded Gate-F trainer for Krea-2 Raw skeleton-control LoRA.

This entry point intentionally requires ``--max-steps`` and accepts at most a
100-step smoke unless ``--allow-extended-training`` is explicitly supplied.
It is not a production-run launcher.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import signal
import tempfile
import time
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from pose_controlnet.checkpointing import (
    HFTrainingCheckpointMirror, load_training_state, resolve_auto_resume,
    save_training_state, validated_hf_checkpoint_for_step,
    validated_local_checkpoint_for_hf_step,
)
from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position, sample_flow_timestep
from pose_controlnet.model import audit_control_model, build_pose_model, load_trainable_state_dict, trainable_params, trainable_state_dict
from pose_controlnet.seed import set_seed
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.wandb_logging import TrainingTelemetry


# This is deliberately a one-purpose branch, rather than a permissive
# "resume from arbitrary archive step" interface.  It prevents an accidental
# latest/nearby checkpoint substitution or writing into the source run.
LR_BRANCH_SOURCE_HF_REPO = "adhit-420/Krea-2-PoseControl-LoRA-checkpoints"
LR_BRANCH_SOURCE_RUN = "pose-learning-1500"
LR_BRANCH_SOURCE_STEP = 900
LR_BRANCH_RUN = "pose-learning-900-lr5e5-to1500"
LR_BRANCH_TARGET_STEP = 1500
LR_BRANCH_LEARNING_RATE = 5e-5
LR_BRANCH_CHECKPOINT_DIR = "/lambda/nfs/adhit/krea2-pose/checkpoints"

# This continuation is intentionally isolated from the completed LR-only
# branch.  Its only scientific difference is pre-shift timestep exposure.
TIMESTEP_BRANCH_SOURCE_RUN = LR_BRANCH_RUN
TIMESTEP_BRANCH_SOURCE_STEP = 1500
TIMESTEP_BRANCH_RUN = "pose-learning-1500-timestep-lowmid20-to1800"
TIMESTEP_BRANCH_TARGET_STEP = 1800
TIMESTEP_BRANCH_REQUIRED_CHECKPOINT_STEPS = (1600, 1700, 1800)
TIMESTEP_BRANCH_SOURCE_CHECKPOINT = (
    Path(LR_BRANCH_CHECKPOINT_DIR) / TIMESTEP_BRANCH_SOURCE_RUN / "step_001500.pt"
)
# These are inverse-shift bounds for final t=[0.1, 0.6] over the actual
# 3,952..4,096-token training buckets (mu=0.891015625..0.90625).
TIMESTEP_BRANCH_AUX_PROB = 0.20
TIMESTEP_BRANCH_AUX_MIN = 0.04359494981207863
TIMESTEP_BRANCH_AUX_MAX = 0.3773562340267345
TIMESTEP_CONFIG_FIELDS = frozenset({"timestep_aux_prob", "timestep_aux_min", "timestep_aux_max"})

# The overnight continuation intentionally changes exactly one optimization
# value: model.first (ControlInputLayer) receives 2x the LoRA LR.  It is not a
# timestep experiment and can only originate from the exact completed LR-only
# step-1500 checkpoint.
CONTROLINPUT_BRANCH_SOURCE_RUN = LR_BRANCH_RUN
CONTROLINPUT_BRANCH_SOURCE_STEP = 1500
CONTROLINPUT_BRANCH_RUN = "pose-learning-1500-controlinput-lr2x-to2800"
CONTROLINPUT_BRANCH_TARGET_STEP = 2800
CONTROLINPUT_BRANCH_LORA_LR = 5e-5
CONTROLINPUT_BRANCH_CONTROL_LR = 1e-4
CONTROLINPUT_BRANCH_CONTROL_LR_MULTIPLIER = 2.0
CONTROLINPUT_BRANCH_REQUIRED_CHECKPOINT_STEPS = tuple(range(1600, 2801, 100))
CONTROLINPUT_BRANCH_SOURCE_CHECKPOINT = (
    Path(LR_BRANCH_CHECKPOINT_DIR) / CONTROLINPUT_BRANCH_SOURCE_RUN / "step_001500.pt"
)
CONTROLINPUT_CONFIG_FIELDS = frozenset({
    "source_checkpoint", "source_step", "target_step", "control_input_lr",
    "control_input_lr_multiplier", "required_checkpoint_steps",
})


def lr_branch_checkpoint_dir() -> Path:
    """The branch's isolated local root; never the source-run directory."""
    return Path(LR_BRANCH_CHECKPOINT_DIR) / LR_BRANCH_RUN


def timestep_branch_checkpoint_dir() -> Path:
    return Path(LR_BRANCH_CHECKPOINT_DIR) / TIMESTEP_BRANCH_RUN


def controlinput_branch_checkpoint_dir() -> Path:
    """The isolated root for the 1500→2800 ControlInput-LR experiment."""
    return Path(LR_BRANCH_CHECKPOINT_DIR) / CONTROLINPUT_BRANCH_RUN


def train_config_from_checkpoint_values(values: dict) -> TrainConfig:
    """Load checkpoint config while permitting only later defaulted metadata."""
    expected = {field.name for field in fields(TrainConfig)}
    missing, extra = expected - set(values), set(values) - expected
    defaultable = {field.name for field in fields(TrainConfig) if field.default is not MISSING}
    if extra or missing - defaultable:
        raise ValueError(f"Checkpoint config is incompatible; missing={sorted(missing)} extra={sorted(extra)}")
    expanded = dict(values)
    for field in fields(TrainConfig):
        if field.name in missing:
            expanded[field.name] = field.default
    return TrainConfig(**expanded)


def _assert_exact_learning_rate(optimizer: torch.optim.Optimizer, expected: float) -> None:
    observed = [group["lr"] for group in optimizer.param_groups]
    if not observed or any(rate != expected for rate in observed):
        raise AssertionError(f"Resumed optimizer LR must be exactly {expected}, got {observed}")


def apply_resumed_learning_rate_override(optimizer: torch.optim.Optimizer,
                                         scheduler: "OptimizerStepWarmup", learning_rate: float,
                                         *, resumed_global_step: int) -> None:
    """Keep a restored warmup scheduler at its restored progress but at a new LR.

    ``OptimizerStepWarmup`` reapplies ``base_lrs`` after every optimizer update.
    Updating optimizer groups alone would therefore silently restore 1e-4 on
    the first scheduler step.  Changing both base rates and installed rates
    preserves AdamW state and the scheduler's post-warmup position.
    """
    if learning_rate <= 0:
        raise ValueError("Resumed learning-rate override must be positive")
    if resumed_global_step < scheduler.warmup_steps:
        raise ValueError("LR branch cannot restart or alter an in-progress warmup")
    if scheduler.step_count != resumed_global_step:
        raise ValueError("Checkpoint scheduler progress must equal global_step for a resumed LR branch")
    scheduler.base_lrs = [learning_rate for _ in optimizer.param_groups]
    scheduler._apply_for_update(scheduler.step_count + 1)
    _assert_exact_learning_rate(optimizer, learning_rate)


def lr_branch_config_from_source_state(source_state: dict) -> TrainConfig:
    """Derive the branch configuration from the archived run, changing only run plumbing and LR."""
    if source_state["global_step"] != LR_BRANCH_SOURCE_STEP:
        raise ValueError(f"LR branch requires embedded global_step {LR_BRANCH_SOURCE_STEP}")
    source_values = source_state["config"]
    source_cfg = train_config_from_checkpoint_values(source_values)
    if source_cfg.run_name != LR_BRANCH_SOURCE_RUN:
        raise ValueError(f"LR branch source config must name {LR_BRANCH_SOURCE_RUN}")
    if source_cfg.lr != 1e-4:
        raise ValueError(f"LR branch source config must have LR 1e-4, got {source_cfg.lr}")
    if source_cfg.max_steps != LR_BRANCH_TARGET_STEP:
        raise ValueError(f"LR branch source config must target step {LR_BRANCH_TARGET_STEP}")
    if not source_cfg.allow_extended_training:
        raise ValueError("LR branch source config must retain its explicit extended-training authorization")
    if source_cfg.warmup_steps >= LR_BRANCH_SOURCE_STEP:
        raise ValueError("LR branch source is not past warmup")
    if 100 % source_cfg.save_every:
        raise ValueError("Source checkpoint cadence cannot preserve required branch checkpoints at 1000..1500")
    if source_cfg.hf_repo_id != LR_BRANCH_SOURCE_HF_REPO:
        raise ValueError("LR branch source config must use the required checkpoint mirror repository")
    branch_cfg = replace(
        source_cfg,
        ckpt_dir=LR_BRANCH_CHECKPOINT_DIR,
        run_name=LR_BRANCH_RUN,
        lr=LR_BRANCH_LEARNING_RATE,
        max_steps=LR_BRANCH_TARGET_STEP,
        metrics_jsonl_path=str(lr_branch_checkpoint_dir() / "metrics.jsonl"),
        hf_repo_id=LR_BRANCH_SOURCE_HF_REPO,
    )
    assert_lr_branch_output_namespace(branch_cfg)
    return branch_cfg


def assert_lr_branch_output_namespace(cfg: TrainConfig) -> None:
    """Fail before any training-side write if this branch could touch the source run."""
    expected = lr_branch_checkpoint_dir()
    actual = Path(cfg.ckpt_dir) / str(cfg.run_name)
    if (cfg.run_name != LR_BRANCH_RUN or actual != expected or cfg.metrics_jsonl_path != str(expected / "metrics.jsonl")
            or cfg.hf_repo_id != LR_BRANCH_SOURCE_HF_REPO):
        raise AssertionError("LR branch output namespace is not isolated")
    if LR_BRANCH_SOURCE_RUN in actual.parts or cfg.run_name == LR_BRANCH_SOURCE_RUN:
        raise AssertionError("LR branch must never write into pose-learning-1500")


def assert_timestep_branch_output_namespace(cfg: TrainConfig) -> None:
    expected = timestep_branch_checkpoint_dir()
    actual = Path(cfg.ckpt_dir) / str(cfg.run_name)
    protected = {LR_BRANCH_SOURCE_RUN, "pose-learning-1500", TIMESTEP_BRANCH_RUN}
    if (cfg.run_name != TIMESTEP_BRANCH_RUN or actual != expected
            or cfg.metrics_jsonl_path != str(expected / "metrics.jsonl")
            or cfg.hf_repo_id != LR_BRANCH_SOURCE_HF_REPO):
        raise AssertionError("Timestep branch output namespace is not isolated")
    if actual == lr_branch_checkpoint_dir() or any(name in actual.parts for name in protected - {TIMESTEP_BRANCH_RUN}):
        raise AssertionError("Timestep branch must never write into an existing branch namespace")


def _assert_only_timestep_branch_config_changes(source: TrainConfig, branch: TrainConfig) -> None:
    allowed = TIMESTEP_CONFIG_FIELDS | {"run_name", "max_steps", "metrics_jsonl_path"}
    changed = {key for key, value in asdict(branch).items() if asdict(source)[key] != value}
    if changed - allowed:
        raise ValueError(f"Timestep continuation changed non-timestep settings: {sorted(changed - allowed)}")
    if changed != allowed:
        raise ValueError(f"Timestep continuation config delta is incomplete or unexpected: {sorted(changed)}")


def timestep_branch_config_from_source_state(source_state: dict) -> TrainConfig:
    """Derive the 1500→1800 ablation, proving all non-timestep settings match."""
    if source_state["global_step"] != TIMESTEP_BRANCH_SOURCE_STEP:
        raise ValueError(f"Timestep branch requires embedded global_step {TIMESTEP_BRANCH_SOURCE_STEP}")
    source_cfg = train_config_from_checkpoint_values(source_state["config"])
    if source_cfg.run_name != TIMESTEP_BRANCH_SOURCE_RUN:
        raise ValueError(f"Timestep branch source config must name {TIMESTEP_BRANCH_SOURCE_RUN}")
    if source_cfg.lr != LR_BRANCH_LEARNING_RATE or source_cfg.max_steps != TIMESTEP_BRANCH_SOURCE_STEP:
        raise ValueError("Timestep branch source must be the completed LR=5e-5 step-1500 run")
    if source_cfg.warmup_steps >= TIMESTEP_BRANCH_SOURCE_STEP or not source_cfg.allow_extended_training:
        raise ValueError("Timestep branch source is incompatible with preserved completed warmup/authorization")
    if source_cfg.hf_repo_id != LR_BRANCH_SOURCE_HF_REPO or source_cfg.save_every <= 0:
        raise ValueError("Timestep branch source checkpoint mirror or save cadence is incompatible")
    if any(step % source_cfg.save_every for step in TIMESTEP_BRANCH_REQUIRED_CHECKPOINT_STEPS):
        raise ValueError("Preserved save cadence cannot create required timestep branch checkpoints")
    if source_cfg.hf_mirror_every_steps and any(step % source_cfg.hf_mirror_every_steps for step in TIMESTEP_BRANCH_REQUIRED_CHECKPOINT_STEPS):
        raise ValueError("Preserved HF mirror cadence cannot mirror required timestep branch checkpoints")
    if any(getattr(source_cfg, key) != default for key, default in
           (("timestep_aux_prob", 0.0), ("timestep_aux_min", 0.0), ("timestep_aux_max", 1.0))):
        raise ValueError("Timestep branch source must have the original sampler disabled")
    scheduler = source_state["scheduler"]
    if scheduler["step_count"] != TIMESTEP_BRANCH_SOURCE_STEP or scheduler["warmup_steps"] != source_cfg.warmup_steps:
        raise ValueError("Timestep branch source scheduler progress is incompatible")
    if list(scheduler["base_lrs"]) != [LR_BRANCH_LEARNING_RATE]:
        raise ValueError("Timestep branch source scheduler LR must be exactly 5e-5")
    branch_cfg = replace(
        source_cfg, ckpt_dir=LR_BRANCH_CHECKPOINT_DIR, run_name=TIMESTEP_BRANCH_RUN,
        max_steps=TIMESTEP_BRANCH_TARGET_STEP,
        metrics_jsonl_path=str(timestep_branch_checkpoint_dir() / "metrics.jsonl"),
        timestep_aux_prob=TIMESTEP_BRANCH_AUX_PROB,
        timestep_aux_min=TIMESTEP_BRANCH_AUX_MIN,
        timestep_aux_max=TIMESTEP_BRANCH_AUX_MAX,
    )
    _assert_only_timestep_branch_config_changes(source_cfg, branch_cfg)
    assert_timestep_branch_output_namespace(branch_cfg)
    return branch_cfg


def resolve_timestep_branch_source_checkpoint() -> Path:
    """Use only the explicitly named completed local step-1500 checkpoint."""
    source = TIMESTEP_BRANCH_SOURCE_CHECKPOINT
    if source.name != "step_001500.pt" or not source.is_file():
        raise FileNotFoundError(f"Required exact local source checkpoint is unavailable: {source}")
    state = load_training_state(source)
    if state["global_step"] != TIMESTEP_BRANCH_SOURCE_STEP:
        raise ValueError("Exact timestep source checkpoint identity validation failed")
    return source


def _assert_byte_rng_state(value: object, label: str, *, allow_list: bool = False) -> None:
    values = value if allow_list else [value]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"Timestep recovery checkpoint {label} is missing")
    if any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.numel() == 0
           for item in values):
        raise ValueError(f"Timestep recovery checkpoint {label} is malformed")


def _assert_finite_adamw_state(optimizer_state: dict, expected_lr: float) -> None:
    """Check the serialized AdamW state before this narrow recovery accepts it."""
    groups, moments = optimizer_state.get("param_groups"), optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(moments, dict) or not moments:
        raise ValueError("Timestep recovery checkpoint has no complete AdamW state")
    group = groups[0]
    if (group.get("lr") != expected_lr or tuple(group.get("betas", ())) != (0.9, 0.99)
            or group.get("weight_decay") != 0.0):
        raise ValueError("Timestep recovery checkpoint AdamW hyperparameters differ from the experiment")
    parameter_ids = group.get("params")
    if not isinstance(parameter_ids, list) or not parameter_ids or set(parameter_ids) != set(moments):
        raise ValueError("Timestep recovery checkpoint AdamW parameter/moment identities are incomplete")
    for parameter_id, moment in moments.items():
        if not isinstance(moment, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(moment):
            raise ValueError(f"Timestep recovery checkpoint AdamW moments are incomplete for parameter {parameter_id}")
        step = moment["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all() or step.item() < 1:
                raise ValueError(f"Timestep recovery checkpoint AdamW step is malformed for parameter {parameter_id}")
        elif not isinstance(step, (int, float)) or not math.isfinite(step) or step < 1:
            raise ValueError(f"Timestep recovery checkpoint AdamW step is malformed for parameter {parameter_id}")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = moment[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.numel() == 0
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"Timestep recovery checkpoint AdamW {name} is malformed for parameter {parameter_id}")


def _validate_single_group_source_optimizer(optimizer_state: dict, model_state: dict[str, torch.Tensor],
                                            expected_names: list[str]) -> dict[str, int]:
    """Map a legacy one-group state to stable serialized trainable names.

    The step-1500 checkpoint predates named optimizer groups.  Its parameter
    IDs are positional serialization IDs, so the only safe migration is to
    prove the original trainable-state ordering and map each ID to its stable
    parameter name before constructing the two new groups.
    """
    groups, moments = optimizer_state.get("param_groups"), optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(moments, dict):
        raise ValueError("ControlInput source checkpoint must contain exactly one complete AdamW group")
    group = groups[0]
    if (group.get("lr") != CONTROLINPUT_BRANCH_LORA_LR
            or tuple(group.get("betas", ())) != (0.9, 0.99)
            or group.get("eps") != 1e-8 or group.get("weight_decay") != 0.0):
        raise ValueError("ControlInput source AdamW settings differ from the completed LR-only run")
    identifiers = group.get("params")
    if (not isinstance(identifiers, list) or len(identifiers) != len(expected_names)
            or len(set(identifiers)) != len(identifiers) or set(identifiers) != set(moments)):
        raise ValueError("ControlInput source optimizer has unmapped, duplicated, or missing parameter state")
    if list(model_state) != expected_names:
        raise ValueError("ControlInput source trainable tensor order cannot prove the optimizer state mapping")
    mapping = dict(zip(expected_names, identifiers))
    for name, identifier in mapping.items():
        moment = moments[identifier]
        if not isinstance(moment, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(moment):
            raise ValueError(f"ControlInput source optimizer state is incomplete for {name}")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            tensor = moment[moment_name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != model_state[name].shape
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"ControlInput source optimizer {moment_name} is invalid for {name}")
        step = moment["step"]
        numeric_step = float(step.item()) if isinstance(step, torch.Tensor) and step.numel() == 1 else step
        if not isinstance(numeric_step, (int, float)) or not math.isfinite(numeric_step) or numeric_step < 1:
            raise ValueError(f"ControlInput source optimizer step is invalid for {name}")
    return mapping


def _migrate_controlinput_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                          source_optimizer_state: dict, source_model_state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Transfer every legacy AdamW moment by stable name into two exact groups."""
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    lora, control = _controlinput_named_groups(named)
    expected_names = [name for name, _ in named]
    source_name_to_id = _validate_single_group_source_optimizer(
        source_optimizer_state, source_model_state, expected_names,
    )
    source_moments = source_optimizer_state["state"]
    exported = optimizer.state_dict()
    target_name_to_id: dict[str, int] = {}
    for live_group, serialized_group in zip(optimizer.param_groups, exported["param_groups"]):
        for parameter, identifier in zip(live_group["params"], serialized_group["params"]):
            name = next((candidate for candidate, candidate_parameter in named if candidate_parameter is parameter), None)
            if name is None or name in target_name_to_id:
                raise ValueError("ControlInput optimizer target parameter mapping is ambiguous")
            target_name_to_id[name] = identifier
    if set(target_name_to_id) != set(source_name_to_id):
        raise ValueError("ControlInput optimizer migration leaves trainable parameters unmapped")
    migrated_state = {target_name_to_id[name]: copy.deepcopy(source_moments[source_name_to_id[name]])
                      for name in expected_names}
    optimizer.load_state_dict({"state": migrated_state, "param_groups": exported["param_groups"]})
    _assert_controlinput_optimizer(optimizer, require_active_lrs=False)
    for name, parameter in named:
        restored = optimizer.state[parameter]
        source = source_moments[source_name_to_id[name]]
        if (not torch.equal(restored["exp_avg"], source["exp_avg"].to(restored["exp_avg"].device))
                or not torch.equal(restored["exp_avg_sq"], source["exp_avg_sq"].to(restored["exp_avg_sq"].device))
                or float(restored["step"].item() if isinstance(restored["step"], torch.Tensor) else restored["step"])
                != float(source["step"].item() if isinstance(source["step"], torch.Tensor) else source["step"])):
            raise ValueError(f"ControlInput optimizer state migration failed for {name}")
    return source_name_to_id


def _restore_controlinput_source_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                       scheduler: "OptimizerStepWarmup", state: dict) -> tuple[int, int, int, object | None]:
    """Restore exact step-1500 state while changing only optimizer LR grouping."""
    load_trainable_state_dict(model, state["model"])
    _migrate_controlinput_optimizer_state(model, optimizer, state["optimizer"], state["model"])
    scheduler.load_state_dict({
        "step_count": state["scheduler"]["step_count"],
        "base_lrs": [CONTROLINPUT_BRANCH_LORA_LR, CONTROLINPUT_BRANCH_CONTROL_LR],
        "warmup_steps": state["scheduler"]["warmup_steps"],
    })
    if scheduler.step_count != state["global_step"]:
        raise ValueError("ControlInput scheduler progress was not preserved from the source checkpoint")
    _assert_controlinput_optimizer(optimizer)
    _restore_rng(state["rng"])
    return state["global_step"], state["epoch"], state["batch_position"], state.get("flow_generator_state")


def validate_timestep_branch_recovery_state(path: Path, state: dict, source_state: dict) -> TrainConfig:
    """Accept only a complete continuation checkpoint from this exact experiment.

    This deliberately is not a general arbitrary-checkpoint resume mechanism:
    it proves the recovered state has the fixed 1500→1800 configuration and
    progression before any training-side write can occur.
    """
    expected_cfg = timestep_branch_config_from_source_state(source_state)
    expected_name = f"step_{state['global_step']:06d}.pt"
    if path.name != expected_name:
        raise ValueError(f"Timestep recovery checkpoint filename must be {expected_name}, got {path.name}")
    if not TIMESTEP_BRANCH_SOURCE_STEP < state["global_step"] < TIMESTEP_BRANCH_TARGET_STEP:
        raise ValueError("Timestep recovery checkpoint step is outside the unfinished 1500→1800 interval")
    recovered_cfg = train_config_from_checkpoint_values(state["config"])
    if asdict(recovered_cfg) != asdict(expected_cfg):
        raise ValueError("Timestep recovery checkpoint config is not identical to the fixed continuation experiment")
    assert_timestep_branch_output_namespace(recovered_cfg)
    scheduler = state["scheduler"]
    if (scheduler.get("step_count") != state["global_step"]
            or scheduler.get("warmup_steps") != expected_cfg.warmup_steps
            or list(scheduler.get("base_lrs", [])) != [expected_cfg.lr]):
        raise ValueError("Timestep recovery checkpoint scheduler progress or LR is incompatible")
    _assert_finite_adamw_state(state["optimizer"], expected_cfg.lr)
    rng = state["rng"]
    if not isinstance(rng.get("python"), tuple) or not isinstance(rng.get("numpy"), tuple):
        raise ValueError("Timestep recovery checkpoint CPU RNG state is malformed")
    _assert_byte_rng_state(rng.get("torch"), "torch RNG state")
    _assert_byte_rng_state(rng.get("cuda"), "CUDA RNG state", allow_list=True)
    _assert_byte_rng_state(state["flow_generator_state"], "flow_generator_state")
    source_model, recovered_model = source_state["model"], state["model"]
    if source_model.keys() != recovered_model.keys() or any(
            not isinstance(source_model[key], torch.Tensor)
            or not isinstance(recovered_model[key], torch.Tensor)
            or recovered_model[key].shape != source_model[key].shape
            or recovered_model[key].numel() == 0
            or not torch.isfinite(recovered_model[key]).all()
            for key in source_model):
        raise ValueError("Timestep recovery checkpoint trainable-model state differs from its immutable source")
    if state["epoch"] < source_state["epoch"] or state["batch_position"] < 0:
        raise ValueError("Timestep recovery checkpoint data-progress state is malformed")
    return recovered_cfg


def resolve_timestep_branch_recovery_checkpoint(source_state: dict) -> tuple[Path, dict, TrainConfig]:
    """Choose the newest semantically valid local checkpoint in this run only."""
    directory = timestep_branch_checkpoint_dir()
    failures: list[str] = []
    for path in sorted(directory.glob("step_*.pt"), reverse=True):
        try:
            state = load_training_state(path)
            cfg = validate_timestep_branch_recovery_state(path, state, source_state)
            return path, state, cfg
        except ValueError as error:
            failures.append(f"{path.name}: {error}")
    detail = "; ".join(failures) if failures else "no step_*.pt files"
    raise FileNotFoundError(f"No valid local timestep-continuation recovery checkpoint found in {directory}: {detail}")


def _controlinput_named_groups(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """Return the exact disjoint LoRA and ControlInputLayer trainable groups."""
    named = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    control = [(name, parameter) for name, parameter in named if name.startswith("first.")]
    lora = [(name, parameter) for name, parameter in named if not name.startswith("first.")]
    control_names, lora_names = {name for name, _ in control}, {name for name, _ in lora}
    if control_names != {"first.weight", "first.bias"}:
        raise ValueError(f"ControlInputLayer trainables must be exactly first.weight/first.bias, got {sorted(control_names)}")
    if not lora or any(not (name.endswith(".A") or name.endswith(".B")) for name, _ in lora):
        raise ValueError("Unexpected non-ControlInputLayer trainable parameter")
    if control_names & lora_names or len(named) != len(control) + len(lora):
        raise ValueError("Trainable parameter grouping is duplicated or incomplete")
    return lora, control


def _controlinput_state_names(model_state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Classify serialized trainable tensors with the same strict contract."""
    fake_named = []
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"Serialized trainable state is invalid for {name}")
        parameter = torch.nn.Parameter(torch.empty_like(tensor), requires_grad=True)
        fake_named.append((name, parameter))
    lora, control = _controlinput_named_groups(fake_named)
    return [name for name, _ in lora], [name for name, _ in control]


def controlinput_optimizer_group_summary(model: torch.nn.Module) -> dict[str, object]:
    lora, control = _controlinput_named_groups(model.named_parameters())
    all_ids = [id(parameter) for _, parameter in lora + control]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != expected_ids:
        raise AssertionError("Every trainable parameter must appear exactly once in a ControlInput optimizer group")
    return {
        "lora_tensor_count": len(lora), "control_input_tensor_count": len(control),
        "lora_scalar_count": sum(parameter.numel() for _, parameter in lora),
        "control_input_scalar_count": sum(parameter.numel() for _, parameter in control),
        "duplicated_parameter_count": len(all_ids) - len(set(all_ids)),
        "assigned_trainable_tensor_count": len(all_ids),
    }


def _assert_controlinput_optimizer(optimizer: torch.optim.Optimizer, *, require_active_lrs: bool = True) -> None:
    if len(optimizer.param_groups) != 2:
        raise AssertionError("ControlInput continuation requires exactly two AdamW parameter groups")
    expected = (("lora", CONTROLINPUT_BRANCH_LORA_LR), ("control_input", CONTROLINPUT_BRANCH_CONTROL_LR))
    observed_ids: list[int] = []
    for group, (name, learning_rate) in zip(optimizer.param_groups, expected):
        if (group.get("group_name") != name or (require_active_lrs and group.get("lr") != learning_rate)
                or tuple(group.get("betas", ())) != (0.9, 0.99)
                or group.get("eps") != 1e-8 or group.get("weight_decay") != 0.0):
            raise AssertionError(f"ControlInput optimizer group {name} does not have the fixed AdamW settings")
        observed_ids.extend(id(parameter) for parameter in group["params"])
    if len(observed_ids) != len(set(observed_ids)):
        raise AssertionError("ControlInput optimizer parameter groups overlap")


def controlinput_branch_config_from_source_state(source_state: dict) -> TrainConfig:
    """Derive the strict 1500→2800 experiment without changing sampler/data settings."""
    if source_state["global_step"] != CONTROLINPUT_BRANCH_SOURCE_STEP:
        raise ValueError("ControlInput continuation requires embedded global_step 1500")
    source_cfg = train_config_from_checkpoint_values(source_state["config"])
    if (source_cfg.run_name != CONTROLINPUT_BRANCH_SOURCE_RUN or source_cfg.lr != CONTROLINPUT_BRANCH_LORA_LR
            or source_cfg.max_steps != CONTROLINPUT_BRANCH_SOURCE_STEP
            or source_cfg.hf_repo_id != LR_BRANCH_SOURCE_HF_REPO):
        raise ValueError("ControlInput continuation source must be the exact completed LR-only step-1500 run")
    if not source_cfg.allow_extended_training or source_cfg.warmup_steps >= CONTROLINPUT_BRANCH_SOURCE_STEP:
        raise ValueError("ControlInput continuation source is incompatible with preserved warmup/authorization")
    if source_cfg.control_input_lr is not None or source_cfg.control_input_lr_multiplier != 1.0:
        raise ValueError("ControlInput continuation source must not already use a ControlInput LR override")
    if any(getattr(source_cfg, key) != default for key, default in
           (("timestep_aux_prob", 0.0), ("timestep_aux_min", 0.0), ("timestep_aux_max", 1.0))):
        raise ValueError("ControlInput continuation must use the original source timestep sampler, not lowmid20")
    scheduler = source_state["scheduler"]
    if (scheduler.get("step_count") != CONTROLINPUT_BRANCH_SOURCE_STEP
            or scheduler.get("warmup_steps") != source_cfg.warmup_steps
            or list(scheduler.get("base_lrs", [])) != [CONTROLINPUT_BRANCH_LORA_LR]):
        raise ValueError("ControlInput continuation source scheduler is not the exact post-warmup LR-only state")
    _controlinput_state_names(source_state["model"])
    _validate_single_group_source_optimizer(source_state["optimizer"], source_state["model"], list(source_state["model"]))
    branch_cfg = replace(
        source_cfg,
        ckpt_dir=LR_BRANCH_CHECKPOINT_DIR,
        run_name=CONTROLINPUT_BRANCH_RUN,
        max_steps=CONTROLINPUT_BRANCH_TARGET_STEP,
        save_every=100,
        hf_mirror_every_steps=100,
        metrics_jsonl_path=str(controlinput_branch_checkpoint_dir() / "metrics.jsonl"),
        source_checkpoint=str(CONTROLINPUT_BRANCH_SOURCE_CHECKPOINT),
        source_step=CONTROLINPUT_BRANCH_SOURCE_STEP,
        target_step=CONTROLINPUT_BRANCH_TARGET_STEP,
        control_input_lr=CONTROLINPUT_BRANCH_CONTROL_LR,
        control_input_lr_multiplier=CONTROLINPUT_BRANCH_CONTROL_LR_MULTIPLIER,
        required_checkpoint_steps=CONTROLINPUT_BRANCH_REQUIRED_CHECKPOINT_STEPS,
    )
    assert_controlinput_branch_output_namespace(branch_cfg)
    return branch_cfg


def assert_controlinput_branch_output_namespace(cfg: TrainConfig) -> None:
    expected = controlinput_branch_checkpoint_dir()
    actual = Path(cfg.ckpt_dir) / str(cfg.run_name)
    if (cfg.run_name != CONTROLINPUT_BRANCH_RUN or actual != expected
            or cfg.metrics_jsonl_path != str(expected / "metrics.jsonl")
            or cfg.hf_repo_id != LR_BRANCH_SOURCE_HF_REPO
            or cfg.source_checkpoint != str(CONTROLINPUT_BRANCH_SOURCE_CHECKPOINT)
            or cfg.source_step != CONTROLINPUT_BRANCH_SOURCE_STEP
            or cfg.target_step != CONTROLINPUT_BRANCH_TARGET_STEP
            or cfg.lr != CONTROLINPUT_BRANCH_LORA_LR
            or cfg.control_input_lr != CONTROLINPUT_BRANCH_CONTROL_LR
            or cfg.control_input_lr_multiplier != CONTROLINPUT_BRANCH_CONTROL_LR_MULTIPLIER
            or tuple(cfg.required_checkpoint_steps) != CONTROLINPUT_BRANCH_REQUIRED_CHECKPOINT_STEPS
            or cfg.save_every != 100 or cfg.hf_mirror_every_steps != 100):
        raise AssertionError("ControlInput continuation output namespace/configuration is not exact")
    if actual == CONTROLINPUT_BRANCH_SOURCE_CHECKPOINT.parent or CONTROLINPUT_BRANCH_SOURCE_RUN in actual.parts:
        raise AssertionError("ControlInput continuation must never write into the LR-only source namespace")
    if any(getattr(cfg, key) != default for key, default in
           (("timestep_aux_prob", 0.0), ("timestep_aux_min", 0.0), ("timestep_aux_max", 1.0))):
        raise AssertionError("ControlInput continuation must keep the original timestep sampler active")


def assert_controlinput_branch_destination_is_new() -> None:
    """Refuse a fresh launch that could overwrite an existing experiment checkpoint."""
    destination = controlinput_branch_checkpoint_dir()
    existing = sorted(destination.glob("step_*.pt")) if destination.is_dir() else []
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing ControlInput continuation checkpoints in {destination}")


def resolve_controlinput_branch_source_checkpoint() -> Path:
    """Validate only the named local step-1500 file against its exact HF marker."""
    source = CONTROLINPUT_BRANCH_SOURCE_CHECKPOINT
    if source.name != "step_001500.pt" or not source.is_file():
        raise FileNotFoundError(f"Required exact local source checkpoint is unavailable: {source}")
    with tempfile.TemporaryDirectory(prefix="krea2-controlinput-source-marker-") as temporary:
        validated = validated_local_checkpoint_for_hf_step(
            checkpoint=source, repo_id=LR_BRANCH_SOURCE_HF_REPO,
            run_name=CONTROLINPUT_BRANCH_SOURCE_RUN, step=CONTROLINPUT_BRANCH_SOURCE_STEP,
            marker_download_dir=temporary,
        )
    if validated != source:
        raise ValueError("Exact LR-only step-1500 source failed HF completion-marker/SHA/schema validation")
    state = load_training_state(source)
    if state["global_step"] != CONTROLINPUT_BRANCH_SOURCE_STEP:
        raise ValueError("Exact ControlInput source checkpoint embedded global_step is not 1500")
    return source


def _validate_controlinput_recovery_optimizer(optimizer_state: dict, model_state: dict[str, torch.Tensor]) -> None:
    lora_names, control_names = _controlinput_state_names(model_state)
    groups, moments = optimizer_state.get("param_groups"), optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(moments, dict):
        raise ValueError("ControlInput recovery checkpoint must contain exactly two AdamW groups")
    expected_groups = (("lora", CONTROLINPUT_BRANCH_LORA_LR, lora_names),
                       ("control_input", CONTROLINPUT_BRANCH_CONTROL_LR, control_names))
    seen: set[int] = set()
    for group, (group_name, learning_rate, names) in zip(groups, expected_groups):
        identifiers = group.get("params")
        if (group.get("group_name") != group_name or group.get("lr") != learning_rate
                or tuple(group.get("betas", ())) != (0.9, 0.99) or group.get("eps") != 1e-8
                or group.get("weight_decay") != 0.0 or not isinstance(identifiers, list)
                or len(identifiers) != len(names) or len(set(identifiers)) != len(identifiers)):
            raise ValueError(f"ControlInput recovery {group_name} AdamW group is incompatible")
        if seen & set(identifiers):
            raise ValueError("ControlInput recovery optimizer groups duplicate parameters")
        seen.update(identifiers)
        for name, identifier in zip(names, identifiers):
            moment = moments.get(identifier)
            if not isinstance(moment, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(moment):
                raise ValueError(f"ControlInput recovery optimizer state is incomplete for {name}")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                value = moment[moment_name]
                if (not isinstance(value, torch.Tensor) or value.shape != model_state[name].shape
                        or not torch.isfinite(value).all()):
                    raise ValueError(f"ControlInput recovery optimizer {moment_name} is invalid for {name}")
            step = moment["step"]
            number = float(step.item()) if isinstance(step, torch.Tensor) and step.numel() == 1 else step
            if not isinstance(number, (float, int)) or not math.isfinite(number) or number < 1:
                raise ValueError(f"ControlInput recovery optimizer step is invalid for {name}")
    if seen != set(moments):
        raise ValueError("ControlInput recovery optimizer has unmapped state")


def validate_controlinput_branch_recovery_state(path: Path, state: dict, source_state: dict) -> TrainConfig:
    expected_cfg = controlinput_branch_config_from_source_state(source_state)
    expected_name = f"step_{state['global_step']:06d}.pt"
    if path.name != expected_name:
        raise ValueError(f"ControlInput recovery filename must be {expected_name}, got {path.name}")
    if not CONTROLINPUT_BRANCH_SOURCE_STEP < state["global_step"] < CONTROLINPUT_BRANCH_TARGET_STEP:
        raise ValueError("ControlInput recovery step is outside the unfinished 1500→2800 interval")
    recovered_cfg = train_config_from_checkpoint_values(state["config"])
    if asdict(recovered_cfg) != asdict(expected_cfg):
        raise ValueError("ControlInput recovery checkpoint config is not identical to the fixed experiment")
    assert_controlinput_branch_output_namespace(recovered_cfg)
    scheduler = state["scheduler"]
    if (scheduler.get("step_count") != state["global_step"]
            or scheduler.get("warmup_steps") != expected_cfg.warmup_steps
            or list(scheduler.get("base_lrs", [])) != [CONTROLINPUT_BRANCH_LORA_LR, CONTROLINPUT_BRANCH_CONTROL_LR]):
        raise ValueError("ControlInput recovery scheduler progress was restarted or is incompatible")
    _validate_controlinput_recovery_optimizer(state["optimizer"], state["model"])
    rng = state["rng"]
    if not isinstance(rng.get("python"), tuple) or not isinstance(rng.get("numpy"), tuple):
        raise ValueError("ControlInput recovery CPU RNG state is malformed")
    _assert_byte_rng_state(rng.get("torch"), "torch RNG state")
    _assert_byte_rng_state(rng.get("cuda"), "CUDA RNG state", allow_list=True)
    _assert_byte_rng_state(state["flow_generator_state"], "flow_generator_state")
    source_model, recovered_model = source_state["model"], state["model"]
    if source_model.keys() != recovered_model.keys() or any(
            not isinstance(recovered_model[key], torch.Tensor)
            or recovered_model[key].shape != source_model[key].shape
            or not torch.isfinite(recovered_model[key]).all()
            for key in source_model):
        raise ValueError("ControlInput recovery trainable-model state differs from its exact source contract")
    if state["epoch"] < source_state["epoch"] or state["batch_position"] < 0:
        raise ValueError("ControlInput recovery data-progress state is malformed")
    return recovered_cfg


def resolve_controlinput_branch_recovery_checkpoint(source_state: dict) -> tuple[Path, dict, TrainConfig]:
    """Choose only the newest valid local checkpoint from this isolated run."""
    directory = controlinput_branch_checkpoint_dir()
    failures: list[str] = []
    for path in sorted(directory.glob("step_*.pt"), reverse=True):
        try:
            state = load_training_state(path)
            return path, state, validate_controlinput_branch_recovery_state(path, state, source_state)
        except ValueError as error:
            failures.append(f"{path.name}: {error}")
    detail = "; ".join(failures) if failures else "no step_*.pt files"
    raise FileNotFoundError(f"No valid local ControlInput continuation recovery checkpoint found in {directory}: {detail}")


def controlinput_preflight_summary(source_state: dict) -> dict[str, object]:
    """CPU-only proof of source identity/config/state before any training is launched."""
    cfg = controlinput_branch_config_from_source_state(source_state)
    lora_names, control_names = _controlinput_state_names(source_state["model"])
    mapping = _validate_single_group_source_optimizer(
        source_state["optimizer"], source_state["model"], list(source_state["model"]),
    )
    return {
        "run_name": cfg.run_name,
        "source_checkpoint": cfg.source_checkpoint,
        "source_embedded_global_step": source_state["global_step"],
        "target_step": cfg.target_step,
        "destination_root": str(controlinput_branch_checkpoint_dir()),
        "hf_repo_id": cfg.hf_repo_id,
        "hf_destination_namespace": f"{cfg.run_name}/full/",
        "checkpoint_interval": cfg.save_every,
        "required_milestones": list(cfg.required_checkpoint_steps),
        "lora_lr": cfg.lr,
        "control_input_lr": cfg.control_input_lr,
        "control_input_lr_multiplier": cfg.control_input_lr_multiplier,
        "lora_optimizer_tensor_count": len(lora_names),
        "control_input_optimizer_tensor_count": len(control_names),
        "duplicated_optimizer_parameter_count": 0,
        "assigned_trainable_tensor_count": len(mapping),
        "optimizer_state_mapped_tensor_count": len(mapping),
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.99], "eps": 1e-8, "weight_decay": 0.0},
        "scheduler_source_step": source_state["scheduler"]["step_count"],
        "rng_present": True,
        "data_state": {"epoch": source_state["epoch"], "batch_position": source_state["batch_position"]},
        "flow_generator_state_present": isinstance(source_state.get("flow_generator_state"), torch.Tensor),
        "timestep_sampler": "original/source sampler",
        "lowmid20_enabled": False,
        "control_dropout": cfg.control_dropout,
        "caption_dropout": cfg.caption_dropout,
    }


def resolve_lr_branch_source_checkpoint() -> Path:
    """Download only the completion-marked, checksum-validated archive step 900."""
    source = validated_hf_checkpoint_for_step(
        repo_id=LR_BRANCH_SOURCE_HF_REPO,
        run_name=LR_BRANCH_SOURCE_RUN,
        step=LR_BRANCH_SOURCE_STEP,
        download_dir=lr_branch_checkpoint_dir() / "source-step-900-recovery",
    )
    if source is None:
        raise FileNotFoundError(
            "Required exact source checkpoint is unavailable: "
            f"{LR_BRANCH_SOURCE_HF_REPO}/{LR_BRANCH_SOURCE_RUN}/full/step_000900.pt"
        )
    state = load_training_state(source)
    if source.name != "step_000900.pt" or state["global_step"] != LR_BRANCH_SOURCE_STEP:
        raise ValueError("Exact source checkpoint identity validation failed")
    return source


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
    if cfg.control_input_lr is None:
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    else:
        if (cfg.lr != CONTROLINPUT_BRANCH_LORA_LR
                or cfg.control_input_lr != CONTROLINPUT_BRANCH_CONTROL_LR
                or cfg.control_input_lr_multiplier != CONTROLINPUT_BRANCH_CONTROL_LR_MULTIPLIER
                or cfg.control_input_lr != cfg.lr * cfg.control_input_lr_multiplier):
            raise ValueError("ControlInput optimizer LR configuration is not exactly 5e-5 / 1e-4 (2x)")
        lora, control = _controlinput_named_groups(model.named_parameters())
        optimizer = torch.optim.AdamW([
            {"params": [parameter for _, parameter in lora], "lr": cfg.lr, "group_name": "lora"},
            {"params": [parameter for _, parameter in control], "lr": cfg.control_input_lr, "group_name": "control_input"},
        ], betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    if cfg.control_input_lr is not None:
        _assert_controlinput_optimizer(optimizer)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    frozen_ids = {id(parameter) for parameter in model.parameters() if not parameter.requires_grad}
    if optimizer_ids != expected_ids or optimizer_ids & frozen_ids:
        raise AssertionError("Optimizer must contain exactly ControlInput and intended LoRA tensors")
    if sum(len(group["params"]) for group in optimizer.param_groups) != len(optimizer_ids):
        raise AssertionError("Optimizer contains a duplicated parameter")
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


def step_mirror_requested(global_step: int, every_steps: int) -> bool:
    """Whether an exact completed local checkpoint is required on HF at this step."""
    return every_steps > 0 and global_step > 0 and global_step % every_steps == 0


def restore_full_training_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                scheduler: OptimizerStepWarmup, state: dict, *,
                                learning_rate_override: float | None = None) -> tuple[int, int, int, object | None]:
    """Restore every resumable component before optionally changing the effective LR."""
    load_trainable_state_dict(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    global_step, epoch, batch_position = state["global_step"], state["epoch"], state["batch_position"]
    _restore_rng(state["rng"])
    if learning_rate_override is not None:
        apply_resumed_learning_rate_override(
            optimizer, scheduler, learning_rate_override, resumed_global_step=global_step,
        )
    return global_step, epoch, batch_position, state.get("flow_generator_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors")
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning",
                        help="Complete persistent Qwen conditioning root; pass --online-text-conditioning only for diagnostics")
    parser.add_argument("--online-text-conditioning", action="store_true", help="Diagnostic fallback that loads Qwen; never production mode")
    parser.add_argument("--checkpoint-dir", default="/lambda/nfs/adhit/krea2-pose/checkpoints")
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-extended-training", action="store_true",
                        help="explicitly authorize a bounded run beyond the 100-step Gate-F entry limit")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
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
    parser.add_argument("--hf-mirror-every-steps", type=int, default=0,
                        help="mirror each exact saved checkpoint divisible by N; 0 disables (default: 0)")
    parser.add_argument("--timestep-aux-prob", type=float, default=0.0,
                        help="probability of the bounded auxiliary pre-shift timestep sampler")
    parser.add_argument("--timestep-aux-min", type=float, default=0.0,
                        help="inclusive lower bound of the auxiliary pre-shift timestep sampler")
    parser.add_argument("--timestep-aux-max", type=float, default=1.0,
                        help="exclusive upper bound of the auxiliary pre-shift timestep sampler")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--lr-branch-900-to-1500", action="store_true",
                        help="resume only exact HF pose-learning-1500 step 900 into an isolated 5e-5 LR branch")
    parser.add_argument("--timestep-lowmid-1500-to1800", action="store_true",
                        help="resume only exact local completed LR=5e-5 step 1500 into the isolated timestep-only branch")
    parser.add_argument("--recover-timestep-lowmid-1500-to1800", action="store_true",
                        help="resume only the newest fully validated local checkpoint from the fixed timestep-only branch")
    parser.add_argument("--controlinput-lr2x-1500-to2800", action="store_true",
                        help="resume only exact validated LR-only step 1500 into the isolated 2x ControlInputLayer-LR run")
    parser.add_argument("--recover-controlinput-lr2x-1500-to2800", action="store_true",
                        help="resume only the newest fully validated local checkpoint from the fixed ControlInputLayer-LR run")
    args = parser.parse_args()
    if sum((args.lr_branch_900_to_1500, args.timestep_lowmid_1500_to1800,
            args.recover_timestep_lowmid_1500_to1800, args.controlinput_lr2x_1500_to2800,
            args.recover_controlinput_lr2x_1500_to2800)) > 1:
        parser.error("continuation branch selectors are mutually exclusive")
    if args.lr_branch_900_to_1500:
        if args.resume:
            parser.error("--lr-branch-900-to-1500 resolves its exact HF source itself; do not pass --resume")
        return args
    if args.timestep_lowmid_1500_to1800:
        if args.resume:
            parser.error("--timestep-lowmid-1500-to1800 resolves its exact local source itself; do not pass --resume")
        if (args.timestep_aux_prob, args.timestep_aux_min, args.timestep_aux_max) != (0.0, 0.0, 1.0):
            parser.error("--timestep-lowmid-1500-to1800 fixes its audited auxiliary sampler; do not pass timestep auxiliary flags")
        return args
    if args.recover_timestep_lowmid_1500_to1800:
        if args.resume:
            parser.error("--recover-timestep-lowmid-1500-to1800 selects its validated local checkpoint itself; do not pass --resume")
        if (args.timestep_aux_prob, args.timestep_aux_min, args.timestep_aux_max) != (0.0, 0.0, 1.0):
            parser.error("--recover-timestep-lowmid-1500-to1800 fixes its audited auxiliary sampler; do not pass timestep auxiliary flags")
        return args
    if args.controlinput_lr2x_1500_to2800 or args.recover_controlinput_lr2x_1500_to2800:
        selector = "--controlinput-lr2x-1500-to2800" if args.controlinput_lr2x_1500_to2800 else "--recover-controlinput-lr2x-1500-to2800"
        if args.resume:
            parser.error(f"{selector} resolves only its fixed exact checkpoint; do not pass --resume")
        if (args.timestep_aux_prob, args.timestep_aux_min, args.timestep_aux_max) != (0.0, 0.0, 1.0):
            parser.error(f"{selector} keeps the original source timestep sampler; do not pass timestep auxiliary flags")
        return args
    if args.max_steps is None or args.run_name is None:
        parser.error("--run-name and --max-steps are required unless --lr-branch-900-to-1500 is used")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.max_steps > 100 and not args.allow_extended_training:
        parser.error("Gate-F entry point permits only explicit bounded 1..100-step runs; pass --allow-extended-training to exceed 100")
    if (args.microbatch_size is None or args.gradient_accumulation_steps is None
            or args.microbatch_size < 1 or args.gradient_accumulation_steps < 1):
        parser.error("batch settings must be positive")
    if args.gradient_checkpointing_blocks is not None and not 0 <= args.gradient_checkpointing_blocks <= 28:
        parser.error("--gradient-checkpointing-blocks must be in [0, 28]")
    if args.hf_mirror_every_seconds < 0: parser.error("--hf-mirror-every-seconds must be non-negative")
    if args.hf_mirror_every_steps < 0: parser.error("--hf-mirror-every-steps must be non-negative")
    if args.save_every < 1: parser.error("--save-every must be positive")
    if args.hf_mirror_every_steps:
        if not args.hf_repo_id:
            parser.error("--hf-mirror-every-steps requires --hf-repo-id")
        if args.hf_mirror_every_steps % args.save_every:
            parser.error("--hf-mirror-every-steps must be divisible by --save-every so exact local checkpoints exist")
    if not 0.0 <= args.timestep_aux_prob <= 1.0:
        parser.error("--timestep-aux-prob must be in [0, 1]")
    if args.timestep_aux_prob and not 0.0 < args.timestep_aux_min < args.timestep_aux_max < 1.0:
        parser.error("enabled timestep auxiliary support must satisfy 0 < --timestep-aux-min < --timestep-aux-max < 1")
    return args


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    gradient_checkpointing_blocks = (
        args.gradient_checkpointing_blocks
        if args.gradient_checkpointing_blocks is not None
        else (28 if args.gradient_checkpointing else 0)
    )
    return TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root, ckpt_dir=args.checkpoint_dir, run_name=args.run_name,
                       max_steps=args.max_steps, microbatch_size=args.microbatch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
                       allow_extended_training=args.allow_extended_training,
                       max_grad_norm=args.max_grad_norm, validation_batches=args.validation_batches, val_every=args.val_every, save_every=args.save_every, diagnostics_every=args.diagnostics_every,
                       compile=args.compile, gradient_checkpointing=gradient_checkpointing_blocks > 0,
                       gradient_checkpointing_blocks=gradient_checkpointing_blocks,
                       wandb_enabled=not args.no_wandb, wandb_mode=args.wandb_mode,
                       metrics_jsonl_path=str(Path(args.checkpoint_dir) / args.run_name / "metrics.jsonl"),
                       hf_repo_id=args.hf_repo_id, hf_push_every_seconds=args.hf_mirror_every_seconds,
                       hf_mirror_every_steps=args.hf_mirror_every_steps,
                       timestep_aux_prob=args.timestep_aux_prob, timestep_aux_min=args.timestep_aux_min,
                       timestep_aux_max=args.timestep_aux_max)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Run Gate-F smoke from the GH200 host shell with CUDA visible")
    branch_source_path = (resolve_lr_branch_source_checkpoint() if args.lr_branch_900_to_1500 else
                          resolve_controlinput_branch_source_checkpoint() if args.controlinput_lr2x_1500_to2800 or args.recover_controlinput_lr2x_1500_to2800 else
                          resolve_timestep_branch_source_checkpoint() if args.timestep_lowmid_1500_to1800 or args.recover_timestep_lowmid_1500_to1800 else None)
    immutable_timestep_source_state = (load_training_state(branch_source_path)
                                       if args.recover_timestep_lowmid_1500_to1800 else None)
    immutable_controlinput_source_state = (load_training_state(branch_source_path)
                                           if args.recover_controlinput_lr2x_1500_to2800 else None)
    branch_source_state = (immutable_timestep_source_state if immutable_timestep_source_state is not None else
                           immutable_controlinput_source_state if immutable_controlinput_source_state is not None else
                           load_training_state(branch_source_path) if branch_source_path is not None else None)
    recovery_path = recovery_state = None
    if args.recover_timestep_lowmid_1500_to1800:
        recovery_path, recovery_state, cfg = resolve_timestep_branch_recovery_checkpoint(immutable_timestep_source_state)
    elif args.recover_controlinput_lr2x_1500_to2800:
        recovery_path, recovery_state, cfg = resolve_controlinput_branch_recovery_checkpoint(immutable_controlinput_source_state)
    else:
        cfg = (lr_branch_config_from_source_state(branch_source_state) if args.lr_branch_900_to_1500 else
               controlinput_branch_config_from_source_state(branch_source_state) if args.controlinput_lr2x_1500_to2800 else
               timestep_branch_config_from_source_state(branch_source_state) if args.timestep_lowmid_1500_to1800 else
               config_from_args(args))
    if args.lr_branch_900_to_1500:
        assert_lr_branch_output_namespace(cfg)
    if args.timestep_lowmid_1500_to1800 or args.recover_timestep_lowmid_1500_to1800:
        assert_timestep_branch_output_namespace(cfg)
    if args.controlinput_lr2x_1500_to2800 or args.recover_controlinput_lr2x_1500_to2800:
        assert_controlinput_branch_output_namespace(cfg)
    if args.controlinput_lr2x_1500_to2800:
        assert_controlinput_branch_destination_is_new()
    set_seed(cfg.seed); device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats()
    print(f"effective_batch={effective_batch_size(cfg.microbatch_size, cfg.gradient_accumulation_steps)} (microbatch × accumulation × world_size=1)", flush=True)
    print(f"runtime: compile={cfg.compile} gradient_checkpointing_blocks={cfg.gradient_checkpointing_blocks} "
          f"allow_extended_training={cfg.allow_extended_training}", flush=True)
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
    if recovery_state is not None:
        global_step, epoch, batch_position, resume_generator_state = restore_full_training_state(
            model, optimizer, scheduler, recovery_state,
        )
        if args.recover_controlinput_lr2x_1500_to2800:
            _assert_controlinput_optimizer(optimizer)
            print(f"[controlinput-recovery] restored {recovery_path} at step {global_step}; "
                  f"lora_lr={CONTROLINPUT_BRANCH_LORA_LR} control_input_lr={CONTROLINPUT_BRANCH_CONTROL_LR}; "
                  "AdamW/scheduler/RNG/data/flow-generator state retained", flush=True)
        else:
            _assert_exact_learning_rate(optimizer, LR_BRANCH_LEARNING_RATE)
            print(f"[timestep-recovery] restored newest validated local checkpoint {recovery_path} at optimizer step {global_step}; "
                  f"AdamW/scheduler/RNG/data state retained; LR={LR_BRANCH_LEARNING_RATE}; "
                  f"aux_prob={cfg.timestep_aux_prob} aux_pre_shift=[{cfg.timestep_aux_min}, {cfg.timestep_aux_max})", flush=True)
    elif branch_source_state is not None:
        if args.controlinput_lr2x_1500_to2800:
            global_step, epoch, batch_position, resume_generator_state = _restore_controlinput_source_state(
                model, optimizer, scheduler, branch_source_state,
            )
            _assert_controlinput_optimizer(optimizer)
            if global_step != CONTROLINPUT_BRANCH_SOURCE_STEP or scheduler.step_count != global_step:
                raise AssertionError("ControlInput continuation did not restore exact global/scheduler progress")
            print(f"[controlinput-branch] restored exact HF-validated local source {branch_source_path} at step {global_step}; "
                  "all AdamW moments and counters migrated by stable trainable parameter name; "
                  f"lora_lr={CONTROLINPUT_BRANCH_LORA_LR} control_input_lr={CONTROLINPUT_BRANCH_CONTROL_LR}", flush=True)
        else:
            global_step, epoch, batch_position, resume_generator_state = restore_full_training_state(
                model, optimizer, scheduler, branch_source_state,
                learning_rate_override=LR_BRANCH_LEARNING_RATE if args.lr_branch_900_to_1500 else None,
            )
            _assert_exact_learning_rate(optimizer, LR_BRANCH_LEARNING_RATE)
        if args.lr_branch_900_to_1500:
            print(f"[lr-branch] restored exact validated source {branch_source_path} at optimizer step {global_step}; "
                  f"AdamW/scheduler/RNG/data state retained and effective LR fixed at {LR_BRANCH_LEARNING_RATE}", flush=True)
        elif not args.controlinput_lr2x_1500_to2800:
            if global_step != TIMESTEP_BRANCH_SOURCE_STEP or scheduler.step_count != global_step:
                raise AssertionError("Timestep branch did not restore exact global/scheduler progress")
            print(f"[timestep-branch] restored exact validated source {branch_source_path} at optimizer step {global_step}; "
                  f"AdamW/scheduler/RNG/data state retained; LR={LR_BRANCH_LEARNING_RATE}; "
                  f"aux_prob={cfg.timestep_aux_prob} aux_pre_shift=[{cfg.timestep_aux_min}, {cfg.timestep_aux_max})", flush=True)
            print(f"[timestep-branch] verify scheduler_step={scheduler.step_count} warmup_steps={scheduler.warmup_steps} "
                  f"save_every={cfg.save_every} required_checkpoints={TIMESTEP_BRANCH_REQUIRED_CHECKPOINT_STEPS} "
                  f"hf={cfg.hf_repo_id}/{cfg.run_name}/full/ wandb_run={cfg.run_name}", flush=True)
    elif args.resume:
        resume_path = (resolve_auto_resume(checkpoint_dir=cfg.ckpt_dir, run_name=cfg.run_name,
                                           repo_id=cfg.hf_repo_id,
                                           remote_download_dir=Path(cfg.ckpt_dir) / cfg.run_name / "hf_recovery")
                       if args.resume == "auto" else Path(args.resume))
        if resume_path is None: raise FileNotFoundError("--resume auto found no valid local or HF full checkpoint")
        state = load_training_state(resume_path)
        global_step, epoch, batch_position, resume_generator_state = restore_full_training_state(model, optimizer, scheduler, state)
        print(f"[resume] loaded validated full checkpoint {resume_path} at optimizer step {global_step} "
              f"(epoch={epoch}, batch_position={batch_position})", flush=True)
    telemetry = TrainingTelemetry(cfg, cfg.run_name)
    mirror = HFTrainingCheckpointMirror(
        repo_id=cfg.hf_repo_id, run_name=cfg.run_name,
        interval_seconds=cfg.hf_push_every_seconds, telemetry=telemetry,
        protected_milestone_steps=(CONTROLINPUT_BRANCH_REQUIRED_CHECKPOINT_STEPS
                                   if args.controlinput_lr2x_1500_to2800 or args.recover_controlinput_lr2x_1500_to2800
                                   else ()),
    )
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
            if args.controlinput_lr2x_1500_to2800 or args.recover_controlinput_lr2x_1500_to2800:
                _assert_controlinput_optimizer(optimizer)
            elif branch_source_state is not None:
                _assert_exact_learning_rate(optimizer, LR_BRANCH_LEARNING_RATE)
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
                if step_mirror_requested(global_step, cfg.hf_mirror_every_steps):
                    mirror.submit(path, reason="step")
                mirror.maybe_submit(path)
    finally:
        mirror.stop()
        telemetry.close()


if __name__ == "__main__": main()
