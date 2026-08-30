"""config.py — single source of truth for all pose-controlnet hyperparameters."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # paths
    raw_ckpt: str
    shard_dir: str
    ckpt_dir: str = "/ckpts"
    run_name: str | None = None

    # model
    rank: int = 64
    alpha: float | None = None

    # optimization
    lr: float = 1e-4
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_steps: int = 6000
    allow_extended_training: bool = False
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    caption_dropout: float = 0.1
    control_dropout: float = 0.0
    compile: bool = False
    # Opt-in only: the benchmark must prove GPU support and throughput before
    # this backend is selected for a production run.  AdamW hyperparameters
    # and parameter membership remain identical.
    fused_adamw: bool = False
    gradient_checkpointing: bool = False
    # Checkpoint the first N main transformer blocks in execution order.
    gradient_checkpointing_blocks: int = 0

    # schedule (resolution-aware flow-matching shift)
    mu_x1: float = 256.0
    mu_y1: float = 0.5
    mu_x2: float = 6400.0
    mu_y2: float = 1.15
    # Optional, deliberately narrow timestep-exposure ablation.  Disabled
    # defaults retain the original sigmoid(N(0, 1)) sampler byte-for-byte in
    # its random-draw path.
    timestep_aux_prob: float = 0.0
    timestep_aux_min: float = 0.0
    timestep_aux_max: float = 1.0

    # cadence
    log_every: int = 10
    val_every: int = 250
    save_every: int = 500
    checkpoint_every_seconds: int = 3600
    validation_batches: int = 8
    diagnostics_every: int = 50

    # guardrails
    seed: int = 42
    wandb_entity: str = "adhit-projects"
    wandb_project: str = "Krea-2-PoseControl-Lora"
    wandb_enabled: bool = True
    wandb_mode: str = "online"
    metrics_jsonl_path: str = "runs/metrics.jsonl"
    hf_repo_id: str = ""
    hf_push_every_seconds: int = 3600
    # Zero keeps the existing wall-clock-only HF mirror behavior.
    hf_mirror_every_steps: int = 0

    # Immutable identity metadata for the narrowly scoped step-1500
    # ControlInputLayer-LR continuation.  Disabled defaults keep historical
    # checkpoints schema-compatible while allowing continuation checkpoints to
    # prove their experiment identity on recovery.
    source_checkpoint: str = ""
    source_step: int = 0
    target_step: int = 0
    control_input_lr: float | None = None
    control_input_lr_multiplier: float = 1.0
    required_checkpoint_steps: tuple[int, ...] = ()

    # eval
    eval_steps: int = 8
    eval_guidance: float = 3.5
