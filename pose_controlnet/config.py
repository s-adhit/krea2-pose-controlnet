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
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    caption_dropout: float = 0.1
    control_dropout: float = 0.0

    # schedule (resolution-aware flow-matching shift)
    mu_x1: float = 256.0
    mu_y1: float = 0.5
    mu_x2: float = 6400.0
    mu_y2: float = 1.15

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
    hf_push_every_seconds: int = 7200

    # eval
    eval_steps: int = 8
    eval_guidance: float = 3.5
