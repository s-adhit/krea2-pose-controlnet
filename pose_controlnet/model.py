"""model.py — pose-controlnet model construction on top of Krea-2 RAW.

Only file that touches base_model/ (mmdit.py / k2_lora.py) — the real
pretrained 13B DiT, required to load your actual checkpoint weights.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base_model"))

from k2_lora import (  # noqa: E402
    K2_RAW_CONFIG,
    LORA_TARGETS,
    audit_control_model,
    build_control_model,
    inspect_raw_checkpoint,
    trainable_params as _trainable_params,
    trainable_state_dict as _trainable_state_dict,
    load_trainable_state_dict as _load_trainable_state_dict,
)

POSE_CONFIG = K2_RAW_CONFIG


def build_pose_model(raw_ckpt: str, rank: int, alpha: float | None, device: str):
    print(f"[model] building pose-controlnet: rank={rank} alpha={alpha or rank}")
    model = build_control_model(raw_ckpt, rank=rank, alpha=alpha, device=device)
    n_train = sum(p.numel() for p in _trainable_params(model))
    print(f"[model] trainable params: {n_train / 1e6:.1f}M")
    return model


def trainable_params(model):
    return _trainable_params(model)


def trainable_state_dict(model):
    return _trainable_state_dict(model)


def load_trainable_state_dict(model, sd):
    return _load_trainable_state_dict(model, sd)
