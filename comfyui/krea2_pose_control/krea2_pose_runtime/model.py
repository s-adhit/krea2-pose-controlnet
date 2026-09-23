"""Release-scoped Krea-2 Turbo control-model construction.

This is a package-relative version of the project model surgery; it contains
no dependency on the source training checkout.
"""
from .base_model.k2_lora import (
    K2_RAW_CONFIG, LORA_TARGETS, build_control_model,
    trainable_params as _trainable_params,
    trainable_state_dict as _trainable_state_dict,
    load_trainable_state_dict as _load_trainable_state_dict,
)

POSE_CONFIG = K2_RAW_CONFIG


def build_turbo_pose_model(turbo_ckpt: str, rank: int = 64, alpha: float | None = 64,
                           device: str = "cuda"):
    return build_control_model(turbo_ckpt, rank=rank, alpha=alpha, device=device,
                               checkpoint_name="Turbo")


def trainable_params(model):
    return _trainable_params(model)


def trainable_state_dict(model):
    return _trainable_state_dict(model)


def load_trainable_state_dict(model, state):
    return _load_trainable_state_dict(model, state)
