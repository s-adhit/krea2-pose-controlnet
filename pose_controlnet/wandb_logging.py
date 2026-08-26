"""Failure-isolated experiment telemetry for Pose Control-LoRA training.

The local JSONL stream is the durable, dependency-free metrics record. W&B is
an optional mirror: no import, initialization, network, or logging exception
from it is allowed to interrupt training.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WANDB_ENTITY = "adhit-projects"
DEFAULT_WANDB_PROJECT = "Krea-2-PoseControl-Lora"
_SECRET_MARKERS = ("api_key", "apikey", "token", "password", "secret")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_value(value: Any) -> Any:
    """Convert common scalar values to JSON while rejecting unsafe objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    return repr(value)


def _safe_config(cfg: Any) -> dict[str, Any]:
    values = asdict(cfg) if is_dataclass(cfg) else vars(cfg)
    return {
        key: _json_value(value)
        for key, value in values.items()
        if not any(marker in key.lower() for marker in _SECRET_MARKERS)
    }


class TrainingTelemetry:
    """Project-owned local telemetry with an optional W&B mirror.

    Future training code should create one instance per run and call the named
    methods below. All telemetry operations are best-effort; their return value
    only reports whether the local JSONL write succeeded.
    """

    def __init__(
        self,
        cfg: Any,
        run_name: str,
        *,
        metrics_path: str | Path | None = None,
        wandb_module: Any | None = None,
    ) -> None:
        self.run_name = run_name
        self.local_errors: list[str] = []
        self.wandb_errors: list[str] = []
        self.run: Any | None = None
        self._wandb_module = wandb_module
        self.wandb_enabled = bool(getattr(cfg, "wandb_enabled", True))
        self.wandb_enabled = self.wandb_enabled and not _env_flag("WANDB_DISABLED")
        self.wandb_mode = os.getenv("WANDB_MODE", getattr(cfg, "wandb_mode", "online"))
        self.wandb_entity = os.getenv(
            "WANDB_ENTITY", getattr(cfg, "wandb_entity", DEFAULT_WANDB_ENTITY)
        )
        self.wandb_project = os.getenv(
            "WANDB_PROJECT", getattr(cfg, "wandb_project", DEFAULT_WANDB_PROJECT)
        )
        configured_path = getattr(cfg, "metrics_jsonl_path", None)
        self.metrics_path = Path(metrics_path or configured_path or "runs/metrics.jsonl")
        self._open_local_stream()
        self._init_wandb(_safe_config(cfg))

    def _open_local_stream(self) -> None:
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._local_stream = self.metrics_path.open("a", encoding="utf-8", buffering=1)
        except Exception as error:  # logging must never terminate training
            self._local_stream = None
            self.local_errors.append(f"local metrics unavailable: {error}")

    def _init_wandb(self, config: dict[str, Any]) -> None:
        if not self.wandb_enabled:
            return
        try:
            module = self._wandb_module
            if module is None:
                import wandb as module
            self.run = module.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.run_name,
                config=config,
                mode=self.wandb_mode,
            )
        except Exception as error:  # includes missing package, login, and network failures
            self.run = None
            self.wandb_errors.append(f"W&B init unavailable: {error}")

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> bool:
        """Write metrics locally then attempt the remote mirror independently."""
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **_json_value(metrics)}
        if step is not None:
            payload["global_step"] = int(step)
        local_ok = self._write_local(payload)
        if self.run is not None:
            try:
                self.run.log(dict(metrics), step=step)
            except Exception as error:
                self.wandb_errors.append(f"W&B log unavailable: {error}")
        return local_ok

    def _write_local(self, payload: Mapping[str, Any]) -> bool:
        if self._local_stream is None:
            return False
        try:
            self._local_stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            self._local_stream.flush()
            return True
        except Exception as error:
            self.local_errors.append(f"local metrics write failed: {error}")
            return False

    def log_train(
        self, *, loss: float, learning_rate: float, global_grad_norm: float,
        sec_per_step: float, samples_per_second: float, step: int,
    ) -> bool:
        return self.log({"train/loss": loss, "train/learning_rate": learning_rate,
                         "train/global_grad_norm": global_grad_norm,
                         "performance/sec_per_step": sec_per_step,
                         "performance/samples_per_second": samples_per_second}, step=step)

    def log_validation_flow_loss(self, loss: float, *, step: int) -> bool:
        return self.log({"validation/flow_loss": loss}, step=step)

    def log_control_diagnostics(
        self, *, control_latent_rms: float, control_latent_std: float,
        control_input_grad_norms: Mapping[str, float], lora_grad_norms: Mapping[str, float],
        step: int,
    ) -> bool:
        metrics: dict[str, Any] = {"diagnostics/control_latent_rms": control_latent_rms,
                                   "diagnostics/control_latent_std": control_latent_std}
        metrics.update({f"diagnostics/control_input_grad_norm/{key}": value
                        for key, value in control_input_grad_norms.items()})
        metrics.update({f"diagnostics/lora_grad_norm/{key}": value
                        for key, value in lora_grad_norms.items()})
        return self.log(metrics, step=step)

    def log_cuda_memory(self, *, allocated_bytes: int, reserved_bytes: int,
                        peak_allocated_bytes: int, step: int) -> bool:
        return self.log({"cuda/allocated_bytes": allocated_bytes,
                         "cuda/reserved_bytes": reserved_bytes,
                         "cuda/peak_allocated_bytes": peak_allocated_bytes}, step=step)

    def log_checkpoint(self, *, checkpoint_step: int, checkpoint_time: str, step: int) -> bool:
        return self.log({"checkpoint/step": checkpoint_step,
                         "checkpoint/time": checkpoint_time}, step=step)

    def log_hf_upload(self, *, success: bool, remote_checkpoint_age_seconds: float,
                      step: int, uploaded_checkpoint_step: int | None = None,
                      error_status: str | None = None,
                      mirror_reason: str | None = None) -> bool:
        metrics = {"hf/upload_success": success,
                   "hf/remote_checkpoint_age_seconds": remote_checkpoint_age_seconds}
        if uploaded_checkpoint_step is not None:
            metrics["hf/uploaded_checkpoint_step"] = uploaded_checkpoint_step
        if error_status is not None:
            metrics["hf/error_status"] = error_status
        if mirror_reason is not None:
            metrics["hf/mirror_reason"] = mirror_reason
        return self.log(metrics, step=step)

    def log_diagnostic_images(self, images: Mapping[str, Any], *, step: int) -> bool:
        """Mirror sparse images to W&B; record their names locally, never image bytes."""
        local_ok = self.log({"diagnostics/images": sorted(images)}, step=step)
        if self.run is not None:
            try:
                self.run.log(dict(images), step=step)
            except Exception as error:
                self.wandb_errors.append(f"W&B image log unavailable: {error}")
        return local_ok

    def close(self) -> None:
        if self._local_stream is not None:
            try:
                self._local_stream.close()
            except Exception as error:
                self.local_errors.append(f"local metrics close failed: {error}")
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                self.wandb_errors.append(f"W&B finish unavailable: {error}")


def init_wandb(cfg: Any, run_name: str) -> TrainingTelemetry:
    """Compatibility factory; returns failure-isolated project telemetry."""
    return TrainingTelemetry(cfg, run_name)
