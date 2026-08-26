"""Atomic full-training checkpoints and best-effort Hugging Face mirroring."""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import torch


REQUIRED_TRAINING_STATE_KEYS = frozenset({
    "model", "optimizer", "scheduler", "global_step", "epoch", "batch_position",
    "rng", "flow_generator_state", "config",
})


def _validate_training_state(state: object, source: str | Path) -> dict:
    if not isinstance(state, dict) or not REQUIRED_TRAINING_STATE_KEYS.issubset(state):
        missing = REQUIRED_TRAINING_STATE_KEYS - set(state) if isinstance(state, dict) else REQUIRED_TRAINING_STATE_KEYS
        raise ValueError(f"Invalid full training checkpoint {source}; missing={sorted(missing)}")
    if not isinstance(state["model"], dict) or not isinstance(state["optimizer"], dict):
        raise ValueError(f"Invalid full training checkpoint {source}; model/optimizer state is malformed")
    if not isinstance(state["scheduler"], dict) or "step_count" not in state["scheduler"]:
        raise ValueError(f"Invalid full training checkpoint {source}; scheduler state is malformed")
    if any(not isinstance(state[key], int) or state[key] < 0 for key in ("global_step", "epoch", "batch_position")):
        raise ValueError(f"Invalid full training checkpoint {source}; progress state is malformed")
    rng = state["rng"]
    if not isinstance(rng, dict) or not {"python", "numpy", "torch", "cuda"}.issubset(rng):
        raise ValueError(f"Invalid full training checkpoint {source}; RNG state is malformed")
    if not isinstance(state["config"], dict):
        raise ValueError(f"Invalid full training checkpoint {source}; config is malformed")
    return state


def save_training_state(path: str | Path, state: dict) -> Path:
    """Atomically publish a deserialize-validated complete training checkpoint."""
    _validate_training_state(state, "in-memory state")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        load_training_state(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_training_state(path: str | Path) -> dict:
    checkpoint = Path(path)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Could not deserialize full training checkpoint {checkpoint}: {error}") from error
    return _validate_training_state(state, checkpoint)


def newest_valid_local_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the newest valid full checkpoint; names are never trusted alone."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.glob("step_*.pt"), reverse=True):
        try:
            load_training_state(candidate)
            return candidate
        except ValueError:
            continue
    return None


def prune_local_full_checkpoints(run_dir: str | Path, keep_last: int = 2) -> None:
    """Retain at least the two newest deserialize-valid local full checkpoints."""
    if keep_last < 2:
        raise ValueError("keep_last must preserve at least two local checkpoints")
    valid = []
    for candidate in sorted(Path(run_dir).glob("step_*.pt"), reverse=True):
        try:
            load_training_state(candidate)
            valid.append(candidate)
        except ValueError:
            continue
    for stale in valid[keep_last:]:
        try:
            stale.unlink()
        except OSError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_error(error: Exception) -> str:
    """Keep diagnostics useful without allowing ambient credentials into logs."""
    message = f"{type(error).__name__}: {error}"
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.getenv(name)
        if value:
            message = message.replace(value, "[redacted]")
    return re.sub(r"(?i)(token|authorization)=([^\s,&]+)", r"\1=[redacted]", message)


class HFTrainingCheckpointMirror:
    """Background mirror using ambient HF authentication and completion markers."""
    def __init__(self, *, repo_id: str, run_name: str, interval_seconds: float = 3600,
                 max_attempts: int = 3, retry_base_seconds: float = 5.0,
                 telemetry: object | None = None, api: object | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if interval_seconds < 0 or max_attempts < 1 or retry_base_seconds < 0:
            raise ValueError("Invalid HF mirror cadence or retry configuration")
        self.repo_id, self.run_name = repo_id, run_name
        self.interval_seconds, self.max_attempts = interval_seconds, max_attempts
        self.retry_base_seconds, self.telemetry, self._api, self._sleep = retry_base_seconds, telemetry, api, sleep
        self._pending: queue.Queue[Path | None] = queue.Queue(); self._queued: set[Path] = set(); self._lock = threading.Lock()
        self._thread: threading.Thread | None = None; self._last_submission = float("-inf")
        self.last_success_time: float | None = None; self.last_success_step: int | None = None; self.last_error: str | None = None

    def start(self) -> None:
        if not self.repo_id or self._thread is not None: return
        self._thread = threading.Thread(target=self._worker, name="hf-checkpoint-mirror", daemon=True); self._thread.start()

    def maybe_submit(self, checkpoint: str | Path) -> bool:
        path = Path(checkpoint)
        try: load_training_state(path)
        except ValueError as error:
            self._record(False, None, f"local validation failed: {error}"); return False
        now = time.monotonic()
        with self._lock:
            if now - self._last_submission < self.interval_seconds or path in self._queued: return False
            self._last_submission = now; self._queued.add(path)
        self._pending.put(path); return True

    def _remote_paths(self, checkpoint: Path) -> tuple[str, str]:
        prefix = f"{self.run_name}/full"; remote = f"{prefix}/{checkpoint.name}"
        return remote, f"{remote}.complete.json"

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi()
        return self._api

    def _upload(self, checkpoint: Path) -> bool:
        """Synchronous primitive for the background worker and focused tests."""
        try: state, checksum = load_training_state(checkpoint), _sha256(checkpoint)
        except ValueError as error:
            self._record(False, None, f"local validation failed: {error}"); return False
        remote_checkpoint, marker_path = self._remote_paths(checkpoint); error_text = ""
        for attempt in range(self.max_attempts):
            try:
                api = self._get_api(); api.create_repo(self.repo_id, repo_type="model", private=True, exist_ok=True)
                api.upload_file(path_or_fileobj=str(checkpoint), path_in_repo=remote_checkpoint, repo_id=self.repo_id,
                                repo_type="model", commit_message=f"Mirror full checkpoint step {state['global_step']}")
                marker = json.dumps({"format": 1, "checkpoint": remote_checkpoint, "sha256": checksum,
                                     "global_step": state["global_step"]}, sort_keys=True).encode()
                api.upload_file(path_or_fileobj=io.BytesIO(marker), path_in_repo=marker_path, repo_id=self.repo_id,
                                repo_type="model", commit_message=f"Mark full checkpoint step {state['global_step']} complete")
                self.last_success_time, self.last_success_step, self.last_error = time.monotonic(), state["global_step"], None
                prune_local_full_checkpoints(checkpoint.parent)
                self._record(True, state["global_step"], None); return True
            except Exception as error:
                error_text = _safe_error(error)
                if attempt + 1 < self.max_attempts: self._sleep(self.retry_base_seconds * (2 ** attempt))
        self._record(False, state["global_step"], error_text); return False

    def _record(self, success: bool, step: int | None, error: str | None) -> None:
        self.last_error = error
        if self.telemetry is not None:
            age = 0.0 if self.last_success_time is None else max(0.0, time.monotonic() - self.last_success_time)
            self.telemetry.log_hf_upload(success=success, uploaded_checkpoint_step=step,
                                         remote_checkpoint_age_seconds=age, error_status=error, step=step or 0)

    def _worker(self) -> None:
        while True:
            checkpoint = self._pending.get()
            if checkpoint is None: return
            try: self._upload(checkpoint)
            finally:
                with self._lock: self._queued.discard(checkpoint)

    def stop(self) -> None:
        if self._thread is not None:
            self._pending.put(None); self._thread.join(timeout=30)


def newest_valid_hf_checkpoint(*, repo_id: str, run_name: str, download_dir: str | Path,
                               api: object | None = None, download_fn: Callable[..., str] | None = None) -> Path | None:
    """Download the newest marker-backed checkpoint that passes checksum and state validation."""
    if not repo_id: return None
    try:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        if download_fn is None:
            from huggingface_hub import hf_hub_download
            download_fn = hf_hub_download
        markers = sorted((name for name in api.list_repo_files(repo_id, repo_type="model")
                          if name.startswith(f"{run_name}/full/") and name.endswith(".complete.json")), reverse=True)
    except Exception: return None
    target = Path(download_dir); target.mkdir(parents=True, exist_ok=True)
    for marker_name in markers:
        try:
            marker_local = Path(download_fn(repo_id=repo_id, repo_type="model", filename=marker_name,
                                            local_dir=str(target)))
            marker = json.loads(marker_local.read_text(encoding="utf-8")); remote_name = marker["checkpoint"]
            if not isinstance(remote_name, str) or not remote_name.startswith(f"{run_name}/full/"): continue
            checkpoint = Path(download_fn(repo_id=repo_id, repo_type="model", filename=remote_name,
                                           local_dir=str(target)))
            if _sha256(checkpoint) != marker["sha256"]: continue
            state = load_training_state(checkpoint)
            if state["global_step"] != marker["global_step"]: continue
            return checkpoint
        except Exception: continue
    return None


def validated_hf_checkpoint_for_step(*, repo_id: str, run_name: str, step: int,
                                     download_dir: str | Path, api: object | None = None,
                                     download_fn: Callable[..., str] | None = None) -> Path | None:
    """Return one exact completion-marked HF checkpoint, or ``None``.

    This is deliberately stricter than newest-checkpoint recovery: evaluation
    comparisons must never silently replace an unavailable archived step with a
    different checkpoint.
    """
    if not repo_id or step < 0:
        return None
    remote = f"{run_name}/full/step_{step:06d}.pt"
    marker_name = f"{remote}.complete.json"
    try:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        if download_fn is None:
            from huggingface_hub import hf_hub_download
            download_fn = hf_hub_download
        files = set(api.list_repo_files(repo_id, repo_type="model"))
        if remote not in files or marker_name not in files:
            return None
        target = Path(download_dir)
        target.mkdir(parents=True, exist_ok=True)
        marker_local = Path(download_fn(repo_id=repo_id, repo_type="model", filename=marker_name,
                                        local_dir=str(target)))
        marker = json.loads(marker_local.read_text(encoding="utf-8"))
        expected = {"format": 1, "checkpoint": remote, "sha256": marker.get("sha256"), "global_step": step}
        if marker != expected or not isinstance(marker["sha256"], str):
            return None
        checkpoint = Path(download_fn(repo_id=repo_id, repo_type="model", filename=remote,
                                      local_dir=str(target)))
        if _sha256(checkpoint) != marker["sha256"]:
            return None
        state = load_training_state(checkpoint)
        return checkpoint if state["global_step"] == step else None
    except Exception:
        return None


def resolve_auto_resume(*, checkpoint_dir: str | Path, run_name: str, repo_id: str,
                        remote_download_dir: str | Path) -> Path | None:
    local = newest_valid_local_checkpoint(Path(checkpoint_dir) / run_name)
    return local or newest_valid_hf_checkpoint(repo_id=repo_id, run_name=run_name, download_dir=remote_download_dir)
