"""checkpointing.py — guardrail 3: local save/resume + wall-clock HF Hub mirror."""
import os
import threading

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def save_checkpoint(model, ckpt_dir: str, run_name: str, step: int, cfg,
                     trainable_state_dict_fn):
    os.makedirs(os.path.join(ckpt_dir, run_name), exist_ok=True)
    sd = trainable_state_dict_fn(model)
    fname = f"step_{step:06d}.safetensors"
    path = os.path.join(ckpt_dir, run_name, fname)
    save_file(sd, path, metadata={
        "step": str(step),
        "rank": str(cfg.rank),
        "seed": str(cfg.seed),
        "control_type": "pose",
        "base": "krea/Krea-2-Raw",
    })
    print(f"[ckpt] saved {path}")
    return path


def resume_checkpoint(model, resume_path: str, load_trainable_state_dict_fn) -> int:
    load_trainable_state_dict_fn(model, load_file(resume_path))
    with safe_open(resume_path, framework="pt") as f:
        meta = f.metadata() or {}
    step = int(meta.get("step", 0))
    print(f"[ckpt] resumed from {resume_path} at step {step}")
    return step


def prune_local_checkpoints(run_dir: str, keep_last: int = 2):
    """Delete older local checkpoints once newer ones exist and have
    (presumably) already been mirrored to HF. Keeps the last `keep_last`."""
    ckpts = sorted(f for f in os.listdir(run_dir) if f.endswith(".safetensors"))
    for old in ckpts[:-keep_last]:
        path = os.path.join(run_dir, old)
        try:
            os.remove(path)
            print(f"[ckpt] pruned {old}")
        except OSError as e:
            print(f"[ckpt] prune failed for {old}: {e}")


class HFCheckpointMirror:
    """Background thread: pushes the newest local checkpoint to a HF Hub
    model repo every `interval_s` seconds, independent of local save cadence."""

    def __init__(self, ckpt_dir: str, run_name: str, repo_id: str, interval_s: int):
        self.run_dir = os.path.join(ckpt_dir, run_name)
        self.repo_id = repo_id
        self.interval_s = interval_s
        self.run_name = run_name
        self._stop = threading.Event()
        self._thread = None
        self._last_pushed = None

    def start(self):
        from huggingface_hub import HfApi
        self._api = HfApi()
        self._api.create_repo(self.repo_id, exist_ok=True, repo_type="model")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[hf-mirror] watching {self.run_dir} -> {self.repo_id} "
              f"every {self.interval_s}s")

    def _push_latest(self):
        if not os.path.isdir(self.run_dir):
            return
        ckpts = sorted(f for f in os.listdir(self.run_dir) if f.endswith(".safetensors"))
        if not ckpts or ckpts[-1] == self._last_pushed:
            return
        latest = ckpts[-1]
        local_path = os.path.join(self.run_dir, latest)
        print(f"[hf-mirror] pushing {latest} -> {self.repo_id}/{self.run_name}/{latest}")
        try:
            self._api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"{self.run_name}/{latest}",
                repo_id=self.repo_id,
                repo_type="model",
            )
            self._last_pushed = latest
            print(f"[hf-mirror] done: {latest}")
            prune_local_checkpoints(self.run_dir, keep_last=2)
        except Exception as e:
            print(f"[hf-mirror] push FAILED for {latest}: {e} (retry next interval)")

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(self.interval_s)
            if self._stop.is_set():
                break
            self._push_latest()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        self._push_latest()