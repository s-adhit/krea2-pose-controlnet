"""Inspect or mirror one marker-backed full checkpoint; never starts training."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.checkpointing import (HFTrainingCheckpointMirror, _sha256, load_training_state,
                                           validated_hf_checkpoint_for_step)

def remote_status(repo_id: str, run_name: str, checkpoint: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    state = load_training_state(checkpoint); remote = f"{run_name}/full/{checkpoint.name}"; marker_name = remote + ".complete.json"; api = HfApi()
    files = set(api.list_repo_files(repo_id, repo_type="model")); result = {"checkpoint": remote, "checkpoint_exists": remote in files, "marker_exists": marker_name in files, "valid_complete": False}
    if marker_name in files and remote in files:
        with tempfile.TemporaryDirectory() as directory:
            marker = json.loads(Path(hf_hub_download(repo_id=repo_id, repo_type="model", filename=marker_name, local_dir=directory)).read_text())
            remote_local = Path(hf_hub_download(repo_id=repo_id, repo_type="model", filename=remote, local_dir=directory))
            try:
                remote_state = load_training_state(remote_local)
                result["valid_complete"] = (marker == {"format": 1, "checkpoint": remote, "sha256": _sha256(checkpoint), "global_step": state["global_step"]}
                                            and _sha256(remote_local) == marker["sha256"] and remote_state["global_step"] == marker["global_step"])
            except (KeyError, ValueError):
                result["valid_complete"] = False
    return result


def remote_checkpoints(repo_id: str, run_name: str) -> list[str]:
    """List only completion-marked full checkpoints in one isolated namespace."""
    from huggingface_hub import HfApi
    prefix = f"{run_name.strip('/')}/full/"
    files = set(HfApi().list_repo_files(repo_id, repo_type="model"))
    checkpoints = []
    for filename in sorted(files):
        if filename.startswith(prefix) and filename.endswith(".pt") and f"{filename}.complete.json" in files:
            checkpoints.append(filename)
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("status", "mirror", "list", "fetch")); parser.add_argument("--repo-id", required=True); parser.add_argument("--run-name", default="pose-learning-500"); parser.add_argument("--checkpoint"); parser.add_argument("--step", type=int); parser.add_argument("--download-dir"); args = parser.parse_args()
    if args.action == "list":
        print(json.dumps({"repo_id": args.repo_id, "run_name": args.run_name, "checkpoints": remote_checkpoints(args.repo_id, args.run_name)}, indent=2, sort_keys=True)); return
    if args.action == "fetch":
        if args.step is None or not args.download_dir:
            parser.error("--step and --download-dir are required for fetch")
        checkpoint = validated_hf_checkpoint_for_step(repo_id=args.repo_id, run_name=args.run_name,
                                                      step=args.step, download_dir=args.download_dir)
        if checkpoint is None:
            raise RuntimeError("remote checkpoint is absent, incomplete, or failed checksum/state validation")
        print(checkpoint); return
    if not args.checkpoint:
        parser.error("--checkpoint is required for status and mirror")
    path = Path(args.checkpoint); status = remote_status(args.repo_id, args.run_name, path); print(json.dumps(status, indent=2, sort_keys=True))
    if args.action == "mirror" and not status["valid_complete"]:
        mirror = HFTrainingCheckpointMirror(repo_id=args.repo_id, run_name=args.run_name)
        if not mirror._upload(path): raise RuntimeError(f"mirror failed: {mirror.last_error}")
        status = remote_status(args.repo_id, args.run_name, path); print(json.dumps(status, indent=2, sort_keys=True))
        if not status["valid_complete"]: raise RuntimeError("remote marker verification failed")
if __name__ == "__main__": main()
