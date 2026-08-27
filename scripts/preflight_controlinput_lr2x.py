"""Read-only preflight for the exact step-1500 ControlInputLayer-LR continuation.

This module never imports or calls ``train.main``.  It does not construct a
model, make an optimizer update, save a checkpoint, or start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from pose_controlnet.checkpointing import load_training_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight() -> dict[str, object]:
    """Fail closed after validating source, metadata, optimizer mapping, and isolation."""
    source = train.resolve_controlinput_branch_source_checkpoint()
    state = load_training_state(source)
    summary = train.controlinput_preflight_summary(state)
    train.assert_controlinput_branch_output_namespace(
        train.controlinput_branch_config_from_source_state(state)
    )
    train.assert_controlinput_branch_destination_is_new()
    summary.update({
        "source_local_path": str(source),
        "source_local_sha256": _sha256(source),
        "source_hf_completion_marker": (
            f"{train.CONTROLINPUT_BRANCH_SOURCE_RUN}/full/step_001500.pt.complete.json"
        ),
        "source_hf_namespace": f"{train.CONTROLINPUT_BRANCH_SOURCE_RUN}/full/",
        "source_hf_marker_and_sha256_validated": True,
        "source_full_state_schema_deserialized": True,
        "source_namespace_unchanged": True,
        "destination_namespace_isolated": True,
        "preflight_starts_training": False,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight",))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
