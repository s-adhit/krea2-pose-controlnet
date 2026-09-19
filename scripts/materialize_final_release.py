#!/usr/bin/env python3
"""Materialize the frozen final mix-025 as a compact public safetensors artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_controlnet.release_artifact import (
    interpolate_model_tensors_fp32,
    load_release_artifact,
    save_release_artifact,
    sha256_file,
    write_provenance,
)


CONTRACT_PATH = ROOT / "docs/evaluation/release/final_release_v1.json"
FROZEN_CONTRACT_SHA256 = "9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b"
DEFAULT_OUTPUT = Path("/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors")


def _load_frozen_contract(path: Path) -> dict[str, Any]:
    if sha256_file(path) != FROZEN_CONTRACT_SHA256:
        raise ValueError(f"Frozen release contract SHA-256 mismatch: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    candidate = contract.get("candidate")
    interpolation = candidate.get("interpolation") if isinstance(candidate, dict) else None
    if (not isinstance(interpolation, dict) or contract.get("release_id") != "krea2-pose-control-lora-v1"
            or candidate.get("id") != "mix-025" or interpolation.get("alpha") != 0.25
            or interpolation.get("formula") != "(1 - alpha) * parent-4000 + alpha * finish-control-a4300"
            or interpolation.get("tensor_scope") != "state['model'] trainable control/LoRA tensors only"
            or interpolation.get("compute_dtype") != "float32"):
        raise ValueError("Frozen release contract does not define the exact mix-025 materialization")
    return contract


def _endpoint(contract: dict[str, Any], endpoint_id: str) -> dict[str, str]:
    endpoints = contract["candidate"]["interpolation"].get("endpoints", [])
    endpoint = next((row for row in endpoints if isinstance(row, dict) and row.get("id") == endpoint_id), None)
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("path"), str) or not isinstance(endpoint.get("sha256"), str):
        raise ValueError(f"Frozen release contract is missing endpoint {endpoint_id}")
    return {"id": endpoint_id, "path": endpoint["path"], "sha256": endpoint["sha256"]}


def _verified_endpoint_path(endpoint: dict[str, str]) -> Path:
    path = Path(endpoint["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Frozen endpoint is missing: {path}")
    if sha256_file(path) != endpoint["sha256"]:
        raise ValueError(f"Frozen endpoint SHA-256 mismatch: {path}")
    return path


def _load_verified_checkpoint(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError(f"Frozen endpoint is not a full checkpoint with model tensors: {path}")
    return state


def materialize(*, contract_path: Path = CONTRACT_PATH, output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Verify frozen inputs, blend only model tensors, publish, then reload/verify."""
    contract = _load_frozen_contract(contract_path)
    parent, finish = _endpoint(contract, "parent-4000"), _endpoint(contract, "finish-control-a4300")
    # Verify all three frozen inputs before either endpoint is deserialized or an output path is created.
    parent_path, finish_path = _verified_endpoint_path(parent), _verified_endpoint_path(finish)
    parent_state, finish_state = _load_verified_checkpoint(parent_path), _load_verified_checkpoint(finish_path)
    parent_raw = parent_state.get("config", {}).get("raw_ckpt")
    finish_raw = finish_state.get("config", {}).get("raw_ckpt")
    if not isinstance(parent_raw, str) or not parent_raw or parent_raw != finish_raw:
        raise ValueError("Frozen endpoints do not share exact Krea-2 Raw provenance")
    alpha = contract["candidate"]["interpolation"]["alpha"]
    model = interpolate_model_tensors_fp32(parent_state, finish_state, alpha)
    artifact = save_release_artifact(
        output, model, release_id=contract["release_id"], candidate=contract["candidate"]["id"], alpha=alpha,
        raw_checkpoint=parent_raw, release_contract_sha256=FROZEN_CONTRACT_SHA256,
    )
    artifact_sha256 = sha256_file(artifact)
    # Independently reload the serialized artifact and compare every saved tensor with FP32 source math.
    reloaded = load_release_artifact(artifact)
    if set(reloaded["model"]) != set(model):
        raise RuntimeError("Reloaded release artifact tensor keys differ from materialized model")
    for name, expected in model.items():
        observed = reloaded["model"][name]
        if observed.dtype != torch.float32 or not torch.equal(observed, expected):
            raise RuntimeError(f"Reloaded release artifact tensor differs from FP32 interpolation: {name}")
    provenance = {
        "release_id": contract["release_id"], "candidate": contract["candidate"]["id"], "alpha": alpha,
        "source_checkpoints": [parent, finish], "release_artifact_sha256": artifact_sha256,
        "tensor_count": len(model), "parameter_count": sum(t.numel() for t in model.values()),
        "serialization_format": "safetensors", "creation_method": "scripts/materialize_final_release.py",
    }
    provenance_path = artifact.with_suffix(artifact.suffix + ".provenance.json")
    write_provenance(provenance_path, provenance)
    return {"artifact": artifact, "provenance": provenance_path, **provenance}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return command


def main() -> None:
    args = parser().parse_args()
    result = materialize(contract_path=args.contract, output=args.output)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
