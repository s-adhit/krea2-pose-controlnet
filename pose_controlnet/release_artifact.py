"""Compact, strict release artifacts for Pose Control-LoRA inference."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from pose_controlnet.trainable_interpolation import validate_trainable_interpolation


RELEASE_ARTIFACT_FORMAT = "krea2-pose-control-lora-release"
RELEASE_ARTIFACT_FORMAT_VERSION = 1


class ReleaseArtifactError(ValueError):
    """Raised when a compact public release artifact is invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(state: Mapping[str, object], label: str) -> Mapping[str, object]:
    model = state.get("model")
    if not isinstance(model, Mapping):
        raise ReleaseArtifactError(f"{label} checkpoint has no model tensor mapping")
    return model


def interpolate_model_tensors_fp32(parent_state: Mapping[str, object], finish_state: Mapping[str, object],
                                   alpha: float) -> dict[str, torch.Tensor]:
    """Blend exactly the two full-checkpoint ``model`` mappings in FP32.

    No other full-training checkpoint field is read or emitted.  The strict
    mapping validator rejects missing keys, unexpected keys, non-tensors, and
    incompatible shapes before any output is written.
    """
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise ReleaseArtifactError(f"Interpolation alpha must be strictly between zero and one, got {alpha!r}")
    parent = _model_state(parent_state, "parent")
    finish = _model_state(finish_state, "finish")
    try:
        validate_trainable_interpolation(parent, finish)
    except ValueError as error:
        raise ReleaseArtifactError(str(error)) from error
    return {
        key: (
            parent[key].detach().to(device="cpu", dtype=torch.float32) * (1.0 - alpha)
            + finish[key].detach().to(device="cpu", dtype=torch.float32) * alpha
        ).contiguous()
        for key in sorted(parent)
    }


def _artifact_metadata(*, release_id: str, candidate: str, alpha: float, raw_checkpoint: str,
                       release_contract_sha256: str) -> dict[str, str]:
    if not all(isinstance(value, str) and value for value in (release_id, candidate, raw_checkpoint, release_contract_sha256)):
        raise ReleaseArtifactError("Release artifact metadata requires non-empty string provenance")
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise ReleaseArtifactError("Release artifact metadata alpha must be strictly between zero and one")
    return {
        "format": RELEASE_ARTIFACT_FORMAT,
        "format_version": str(RELEASE_ARTIFACT_FORMAT_VERSION),
        "release_id": release_id,
        "candidate": candidate,
        "alpha": repr(alpha),
        "raw_checkpoint": raw_checkpoint,
        "release_contract_sha256": release_contract_sha256,
    }


def _canonicalize_safetensors_header(path: Path) -> None:
    """Remove safetensors metadata HashMap ordering from the published bytes."""
    with path.open("r+b") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little")
        original = handle.read(header_size)
        try:
            parsed = json.loads(original)
        except json.JSONDecodeError as error:
            raise ReleaseArtifactError("Safetensors writer produced an invalid header") from error
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(canonical) > header_size:
            raise ReleaseArtifactError("Canonical safetensors header unexpectedly grew")
        handle.seek(8)
        handle.write(canonical)
        handle.write(b" " * (header_size - len(canonical)))


def save_release_artifact(path: str | Path, model: Mapping[str, torch.Tensor], *, release_id: str,
                          candidate: str, alpha: float, raw_checkpoint: str,
                          release_contract_sha256: str) -> Path:
    """Atomically publish a direct-tensor safetensors release without overwrite."""
    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ReleaseArtifactError("Compact release artifacts must use the .safetensors format")
    tensors = dict(sorted(model.items()))
    if not tensors or any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ReleaseArtifactError("Release artifact model must contain tensors only")
    if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in tensors.values()):
        raise ReleaseArtifactError("Release artifact tensors must be CPU float32 tensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _artifact_metadata(
        release_id=release_id, candidate=candidate, alpha=alpha, raw_checkpoint=raw_checkpoint,
        release_contract_sha256=release_contract_sha256,
    )
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        _canonicalize_safetensors_header(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite existing release artifact: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_release_artifact(path: str | Path) -> dict[str, Any]:
    """Reload and structurally validate a standalone compact release artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Release artifact is missing or not a file: {artifact}")
    if artifact.suffix != ".safetensors":
        raise ReleaseArtifactError("Release artifact must be a .safetensors file")
    try:
        with safe_open(artifact, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != RELEASE_ARTIFACT_FORMAT:
                raise ReleaseArtifactError("Unsupported release artifact format")
            if metadata.get("format_version") != str(RELEASE_ARTIFACT_FORMAT_VERSION):
                raise ReleaseArtifactError("Unsupported release artifact format version")
            required = ("release_id", "candidate", "alpha", "raw_checkpoint", "release_contract_sha256")
            if any(not isinstance(metadata.get(key), str) or not metadata[key] for key in required):
                raise ReleaseArtifactError("Release artifact metadata is incomplete")
            try:
                alpha = float(metadata["alpha"])
            except ValueError as error:
                raise ReleaseArtifactError("Release artifact alpha is invalid") from error
            if not 0.0 < alpha < 1.0:
                raise ReleaseArtifactError("Release artifact alpha is outside (0, 1)")
            model = {key: handle.get_tensor(key) for key in sorted(handle.keys())}
    except ReleaseArtifactError:
        raise
    except Exception as error:
        raise ReleaseArtifactError(f"Could not load release artifact {artifact}: {error}") from error
    if not model or any(value.dtype != torch.float32 or value.device.type != "cpu" for value in model.values()):
        raise ReleaseArtifactError("Release artifact tensors must be non-empty CPU float32 tensors")
    return {
        "format_version": RELEASE_ARTIFACT_FORMAT_VERSION,
        "release": {
            "release_id": metadata["release_id"], "candidate": metadata["candidate"], "alpha": alpha,
            "release_contract_sha256": metadata["release_contract_sha256"],
        },
        "config": {"raw_ckpt": metadata["raw_checkpoint"]},
        "model": model,
    }


def write_provenance(path: str | Path, provenance: Mapping[str, object]) -> Path:
    """Write deterministic human-readable provenance once, without overwrite."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite existing release provenance: {destination}") from error
    return destination
