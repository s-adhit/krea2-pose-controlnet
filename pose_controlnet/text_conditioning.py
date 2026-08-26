"""Persistent, exact Qwen text-conditioning archives for pose training."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pose_controlnet.dataset_index import EXPECTED_SPLIT_COUNTS, ManifestRecord, validate_posebridge_snapshot
from pose_controlnet.text_encoder import PoseTextConditioner


FORMAT_VERSION = 2
METADATA_NAME = "text_conditioning.json"
DEFAULT_SHARD_SAMPLES = 64


class TextConditioningError(ValueError):
    pass


def compact_valid_conditioning(contexts: torch.Tensor, masks: torch.Tensor, index: int) -> dict[str, torch.Tensor]:
    """Extract one conditioning sequence by boolean validity, never by length.

    `PoseTextConditioner` returns batch-right-padded sequences.  Selecting valid
    positions is the canonical archive representation and preserves suffix tokens
    even when an older caller has padding between a prompt and its suffix.
    """
    if contexts.ndim != 4 or masks.ndim != 2 or contexts.shape[:2] != masks.shape:
        raise TextConditioningError("Conditioner returned incompatible context/mask shapes")
    if not 0 <= index < contexts.shape[0]:
        raise TextConditioningError(f"Conditioning index out of range: {index}")
    valid = masks[index].to(torch.bool)
    if not valid.any().item():
        raise TextConditioningError(f"Conditioning entry {index} has no valid tokens")
    return {"context": contexts[index][valid], "mask": valid[valid]}


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            torch.save(payload, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _archive_path(root: Path, split: str, number: int) -> Path:
    return root / split / f"{split}-{number:05d}.pt"


def _validate_entry(entry: Any, *, path: Path | str, expected_stem: str | None = None,
                    dimensions: tuple[int, int] | None = None) -> tuple[str, tuple[int, int]]:
    if not isinstance(entry, dict): raise TextConditioningError(f"Malformed entry in {path}")
    stem, context, mask = entry.get("stem"), entry.get("context"), entry.get("mask")
    if not isinstance(stem, str) or not stem: raise TextConditioningError(f"Invalid stem in {path}")
    if expected_stem is not None and stem != expected_stem: raise TextConditioningError(f"Stem mismatch in {path}: {stem} != {expected_stem}")
    if not isinstance(context, torch.Tensor) or context.dtype != torch.bfloat16 or context.ndim != 3:
        raise TextConditioningError(f"Context must be rank-3 BF16 in {path}:{stem}")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != context.shape[0]:
        raise TextConditioningError(f"Invalid boolean mask in {path}:{stem}")
    if context.shape[0] < 1 or not mask.all().item() or not torch.isfinite(context).all().item():
        raise TextConditioningError(f"Empty or non-finite conditioning in {path}:{stem}")
    observed = tuple(context.shape[1:])
    if dimensions is not None and observed != dimensions:
        raise TextConditioningError(f"Conditioning dimensions differ in {path}:{stem}: {observed} != {dimensions}")
    return stem, observed


def _latent_manifest_alignment(latent_root: str | Path, records_by_split: Mapping[str, Sequence[ManifestRecord]]) -> None:
    """Prove immutable manifest captions/stems are exactly those in latent archives."""
    root = Path(latent_root)
    metadata = json.loads((root / "shards.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or not metadata.get("complete"):
        raise TextConditioningError(f"Latent root is not complete: {root}")
    for split, records in records_by_split.items():
        observed: list[tuple[str, str]] = []
        for path in sorted((root / split).glob(f"{split}-*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format_version") != 1 or payload.get("split") != split: raise TextConditioningError(f"Malformed latent archive {path}")
            for sample in payload.get("samples", []):
                observed.append((sample.get("stem"), sample.get("text")))
        expected = [(record.stem, record.text) for record in records]
        if observed != expected: raise TextConditioningError(f"Latent shard stem/caption identity differs from immutable {split} manifest")


def _write_metadata(root: Path, *, counts: Mapping[str, int], shard_samples: int, complete: bool,
                    dimensions: tuple[int, int] | None = None) -> None:
    payload: dict[str, Any] = {"format_version": FORMAT_VERSION, "expected_counts": dict(counts),
        "total_samples": sum(counts.values()), "shard_samples": shard_samples, "complete": complete,
        "context_dtype": "bfloat16", "mask_dtype": "bool", "select_layers": list(PoseTextConditioner.SELECT_LAYERS)}
    if dimensions is not None: payload["context_dimensions"] = list(dimensions)
    _atomic_json(root / METADATA_NAME, payload)


def _resolve_dataset_root(dataset_root: str | Path | None, latent_root: str | Path) -> Path:
    if dataset_root is not None: return Path(dataset_root)
    try: value = json.loads((Path(latent_root) / "shards.json").read_text(encoding="utf-8"))["dataset_root"]
    except (OSError, KeyError, json.JSONDecodeError) as exc: raise TextConditioningError("--dataset-root is required when latent shards lack dataset_root metadata") from exc
    return Path(value)


def prepare_text_conditioning(*, dataset_root: str | Path | None, latent_root: str | Path, output_root: str | Path,
                              device: str = "cuda", shard_samples: int = DEFAULT_SHARD_SAMPLES,
                              conditioner: Any | None = None) -> dict[str, int]:
    if shard_samples < 1: raise TextConditioningError("shard_samples must be positive")
    dataset_root = _resolve_dataset_root(dataset_root, latent_root)
    validation = validate_posebridge_snapshot(dataset_root)
    records_by_split = validation.records_by_split
    if dict(validation.split_counts) != dict(EXPECTED_SPLIT_COUNTS): raise TextConditioningError("Unexpected immutable split counts")
    _latent_manifest_alignment(latent_root, records_by_split)
    root = Path(output_root).expanduser().resolve(); counts = {split: len(records) for split, records in records_by_split.items()}
    _write_metadata(root, counts=counts, shard_samples=shard_samples, complete=False)
    encoder = conditioner or PoseTextConditioner(device=device, dtype=torch.bfloat16)
    dimensions: tuple[int, int] | None = None
    for split, records in records_by_split.items():
        for number, start in enumerate(range(0, len(records), shard_samples)):
            chunk = records[start:start + shard_samples]; path = _archive_path(root, split, number)
            expected_stems = [record.stem for record in chunk]
            if path.exists():
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    entries = payload.get("samples", [])
                    if payload.get("format_version") == FORMAT_VERSION and payload.get("split") == split and [e.get("stem") for e in entries] == expected_stems:
                        for entry, stem in zip(entries, expected_stems): _, dimensions = _validate_entry(entry, path=path, expected_stem=stem, dimensions=dimensions)
                        continue
                except (OSError, RuntimeError, EOFError, TextConditioningError): pass
            contexts, masks = encoder([record.text for record in chunk])
            if contexts.dtype != torch.bfloat16: contexts = contexts.to(torch.bfloat16)
            entries = []
            for index, record in enumerate(chunk):
                # Only valid tokens are persisted; collate restores right-padding dynamically.
                entry = {
                    "stem": record.stem,
                    **{
                        key: value.detach().cpu().to(torch.bfloat16 if key == "context" else torch.bool).contiguous()
                        for key, value in compact_valid_conditioning(contexts, masks, index).items()
                    },
                }
                _, dimensions = _validate_entry(entry, path=path, expected_stem=record.stem, dimensions=dimensions)
                entries.append(entry)
            _atomic_torch_save(path, {"format_version": FORMAT_VERSION, "split": split, "samples": entries})
    unconditional_context, unconditional_mask = encoder([""])
    unconditional = {key: value.detach().cpu().to(torch.bfloat16 if key == "context" else torch.bool).contiguous()
                     for key, value in compact_valid_conditioning(unconditional_context, unconditional_mask, 0).items()}
    _, dimensions = _validate_entry({"stem": "__unconditional__", **unconditional}, path=root / "unconditional.pt", dimensions=dimensions)
    _atomic_torch_save(root / "unconditional.pt", {"format_version": FORMAT_VERSION, **unconditional})
    verify_text_conditioning(dataset_root=dataset_root, latent_root=latent_root, output_root=root)
    _write_metadata(root, counts=counts, shard_samples=shard_samples, complete=True, dimensions=dimensions)
    return counts


def verify_text_conditioning(*, dataset_root: str | Path | None, latent_root: str | Path, output_root: str | Path) -> dict[str, int]:
    validation = validate_posebridge_snapshot(_resolve_dataset_root(dataset_root, latent_root)); _latent_manifest_alignment(latent_root, validation.records_by_split)
    root = Path(output_root); metadata = json.loads((root / METADATA_NAME).read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("context_dtype") != "bfloat16" or metadata.get("mask_dtype") != "bool": raise TextConditioningError("Malformed text-conditioning metadata")
    counts = metadata.get("expected_counts")
    if counts != dict(EXPECTED_SPLIT_COUNTS): raise TextConditioningError("Text-conditioning counts do not match immutable splits")
    dimensions = tuple(metadata["context_dimensions"]) if metadata.get("complete") and metadata.get("context_dimensions") else None
    for split, records in validation.records_by_split.items():
        observed: list[str] = []
        paths = sorted((root / split).glob(f"{split}-*.pt"))
        planned = [_archive_path(root, split, number) for number in range((len(records) + metadata["shard_samples"] - 1) // metadata["shard_samples"])]
        if paths != planned: raise TextConditioningError(f"Text archive set does not match deterministic {split} plan")
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format_version") != FORMAT_VERSION or payload.get("split") != split: raise TextConditioningError(f"Malformed text archive {path}")
            for entry in payload.get("samples", []):
                stem, observed_dimensions = _validate_entry(entry, path=path, dimensions=dimensions)
                dimensions = dimensions or observed_dimensions
                observed.append(stem)
        expected = [record.stem for record in records]
        if observed != expected or len(observed) != len(set(observed)): raise TextConditioningError(f"Missing, duplicate, or misordered {split} text conditioning stems")
    payload = torch.load(root / "unconditional.pt", map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise TextConditioningError(f"Malformed unconditional archive {root / 'unconditional.pt'}")
    _, unconditional_dimensions = _validate_entry({"stem": "__unconditional__", "context": payload.get("context"), "mask": payload.get("mask")}, path=root / "unconditional.pt", dimensions=dimensions)
    if dimensions is None: dimensions = unconditional_dimensions
    return dict(counts)


@torch.no_grad()
def smoke_online_cached_equivalence(*, dataset_root: str | Path, output_root: str | Path,
                                    device: str = "cuda", samples_per_split: int = 2,
                                    stems: Sequence[str] | None = None) -> dict[str, float]:
    """Hard smoke: cached BF16 tensors must be byte-identical to fresh Qwen output.

    Each prompt is encoded independently because cached records retain only valid
    (un-padded) tokens; dynamic batch padding is restored by the training collate.
    """
    if samples_per_split < 1: raise TextConditioningError("samples_per_split must be positive")
    validation = validate_posebridge_snapshot(dataset_root)
    encoder = PoseTextConditioner(device=device, dtype=torch.bfloat16)
    maximum_difference = 0.0; checked = 0
    requested = set(stems or ())
    found: set[str] = set()
    for split, records in validation.records_by_split.items():
        cached = CachedTextConditioning(output_root, split)
        selected = [record for record in records if record.stem in requested] if requested else records[:samples_per_split]
        for record in selected:
            found.add(record.stem)
            online_context, online_mask = encoder([record.text])
            entry = cached.get(record.stem)
            if online_context.dtype != entry["context"].dtype or online_mask.dtype != entry["mask"].dtype:
                raise TextConditioningError(f"Online/cached dtype mismatch for {record.stem}")
            online_entry = compact_valid_conditioning(online_context, online_mask, 0)
            if not torch.equal(online_entry["mask"].cpu(), entry["mask"]):
                raise TextConditioningError(f"Online/cached mask mismatch for {record.stem}")
            online = online_entry["context"].cpu()
            if online.shape != entry["context"].shape: raise TextConditioningError(f"Online/cached shape mismatch for {record.stem}")
            if not torch.equal(online, entry["context"]):
                difference = (online.float() - entry["context"].float()).abs().max().item()
                raise TextConditioningError(f"Online/cached value mismatch for {record.stem}: max_abs_diff={difference}")
            difference = (online.float() - entry["context"].float()).abs().max().item()
            maximum_difference = max(maximum_difference, difference)
            if difference != 0.0: raise TextConditioningError(f"Online/cached value mismatch for {record.stem}: max_abs_diff={difference}")
            checked += 1
    missing = requested - found
    if missing: raise TextConditioningError(f"Requested stems absent from immutable manifests: {sorted(missing)}")
    return {"checked": checked, "max_abs_difference": maximum_difference}


class CachedTextConditioning:
    """Read-only, one-archive cached conditioning lookup keyed by immutable stem."""
    def __init__(self, root: str | Path, split: str) -> None:
        self.root, self.split = Path(root), split
        metadata = json.loads((self.root / METADATA_NAME).read_text(encoding="utf-8"))
        if metadata.get("format_version") != FORMAT_VERSION or not metadata.get("complete"): raise TextConditioningError("Text-conditioning root is incomplete")
        self.dimensions = tuple(metadata["context_dimensions"])
        self.index: dict[str, tuple[Path, int]] = {}
        for path in sorted((self.root / split).glob(f"{split}-*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format_version") != FORMAT_VERSION or payload.get("split") != split:
                raise TextConditioningError(f"Malformed text archive {path}")
            for offset, entry in enumerate(payload.get("samples", [])):
                stem, _ = _validate_entry(entry, path=path, dimensions=self.dimensions)
                if stem in self.index: raise TextConditioningError(f"Duplicate cached stem {stem}")
                self.index[stem] = (path, offset)
        self._cached_path: Path | None = None; self._cached_samples: list[dict] | None = None
        uncond = torch.load(self.root / "unconditional.pt", map_location="cpu", weights_only=False)
        if uncond.get("format_version") != FORMAT_VERSION:
            raise TextConditioningError(f"Malformed unconditional archive {self.root / 'unconditional.pt'}")
        _validate_entry({"stem": "__unconditional__", "context": uncond.get("context"), "mask": uncond.get("mask")}, path=self.root / "unconditional.pt", dimensions=self.dimensions)
        self.unconditional = {"context": uncond["context"], "mask": uncond["mask"]}

    def get(self, stem: str) -> dict[str, torch.Tensor]:
        path, offset = self.index[stem]
        if path != self._cached_path:
            self._cached_path = path; self._cached_samples = torch.load(path, map_location="cpu", weights_only=False)["samples"]
        entry = self._cached_samples[offset]  # type: ignore[index]
        return {"context": entry["context"], "mask": entry["mask"]}
