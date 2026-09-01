"""Deterministic alternate-resolution inputs for isolated capacity runs.

Native capacity experiments consume the verified persisted latents verbatim.
An alternate policy never rescales those latents: it re-applies paired pixel
geometry to the read-only RGB/control sources and VAE-encodes a separately
versioned, experiment-scoped cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
from pose_controlnet.overfit_capacity import (
    CapacityScientificConfig, NATIVE_RESOLUTION_POLICY, buckets_for_resolution,
    canonical_resolution_policy, deterministic_seed,
)
from pose_controlnet.paired_preprocessing import (
    PreprocessedPair, ResizeCropGeometry, preprocess_pair,
    preprocess_pair_with_persisted_geometry,
)
from pose_controlnet.vae_preprocessing import encode_preprocessed_pair


FORMAT_VERSION = 1
MANIFEST_NAME = "resolution_manifest.json"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def geometry_dict(pair: PreprocessedPair) -> dict[str, list[int]]:
    return {
        "source_size": list(pair.geometry.source_size), "resized_size": list(pair.geometry.resized_size),
        "crop_box": list(pair.geometry.crop_box), "bucket": list(pair.geometry.bucket),
        "latent_size": [pair.geometry.bucket[1] // 8, pair.geometry.bucket[0] // 8],
    }


def _validate_geometry(geometry: dict[str, Any], *, stem: str) -> None:
    fields = (("source_size", 2), ("resized_size", 2), ("crop_box", 4), ("bucket", 2), ("latent_size", 2))
    for field, length in fields:
        value = geometry.get(field)
        minimum = 0 if field == "crop_box" else 1
        if not isinstance(value, list) or len(value) != length or any(not isinstance(v, int) or v < minimum for v in value):
            raise ValueError(f"{stem}: malformed resolution geometry {field}")
    left, top, right, bottom = geometry["crop_box"]
    width, height = geometry["bucket"]
    if right - left != width or bottom - top != height:
        raise ValueError(f"{stem}: crop and bucket geometry disagree")
    resized_width, resized_height = geometry["resized_size"]
    if right > resized_width or bottom > resized_height:
        raise ValueError(f"{stem}: crop lies outside persisted resized geometry")
    if geometry["latent_size"] != [height // 8, width // 8] or height % 8 or width % 8:
        raise ValueError(f"{stem}: latent geometry is incompatible with the VAE spatial factor")


def native_resolution_provenance(data: Dataset, stems: Iterable[str]) -> dict[str, Any]:
    """Read exact persisted geometry; do not recompute native capacity geometry."""
    items = [data[index] for index in range(len(data))]
    by_stem = {item["stem"]: item for item in items}
    if len(by_stem) != len(items):
        raise ValueError("Persisted native latent geometry has duplicate stems")
    samples: dict[str, Any] = {}
    for stem in stems:
        try:
            sample = by_stem[stem]
        except KeyError as exc:
            raise ValueError(f"{stem}: persisted native latent geometry is missing") from exc
        latent, control = sample.get("latent"), sample.get("control")
        if not isinstance(latent, torch.Tensor) or not isinstance(control, torch.Tensor):
            raise ValueError(f"{stem}: persisted native RGB/control latents are missing")
        if latent.shape != control.shape:
            raise ValueError(f"{stem}: persisted native RGB/control latent geometry disagrees")
        geometry = {
            "source_size": list(sample.get("source_size") or ()), "resized_size": list(sample.get("resized_size") or ()),
            "crop_box": list(sample.get("crop_box") or ()), "bucket": list(sample.get("bucket") or ()),
            "latent_size": list(latent.shape[-2:]),
        }
        _validate_geometry(geometry, stem=stem)
        samples[stem] = geometry
    return {"format_version": FORMAT_VERSION, "resolution_policy": NATIVE_RESOLUTION_POLICY,
            "source": "verified_persisted_latent_geometry", "samples": samples}


def preprocess_native_evaluation_pair(record: ManifestRecord, sample: dict[str, Any]) -> PreprocessedPair:
    """Rebuild a native display pair from its exact persisted shard geometry."""
    geometry = {
        "source_size": list(sample.get("source_size") or ()),
        "resized_size": list(sample.get("resized_size") or ()),
        "crop_box": list(sample.get("crop_box") or ()),
        "bucket": list(sample.get("bucket") or ()),
        "latent_size": list(sample["latent"].shape[-2:]),
    }
    _validate_geometry(geometry, stem=record.stem)
    source_width, source_height = geometry["source_size"]
    bucket_width, bucket_height = geometry["bucket"]
    persisted = ResizeCropGeometry(
        source_size=(source_width, source_height),
        bucket=(bucket_width, bucket_height),
        scale=max(bucket_width / source_width, bucket_height / source_height),
        resized_size=tuple(geometry["resized_size"]),
        crop_box=tuple(geometry["crop_box"]),
    )
    return preprocess_pair_with_persisted_geometry(record, persisted)


def prepare_alternate_resolution_cache(*, selected: Dataset, config: CapacityScientificConfig,
                                       dataset_root: str | Path, cache_root: str | Path,
                                       vae: Any, device: torch.device) -> dict[str, Any]:
    """Encode the exact selected sources under an alternate paired policy.

    The output namespace belongs to one dynamic experiment and its manifest is
    written only after every per-stem tensor has passed alignment checks.
    """
    policy = canonical_resolution_policy(config.resolution)
    buckets = buckets_for_resolution(policy)
    if buckets is None:
        raise ValueError("native resolution must use the persisted-latent dataset, not an alternate cache")
    root = Path(cache_root) / config.experiment_name
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        return load_alternate_resolution_cache(selected=selected, config=config, cache_root=cache_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to reuse incomplete alternate-resolution cache namespace: {root}")
    index = DatasetIndex.discover(dataset_root)
    samples: dict[str, Any] = {}
    root.mkdir(parents=True, exist_ok=True)
    for item_index in range(len(selected)):
        item = selected[item_index]
        stem = item["stem"]
        record = ManifestRecord(split="train", stem=stem, file_name=f"{stem}.jpg", text=item["prompt"],
                                rgb_path=index.rgb_by_stem[stem], control_path=index.control_by_stem[stem])
        pair = preprocess_pair(record, buckets=buckets)
        geometry = geometry_dict(pair); _validate_geometry(geometry, stem=stem)
        generator = torch.Generator(device=device).manual_seed(deterministic_seed(stem))
        encoded = encode_preprocessed_pair(vae, pair, device=device, generator=generator)
        latent, control = encoded.latent.detach().float().cpu(), encoded.control.detach().float().cpu()
        expected = tuple(geometry["latent_size"])
        if latent.shape != control.shape or tuple(latent.shape[-2:]) != expected or not torch.isfinite(latent).all() or not torch.isfinite(control).all() or control.abs().max().item() == 0:
            raise ValueError(f"{stem}: alternate RGB/control VAE encoding violates the paired latent contract")
        path = root / f"{stem}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"format_version": FORMAT_VERSION, "stem": stem, "geometry": geometry,
                    "image_latent": latent, "control_latent": control}, temporary)
        temporary.replace(path)
        samples[stem] = {"geometry": geometry, "latent_file": path.name}
    manifest = {"format_version": FORMAT_VERSION, "experiment": config.experiment_name,
                "scientific_config": config.__dict__, "resolution_policy": policy,
                "bucket_shapes": [list(bucket) for bucket in buckets],
                "stems": [selected[index]["stem"] for index in range(len(selected))], "samples": samples}
    _atomic_json(manifest_path, manifest)
    return load_alternate_resolution_cache(selected=selected, config=config, cache_root=cache_root)


def load_alternate_resolution_cache(*, selected: Dataset, config: CapacityScientificConfig,
                                    cache_root: str | Path) -> dict[str, Any]:
    root = Path(cache_root) / config.experiment_name
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Alternate resolution cache is missing or malformed: {root / MANIFEST_NAME}") from exc
    expected_stems = [selected[index]["stem"] for index in range(len(selected))]
    if (manifest.get("format_version") != FORMAT_VERSION or manifest.get("experiment") != config.experiment_name
            or manifest.get("scientific_config") != config.__dict__ or manifest.get("resolution_policy") != config.resolution
            or manifest.get("stems") != expected_stems):
        raise ValueError("Alternate resolution cache does not prove the requested scientific configuration and exact Mixed-32 order")
    samples = manifest.get("samples")
    if not isinstance(samples, dict) or set(samples) != set(expected_stems):
        raise ValueError("Alternate resolution cache has incomplete or unexpected stem membership")
    for stem in expected_stems:
        entry = samples[stem]
        if not isinstance(entry, dict) or not isinstance(entry.get("geometry"), dict) or not isinstance(entry.get("latent_file"), str):
            raise ValueError(f"{stem}: malformed alternate-resolution cache entry")
        _validate_geometry(entry["geometry"], stem=stem)
        path = root / entry["latent_file"]
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, ValueError) as exc:
            raise ValueError(f"{stem}: unreadable alternate latent cache") from exc
        latent, control = payload.get("image_latent"), payload.get("control_latent")
        expected = tuple(entry["geometry"]["latent_size"])
        if payload.get("stem") != stem or not isinstance(latent, torch.Tensor) or not isinstance(control, torch.Tensor) or latent.shape != control.shape or tuple(latent.shape[-2:]) != expected:
            raise ValueError(f"{stem}: stale or incompatible alternate latent geometry")
    return manifest


class AlternateResolutionDataset(Dataset):
    """Add exact alternate latents to the existing selected text-conditioning dataset."""
    def __init__(self, selected: Dataset, manifest: dict[str, Any], cache_root: str | Path, experiment_name: str) -> None:
        self.selected, self.manifest = selected, manifest
        self.root = Path(cache_root) / experiment_name
        self.stems = tuple(manifest["stems"])
        self.records = getattr(selected, "records", [])
        self.text_conditioning = getattr(selected, "text_conditioning", None)

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.selected[index]); stem = item["stem"]
        entry = self.manifest["samples"].get(stem)
        if entry is None:
            raise ValueError(f"{stem}: selected sample is absent from alternate-resolution manifest")
        payload = torch.load(self.root / entry["latent_file"], map_location="cpu", weights_only=False)
        item["latent"], item["control"] = payload["image_latent"], payload["control_latent"]
        item.update(entry["geometry"])
        return item
