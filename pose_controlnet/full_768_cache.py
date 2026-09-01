"""Immutable full-train 768 latent cache and pose-sidecar contract.

This is deliberately separate from the Mixed-32 alternate-resolution cache.
It is the only cache format accepted by the production-throughput benchmark:
all 16,503 immutable training records are re-encoded from paired pixels under
the locked 768 policy, and the pose targets are reprojected from the reviewed
source-space v3 sidecar.  No native latent is ever read or resampled here.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image

from prepare_shards import ShardError, validate_shard, write_shard_atomically
from pose_controlnet.dataset_index import ManifestRecord, validate_posebridge_snapshot
from pose_controlnet.overfit_capacity import deterministic_seed
from pose_controlnet.resolution_policy import RESOLUTION_768_BUCKETS
from pose_controlnet.paired_preprocessing import preprocess_pair, resize_center_crop_geometry
from pose_controlnet.pose_targets import (
    POSE_TARGET_SIDECAR_VERSION, PoseTargetError, common_body_mapping,
    load_sidecar, source_for_stem, transform_person, write_sidecar,
)
from pose_controlnet.vae_preprocessing import encode_preprocessed_pair, load_krea_vae


FULL_TRAIN_COUNT = 16_503
POLICY = "posebridge_full_train_768_v1"
FORMAT_VERSION = 1
METADATA_NAME = "shards.json"
IDENTITY_NAME = "train_manifest_identity.json"
POSE_SOURCE_SHA256 = "dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f"
DEFAULT_SHARD_SAMPLES = 256
DEFAULT_TRAIN_MANIFEST = Path(__file__).resolve().parents[1] / "data/manifests/train.jsonl"


class Full768CacheError(ValueError):
    """A full-production 768 artifact violates its immutable contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_rows(path: Path, *, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    """Load JSONL records without making raw serialization part of their identity."""
    try:
        raw = path.read_bytes()
        parsed = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Full768CacheError(f"{label} is unreadable: {path}") from exc
    if not all(isinstance(row, dict) for row in parsed):
        raise Full768CacheError(f"{label} contains a non-object JSONL record: {path}")
    return raw, parsed


def _identity(records: Sequence[ManifestRecord], manifest_path: Path,
              snapshot_manifest_path: Path) -> dict[str, Any]:
    """Prove exact parsed project/snapshot manifest identity before cache work.

    Raw file hashes are retained only as provenance.  The scientific identity
    is the complete ordered parsed-record sequence plus its resolved stem order.
    This deliberately preserves fields such as ``conditioning_image`` that the
    physical-file resolver does not carry in ``ManifestRecord``.
    """
    stems = [record.stem for record in records]
    if len(stems) != FULL_TRAIN_COUNT or len(stems) != len(set(stems)):
        raise Full768CacheError("full 768 cache requires exactly 16,503 unique immutable train stems")
    raw, manifest_rows = _manifest_rows(manifest_path, label="authoritative full train manifest")
    snapshot_raw, snapshot_rows = _manifest_rows(snapshot_manifest_path, label="snapshot full train manifest")
    if len(manifest_rows) != FULL_TRAIN_COUNT or len(snapshot_rows) != FULL_TRAIN_COUNT:
        raise Full768CacheError("full 768 cache requires exactly 16,503 parsed train manifest records on both sides")
    if manifest_rows != snapshot_rows:
        raise Full768CacheError(
            "authoritative full train manifest does not exactly match the dataset snapshot's ordered records"
        )
    snapshot_file_names = [row.get("file_name") for row in snapshot_rows]
    if snapshot_file_names != [record.file_name for record in records]:
        raise Full768CacheError("snapshot parsed manifest order disagrees with its resolved train records")
    return {
        "schema_version": 2,
        "sample_count": FULL_TRAIN_COUNT,
        "train_manifest_path": str(manifest_path.resolve()),
        "snapshot_train_manifest_path": str(snapshot_manifest_path.resolve()),
        "authoritative_train_manifest_raw_sha256": _sha256_bytes(raw),
        "snapshot_train_manifest_raw_sha256": _sha256_bytes(snapshot_raw),
        "manifest_records_sha256": _sha256_bytes(_canonical_bytes(manifest_rows)),
        "ordered_stems": stems,
        "ordered_stems_sha256": _sha256_bytes(_canonical_bytes(stems)),
    }


def _cache_metadata(identity: Mapping[str, Any], *, dataset_root: Path, shard_samples: int,
                    complete: bool) -> dict[str, Any]:
    immutable = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": "posebridge_full_train_768_latents",
        "resolution_policy": POLICY,
        "policy_buckets": [list(bucket) for bucket in RESOLUTION_768_BUCKETS],
        "vae_spatial_factor": 8,
        "expected_counts": {"train": FULL_TRAIN_COUNT},
        "total_samples": FULL_TRAIN_COUNT,
        "shard_samples": shard_samples,
        "dataset_root": str(dataset_root.resolve()),
        "manifest_records_sha256": identity["manifest_records_sha256"],
        "ordered_stems_sha256": identity["ordered_stems_sha256"],
        "identity_file": IDENTITY_NAME,
        "builder_version": POLICY,
        "vae_encoding": "qwen_posterior_sample_seeded_per_stem_v1",
    }
    immutable["cache_contract_sha256"] = _sha256_bytes(_canonical_bytes(immutable))
    return immutable | {"complete": complete}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Full768CacheError(f"Unreadable {label}: {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise Full768CacheError(f"Malformed {label}: {path}")
    return result


def _planned_shards(records: Sequence[ManifestRecord], shard_samples: int) -> list[tuple[Path, Sequence[ManifestRecord]]]:
    if shard_samples < 1:
        raise Full768CacheError("shard_samples must be positive")
    return [(Path("train") / f"train-{number:05d}.pt", records[start:start + shard_samples])
            for number, start in enumerate(range(0, len(records), shard_samples))]


def _validate_sample_768(sample: Mapping[str, Any], *, stem: str) -> None:
    bucket = sample.get("bucket")
    if not isinstance(bucket, list) or tuple(bucket) not in RESOLUTION_768_BUCKETS:
        raise Full768CacheError(f"{stem}: native or non-768-policy bucket contamination: {bucket!r}")
    if any(value % 64 for value in bucket):
        raise Full768CacheError(f"{stem}: bucket is not 64-pixel aligned")
    for field, length in (("source_size", 2), ("resized_size", 2), ("crop_box", 4)):
        value = sample.get(field)
        if not isinstance(value, list) or len(value) != length or any(not isinstance(x, int) for x in value):
            raise Full768CacheError(f"{stem}: malformed {field}")
    source_size = tuple(sample["source_size"])
    expected = resize_center_crop_geometry(source_size, tuple(bucket))
    observed = (tuple(sample["resized_size"]), tuple(sample["crop_box"]))
    if observed != (expected.resized_size, expected.crop_box):
        raise Full768CacheError(f"{stem}: stale or non-deterministic 768 paired geometry")
    image, control = sample.get("image_latent"), sample.get("control_latent")
    if not isinstance(image, torch.Tensor) or not isinstance(control, torch.Tensor):
        raise Full768CacheError(f"{stem}: missing paired latents")
    expected_shape = (16, bucket[1] // 8, bucket[0] // 8)
    if tuple(image.shape) != expected_shape or image.shape != control.shape:
        raise Full768CacheError(f"{stem}: RGB/control latent VAE-factor-eight shape mismatch")
    if not torch.isfinite(image).all().item() or not torch.isfinite(control).all().item():
        raise Full768CacheError(f"{stem}: non-finite latent")
    if control.abs().max().item() == 0:
        raise Full768CacheError(f"{stem}: empty control latent")


def _validate_existing_shard(path: Path, stems: Sequence[str]) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_shard(payload, path=path, expected_split="train", expected_stems=stems)
        for sample in payload["samples"]:
            _validate_sample_768(sample, stem=str(sample["stem"]))
        return True
    except (ShardError, Full768CacheError, OSError, RuntimeError, EOFError, ValueError):
        return False


def _sample(record: ManifestRecord, encoded: Any) -> dict[str, Any]:
    geometry = encoded.pair.geometry
    result = {
        "stem": record.stem, "file_name": record.file_name, "text": record.text, "split": "train",
        "bucket": list(geometry.bucket), "source_size": list(geometry.source_size),
        "resized_size": list(geometry.resized_size), "crop_box": list(geometry.crop_box),
        "image_latent": encoded.latent.detach().to("cpu", dtype=torch.float32).contiguous(),
        "control_latent": encoded.control.detach().to("cpu", dtype=torch.float32).contiguous(),
    }
    _validate_sample_768(result, stem=record.stem)
    return result


def prepare_full_768_cache(*, dataset_root: str | Path, output_root: str | Path, device: str,
                           shard_samples: int = DEFAULT_SHARD_SAMPLES,
                           train_manifest: str | Path = DEFAULT_TRAIN_MANIFEST) -> dict[str, Any]:
    """Build/resume only the exact immutable 16,503-sample train cache."""
    snapshot = validate_posebridge_snapshot(dataset_root)
    records = snapshot.records_by_split["train"]
    root = Path(output_root).expanduser().resolve()
    identity = _identity(records, Path(train_manifest).expanduser().resolve(),
                         Path(dataset_root).expanduser().resolve() / "manifests/train.jsonl")
    wanted = _cache_metadata(identity, dataset_root=Path(dataset_root), shard_samples=shard_samples, complete=False)
    metadata_path, identity_path = root / METADATA_NAME, root / IDENTITY_NAME
    if root.exists() and not root.is_dir():
        raise Full768CacheError(f"Cache root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        existing = _load_json(metadata_path, "full 768 cache metadata")
        for key, value in wanted.items():
            if key != "complete" and existing.get(key) != value:
                raise Full768CacheError(f"Refusing scientifically conflicting cache root {root}: {key} differs")
        existing_identity = _load_json(identity_path, "full 768 identity")
        if existing_identity != identity:
            raise Full768CacheError(f"Refusing cache root with mismatched immutable manifest identity: {root}")
    else:
        if any(root.iterdir()):
            raise Full768CacheError(f"Refusing nonempty unrecognized cache root: {root}")
        _atomic_json(identity_path, identity)
        _atomic_json(metadata_path, wanted)

    planned = _planned_shards(records, shard_samples)
    expected_paths = {root / relative for relative, _ in planned}
    present_paths = set((root / "train").glob("train-*.pt")) if (root / "train").is_dir() else set()
    extra = present_paths - expected_paths
    if extra:
        raise Full768CacheError(f"Refusing cache root with unexpected shard files: {sorted(map(str, extra))[:3]}")
    vae: Any | None = None
    rebuilt = reused = 0
    for relative, chunk in planned:
        path, stems = root / relative, [record.stem for record in chunk]
        if path.is_file() and _validate_existing_shard(path, stems):
            reused += len(chunk)
            continue
        if vae is None:
            vae = load_krea_vae(device)
        samples = []
        for record in chunk:
            pair = preprocess_pair(record, buckets=RESOLUTION_768_BUCKETS)
            generator = torch.Generator(device=device).manual_seed(deterministic_seed(record.stem))
            samples.append(_sample(record, encode_preprocessed_pair(vae, pair, device=device, generator=generator)))
        write_shard_atomically(path, "train", samples)
        rebuilt += len(chunk)
    # Validate every planned final shard before publishing the completion marker.
    # The metadata marker is deliberately the last cache mutation.
    for relative, chunk in planned:
        if not _validate_existing_shard(root / relative, [record.stem for record in chunk]):
            raise Full768CacheError(f"failed final validation for 768 shard: {root / relative}")
    _atomic_json(metadata_path, _cache_metadata(identity, dataset_root=Path(dataset_root), shard_samples=shard_samples, complete=True))
    return {"cache_root": str(root), "sample_count": FULL_TRAIN_COUNT, "reused_samples": reused, "rebuilt_samples": rebuilt}


def _geometry(sample: Mapping[str, Any]) -> dict[str, list[int]]:
    return {field: list(sample[field]) for field in ("source_size", "resized_size", "crop_box", "bucket")}


def _unavailable(stem: str, source: str, geometry: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": POSE_TARGET_SIDECAR_VERSION, "stem": stem, "source": source,
            "pose_reward_available": False, "target_provenance": "unavailable", "annotation_source": None,
            **{field: list(geometry[field]) for field in ("source_size", "resized_size", "crop_box", "bucket")}, "people": None}


def build_full_768_pose_sidecar(*, cache_root: str | Path, authoritative_source: str | Path,
                                output_dir: str | Path, expected_source_sha256: str = POSE_SOURCE_SHA256) -> dict[str, Any]:
    """Project source-space targets from the immutable v3 sidecar into 768 geometry."""
    cache = _load_complete_cache(Path(cache_root))
    source_metadata, source_records = load_sidecar(authoritative_source)
    if source_metadata.get("records_sha256") != expected_source_sha256:
        raise Full768CacheError("authoritative pose source SHA does not match the locked v3 records SHA")
    source_by_stem = {str(record["stem"]): record for record in source_records}
    records: list[dict[str, Any]] = []
    for sample in _iterate_cache_samples(Path(cache_root), cache["ordered_stems"]):
        stem, geometry = str(sample["stem"]), _geometry(sample)
        source = source_for_stem(stem)
        original = source_by_stem.get(stem)
        if original is None:
            raise Full768CacheError(f"{stem}: no authoritative v3 source record")
        if source == "danbooru":
            if original.get("pose_reward_available") is not False:
                raise Full768CacheError(f"{stem}: Danbooru source must remain explicitly unavailable")
            records.append(_unavailable(stem, source, geometry)); continue
        if original.get("pose_reward_available") is not True or not isinstance(original.get("people"), list):
            raise Full768CacheError(f"{stem}: pose-eligible source has no authoritative target")
        if original.get("source_size") != geometry["source_size"]:
            raise Full768CacheError(f"{stem}: source-sidecar/source-image size disagreement")
        # v3 stores the authoritative source bbox under its explicit
        # ``bbox_source_xywh`` name; transform_person consumes the original
        # annotation spelling.  Keep the source-space points/visibility intact.
        people = [transform_person({**person, "bbox_xywh": person.get("bbox_source_xywh")}, **geometry)
                  for person in original["people"]]
        records.append({
            "schema_version": POSE_TARGET_SIDECAR_VERSION, "stem": stem, "source": source,
            "pose_reward_available": True, "target_provenance": "original_annotation",
            "annotation_source": "pose_targets_v3_source_space_reprojection",
            "source_image_id": original.get("source_image_id"), "source_image_name": original.get("source_image_name"),
            "source_annotation_split": original.get("source_annotation_split"), **geometry,
            "geometry_transform": "x_final=clip(x_source*resized_width/source_width-crop_left,0,bucket_width-1); y_final=clip(y_source*resized_height/source_height-crop_top,0,bucket_height-1)",
            "joint_schema": "coco17", "common_body_mapping": common_body_mapping("coco17"),
            "person_grouping": "original image-level people list preserved; renderer-only neck is never a target joint",
            "people": people, "renderer": original.get("renderer"),
            "provenance_metadata": {"authoritative_pose_source_records_sha256": expected_source_sha256,
                                    "source_sidecar_record_schema": original.get("schema_version"),
                                    "source_provenance": original.get("provenance_metadata")},
        })
    availability = Counter("available" if row["pose_reward_available"] else "unavailable" for row in records)
    metadata = write_sidecar(records, output_dir, build_metadata={
        "artifact_kind": "posebridge_full_train_768_pose_targets", "resolution_policy": POLICY,
        "sample_count": FULL_TRAIN_COUNT, "manifest_records_sha256": cache["manifest_records_sha256"],
        "ordered_stems_sha256": cache["ordered_stems_sha256"], "cache_contract_sha256": cache["cache_contract_sha256"],
        "authoritative_pose_source_sha256": expected_source_sha256,
        "authoritative_pose_source_path": str(Path(authoritative_source).resolve()),
        "pose_available_count": availability["available"], "pose_unavailable_count": availability["unavailable"],
        "builder_version": POLICY,
    })
    return metadata


def _load_complete_cache(root: Path) -> dict[str, Any]:
    metadata = _load_json(root / METADATA_NAME, "full 768 cache metadata")
    identity = _load_json(root / IDENTITY_NAME, "full 768 identity")
    if metadata.get("complete") is not True:
        raise Full768CacheError("full 768 cache completion marker is absent")
    if metadata.get("resolution_policy") != POLICY or metadata.get("expected_counts") != {"train": FULL_TRAIN_COUNT}:
        raise Full768CacheError("cache is not the isolated full-train 768 production artifact")
    if (identity.get("sample_count") != FULL_TRAIN_COUNT
            or metadata.get("manifest_records_sha256") != identity.get("manifest_records_sha256")
            or metadata.get("ordered_stems_sha256") != identity.get("ordered_stems_sha256")):
        raise Full768CacheError("cache identity metadata is inconsistent")
    stems = identity.get("ordered_stems")
    if not isinstance(stems, list) or len(stems) != FULL_TRAIN_COUNT or len(set(stems)) != FULL_TRAIN_COUNT:
        raise Full768CacheError("cache identity does not contain the exact full ordered manifest")
    return metadata | identity


def _iterate_cache_samples(root: Path, stems: Sequence[str]) -> Iterable[dict[str, Any]]:
    metadata = _load_complete_cache(root)
    observed: list[str] = []
    for relative, expected_chunk in _planned_shards([_StubRecord(stem) for stem in stems], metadata["shard_samples"]):
        path = root / relative
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected_stems = [record.stem for record in expected_chunk]
            validate_shard(payload, path=path, expected_split="train", expected_stems=expected_stems)
        except (ShardError, OSError, RuntimeError, EOFError, ValueError) as exc:
            raise Full768CacheError(f"invalid/missing production shard: {path}") from exc
        for sample in payload["samples"]:
            _validate_sample_768(sample, stem=str(sample["stem"]))
            observed.append(str(sample["stem"])); yield sample
    if observed != list(stems):
        raise Full768CacheError("cache shard order differs from immutable full manifest")


class _StubRecord:
    def __init__(self, stem: str) -> None: self.stem = stem


def verify_full_768_cache(*, dataset_root: str | Path, cache_root: str | Path,
                          pose_sidecar: str | Path | None, expected_source_sha256: str = POSE_SOURCE_SHA256,
                          train_manifest: str | Path = DEFAULT_TRAIN_MANIFEST) -> dict[str, Any]:
    """No-GPU, no-network fail-closed verifier for cache + matching sidecar."""
    snapshot = validate_posebridge_snapshot(dataset_root)
    records = snapshot.records_by_split["train"]
    root = Path(cache_root).expanduser().resolve()
    cache = _load_complete_cache(root)
    identity = _identity(records, Path(train_manifest).expanduser().resolve(),
                         Path(dataset_root).expanduser().resolve() / "manifests/train.jsonl")
    for key in ("manifest_records_sha256", "ordered_stems_sha256", "ordered_stems"):
        if cache.get(key) != identity.get(key):
            raise Full768CacheError(f"cache immutable manifest identity mismatch: {key}")
    samples = list(_iterate_cache_samples(root, identity["ordered_stems"]))
    if len(samples) != FULL_TRAIN_COUNT:
        raise Full768CacheError("full 768 cache sample count mismatch")
    # Check physical source sizes too: stale cached geometry is not accepted.
    by_stem = {record.stem: record for record in records}
    for sample in samples:
        record = by_stem[str(sample["stem"])]
        with Image.open(record.rgb_path) as rgb, Image.open(record.control_path) as control:
            if rgb.size != control.size or list(rgb.size) != sample["source_size"]:
                raise Full768CacheError(f"{record.stem}: source provenance no longer matches persisted paired geometry")
    result: dict[str, Any] = {"cache_samples": len(samples), "resolution_policy": POLICY,
                              "cache_contract_sha256": cache["cache_contract_sha256"]}
    if pose_sidecar is not None:
        metadata, sidecar_records = load_sidecar(pose_sidecar)
        required = {"artifact_kind": "posebridge_full_train_768_pose_targets", "resolution_policy": POLICY,
                    "sample_count": FULL_TRAIN_COUNT, "manifest_records_sha256": identity["manifest_records_sha256"],
                    "ordered_stems_sha256": identity["ordered_stems_sha256"], "cache_contract_sha256": cache["cache_contract_sha256"],
                    "authoritative_pose_source_sha256": expected_source_sha256}
        for key, value in required.items():
            if metadata.get(key) != value:
                raise Full768CacheError(f"pose sidecar does not match verified full 768 cache: {key}")
        by_pose = {str(row["stem"]): row for row in sidecar_records}
        if set(by_pose) != set(identity["ordered_stems"]) or len(by_pose) != FULL_TRAIN_COUNT:
            raise Full768CacheError("pose sidecar membership does not exactly equal the full train manifest")
        available = unavailable = 0
        for sample in samples:
            row = by_pose[str(sample["stem"])]
            if _geometry(sample) != {field: row.get(field) for field in ("source_size", "resized_size", "crop_box", "bucket")}:
                raise Full768CacheError(f"{sample['stem']}: pose sidecar geometry is not the latent geometry")
            if source_for_stem(str(sample["stem"])) == "danbooru":
                if row.get("pose_reward_available") is not False or row.get("people") is not None:
                    raise Full768CacheError(f"{sample['stem']}: Danbooru pose reward was fabricated")
                unavailable += 1
            elif row.get("pose_reward_available") is not True or not isinstance(row.get("people"), list):
                raise Full768CacheError(f"{sample['stem']}: eligible pose target unavailable")
            else:
                available += 1
        if metadata.get("pose_available_count") != available or metadata.get("pose_unavailable_count") != unavailable:
            raise Full768CacheError("pose sidecar available/unavailable counts are not sane")
        result |= {"pose_available": available, "pose_unavailable": unavailable,
                   "pose_records_sha256": metadata["records_sha256"]}
    return result
