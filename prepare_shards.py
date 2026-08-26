"""Create resumable, verified latent shards for the immutable PoseBridge splits.

Each ``.pt`` file is a self-contained Torch archive containing a list of
heterogeneous-bucket samples.  Keeping 256 samples together produces roughly
0.5 GiB shards for the 16-channel float32 image/control pair: large sequential
NFS reads, while leaving resume/recovery at a practical granularity.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pose_controlnet.dataset_index import (
    EXPECTED_SPLIT_COUNTS,
    DatasetIndexError,
    ManifestRecord,
    validate_posebridge_snapshot,
)
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.vae_preprocessing import (
    VAEPreprocessingError,
    encode_preprocessed_pair,
    load_krea_vae,
)


SHARD_FORMAT_VERSION = 1
DEFAULT_SHARD_SAMPLES = 256
METADATA_FILE_NAME = "shards.json"


class ShardError(ValueError):
    """Raised when a latent shard violates the on-disk contract."""


def shard_path(output_root: Path, split: str, shard_number: int) -> Path:
    return output_root / split / f"{split}-{shard_number:05d}.pt"


def load_shard(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ShardError(f"Unreadable shard {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShardError(f"Malformed shard {path}: expected dictionary payload")
    return payload


def validate_shard(
    payload: Mapping[str, Any], *, path: str | Path = "<memory>", expected_split: str | None = None,
    expected_stems: Sequence[str] | None = None,
) -> list[str]:
    """Hard-validate one archive and return its ordered unique stems."""
    if payload.get("format_version") != SHARD_FORMAT_VERSION:
        raise ShardError(f"Malformed shard {path}: unsupported format_version")
    split = payload.get("split")
    if not isinstance(split, str) or not split:
        raise ShardError(f"Malformed shard {path}: invalid split")
    if expected_split is not None and split != expected_split:
        raise ShardError(f"Wrong split membership in {path}: expected {expected_split}, got {split}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ShardError(f"Malformed shard {path}: samples must be a non-empty list")
    stems: list[str] = []
    for sample_number, sample in enumerate(samples):
        _validate_sample(sample, split=split, path=path, sample_number=sample_number)
        stems.append(sample["stem"])
    if len(stems) != len(set(stems)):
        raise ShardError(f"Duplicate samples within shard {path}")
    if expected_stems is not None and stems != list(expected_stems):
        raise ShardError(f"Shard {path} does not match its deterministic planned sample range")
    return stems


def validate_shard_file(
    path: str | Path, *, expected_split: str | None = None,
    expected_stems: Sequence[str] | None = None,
) -> list[str]:
    path = Path(path)
    return validate_shard(
        load_shard(path), path=path, expected_split=expected_split, expected_stems=expected_stems
    )


def make_sample(record: ManifestRecord, encoded: Any) -> dict[str, Any]:
    """Convert one helper-produced encoded pair into the durable shard schema."""
    image = encoded.latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
    control = encoded.control.detach().to(device="cpu", dtype=torch.float32).contiguous()
    geometry = encoded.pair.geometry
    return {
        "stem": record.stem,
        "file_name": record.file_name,
        "text": record.text,
        "split": record.split,
        "bucket": list(geometry.bucket),
        "source_size": list(geometry.source_size),
        "resized_size": list(geometry.resized_size),
        "crop_box": list(geometry.crop_box),
        "image_latent": image,
        "control_latent": control,
    }


def write_shard_atomically(path: Path, split: str, samples: list[dict[str, Any]]) -> None:
    """Save, fsync, validate, then atomically publish one completed shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": SHARD_FORMAT_VERSION, "split": split, "samples": samples}
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        validate_shard_file(temp_name, expected_split=split, expected_stems=[s["stem"] for s in samples])
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def prepare_shards(
    *, dataset_root: str | Path, output_root: str | Path, device: str,
    shard_samples: int = DEFAULT_SHARD_SAMPLES, max_samples_per_split: int | None = None,
) -> dict[str, int]:
    if shard_samples < 1:
        raise ShardError("shard_samples must be at least 1")
    if max_samples_per_split is not None and max_samples_per_split < 1:
        raise ShardError("max_samples_per_split must be at least 1")
    validation = validate_posebridge_snapshot(dataset_root)
    records_by_split = {
        split: records if max_samples_per_split is None else records[:max_samples_per_split]
        for split, records in validation.records_by_split.items()
    }
    expected_counts = {split: len(records) for split, records in records_by_split.items()}
    output = Path(output_root).expanduser().resolve()
    # A process may fail before it writes its first shard.  Publish only an
    # explicitly incomplete manifest until every physical archive is validated.
    _write_metadata(
        output,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        expected_counts=expected_counts,
        shard_samples=shard_samples,
        complete=False,
    )
    _remove_stale_temporary_files(output)
    vae: Any | None = None
    for split, records in records_by_split.items():
        reporter = _ProgressReporter(split=split, total=len(records))
        for shard_number, chunk in enumerate(_chunks(records, shard_samples)):
            path = shard_path(output, split, shard_number)
            stems = [record.stem for record in chunk]
            if path.is_file():
                try:
                    validate_shard_file(path, expected_split=split, expected_stems=stems)
                    reporter.advance(len(chunk), shard_number, force=True, reused=True)
                    continue
                except ShardError:
                    # A final-named corrupt shard is never accepted; replace it atomically.
                    pass
            samples = []
            if vae is None:
                vae = load_krea_vae(device)
            for record in chunk:
                samples.append(
                    make_sample(record, encode_preprocessed_pair(vae, preprocess_pair(record), device=device))
                )
                reporter.advance(1, shard_number)
            write_shard_atomically(path, split, samples)
            reporter.report(shard_number, force=True)
    actual_counts = validate_shard_set(output, records_by_split, shard_samples=shard_samples)
    if actual_counts != expected_counts:
        raise ShardError(f"Completed shard counts differ: expected {expected_counts}, got {actual_counts}")
    _write_metadata(
        output,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        expected_counts=expected_counts,
        shard_samples=shard_samples,
        complete=max_samples_per_split is None,
    )
    return expected_counts


def _write_metadata(output: Path, *, dataset_root: Path, expected_counts: Mapping[str, int],
                    shard_samples: int, complete: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_root": str(dataset_root),
        "expected_counts": dict(expected_counts),
        "shard_samples": shard_samples,
        "complete": complete,
        "total_samples": sum(expected_counts.values()),
    }
    _atomic_json(output / METADATA_FILE_NAME, metadata)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_shard_set(
    output_root: str | Path, records_by_split: Mapping[str, Sequence[ManifestRecord]], *, shard_samples: int,
) -> dict[str, int]:
    """Validate every planned final shard and exact aggregate membership.

    Metadata is deliberately not an input here: recovery treats physical,
    deterministic shard ranges as the source of truth.
    """
    root = Path(output_root).expanduser().resolve()
    if shard_samples < 1:
        raise ShardError("shard_samples must be at least 1")
    planned: dict[Path, tuple[str, list[str]]] = {}
    expected_stems: dict[str, list[str]] = {}
    for split, records in records_by_split.items():
        stems = [record.stem for record in records]
        if len(stems) != len(set(stems)):
            raise ShardError(f"Duplicate planned manifest samples in split {split}")
        expected_stems[split] = stems
        for shard_number, chunk in enumerate(_chunks(records, shard_samples)):
            path = shard_path(root, split, shard_number)
            planned[path] = (split, [record.stem for record in chunk])

    actual_paths = set(root.glob("*/*.pt"))
    planned_paths = set(planned)
    missing_paths = sorted(planned_paths - actual_paths)
    extra_paths = sorted(actual_paths - planned_paths)
    if missing_paths or extra_paths:
        raise ShardError(
            "Shard file set does not match the required deterministic plan: "
            f"missing={[str(path) for path in missing_paths[:3]]}, "
            f"extra={[str(path) for path in extra_paths[:3]]}"
        )

    observed: dict[str, list[str]] = {split: [] for split in records_by_split}
    for path, (split, stems) in planned.items():
        observed[split].extend(validate_shard_file(path, expected_split=split, expected_stems=stems))
    all_stems: list[str] = []
    for split, stems in observed.items():
        if len(stems) != len(set(stems)):
            raise ShardError(f"Duplicate samples in split {split}")
        if stems != expected_stems[split]:
            raise ShardError(f"Wrong or missing aggregate sample membership in split {split}")
        all_stems.extend(stems)
    if len(all_stems) != len(set(all_stems)):
        raise ShardError("Duplicate samples across splits")
    return {split: len(stems) for split, stems in observed.items()}


def _remove_stale_temporary_files(output: Path) -> None:
    """Remove only same-directory temporary files created by our atomic writers."""
    for directory in (output, *(output / split for split in EXPECTED_SPLIT_COUNTS)):
        if directory.is_dir():
            for temporary in directory.glob(".*.tmp"):
                if temporary.is_file():
                    temporary.unlink()


class _ProgressReporter:
    """Bounded, live per-split preprocessing progress for long host runs."""

    def __init__(self, *, split: str, total: int) -> None:
        self.split = split
        self.total = total
        self.completed = 0
        self.started = time.monotonic()
        self.last_report = self.started

    def advance(self, count: int, shard_number: int, *, force: bool = False, reused: bool = False) -> None:
        self.completed += count
        self.report(shard_number, force=force, reused=reused)

    def report(self, shard_number: int, *, force: bool = False, reused: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < 5.0:
            return
        elapsed = max(now - self.started, 1e-6)
        rate = self.completed / elapsed
        remaining = (self.total - self.completed) / rate if rate else float("inf")
        eta = _format_duration(remaining)
        suffix = " (reused)" if reused else ""
        print(
            f"[{self.split}] {self.completed}/{self.total} samples | shard {shard_number} | "
            f"{rate:.2f} samples/s | elapsed {_format_duration(elapsed)} | ETA {eta}{suffix}",
            flush=True,
        )
        self.last_report = now


def _format_duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _chunks(records: Sequence[ManifestRecord], size: int) -> Iterable[Sequence[ManifestRecord]]:
    for start in range(0, len(records), size):
        yield records[start:start + size]


def _validate_sample(sample: Any, *, split: str, path: str | Path, sample_number: int) -> None:
    if not isinstance(sample, dict):
        raise ShardError(f"Malformed metadata in {path} sample {sample_number}: expected dictionary")
    for key in ("stem", "file_name", "text", "split"):
        value = sample.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ShardError(f"Malformed metadata in {path} sample {sample_number}: invalid {key}")
    if sample["split"] != split:
        raise ShardError(f"Wrong split membership in {path} sample {sample_number}")
    if sample["file_name"] != f"{sample['stem']}.jpg":
        raise ShardError(f"Malformed metadata in {path} sample {sample_number}: file_name/stem mismatch")
    for key, length in (("bucket", 2), ("source_size", 2), ("resized_size", 2), ("crop_box", 4)):
        value = sample.get(key)
        if not isinstance(value, list) or len(value) != length or not all(isinstance(x, int) and x >= 0 for x in value):
            raise ShardError(f"Malformed metadata in {path} sample {sample_number}: invalid {key}")
    if any(x <= 0 for x in sample["bucket"] + sample["source_size"] + sample["resized_size"]):
        raise ShardError(f"Malformed metadata in {path} sample {sample_number}: non-positive dimensions")
    left, top, right, bottom = sample["crop_box"]
    bucket_width, bucket_height = sample["bucket"]
    resized_width, resized_height = sample["resized_size"]
    if right - left != bucket_width or bottom - top != bucket_height or right > resized_width or bottom > resized_height:
        raise ShardError(f"Malformed metadata in {path} sample {sample_number}: invalid crop_box")
    image, control = sample.get("image_latent"), sample.get("control_latent")
    if not isinstance(image, torch.Tensor) or not isinstance(control, torch.Tensor):
        raise ShardError(f"Malformed shard {path} sample {sample_number}: missing latent tensors")
    if image.dtype != torch.float32 or control.dtype != torch.float32:
        raise ShardError(f"Shard {path} sample {sample_number}: latents must be float32")
    if image.ndim != 3 or image.shape[0] != 16 or image.shape != control.shape:
        raise ShardError(f"Shard {path} sample {sample_number}: mismatched latent shapes")
    if image.shape[1:] != (sample["bucket"][1] // 8, sample["bucket"][0] // 8):
        raise ShardError(f"Shard {path} sample {sample_number}: latent shape/bucket mismatch")
    if not torch.isfinite(image).all().item() or not torch.isfinite(control).all().item():
        raise ShardError(f"Shard {path} sample {sample_number}: nonfinite latent values")
    if control.abs().max().item() == 0.0:
        raise ShardError(f"Shard {path} sample {sample_number}: empty control latent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create atomic resumable float32 Qwen latent shards.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-samples", type=int, default=DEFAULT_SHARD_SAMPLES)
    parser.add_argument("--max-samples-per-split", type=int, help="Smoke-only bounded creation; never a full gate.")
    args = parser.parse_args()
    try:
        counts = prepare_shards(dataset_root=args.dataset_root, output_root=args.output_root, device=args.device,
                                shard_samples=args.shard_samples, max_samples_per_split=args.max_samples_per_split)
    except (DatasetIndexError, VAEPreprocessingError, ShardError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", "counts": counts, "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
