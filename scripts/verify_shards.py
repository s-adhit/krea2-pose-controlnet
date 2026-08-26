"""Hard verification gate for PoseBridge latent shards."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Direct ``python scripts/verify_shards.py`` execution needs the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_shards import METADATA_FILE_NAME, SHARD_FORMAT_VERSION, ShardError, validate_shard_file
from pose_controlnet.dataset_index import EXPECTED_SPLIT_COUNTS, DatasetIndexError, validate_posebridge_snapshot


def verify_shards(*, dataset_root: str | Path, output_root: str | Path, allow_partial: bool = False) -> dict[str, int]:
    root = Path(output_root).expanduser().resolve()
    metadata = _load_metadata(root)
    validation = validate_posebridge_snapshot(dataset_root)
    expected_counts = dict(EXPECTED_SPLIT_COUNTS) if not allow_partial else metadata["expected_counts"]
    if not allow_partial and metadata.get("complete") is not True:
        raise ShardError("Shard set is marked incomplete; full verification requires a complete set")
    if set(expected_counts) != set(EXPECTED_SPLIT_COUNTS):
        raise ShardError("Malformed metadata: expected_counts must name all immutable splits")
    expected_stems = {split: [record.stem for record in records] for split, records in validation.records_by_split.items()}
    observed: dict[str, list[str]] = {split: [] for split in EXPECTED_SPLIT_COUNTS}
    shard_files = sorted(root.glob("*/*.pt"))
    if not shard_files:
        raise ShardError(f"No shard files found beneath {root}")
    for path in shard_files:
        split = path.parent.name
        if split not in observed:
            raise ShardError(f"Malformed shard location outside immutable split directories: {path}")
        observed[split].extend(validate_shard_file(path, expected_split=split))
    for split, stems in observed.items():
        duplicates = [stem for stem, count in Counter(stems).items() if count > 1]
        if duplicates:
            raise ShardError(f"Duplicate samples in split {split}: {duplicates[:3]}")
        wanted = expected_stems[split][:expected_counts[split]] if allow_partial else expected_stems[split]
        if stems != wanted:
            missing = sorted(set(wanted) - set(stems))
            extra = sorted(set(stems) - set(wanted))
            raise ShardError(f"Wrong/missing samples for {split}: missing={missing[:3]}, extra={extra[:3]}")
        if len(stems) != expected_counts[split]:
            raise ShardError(f"Wrong total for {split}: expected {expected_counts[split]}, got {len(stems)}")
    total = sum(len(stems) for stems in observed.values())
    expected_total = sum(expected_counts.values())
    if total != expected_total:
        raise ShardError(f"Wrong total count: expected {expected_total}, got {total}")
    return {split: len(stems) for split, stems in observed.items()}


def _load_metadata(root: Path) -> dict[str, Any]:
    path = root / METADATA_FILE_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardError(f"Unreadable shard metadata {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format_version") != SHARD_FORMAT_VERSION:
        raise ShardError(f"Malformed shard metadata {path}")
    counts = value.get("expected_counts")
    if not isinstance(counts, dict) or any(not isinstance(v, int) or v < 1 for v in counts.values()):
        raise ShardError(f"Malformed shard metadata {path}: expected_counts")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard verification for PoseBridge latent shards.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Only for bounded smoke shard sets.")
    args = parser.parse_args()
    try:
        counts = verify_shards(dataset_root=args.dataset_root, output_root=args.output_root, allow_partial=args.allow_partial)
    except (DatasetIndexError, ShardError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", "counts": counts, "total": sum(counts.values())}, indent=2))


if __name__ == "__main__":
    main()
