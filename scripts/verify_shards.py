"""Hard verification gate for PoseBridge latent shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Direct ``python scripts/verify_shards.py`` execution needs the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_shards import METADATA_FILE_NAME, SHARD_FORMAT_VERSION, ShardError, validate_shard_set
from pose_controlnet.dataset_index import EXPECTED_SPLIT_COUNTS, DatasetIndexError, validate_posebridge_snapshot


def verify_shards(*, dataset_root: str | Path, output_root: str | Path, allow_partial: bool = False) -> dict[str, int]:
    root = Path(output_root).expanduser().resolve()
    metadata = _load_metadata(root)
    validation = validate_posebridge_snapshot(dataset_root)
    metadata_counts = metadata["expected_counts"]
    if not allow_partial and metadata.get("complete") is not True:
        raise ShardError("Shard set is marked incomplete; full verification requires a complete set")
    if not allow_partial and metadata_counts != dict(EXPECTED_SPLIT_COUNTS):
        raise ShardError("Complete metadata counts do not match immutable split counts")
    records_by_split = {
        split: records[:metadata_counts[split]] if allow_partial else records
        for split, records in validation.records_by_split.items()
    }
    observed = validate_shard_set(
        root, records_by_split, shard_samples=metadata["shard_samples"]
    )
    if observed != metadata_counts:
        raise ShardError(f"Metadata/physical count mismatch: metadata={metadata_counts}, actual={observed}")
    if sum(observed.values()) != metadata["total_samples"]:
        raise ShardError("Metadata/physical total mismatch")
    return observed


def _load_metadata(root: Path) -> dict[str, Any]:
    path = root / METADATA_FILE_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardError(f"Unreadable shard metadata {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format_version") != SHARD_FORMAT_VERSION:
        raise ShardError(f"Malformed shard metadata {path}")
    counts = value.get("expected_counts")
    if not isinstance(counts, dict) or set(counts) != set(EXPECTED_SPLIT_COUNTS) or any(
        not isinstance(v, int) or v < 1 or v > EXPECTED_SPLIT_COUNTS[split]
        for split, v in counts.items()
    ):
        raise ShardError(f"Malformed shard metadata {path}: expected_counts")
    if not isinstance(value.get("complete"), bool):
        raise ShardError(f"Malformed shard metadata {path}: complete")
    if not isinstance(value.get("shard_samples"), int) or value["shard_samples"] < 1:
        raise ShardError(f"Malformed shard metadata {path}: shard_samples")
    if value.get("total_samples") != sum(counts.values()):
        raise ShardError(f"Malformed shard metadata {path}: total_samples")
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
