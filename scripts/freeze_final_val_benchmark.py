"""Validate human selections and freeze the final 48-image validation benchmark.

The candidate files and split manifests are inputs only.  This script writes a
deterministic JSONL selection that can later be supplied to evaluation-spec
creation without re-running manual selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CANDIDATE_POOL_SHA256 = (
    "a72607f65d104ed09a083588bb210b8fb4e7ab22db3f2224ba939b838d906056"
)
QUOTAS = {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8}
REVIEW_FIELDS = (
    "keep", "difficulty", "pose_type", "multi_person", "notes", "stem", "source",
    "orientation", "width", "height", "aspect_ratio", "rgb_path", "pose_path", "text",
)
PRESERVED_REVIEW_FIELDS = ("difficulty", "pose_type", "multi_person", "notes")


class BenchmarkFreezeError(ValueError):
    """Raised when a review cannot safely become a frozen benchmark."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkFreezeError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise BenchmarkFreezeError(f"{path}:{line_number}: record must be an object")
            rows.append(row)
    return rows


def _stem(record: dict[str, Any], path: Path) -> str:
    file_name = record.get("file_name")
    if not isinstance(file_name, str) or Path(file_name).suffix.lower() != ".jpg":
        raise BenchmarkFreezeError(f"{path}: manifest record has invalid file_name: {file_name!r}")
    return Path(file_name).stem


def _unique_by_stem(rows: list[dict[str, Any]], *, path: Path, manifest: bool = False) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        stem = _stem(row, path) if manifest else row.get("stem")
        if not isinstance(stem, str) or not stem:
            raise BenchmarkFreezeError(f"{path}: record has invalid stem: {stem!r}")
        if stem in indexed:
            raise BenchmarkFreezeError(f"{path}: duplicate stem: {stem}")
        indexed[stem] = row
    return indexed


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BenchmarkFreezeError(f"{path}: missing CSV header")
        missing = [field for field in REVIEW_FIELDS if field not in reader.fieldnames]
        if missing:
            raise BenchmarkFreezeError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)
    return rows


def freeze_benchmark(
    *,
    candidate_pool: Path,
    candidate_review: Path,
    val_manifest: Path,
    diagnostic_manifest: Path,
    output: Path,
    expected_pool_sha256: str = EXPECTED_CANDIDATE_POOL_SHA256,
) -> list[dict[str, Any]]:
    """Validate inputs and atomically write sorted frozen-selection JSONL."""
    observed_hash = _sha256(candidate_pool)
    if observed_hash != expected_pool_sha256:
        raise BenchmarkFreezeError(
            f"candidate pool SHA256 mismatch: expected {expected_pool_sha256}, got {observed_hash}"
        )

    candidates = _unique_by_stem(_read_jsonl(candidate_pool), path=candidate_pool)
    val = _unique_by_stem(_read_jsonl(val_manifest), path=val_manifest, manifest=True)
    diagnostic = _unique_by_stem(_read_jsonl(diagnostic_manifest), path=diagnostic_manifest, manifest=True)
    overlap = set(val) & set(diagnostic)
    if overlap:
        raise BenchmarkFreezeError(f"validation/diagnostic manifests overlap at stem: {sorted(overlap)[0]}")

    review_rows = _read_review(candidate_review)
    reviewed: set[str] = set()
    selected: list[dict[str, Any]] = []
    for line_number, review in enumerate(review_rows, 2):
        stem = (review.get("stem") or "").strip()
        if not stem:
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: missing stem")
        if stem in reviewed:
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: duplicate stem: {stem}")
        reviewed.add(stem)
        if stem not in candidates:
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: non-candidate stem: {stem}")
        if stem not in val:
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: stem is not in val manifest: {stem}")
        if stem in diagnostic:
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: stem overlaps diagnostic manifest: {stem}")

        candidate, val_row = candidates[stem], val[stem]
        if review.get("text") != candidate.get("text") or candidate.get("text") != val_row.get("text"):
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: caption mismatch for stem: {stem}")
        if review.get("source") != candidate.get("source"):
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: source mismatch for stem: {stem}")

        keep = (review.get("keep") or "").strip().lower()
        if keep not in {"", "no", "yes"}:
            raise BenchmarkFreezeError(f"{candidate_review}:{line_number}: keep must be yes, no, or blank")
        if keep == "yes":
            frozen = dict(candidate)
            frozen["candidate_pool_sha256"] = observed_hash
            for field in PRESERVED_REVIEW_FIELDS:
                frozen[field] = review.get(field, "")
            selected.append(frozen)

    if len(selected) != 48:
        raise BenchmarkFreezeError(f"expected exactly 48 keep=yes rows, found {len(selected)}")
    counts = Counter(row["source"] for row in selected)
    if dict(sorted(counts.items())) != QUOTAS:
        raise BenchmarkFreezeError(f"source quotas must be {QUOTAS}, found {dict(sorted(counts.items()))}")

    selected.sort(key=lambda row: (row["source"], row["stem"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, output)
    return selected


def main() -> None:
    root = Path("docs/evaluation/final-val-benchmark-selection")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=root / "candidate_pool_96.jsonl")
    parser.add_argument("--candidate-review", type=Path, default=root / "candidate_review.csv")
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.jsonl"))
    parser.add_argument("--diagnostic-manifest", type=Path, default=Path("data/manifests/diagnostic_val.jsonl"))
    parser.add_argument("--output", type=Path, default=root / "final_val_benchmark_48.jsonl")
    args = parser.parse_args()
    frozen = freeze_benchmark(
        candidate_pool=args.candidate_pool, candidate_review=args.candidate_review,
        val_manifest=args.val_manifest, diagnostic_manifest=args.diagnostic_manifest,
        output=args.output,
    )
    print(f"froze {len(frozen)} validation selections at {args.output}")


if __name__ == "__main__":
    main()
