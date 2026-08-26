"""Read-only physical-file and manifest index for the PoseBridge dataset.

The immutable manifests define split membership.  This module only discovers
where each manifest stem is physically stored in a Hugging Face snapshot; it
never derives membership from directory names or modifies dataset files.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


EXPECTED_PHYSICAL_PAIRS = 17_495
EXPECTED_SPLIT_COUNTS = MappingProxyType(
    {"train": 16_503, "val": 889, "diagnostic_val": 24}
)


class DatasetIndexError(ValueError):
    """Raised when physical files or immutable manifests violate invariants."""


@dataclass(frozen=True)
class ManifestRecord:
    """One manifest entry resolved to its unique physical RGB/control files."""

    split: str
    stem: str
    file_name: str
    text: str
    rgb_path: Path
    control_path: Path


@dataclass(frozen=True)
class ManifestValidation:
    """Resolved manifest records and their verified split counts."""

    records_by_split: Mapping[str, tuple[ManifestRecord, ...]]

    @property
    def split_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {split: len(records) for split, records in self.records_by_split.items()}
        )

    @property
    def total_records(self) -> int:
        return sum(self.split_counts.values())


@dataclass(frozen=True)
class DatasetIndex:
    """Unique stem-to-path maps discovered recursively from a dataset root."""

    dataset_root: Path
    rgb_by_stem: Mapping[str, Path]
    control_by_stem: Mapping[str, Path]

    @classmethod
    def discover(cls, dataset_root: str | Path) -> "DatasetIndex":
        root = Path(dataset_root).expanduser().resolve()
        images_root = root / "images"
        controls_root = root / "conditioning_images"
        if not images_root.is_dir() or not controls_root.is_dir():
            raise DatasetIndexError(
                "Dataset root must contain images/ and conditioning_images/: "
                f"{root}"
            )

        rgb_by_stem = _build_stem_index(images_root.rglob("*.jpg"), "RGB JPG")
        control_by_stem = _build_stem_index(
            controls_root.rglob("*.png"), "pose-control PNG"
        )
        rgb_stems = set(rgb_by_stem)
        control_stems = set(control_by_stem)
        missing_controls = sorted(rgb_stems - control_stems)
        missing_rgbs = sorted(control_stems - rgb_stems)
        if missing_controls or missing_rgbs:
            parts = []
            if missing_controls:
                parts.append(
                    f"{len(missing_controls)} RGB stem(s) lack a control counterpart "
                    f"(examples: {_examples(missing_controls)})"
                )
            if missing_rgbs:
                parts.append(
                    f"{len(missing_rgbs)} control stem(s) lack an RGB counterpart "
                    f"(examples: {_examples(missing_rgbs)})"
                )
            raise DatasetIndexError("; ".join(parts))

        return cls(
            dataset_root=root,
            rgb_by_stem=MappingProxyType(rgb_by_stem),
            control_by_stem=MappingProxyType(control_by_stem),
        )

    def resolve(self, file_name: str) -> tuple[Path, Path]:
        stem = _parse_manifest_file_name(file_name, "manifest record")
        try:
            return self.rgb_by_stem[stem], self.control_by_stem[stem]
        except KeyError as exc:
            raise DatasetIndexError(
                f"Manifest file_name {file_name!r} has no unique RGB/control pair"
            ) from exc

    def validate_manifests(
        self,
        manifests: Mapping[str, str | Path],
        *,
        expected_counts: Mapping[str, int] | None = None,
        expected_total: int | None = None,
    ) -> ManifestValidation:
        """Strictly parse manifests, resolve all rows, and require disjoint splits."""
        if not manifests:
            raise DatasetIndexError("At least one named manifest is required")
        records_by_split: dict[str, tuple[ManifestRecord, ...]] = {}
        all_stems: dict[str, str] = {}
        for split, manifest_path in manifests.items():
            if not isinstance(split, str) or not split:
                raise DatasetIndexError(f"Invalid split name: {split!r}")
            records = []
            seen_in_split: set[str] = set()
            path = Path(manifest_path)
            if not path.is_file():
                raise DatasetIndexError(f"Manifest for split {split!r} is missing: {path}")
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    record = _parse_manifest_line(path, line_number, line)
                    file_name = record["file_name"]
                    text = record["text"]
                    stem = _parse_manifest_file_name(file_name, f"{path}:{line_number}")
                    if not isinstance(text, str) or not text.strip():
                        raise DatasetIndexError(
                            f"Missing or empty caption at {path}:{line_number}"
                        )
                    if stem in seen_in_split:
                        raise DatasetIndexError(
                            f"Duplicate manifest stem {stem!r} within split {split!r}"
                        )
                    if stem in all_stems:
                        raise DatasetIndexError(
                            f"Manifest stem {stem!r} appears in both {all_stems[stem]!r} "
                            f"and {split!r} splits"
                        )
                    rgb_path, control_path = self.resolve(file_name)
                    seen_in_split.add(stem)
                    all_stems[stem] = split
                    records.append(
                        ManifestRecord(
                            split=split,
                            stem=stem,
                            file_name=file_name,
                            text=text,
                            rgb_path=rgb_path,
                            control_path=control_path,
                        )
                    )
            records_by_split[split] = tuple(records)

        validation = ManifestValidation(MappingProxyType(records_by_split))
        if expected_counts is not None:
            actual = validation.split_counts
            if dict(actual) != dict(expected_counts):
                raise DatasetIndexError(
                    f"Manifest split counts differ: expected {dict(expected_counts)}, "
                    f"got {dict(actual)}"
                )
        if expected_total is not None and validation.total_records != expected_total:
            raise DatasetIndexError(
                f"Manifest total differs: expected {expected_total}, "
                f"got {validation.total_records}"
            )
        return validation


def validate_posebridge_snapshot(dataset_root: str | Path) -> ManifestValidation:
    """Validate the production snapshot's required counts and immutable splits."""
    index = DatasetIndex.discover(dataset_root)
    if len(index.rgb_by_stem) != EXPECTED_PHYSICAL_PAIRS:
        raise DatasetIndexError(
            f"Expected {EXPECTED_PHYSICAL_PAIRS} unique RGB stems, got {len(index.rgb_by_stem)}"
        )
    if len(index.control_by_stem) != EXPECTED_PHYSICAL_PAIRS:
        raise DatasetIndexError(
            f"Expected {EXPECTED_PHYSICAL_PAIRS} unique control stems, "
            f"got {len(index.control_by_stem)}"
        )
    root = index.dataset_root
    return index.validate_manifests(
        {
            "train": root / "manifests/train.jsonl",
            "val": root / "manifests/val.jsonl",
            "diagnostic_val": root / "manifests/diagnostic_val.jsonl",
        },
        expected_counts=EXPECTED_SPLIT_COUNTS,
        expected_total=17_416,
    )


def _build_stem_index(paths: Sequence[Path] | object, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(paths):
        stem = path.stem
        if not stem:
            raise DatasetIndexError(f"Malformed {label} filename with empty stem: {path}")
        previous = result.get(stem)
        if previous is not None:
            raise DatasetIndexError(
                f"Duplicate {label} stem {stem!r}: {previous} and {path}"
            )
        result[stem] = path.resolve()
    return result


def _parse_manifest_line(path: Path, line_number: int, line: str) -> dict[str, object]:
    if not line.strip():
        raise DatasetIndexError(f"Malformed manifest: blank line at {path}:{line_number}")
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetIndexError(
            f"Malformed manifest JSON at {path}:{line_number}: {exc.msg}"
        ) from exc
    if not isinstance(record, dict):
        raise DatasetIndexError(f"Malformed manifest record at {path}:{line_number}: expected object")
    if "file_name" not in record or "text" not in record:
        raise DatasetIndexError(
            f"Malformed manifest record at {path}:{line_number}: requires file_name and text"
        )
    if not isinstance(record["file_name"], str):
        raise DatasetIndexError(f"Malformed file_name at {path}:{line_number}: expected string")
    return record


def _parse_manifest_file_name(file_name: str, context: str) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise DatasetIndexError(f"Malformed file_name in {context}: expected non-empty string")
    if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
        raise DatasetIndexError(f"Manifest file_name must be bare <stem>.jpg in {context}: {file_name!r}")
    path = Path(file_name)
    if path.suffix != ".jpg" or not path.stem:
        raise DatasetIndexError(f"Manifest file_name must be bare <stem>.jpg in {context}: {file_name!r}")
    return path.stem


def _examples(stems: list[str], limit: int = 3) -> str:
    return ", ".join(repr(stem) for stem in stems[:limit])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a read-only PoseBridge HF snapshot.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    validation = validate_posebridge_snapshot(args.dataset_root)
    print(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "unique_rgb_stems": EXPECTED_PHYSICAL_PAIRS,
                "unique_control_stems": EXPECTED_PHYSICAL_PAIRS,
                "split_counts": dict(validation.split_counts),
                "total_used_manifest_records": validation.total_records,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
