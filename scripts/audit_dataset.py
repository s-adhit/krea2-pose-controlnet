#!/usr/bin/env python3
"""
Pose-ControlNet Dataset Fingerprint / Audit

Read-only audit of the consolidated training dataset.

Expected layout:

Krea-2-Pose-ControlNet/
├── data/
│   └── full/
│       ├── images/
│       ├── conditioning_images/
│       └── metadata.jsonl
└── scripts/
    └── audit_dataset.py

The script does NOT modify the dataset.

It checks:
- image count
- conditioning-image count
- metadata row count
- image/control filename alignment
- duplicate filenames
- missing/orphan files
- corrupt images
- image dimensions/formats
- conditioning dimensions/formats
- caption lengths
- source/bucket counts when source information is available
- aspect-ratio distribution
- control-map sparsity

Outputs:
data/stats/dataset_fingerprint.json
data/stats/dataset_fingerprint.txt
data/stats/image_dimensions.csv
data/stats/caption_lengths.csv
data/stats/aspect_ratios.csv
data/stats/control_sparsity.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageFile
except ImportError:
    print("ERROR: Pillow is required.")
    print("Install with: pip install pillow")
    sys.exit(1)

# Allow Pillow to detect truncated/corrupt images rather than silently accepting them.
ImageFile.LOAD_TRUNCATED_IMAGES = False


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

CONTROL_EXTENSIONS = IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a read-only fingerprint of the PoseBridge training dataset."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Path to data/full, containing images/, "
            "conditioning_images/, and metadata.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <project>/data/stats, "
            "where <project> is the parent of data/full."
        ),
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=17495,
        help="Expected number of examples. Default: 17495.",
    )

    parser.add_argument(
        "--hash",
        action="store_true",
        help=(
            "Calculate SHA256 hashes for files. This is slower but gives "
            "a stronger duplicate/fingerprint check."
        ),
    )

    parser.add_argument(
        "--control-threshold",
        type=int,
        default=10,
        help=(
            "Pixel intensity threshold for considering a control pixel "
            "non-background. Default: 10."
        ),
    )

    return parser.parse_args()


def normalize_stem(path: Path) -> str:
    """
    Return a normalized filename stem.

    Example:
        person_001.jpg -> person_001
        person_001.png -> person_001
    """
    return path.stem


def discover_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def load_metadata(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load metadata.jsonl.

    Returns:
        rows
        parse_errors
    """
    rows = []
    errors = []

    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                if not isinstance(obj, dict):
                    raise ValueError("JSONL row is not an object")

                rows.append(obj)

            except Exception as exc:
                errors.append(
                    {
                        "line": line_number,
                        "error": str(exc),
                        "raw_prefix": line[:200],
                    }
                )

    return rows, errors


def get_metadata_filename(row: dict[str, Any]) -> str | None:
    """
    Expected field from the Krea data pipeline:
        file_name

    We also tolerate a few common alternatives for diagnostics,
    but the report will explicitly state which field was used.
    """
    for key in ("file_name", "filename", "image", "image_path"):
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def get_caption(row: dict[str, Any]) -> str:
    value = row.get("text", "")

    if value is None:
        return ""

    return str(value)


def get_source(row: dict[str, Any]) -> str | None:
    """
    Prefer explicit source metadata.

    The handoff establishes the five source buckets, but does not state
    that source is necessarily stored as a metadata.jsonl field.
    Therefore we do not silently invent a source field.

    If source information exists, use it.
    """
    for key in (
        "source",
        "dataset",
        "bucket",
        "category",
        "subset",
    ):
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    """
    Open and fully verify an image.

    Returns a structured diagnostic record.
    """
    result: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "ok": False,
        "format": None,
        "mode": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "pixel_count": None,
        "error": None,
    }

    try:
        with Image.open(path) as img:
            result["format"] = img.format
            result["mode"] = img.mode
            result["width"] = img.width
            result["height"] = img.height
            result["pixel_count"] = img.width * img.height

            if img.height:
                result["aspect_ratio"] = img.width / img.height

            # Force decoding of image data.
            img.load()

        # Separate verify pass.
        with Image.open(path) as img:
            img.verify()

        result["ok"] = True

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def analyze_control_sparsity(
    path: Path,
    threshold: int,
) -> dict[str, Any]:
    """
    Analyze how much of a control image contains non-background pixels.

    This is deliberately simple and diagnostic.

    For RGB/RGBA images, grayscale luminance is used.
    For grayscale images, the existing intensity is used.

    Metrics:
        nonzero_fraction
        mean_intensity
        max_intensity
        bbox_fraction

    The bbox fraction measures the fraction of the image occupied by
    pixels above the threshold.
    """
    result: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "width": None,
        "height": None,
        "nonzero_fraction": None,
        "mean_intensity": None,
        "max_intensity": None,
        "bbox_fraction": None,
        "error": None,
    }

    try:
        with Image.open(path) as img:
            img = img.convert("L")
            width, height = img.size

            result["width"] = width
            result["height"] = height

            pixels = list(img.getdata())

            if not pixels:
                raise ValueError("Control image contains no pixels")

            total = len(pixels)

            active = [
                (idx, value)
                for idx, value in enumerate(pixels)
                if value > threshold
            ]

            active_values = [value for _, value in active]

            result["nonzero_fraction"] = len(active) / total
            result["mean_intensity"] = sum(pixels) / total
            result["max_intensity"] = max(pixels)

            if active:
                xs = []
                ys = []

                for idx, _ in active:
                    x = idx % width
                    y = idx // width

                    xs.append(x)
                    ys.append(y)

                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)

                bbox_width = max_x - min_x + 1
                bbox_height = max_y - min_y + 1

                result["bbox_fraction"] = (
                    bbox_width * bbox_height
                ) / (width * height)

            else:
                result["bbox_fraction"] = 0.0

            result["ok"] = True

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"

    if value is None:
        return "N/A"

    return str(value)


def build_text_report(
    fingerprint: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("POSE-CONTROLNET DATASET FINGERPRINT")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"Generated: {fingerprint['generated_at']}")
    lines.append(f"Data root: {fingerprint['data_root']}")
    lines.append("")

    lines.append("EXPECTED / ACTUAL COUNTS")
    lines.append("-" * 72)

    counts = fingerprint["counts"]

    for key, value in counts.items():
        lines.append(f"{key:35s}: {value}")

    lines.append("")

    lines.append("ALIGNMENT")
    lines.append("-" * 72)

    alignment = fingerprint["alignment"]

    for key, value in alignment.items():
        lines.append(f"{key:35s}: {value}")

    lines.append("")

    lines.append("CORRUPT / INVALID FILES")
    lines.append("-" * 72)

    corrupt = fingerprint["corruption"]

    lines.append(
        f"{'Corrupt source images':35s}: "
        f"{len(corrupt['images'])}"
    )
    lines.append(
        f"{'Corrupt conditioning images':35s}: "
        f"{len(corrupt['conditioning_images'])}"
    )
    lines.append(
        f"{'Metadata parse errors':35s}: "
        f"{corrupt['metadata_parse_errors']}"
    )

    lines.append("")

    lines.append("DUPLICATES")
    lines.append("-" * 72)

    duplicates = fingerprint["duplicates"]

    lines.append(
        f"{'Duplicate image stems':35s}: "
        f"{len(duplicates['image_stems'])}"
    )
    lines.append(
        f"{'Duplicate control stems':35s}: "
        f"{len(duplicates['control_stems'])}"
    )

    if "image_sha256" in duplicates:
        lines.append(
            f"{'Duplicate image SHA256 groups':35s}: "
            f"{len(duplicates['image_sha256'])}"
        )

    if "control_sha256" in duplicates:
        lines.append(
            f"{'Duplicate control SHA256 groups':35s}: "
            f"{len(duplicates['control_sha256'])}"
        )

    lines.append("")

    lines.append("IMAGE STATISTICS")
    lines.append("-" * 72)

    image_stats = fingerprint["image_statistics"]

    for key, value in image_stats.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k:31s}: {format_number(v)}")
        else:
            lines.append(f"{key:35s}: {format_number(value)}")

    lines.append("")

    lines.append("CONDITIONING IMAGE STATISTICS")
    lines.append("-" * 72)

    control_stats = fingerprint["conditioning_statistics"]

    for key, value in control_stats.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k:31s}: {format_number(v)}")
        else:
            lines.append(f"{key:35s}: {format_number(value)}")

    lines.append("")

    lines.append("CAPTION STATISTICS")
    lines.append("-" * 72)

    caption_stats = fingerprint["caption_statistics"]

    for key, value in caption_stats.items():
        lines.append(f"{key:35s}: {format_number(value)}")

    lines.append("")

    lines.append("SOURCE COUNTS")
    lines.append("-" * 72)

    source_info = fingerprint["source_statistics"]

    lines.append(
        f"Source field detected: "
        f"{source_info['source_field_detected']}"
    )
    lines.append(
        f"Source field name: "
        f"{source_info['source_field_name'] or 'N/A'}"
    )

    for source, count in source_info["counts"].items():
        lines.append(f"{source:35s}: {count}")

    if source_info["note"]:
        lines.append("")
        lines.append(f"NOTE: {source_info['note']}")

    lines.append("")

    lines.append("ASPECT-RATIO DISTRIBUTION")
    lines.append("-" * 72)

    for bucket, count in fingerprint["aspect_ratio_buckets"].items():
        lines.append(f"{bucket:20s}: {count}")

    lines.append("")

    lines.append("CONTROL SPARSITY")
    lines.append("-" * 72)

    sparsity = fingerprint["control_sparsity"]

    for key, value in sparsity.items():
        lines.append(f"{key:35s}: {format_number(value)}")

    lines.append("")

    lines.append("STATUS")
    lines.append("-" * 72)

    status = fingerprint["status"]

    lines.append(
        f"PASS: {status['pass']}"
    )

    if status["warnings"]:
        lines.append("")
        lines.append("WARNINGS:")

        for warning in status["warnings"]:
            lines.append(f"  - {warning}")

    if status["errors"]:
        lines.append("")
        lines.append("ERRORS:")

        for error in status["errors"]:
            lines.append(f"  - {error}")

    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    data_root = args.data_root.resolve()

    images_dir = data_root / "images"
    controls_dir = data_root / "conditioning_images"
    metadata_path = data_root / "metadata.jsonl"

    if args.output_dir is None:
        # data/full -> project root is ../../
        output_dir = data_root.parent.parent / "data" / "stats"
    else:
        output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    print("=" * 72)
    print("POSE-CONTROLNET DATASET AUDIT")
    print("=" * 72)
    print(f"Data root: {data_root}")
    print(f"Output:    {output_dir}")
    print("")

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------

    print("[1/8] Discovering files...")

    image_files = discover_files(
        images_dir,
        IMAGE_EXTENSIONS,
    )

    control_files = discover_files(
        controls_dir,
        CONTROL_EXTENSIONS,
    )

    print(f"  Images:              {len(image_files):,}")
    print(f"  Conditioning images: {len(control_files):,}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    print("[2/8] Loading metadata.jsonl...")

    metadata_rows, metadata_errors = load_metadata(
        metadata_path
    )

    print(f"  Metadata rows:       {len(metadata_rows):,}")
    print(f"  Parse errors:        {len(metadata_errors):,}")

    # ------------------------------------------------------------------
    # Filename alignment
    # ------------------------------------------------------------------

    print("[3/8] Checking filename alignment...")

    image_by_stem: dict[str, list[Path]] = defaultdict(list)
    control_by_stem: dict[str, list[Path]] = defaultdict(list)

    for path in image_files:
        image_by_stem[normalize_stem(path)].append(path)

    for path in control_files:
        control_by_stem[normalize_stem(path)].append(path)

    duplicate_image_stems = {
        stem: [str(p) for p in paths]
        for stem, paths in image_by_stem.items()
        if len(paths) > 1
    }

    duplicate_control_stems = {
        stem: [str(p) for p in paths]
        for stem, paths in control_by_stem.items()
        if len(paths) > 1
    }

    image_stems = set(image_by_stem)
    control_stems = set(control_by_stem)

    missing_controls = sorted(image_stems - control_stems)
    orphan_controls = sorted(control_stems - image_stems)

    # Metadata filenames
    metadata_filenames: list[str] = []
    metadata_stems: list[str] = []
    metadata_missing_field: list[int] = []

    source_field_counter: Counter[str] = Counter()

    for idx, row in enumerate(metadata_rows, start=1):
        filename = get_metadata_filename(row)

        if filename is None:
            metadata_missing_field.append(idx)
            continue

        metadata_filenames.append(filename)
        metadata_stems.append(Path(filename).stem)

        for key in (
            "source",
            "dataset",
            "bucket",
            "category",
            "subset",
        ):
            if isinstance(row.get(key), str) and row[key].strip():
                source_field_counter[key] += 1

    metadata_stem_set = set(metadata_stems)

    metadata_missing_images = sorted(
        metadata_stem_set - image_stems
    )

    metadata_orphan_image_stems = sorted(
        image_stems - metadata_stem_set
    )

    duplicate_metadata_stems = {
        stem: count
        for stem, count in Counter(metadata_stems).items()
        if count > 1
    }

    # Determine source field if one exists.
    source_field_name = (
        source_field_counter.most_common(1)[0][0]
        if source_field_counter
        else None
    )

    # ------------------------------------------------------------------
    # Inspect source and control images
    # ------------------------------------------------------------------

    print("[4/8] Inspecting source images...")

    image_inspections = []

    for i, path in enumerate(image_files, start=1):
        result = inspect_image(path)
        image_inspections.append(result)

        if i % 500 == 0 or i == len(image_files):
            print(
                f"  Source images: {i:,}/{len(image_files):,}",
                end="\r",
            )

    print("")

    print("[5/8] Inspecting conditioning images...")

    control_inspections = []

    for i, path in enumerate(control_files, start=1):
        result = inspect_image(path)
        control_inspections.append(result)

        if i % 500 == 0 or i == len(control_files):
            print(
                f"  Control images: {i:,}/{len(control_files):,}",
                end="\r",
            )

    print("")

    corrupt_images = [
        r for r in image_inspections if not r["ok"]
    ]

    corrupt_controls = [
        r for r in control_inspections if not r["ok"]
    ]

    # ------------------------------------------------------------------
    # Control sparsity
    # ------------------------------------------------------------------

    print("[6/8] Analyzing control-map sparsity...")

    control_sparsity_rows = []

    sparsity_nonzero = []
    sparsity_mean = []
    sparsity_bbox = []

    for i, path in enumerate(control_files, start=1):
        result = analyze_control_sparsity(
            path,
            args.control_threshold,
        )

        control_sparsity_rows.append(result)

        if result["ok"]:
            sparsity_nonzero.append(
                result["nonzero_fraction"]
            )
            sparsity_mean.append(
                result["mean_intensity"]
            )
            sparsity_bbox.append(
                result["bbox_fraction"]
            )

        if i % 500 == 0 or i == len(control_files):
            print(
                f"  Controls: {i:,}/{len(control_files):,}",
                end="\r",
            )

    print("")

    # ------------------------------------------------------------------
    # Caption statistics
    # ------------------------------------------------------------------

    print("[7/8] Analyzing captions...")

    caption_rows = []

    caption_lengths = []
    caption_words = []

    for idx, row in enumerate(metadata_rows, start=1):
        caption = get_caption(row)

        chars = len(caption)
        words = len(caption.split())

        caption_lengths.append(chars)
        caption_words.append(words)

        caption_rows.append(
            {
                "metadata_row": idx,
                "file_name": get_metadata_filename(row),
                "caption_chars": chars,
                "caption_words": words,
            }
        )

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    print("[8/8] Building fingerprint...")

    image_dimensions = Counter(
        (
            r["width"],
            r["height"],
        )
        for r in image_inspections
        if r["ok"]
    )

    control_dimensions = Counter(
        (
            r["width"],
            r["height"],
        )
        for r in control_inspections
        if r["ok"]
    )

    image_formats = Counter(
        r["format"]
        for r in image_inspections
        if r["ok"]
    )

    control_formats = Counter(
        r["format"]
        for r in control_inspections
        if r["ok"]
    )

    image_modes = Counter(
        r["mode"]
        for r in image_inspections
        if r["ok"]
    )

    control_modes = Counter(
        r["mode"]
        for r in control_inspections
        if r["ok"]
    )

    aspect_ratios = [
        r["aspect_ratio"]
        for r in image_inspections
        if r["ok"] and r["aspect_ratio"] is not None
    ]

    # Broad aspect-ratio bins for a human-readable distribution.
    ratio_bins = [
        (0.00, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.00),
        (1.00, 1.10),
        (1.10, 1.25),
        (1.25, 1.40),
        (1.40, 1.60),
        (1.60, 1.80),
        (1.80, 2.00),
        (2.00, 999.00),
    ]

    aspect_ratio_bucket_counts: Counter[str] = Counter()

    for ratio in aspect_ratios:
        assigned = False

        for low, high in ratio_bins:
            if low <= ratio < high:
                label = f"{low:.2f}-{high:.2f}"
                aspect_ratio_bucket_counts[label] += 1
                assigned = True
                break

        if not assigned:
            aspect_ratio_bucket_counts["other"] += 1

    # ------------------------------------------------------------------
    # Source statistics
    # ------------------------------------------------------------------

    source_counts: Counter[str] = Counter()

    if source_field_name:
        for row in metadata_rows:
            value = row.get(source_field_name)

            if isinstance(value, str) and value.strip():
                source_counts[value.strip()] += 1

        source_note = ""
    else:
        source_note = (
            "No explicit source/dataset/bucket field was found in "
            "metadata.jsonl. Source counts were therefore NOT inferred "
            "from filenames. If you want authoritative source counts, "
            "we should add/use the source information from the original "
            "manifests rather than guessing."
        )

    # ------------------------------------------------------------------
    # Optional SHA256 duplicate detection
    # ------------------------------------------------------------------

    duplicate_sha_image: dict[str, list[str]] = {}
    duplicate_sha_control: dict[str, list[str]] = {}

    if args.hash:
        print("Calculating SHA256 hashes...")

        image_hashes: dict[str, list[str]] = defaultdict(list)

        for i, path in enumerate(image_files, start=1):
            digest = file_sha256(path)
            image_hashes[digest].append(str(path))

            if i % 500 == 0 or i == len(image_files):
                print(
                    f"  Image hashes: {i:,}/{len(image_files):,}",
                    end="\r",
                )

        print("")

        duplicate_sha_image = {
            digest: paths
            for digest, paths in image_hashes.items()
            if len(paths) > 1
        }

        control_hashes: dict[str, list[str]] = defaultdict(list)

        for i, path in enumerate(control_files, start=1):
            digest = file_sha256(path)
            control_hashes[digest].append(str(path))

            if i % 500 == 0 or i == len(control_files):
                print(
                    f"  Control hashes: {i:,}/{len(control_files):,}",
                    end="\r",
                )

        print("")

        duplicate_sha_control = {
            digest: paths
            for digest, paths in control_hashes.items()
            if len(paths) > 1
        }

    # ------------------------------------------------------------------
    # Alignment by filename
    # ------------------------------------------------------------------

    image_names = {p.name for p in image_files}
    control_names = {p.name for p in control_files}

    # The reference pipeline matches controls by stem rather than
    # requiring identical extensions.
    alignment_ok = (
        len(image_files) == len(control_files)
        and not missing_controls
        and not orphan_controls
        and not metadata_missing_images
        and not metadata_orphan_image_stems
        and not metadata_missing_field
        and not duplicate_metadata_stems
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    image_pixel_counts = [
        r["pixel_count"]
        for r in image_inspections
        if r["ok"] and r["pixel_count"] is not None
    ]

    control_pixel_counts = [
        r["pixel_count"]
        for r in control_inspections
        if r["ok"] and r["pixel_count"] is not None
    ]

    fingerprint: dict[str, Any] = {
        "fingerprint_version": "1.0",
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "data_root": str(data_root),
        "expected_count": args.expected_count,

        "counts": {
            "expected_examples": args.expected_count,
            "image_files": len(image_files),
            "conditioning_image_files": len(control_files),
            "metadata_rows": len(metadata_rows),
        },

        "alignment": {
            "image_control_count_match": (
                len(image_files) == len(control_files)
            ),
            "missing_controls": len(missing_controls),
            "orphan_controls": len(orphan_controls),
            "metadata_missing_filename": len(
                metadata_missing_field
            ),
            "metadata_stems_missing_images": len(
                metadata_missing_images
            ),
            "images_missing_metadata": len(
                metadata_orphan_image_stems
            ),
            "duplicate_metadata_stems": len(
                duplicate_metadata_stems
            ),
            "overall_alignment_pass": alignment_ok,
        },

        "corruption": {
            "images": [
                {
                    "path": r["path"],
                    "error": r["error"],
                }
                for r in corrupt_images
            ],
            "conditioning_images": [
                {
                    "path": r["path"],
                    "error": r["error"],
                }
                for r in corrupt_controls
            ],
            "metadata_parse_errors": len(
                metadata_errors
            ),
            "metadata_parse_error_details": metadata_errors,
        },

        "duplicates": {
            "image_stems": duplicate_image_stems,
            "control_stems": duplicate_control_stems,
            "metadata_stems": duplicate_metadata_stems,
        },

        "image_statistics": {
            "valid_images": len(
                [r for r in image_inspections if r["ok"]]
            ),
            "invalid_images": len(corrupt_images),
            "formats": counter_to_dict(image_formats),
            "modes": counter_to_dict(image_modes),
            "dimensions": {
                f"{w}x{h}": count
                for (w, h), count in image_dimensions.items()
            },
            "pixel_count": {
                "min": min(image_pixel_counts)
                if image_pixel_counts
                else None,
                "max": max(image_pixel_counts)
                if image_pixel_counts
                else None,
                "median": percentile(
                    [float(x) for x in image_pixel_counts],
                    0.50,
                ),
            },
        },

        "conditioning_statistics": {
            "valid_images": len(
                [r for r in control_inspections if r["ok"]]
            ),
            "invalid_images": len(corrupt_controls),
            "formats": counter_to_dict(control_formats),
            "modes": counter_to_dict(control_modes),
            "dimensions": {
                f"{w}x{h}": count
                for (w, h), count in control_dimensions.items()
            },
            "pixel_count": {
                "min": min(control_pixel_counts)
                if control_pixel_counts
                else None,
                "max": max(control_pixel_counts)
                if control_pixel_counts
                else None,
                "median": percentile(
                    [float(x) for x in control_pixel_counts],
                    0.50,
                ),
            },
        },

        "caption_statistics": {
            "rows": len(caption_lengths),
            "empty_captions": sum(
                x == 0 for x in caption_lengths
            ),
            "chars": {
                "min": min(caption_lengths)
                if caption_lengths
                else None,
                "max": max(caption_lengths)
                if caption_lengths
                else None,
                "mean": (
                    sum(caption_lengths) / len(caption_lengths)
                    if caption_lengths
                    else None
                ),
                "median": percentile(
                    [float(x) for x in caption_lengths],
                    0.50,
                ),
                "p90": percentile(
                    [float(x) for x in caption_lengths],
                    0.90,
                ),
            },
            "words": {
                "min": min(caption_words)
                if caption_words
                else None,
                "max": max(caption_words)
                if caption_words
                else None,
                "mean": (
                    sum(caption_words) / len(caption_words)
                    if caption_words
                    else None
                ),
                "median": percentile(
                    [float(x) for x in caption_words],
                    0.50,
                ),
                "p90": percentile(
                    [float(x) for x in caption_words],
                    0.90,
                ),
            },
        },

        "source_statistics": {
            "source_field_detected": bool(
                source_field_name
            ),
            "source_field_name": source_field_name,
            "counts": counter_to_dict(source_counts),
            "note": source_note,
        },

        "aspect_ratio_statistics": {
            "valid_images": len(aspect_ratios),
            "min": min(aspect_ratios)
            if aspect_ratios
            else None,
            "max": max(aspect_ratios)
            if aspect_ratios
            else None,
            "mean": (
                sum(aspect_ratios) / len(aspect_ratios)
                if aspect_ratios
                else None
            ),
            "median": percentile(
                aspect_ratios,
                0.50,
            ),
            "p10": percentile(
                aspect_ratios,
                0.10,
            ),
            "p90": percentile(
                aspect_ratios,
                0.90,
            ),
        },

        "aspect_ratio_buckets": counter_to_dict(
            aspect_ratio_bucket_counts
        ),

        "control_sparsity": {
            "threshold": args.control_threshold,
            "valid_controls": len(
                sparsity_nonzero
            ),
            "mean_nonzero_fraction": (
                sum(sparsity_nonzero)
                / len(sparsity_nonzero)
                if sparsity_nonzero
                else None
            ),
            "median_nonzero_fraction": percentile(
                sparsity_nonzero,
                0.50,
            ),
            "p10_nonzero_fraction": percentile(
                sparsity_nonzero,
                0.10,
            ),
            "p90_nonzero_fraction": percentile(
                sparsity_nonzero,
                0.90,
            ),
            "mean_intensity": (
                sum(sparsity_mean)
                / len(sparsity_mean)
                if sparsity_mean
                else None
            ),
            "median_bbox_fraction": percentile(
                sparsity_bbox,
                0.50,
            ),
            "all_black_controls": sum(
                x == 0 for x in sparsity_nonzero
            ),
        },

        "status": {
            "pass": True,
            "warnings": [],
            "errors": [],
        },
    }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    errors = []
    warnings = []

    if len(image_files) != args.expected_count:
        warnings.append(
            f"Expected {args.expected_count:,} images, "
            f"found {len(image_files):,}."
        )

    if len(control_files) != args.expected_count:
        warnings.append(
            f"Expected {args.expected_count:,} conditioning images, "
            f"found {len(control_files):,}."
        )

    if len(metadata_rows) != args.expected_count:
        warnings.append(
            f"Expected {args.expected_count:,} metadata rows, "
            f"found {len(metadata_rows):,}."
        )

    if missing_controls:
        errors.append(
            f"{len(missing_controls):,} source images have no "
            "matching conditioning image."
        )

    if orphan_controls:
        errors.append(
            f"{len(orphan_controls):,} conditioning images have "
            "no matching source image."
        )

    if metadata_missing_field:
        errors.append(
            f"{len(metadata_missing_field):,} metadata rows have "
            "no usable file_name field."
        )

    if metadata_missing_images:
        errors.append(
            f"{len(metadata_missing_images):,} metadata image stems "
            "do not exist in images/."
        )

    if metadata_orphan_image_stems:
        errors.append(
            f"{len(metadata_orphan_image_stems):,} images have "
            "no metadata row."
        )

    if duplicate_metadata_stems:
        errors.append(
            f"{len(duplicate_metadata_stems):,} metadata stems "
            "occur more than once."
        )

    if duplicate_image_stems:
        errors.append(
            f"{len(duplicate_image_stems):,} duplicate image stems."
        )

    if duplicate_control_stems:
        errors.append(
            f"{len(duplicate_control_stems):,} duplicate control stems."
        )

    if corrupt_images:
        errors.append(
            f"{len(corrupt_images):,} corrupt source images."
        )

    if corrupt_controls:
        errors.append(
            f"{len(corrupt_controls):,} corrupt conditioning images."
        )

    if metadata_errors:
        errors.append(
            f"{len(metadata_errors):,} metadata JSONL parse errors."
        )

    if not source_field_name:
        warnings.append(
            "No explicit source field was found in metadata. "
            "Source counts were not guessed from filenames."
        )

    if fingerprint["control_sparsity"]["all_black_controls"] > 0:
        errors.append(
            f"{fingerprint['control_sparsity']['all_black_controls']:,} "
            "conditioning images appear completely black at the "
            f"threshold of {args.control_threshold}."
        )

    fingerprint["status"]["errors"] = errors
    fingerprint["status"]["warnings"] = warnings
    fingerprint["status"]["pass"] = len(errors) == 0

    # ------------------------------------------------------------------
    # Write detailed CSVs
    # ------------------------------------------------------------------

    write_csv(
        output_dir / "image_dimensions.csv",
        image_inspections,
        [
            "path",
            "filename",
            "stem",
            "extension",
            "ok",
            "format",
            "mode",
            "width",
            "height",
            "aspect_ratio",
            "pixel_count",
            "error",
        ],
    )

    write_csv(
        output_dir / "caption_lengths.csv",
        caption_rows,
        [
            "metadata_row",
            "file_name",
            "caption_chars",
            "caption_words",
        ],
    )

    aspect_rows = []

    for result in image_inspections:
        if result["ok"]:
            aspect_rows.append(
                {
                    "filename": result["filename"],
                    "width": result["width"],
                    "height": result["height"],
                    "aspect_ratio": result["aspect_ratio"],
                }
            )

    write_csv(
        output_dir / "aspect_ratios.csv",
        aspect_rows,
        [
            "filename",
            "width",
            "height",
            "aspect_ratio",
        ],
    )

    write_csv(
        output_dir / "control_sparsity.csv",
        control_sparsity_rows,
        [
            "path",
            "ok",
            "width",
            "height",
            "nonzero_fraction",
            "mean_intensity",
            "max_intensity",
            "bbox_fraction",
            "error",
        ],
    )

    # Optional duplicate hashes.
    if args.hash:
        fingerprint["duplicates"]["image_sha256"] = (
            duplicate_sha_image
        )
        fingerprint["duplicates"]["control_sha256"] = (
            duplicate_sha_control
        )

    # Add a few representative alignment details.
    fingerprint["alignment"]["missing_control_stems"] = (
        missing_controls[:100]
    )
    fingerprint["alignment"]["orphan_control_stems"] = (
        orphan_controls[:100]
    )
    fingerprint["alignment"]["metadata_missing_images"] = (
        metadata_missing_images[:100]
    )
    fingerprint["alignment"]["images_missing_metadata"] = (
        metadata_orphan_image_stems[:100]
    )

    # Add duplicate details.
    fingerprint["duplicates"]["image_stems"] = (
        duplicate_image_stems
    )
    fingerprint["duplicates"]["control_stems"] = (
        duplicate_control_stems
    )

    # ------------------------------------------------------------------
    # Write JSON and TXT
    # ------------------------------------------------------------------

    json_path = output_dir / "dataset_fingerprint.json"
    txt_path = output_dir / "dataset_fingerprint.txt"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fingerprint,
            f,
            indent=2,
            ensure_ascii=False,
        )

    report = build_text_report(fingerprint)

    txt_path.write_text(
        report,
        encoding="utf-8",
    )

    elapsed = time.time() - started

    print("")
    print(report)
    print("")
    print(f"Audit completed in {elapsed:.1f} seconds.")
    print("")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")
    print("")

    if fingerprint["status"]["pass"]:
        print("DATASET AUDIT: PASS")
        return 0

    print("DATASET AUDIT: ERRORS FOUND")
    print("Review dataset_fingerprint.txt before proceeding.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())