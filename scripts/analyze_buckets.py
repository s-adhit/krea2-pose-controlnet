#!/usr/bin/env python3
"""
Aspect-Ratio / Resolution Bucket Analysis
==========================================

Read-only analysis of the frozen Pose-ControlNet dataset.

Expected layout:

Krea-2-Pose-ControlNet/
├── data/
│   └── full/
│       ├── images/
│       ├── conditioning_images/
│       └── metadata.jsonl
└── scripts/
    └── analyze_buckets.py

This script DOES NOT modify images or metadata.

It analyzes:
- native dimensions
- aspect-ratio distribution
- candidate ~1MP buckets
- bucket assignment
- resize scale
- crop amount
- crop fraction
- extreme aspect ratios
- bucket utilization

Outputs:

data/stats/
├── bucket_analysis.json
├── bucket_analysis.txt
├── image_bucket_assignments.csv
├── aspect_ratio_distribution.csv
└── bucket_summary.csv

Important:
The bucket schemes in this script are CANDIDATES.
Do not treat them as final training configuration until we inspect
the generated report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ----------------------------------------------------------------------
# Candidate bucket schemes
# ----------------------------------------------------------------------
#
# Each bucket is approximately 1MP.
#
# These are deliberately explicit rather than dynamically generated.
# We will inspect utilization and crop behavior before selecting one.
#
# A bucket is represented as:
#
#     (width, height)
#
# Aspect ratio = width / height
#

BUCKET_SCHEMES: dict[str, list[tuple[int, int]]] = {
    "conservative_9": [
        (512, 2048),
        (576, 1536),
        (640, 1344),
        (768, 1280),
        (896, 1152),
        (1024, 1024),
        (1152, 896),
        (1280, 768),
        (1344, 640),
        (1536, 576),
        (2048, 512),
    ],

    "balanced_13": [
        (512, 2048),
        (576, 1792),
        (640, 1536),
        (704, 1408),
        (768, 1280),
        (832, 1216),
        (896, 1152),
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1280, 768),
        (1408, 704),
        (1536, 640),
        (1792, 576),
        (2048, 512),
    ],

    "core_9": [
        (576, 1536),
        (640, 1344),
        (768, 1280),
        (896, 1152),
        (1024, 1024),
        (1152, 896),
        (1280, 768),
        (1344, 640),
        (1536, 576),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze aspect ratios and candidate resolution buckets."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to data/full",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project>/data/stats",
    )

    parser.add_argument(
        "--min-bucket-count",
        type=int,
        default=25,
        help=(
            "Flag buckets with fewer than this many examples "
            "as low-utilization. Default: 25."
        ),
    )

    parser.add_argument(
        "--crop-warning",
        type=float,
        default=0.20,
        help=(
            "Warn when more than this fraction of the resized "
            "image area is cropped. Default: 0.20."
        ),
    )

    parser.add_argument(
        "--extreme-low",
        type=float,
        default=0.50,
        help="AR below this is considered extremely portrait.",
    )

    parser.add_argument(
        "--extreme-high",
        type=float,
        default=2.00,
        help="AR above this is considered extremely landscape.",
    )

    return parser.parse_args()


def discover_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {images_dir}"
        )

    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def bucket_aspect_ratio(bucket: tuple[int, int]) -> float:
    width, height = bucket
    return width / height


def bucket_area(bucket: tuple[int, int]) -> int:
    width, height = bucket
    return width * height


def bucket_name(bucket: tuple[int, int]) -> str:
    width, height = bucket
    return f"{width}x{height}"


def bucket_distance(
    image_ar: float,
    bucket: tuple[int, int],
) -> float:
    """
    Distance in log aspect-ratio space.

    This treats 0.5 -> 1.0 as equivalent in multiplicative terms
    to 1.0 -> 2.0, which is more appropriate for aspect ratios.
    """
    bucket_ar = bucket_aspect_ratio(bucket)

    return abs(math.log(image_ar / bucket_ar))


def choose_bucket(
    image_ar: float,
    buckets: list[tuple[int, int]],
) -> tuple[int, int]:
    return min(
        buckets,
        key=lambda b: bucket_distance(image_ar, b),
    )


def simulate_resize_crop(
    source_width: int,
    source_height: int,
    bucket_width: int,
    bucket_height: int,
) -> dict[str, Any]:
    """
    Simulate the standard resize-to-cover + center crop operation.

    The source is scaled until BOTH bucket dimensions are covered.
    The excess is then cropped.

    Returns:
        scale
        resized dimensions
        crop pixels
        crop fractions
        retained area fraction

    This is a geometric analysis only. No pixels are actually resized.
    """

    source_ar = source_width / source_height
    bucket_ar = bucket_width / bucket_height

    # Resize source until it covers the target bucket.
    scale = max(
        bucket_width / source_width,
        bucket_height / source_height,
    )

    resized_width = source_width * scale
    resized_height = source_height * scale

    crop_width = max(
        0.0,
        resized_width - bucket_width,
    )

    crop_height = max(
        0.0,
        resized_height - bucket_height,
    )

    resized_area = resized_width * resized_height
    bucket_area_value = bucket_width * bucket_height

    crop_area_fraction = (
        max(0.0, resized_area - bucket_area_value)
        / resized_area
        if resized_area > 0
        else 0.0
    )

    retained_fraction = (
        bucket_area_value / resized_area
        if resized_area > 0
        else 0.0
    )

    # Fraction of each resized dimension removed.
    crop_width_fraction = (
        crop_width / resized_width
        if resized_width > 0
        else 0.0
    )

    crop_height_fraction = (
        crop_height / resized_height
        if resized_height > 0
        else 0.0
    )

    return {
        "source_aspect_ratio": source_ar,
        "bucket_aspect_ratio": bucket_ar,
        "scale": scale,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_width_px": crop_width,
        "crop_height_px": crop_height,
        "crop_area_fraction": crop_area_fraction,
        "retained_area_fraction": retained_fraction,
        "crop_width_fraction": crop_width_fraction,
        "crop_height_fraction": crop_height_fraction,
    }


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


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def percent(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def percentile(
    values: list[float],
    p: float,
) -> float | None:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * p

    low = math.floor(index)
    high = math.ceil(index)

    if low == high:
        return values[low]

    weight = index - low

    return values[low] * (1 - weight) + values[high] * weight


def build_report(
    data_root: Path,
    total_images: int,
    scheme_results: dict[str, Any],
    global_stats: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("POSE-CONTROLNET ASPECT-RATIO / BUCKET ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Data root: {data_root}")
    lines.append(f"Images analyzed: {total_images:,}")
    lines.append("")
    lines.append(
        "IMPORTANT: Bucket schemes below are candidates only."
    )
    lines.append(
        "They have NOT been selected as the final training scheme."
    )
    lines.append("")

    lines.append("GLOBAL ASPECT-RATIO STATISTICS")
    lines.append("-" * 80)

    for key, value in global_stats.items():
        lines.append(
            f"{key:35s}: {fmt(value)}"
        )

    lines.append("")

    lines.append("EXTREME ASPECT-RATIO COUNTS")
    lines.append("-" * 80)

    lines.append(
        f"AR < {args.extreme_low:.2f}: "
        f"{global_stats['extreme_portrait_count']:,}"
    )

    lines.append(
        f"AR > {args.extreme_high:.2f}: "
        f"{global_stats['extreme_landscape_count']:,}"
    )

    lines.append("")

    for scheme_name, result in scheme_results.items():
        lines.append("=" * 80)
        lines.append(f"SCHEME: {scheme_name}")
        lines.append("=" * 80)
        lines.append("")

        lines.append(
            f"Number of buckets: {result['bucket_count']}"
        )

        lines.append(
            f"Images assigned: {result['assigned_images']:,}"
        )

        lines.append(
            f"Low-utilization buckets: "
            f"{result['low_utilization_bucket_count']}"
        )

        lines.append(
            f"Images with crop > {args.crop_warning:.0%}: "
            f"{result['high_crop_images']:,} "
            f"({percent(result['high_crop_fraction'])})"
        )

        lines.append(
            f"Mean cropped area: "
            f"{percent(result['mean_crop_area_fraction'])}"
        )

        lines.append(
            f"Median cropped area: "
            f"{percent(result['median_crop_area_fraction'])}"
        )

        lines.append(
            f"P90 cropped area: "
            f"{percent(result['p90_crop_area_fraction'])}"
        )

        lines.append("")

        lines.append(
            f"{'Bucket':15s}"
            f"{'AR':>10s}"
            f"{'Count':>10s}"
            f"{'Share':>10s}"
            f"{'Mean Crop':>12s}"
            f"{'P90 Crop':>12s}"
        )

        lines.append("-" * 80)

        for bucket in result["buckets"]:
            lines.append(
                f"{bucket['bucket']:15s}"
                f"{bucket['aspect_ratio']:>10.3f}"
                f"{bucket['count']:>10,}"
                f"{percent(bucket['fraction']):>10s}"
                f"{percent(bucket['mean_crop_area_fraction']):>12s}"
                f"{percent(bucket['p90_crop_area_fraction']):>12s}"
            )

        lines.append("")

        lines.append("LOW-UTILIZATION BUCKETS")

        if result["low_utilization_buckets"]:
            for bucket in result["low_utilization_buckets"]:
                lines.append(
                    f"  {bucket['bucket']}: "
                    f"{bucket['count']} images"
                )
        else:
            lines.append("  None")

        lines.append("")

    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "Use bucket utilization and crop statistics together."
    )
    lines.append(
        "A bucket scheme with many tiny buckets may preserve aspect ratio "
        "well but produce poor batching efficiency."
    )
    lines.append(
        "A very small number of buckets may improve batching efficiency "
        "but introduce excessive cropping."
    )
    lines.append(
        "For Pose-ControlNet, inspect high-crop cases visually because "
        "cropping a hand, foot, head, or entire person can damage the "
        "conditioning relationship."
    )
    lines.append("")
    lines.append(
        "The next step after this report should be to choose a final "
        "bucket scheme and then test the exact RGB/control geometric "
        "transform on representative examples."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    data_root = args.data_root.resolve()
    images_dir = data_root / "images"

    if args.output_dir is None:
        output_dir = data_root.parent.parent / "data" / "stats"
    else:
        output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.time()

    print("=" * 80)
    print("POSE-CONTROLNET BUCKET ANALYSIS")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print("")

    image_files = discover_images(images_dir)

    if not image_files:
        raise RuntimeError(
            f"No images found in {images_dir}"
        )

    print(
        f"Found {len(image_files):,} source images."
    )

    # ------------------------------------------------------------------
    # Read dimensions
    # ------------------------------------------------------------------

    print("Reading image dimensions...")

    image_rows: list[dict[str, Any]] = []

    aspect_ratios: list[float] = []

    extreme_portrait = 0
    extreme_landscape = 0

    for index, path in enumerate(image_files, start=1):

        try:
            with Image.open(path) as img:
                width, height = img.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Invalid image dimensions"
                )

            ar = width / height

            aspect_ratios.append(ar)

            if ar < args.extreme_low:
                extreme_portrait += 1

            if ar > args.extreme_high:
                extreme_landscape += 1

            image_rows.append(
                {
                    "filename": path.name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": ar,
                }
            )

        except Exception as exc:
            print(
                f"\nERROR reading {path}: {exc}"
            )
            raise

        if index % 500 == 0 or index == len(image_files):
            print(
                f"  {index:,}/{len(image_files):,}",
                end="\r",
            )

    print("")

    # ------------------------------------------------------------------
    # Global aspect ratio distribution
    # ------------------------------------------------------------------

    ratio_bins = [
        (0.00, 0.40),
        (0.40, 0.50),
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
        (2.00, 2.50),
        (2.50, 3.00),
        (3.00, 999.0),
    ]

    ratio_distribution: Counter[str] = Counter()

    for ar in aspect_ratios:
        assigned = False

        for low, high in ratio_bins:
            if low <= ar < high:
                ratio_distribution[
                    f"{low:.2f}-{high:.2f}"
                ] += 1
                assigned = True
                break

        if not assigned:
            ratio_distribution["other"] += 1

    global_stats = {
        "min_aspect_ratio": min(aspect_ratios),
        "max_aspect_ratio": max(aspect_ratios),
        "mean_aspect_ratio": (
            sum(aspect_ratios) / len(aspect_ratios)
        ),
        "median_aspect_ratio": percentile(
            aspect_ratios,
            0.50,
        ),
        "p10_aspect_ratio": percentile(
            aspect_ratios,
            0.10,
        ),
        "p25_aspect_ratio": percentile(
            aspect_ratios,
            0.25,
        ),
        "p75_aspect_ratio": percentile(
            aspect_ratios,
            0.75,
        ),
        "p90_aspect_ratio": percentile(
            aspect_ratios,
            0.90,
        ),
        "extreme_portrait_count": extreme_portrait,
        "extreme_landscape_count": extreme_landscape,
    }

    # ------------------------------------------------------------------
    # Analyze each candidate scheme
    # ------------------------------------------------------------------

    scheme_results: dict[str, Any] = {}
    all_assignment_rows: list[dict[str, Any]] = []

    for scheme_name, buckets in BUCKET_SCHEMES.items():

        print("")
        print(
            f"Analyzing scheme: {scheme_name} "
            f"({len(buckets)} buckets)"
        )

        assignments: dict[str, list[dict[str, Any]]] = (
            defaultdict(list)
        )

        crop_fractions: list[float] = []

        high_crop_count = 0

        for row in image_rows:

            image_ar = row["aspect_ratio"]

            bucket = choose_bucket(
                image_ar,
                buckets,
            )

            bucket_width, bucket_height = bucket

            geometry = simulate_resize_crop(
                row["width"],
                row["height"],
                bucket_width,
                bucket_height,
            )

            crop_fraction = geometry[
                "crop_area_fraction"
            ]

            crop_fractions.append(
                crop_fraction
            )

            if crop_fraction > args.crop_warning:
                high_crop_count += 1

            bucket_key = bucket_name(bucket)

            assignments[bucket_key].append(
                {
                    **row,
                    **geometry,
                }
            )

        bucket_summaries = []

        low_utilization = []

        for bucket in buckets:

            key = bucket_name(bucket)

            rows = assignments.get(key, [])

            crops = [
                r["crop_area_fraction"]
                for r in rows
            ]

            summary = {
                "bucket": key,
                "width": bucket[0],
                "height": bucket[1],
                "area": bucket_area(bucket),
                "aspect_ratio": bucket_aspect_ratio(
                    bucket
                ),
                "count": len(rows),
                "fraction": (
                    len(rows) / len(image_rows)
                    if image_rows
                    else 0
                ),
                "mean_crop_area_fraction": (
                    sum(crops) / len(crops)
                    if crops
                    else None
                ),
                "median_crop_area_fraction": percentile(
                    crops,
                    0.50,
                ),
                "p90_crop_area_fraction": percentile(
                    crops,
                    0.90,
                ),
                "max_crop_area_fraction": (
                    max(crops)
                    if crops
                    else None
                ),
            }

            bucket_summaries.append(summary)

            if len(rows) < args.min_bucket_count:
                low_utilization.append(summary)

        scheme_result = {
            "bucket_count": len(buckets),
            "assigned_images": len(image_rows),
            "high_crop_images": high_crop_count,
            "high_crop_fraction": (
                high_crop_count / len(image_rows)
            ),
            "mean_crop_area_fraction": (
                sum(crop_fractions)
                / len(crop_fractions)
            ),
            "median_crop_area_fraction": percentile(
                crop_fractions,
                0.50,
            ),
            "p90_crop_area_fraction": percentile(
                crop_fractions,
                0.90,
            ),
            "max_crop_area_fraction": max(
                crop_fractions
            ),
            "low_utilization_bucket_count": len(
                low_utilization
            ),
            "low_utilization_buckets": low_utilization,
            "buckets": bucket_summaries,
        }

        scheme_results[scheme_name] = scheme_result

        # Keep per-image assignments.
        for row in image_rows:
            bucket = choose_bucket(
                row["aspect_ratio"],
                buckets,
            )

            geometry = simulate_resize_crop(
                row["width"],
                row["height"],
                bucket[0],
                bucket[1],
            )

            all_assignment_rows.append(
                {
                    "scheme": scheme_name,
                    "filename": row["filename"],
                    "source_width": row["width"],
                    "source_height": row["height"],
                    "source_aspect_ratio": row[
                        "aspect_ratio"
                    ],
                    "bucket": bucket_name(bucket),
                    "bucket_width": bucket[0],
                    "bucket_height": bucket[1],
                    "bucket_aspect_ratio": bucket_aspect_ratio(
                        bucket
                    ),
                    "scale": geometry["scale"],
                    "resized_width": geometry[
                        "resized_width"
                    ],
                    "resized_height": geometry[
                        "resized_height"
                    ],
                    "crop_width_px": geometry[
                        "crop_width_px"
                    ],
                    "crop_height_px": geometry[
                        "crop_height_px"
                    ],
                    "crop_area_fraction": geometry[
                        "crop_area_fraction"
                    ],
                    "retained_area_fraction": geometry[
                        "retained_area_fraction"
                    ],
                }
            )

    # ------------------------------------------------------------------
    # Write aspect-ratio CSV
    # ------------------------------------------------------------------

    ratio_rows = [
        {
            "range": key,
            "count": value,
            "fraction": value / len(image_rows),
        }
        for key, value in sorted(
            ratio_distribution.items()
        )
    ]

    write_csv(
        output_dir / "aspect_ratio_distribution.csv",
        ratio_rows,
        [
            "range",
            "count",
            "fraction",
        ],
    )

    # ------------------------------------------------------------------
    # Write bucket summary
    # ------------------------------------------------------------------

    bucket_summary_rows = []

    for scheme_name, result in scheme_results.items():
        for bucket in result["buckets"]:
            bucket_summary_rows.append(
                {
                    "scheme": scheme_name,
                    **bucket,
                }
            )

    write_csv(
        output_dir / "bucket_summary.csv",
        bucket_summary_rows,
        [
            "scheme",
            "bucket",
            "width",
            "height",
            "area",
            "aspect_ratio",
            "count",
            "fraction",
            "mean_crop_area_fraction",
            "median_crop_area_fraction",
            "p90_crop_area_fraction",
            "max_crop_area_fraction",
        ],
    )

    # ------------------------------------------------------------------
    # Write per-image assignments
    # ------------------------------------------------------------------

    write_csv(
        output_dir / "image_bucket_assignments.csv",
        all_assignment_rows,
        [
            "scheme",
            "filename",
            "source_width",
            "source_height",
            "source_aspect_ratio",
            "bucket",
            "bucket_width",
            "bucket_height",
            "bucket_aspect_ratio",
            "scale",
            "resized_width",
            "resized_height",
            "crop_width_px",
            "crop_height_px",
            "crop_area_fraction",
            "retained_area_fraction",
        ],
    )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    result_json = {
        "analysis_version": "1.0",
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "data_root": str(data_root),
        "image_count": len(image_rows),
        "parameters": {
            "min_bucket_count": args.min_bucket_count,
            "crop_warning": args.crop_warning,
            "extreme_low": args.extreme_low,
            "extreme_high": args.extreme_high,
        },
        "global_statistics": global_stats,
        "aspect_ratio_distribution": dict(
            ratio_distribution
        ),
        "candidate_schemes": scheme_results,
    }

    json_path = (
        output_dir / "bucket_analysis.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result_json,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    report = build_report(
        data_root,
        len(image_rows),
        scheme_results,
        global_stats,
        args,
    )

    txt_path = (
        output_dir / "bucket_analysis.txt"
    )

    txt_path.write_text(
        report,
        encoding="utf-8",
    )

    elapsed = time.time() - started

    print("")
    print(report)
    print("")
    print(
        f"Analysis completed in {elapsed:.1f} seconds."
    )
    print("")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")
    print(
        f"CSV:  {output_dir / 'bucket_summary.csv'}"
    )
    print(
        f"CSV:  {output_dir / 'image_bucket_assignments.csv'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())