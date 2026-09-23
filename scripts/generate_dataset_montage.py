#!/usr/bin/env python3
"""Render provenance-backed paired RGB and pose-control blog montages.

This script is deliberately read-only with respect to the PoseBridge snapshot.
It resolves selected frozen *training* manifest stems through DatasetIndex and
only writes derived blog assets plus their provenance manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from pose_controlnet.dataset_index import DatasetIndex


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/lambda/nfs/adhit/krea2-pose/posebridge_hf")
PROJECT_TRAIN_MANIFEST = REPOSITORY_ROOT / "data/manifests/train.jsonl"
POSE_TARGETS = REPOSITORY_ROOT / "data/pose_targets_authoritative_v1.jsonl"
ASSET_ROOT = REPOSITORY_ROOT / "docs/blog-assets"
FIGURE_ROOT = ASSET_ROOT / "figures"
PROVENANCE_PATH = ASSET_ROOT / "dataset_montage_manifest.json"

# Ordered left-to-right, top-to-bottom. These are sourced from the existing
# source-diversity review candidates, then independently checked against both
# frozen train manifests below. Categories are domains, not quality ratings.
SELECTION = (
    ("coco_574672_crowd", "coco photograph"),
    ("coco_417031_crowd", "coco photograph"),
    ("coco_258019_438054", "coco photograph"),
    ("coco_139261_2164586", "coco photograph"),
    ("coco_353067_2161019", "coco photograph"),
    ("danbooru_anime_11917293", "anime illustration"),
    ("danbooru_anime_11921130", "anime illustration"),
    ("danbooru_anime_11908697", "anime illustration"),
    ("danbooru_anime_11919827", "anime illustration"),
    ("painting_humanart_2000000001313", "painting / digital illustration"),
    ("painting_humanart_6000000002942", "painting / portrait"),
    ("painting_humanart_2000000001633", "painting / landscape illustration"),
    ("painting_humanart_10000000000489", "painting / ukiyo-e illustration"),
    ("painting_humanart_2000000000974", "painting / science-fiction illustration"),
    ("real_human_humanart_17000000001852", "real human / dance photography"),
    ("real_human_humanart_17000000000973", "real human / stage performance"),
    ("real_human_humanart_15000000002388", "real human / action photography"),
    ("real_human_humanart_15000000001681", "real human / equestrian photography"),
    ("real_human_humanart_15000000001590", "real human / urban photography"),
    ("sculpture_humanart_14000000000666", "sculpture / weathered figurative artifact"),
    ("sculpture_humanart_14000000003911", "sculpture / public monument"),
    ("sculpture_humanart_14000000000005", "sculpture / museum classical figure"),
    ("sculpture_humanart_14000000000036", "sculpture / contemporary metal figure"),
    ("sculpture_humanart_14000000004547", "sculpture / multi-figure marble artwork"),
)

# The row memberships are intentional: they produce a controlled, varied
# editorial composition while keeping each source image whole and uncropped.
ROW_SIZES = (5, 6, 6, 7)
CANVAS_WIDTH = 2048
OUTER_GUTTER = 12
GAP = 10
BACKGROUND = (18, 18, 20)


def _manifest_stems(path: Path) -> set[str]:
    return {json.loads(line)["file_name"].removesuffix(".jpg") for line in path.read_text(encoding="utf-8").splitlines()}


def _pose_metadata() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in POSE_TARGETS.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        stem = record.get("stem")
        people = record.get("people")
        if isinstance(stem, str) and isinstance(people, list):
            result[stem] = {"person_count": len(people), "pose_annotation_source": record.get("source")}
    return result


def _render_cell(path: Path, displayed_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return image.resize(displayed_size, Image.Resampling.LANCZOS)


def _justified_layout(sizes: list[tuple[int, int]]) -> tuple[list[dict[str, int]], int]:
    """Return full-image cells with shared paired positions and no cropping."""
    if sum(ROW_SIZES) != len(sizes):
        raise ValueError("Row sizes must exactly cover the selected samples")

    available_width = CANVAS_WIDTH - 2 * OUTER_GUTTER
    cells: list[dict[str, int]] = []
    y = OUTER_GUTTER
    offset = 0
    for row, row_size in enumerate(ROW_SIZES):
        row_sizes = sizes[offset : offset + row_size]
        ratios = [width / height for width, height in row_sizes]
        row_height = max(1, round((available_width - (row_size - 1) * GAP) / sum(ratios)))
        widths = [max(1, round(ratio * row_height)) for ratio in ratios]
        x = OUTER_GUTTER
        for column, (width, source_size) in enumerate(zip(widths, row_sizes)):
            cells.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": row_height,
                    "row": row + 1,
                    "column": column + 1,
                    "source_width": source_size[0],
                    "source_height": source_size[1],
                }
            )
            x += width + GAP
        y += row_height + GAP
        offset += row_size
    return cells, y - GAP + OUTER_GUTTER


def _render_montage(paths: list[Path], cells: list[dict[str, int]], canvas_height: int) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_WIDTH, canvas_height), BACKGROUND)
    for path, cell in zip(paths, cells):
        image = _render_cell(path, (cell["width"], cell["height"]))
        canvas.paste(image, (cell["x"], cell["y"]))
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(SELECTION) != sum(ROW_SIZES):
        raise ValueError("Selection must exactly fill the declared justified rows")
    index = DatasetIndex.discover(DATASET_ROOT)
    snapshot_train = _manifest_stems(DATASET_ROOT / "manifests/train.jsonl")
    project_train = _manifest_stems(PROJECT_TRAIN_MANIFEST)
    pose_metadata = _pose_metadata()

    rgb_paths: list[Path] = []
    control_paths: list[Path] = []
    source_sizes: list[tuple[int, int]] = []
    samples: list[dict[str, Any]] = []
    for order, (stem, category) in enumerate(SELECTION, start=1):
        if stem not in snapshot_train or stem not in project_train:
            raise ValueError(f"Selected stem is not present in both frozen train manifests: {stem}")
        rgb_path = index.rgb_by_stem[stem]
        control_path = index.control_by_stem[stem]
        with Image.open(rgb_path) as rgb, Image.open(control_path) as control:
            rgb_size, control_size = rgb.size, control.size
        if rgb_size != control_size:
            raise ValueError(f"Paired source/control geometry mismatch for {stem}: {rgb_size} != {control_size}")
        rgb_paths.append(rgb_path)
        control_paths.append(control_path)
        source_sizes.append(rgb_size)
    cells, canvas_height = _justified_layout(source_sizes)

    for order, ((stem, category), rgb_path, control_path, cell) in enumerate(
        zip(SELECTION, rgb_paths, control_paths, cells), start=1
    ):
        rgb_size = (cell["source_width"], cell["source_height"])
        sample: dict[str, Any] = {
            "order": order,
            "stem": stem,
            "split": "train",
            "source_rgb_path": str(rgb_path),
            "pose_condition_path": str(control_path),
            "source_dimensions": {"width": rgb_size[0], "height": rgb_size[1]},
            "pose_condition_dimensions": {"width": rgb_size[0], "height": rgb_size[1]},
            "displayed_dimensions": {"width": cell["width"], "height": cell["height"]},
            "layout_position": {"x": cell["x"], "y": cell["y"], "row": cell["row"], "column": cell["column"]},
            "source_crop_box_xyxy": {"left": 0, "top": 0, "right": rgb_size[0], "bottom": rgb_size[1]},
            "crop_policy": "full source image displayed without cropping or distortion; the exact paired RGB and existing pose render use the same aspect-preserving layout cell",
            "category_domain": category,
            "rgb_sha256": _sha256(rgb_path),
            "pose_condition_sha256": _sha256(control_path),
        }
        if stem in pose_metadata:
            sample.update(pose_metadata[stem])
        else:
            sample["person_count"] = None
            sample["pose_annotation_source"] = "not available in pose_targets_authoritative_v1.jsonl"
        samples.append(sample)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rgb_output = FIGURE_ROOT / "dataset_rgb_montage.png"
    control_output = FIGURE_ROOT / "dataset_condition_montage.png"
    _render_montage(rgb_paths, cells, canvas_height).save(rgb_output, optimize=True)
    _render_montage(control_paths, cells, canvas_height).save(control_output, optimize=True)
    payload = {
        "schema_version": 2,
        "description": "Representative exact paired RGB and existing pose-condition renders from the frozen PoseBridge training split, in shared left-to-right/top-to-bottom order.",
        "dataset_root": str(DATASET_ROOT),
        "frozen_train_manifests_verified": [str(DATASET_ROOT / "manifests/train.jsonl"), str(PROJECT_TRAIN_MANIFEST)],
        "layout": {
            "type": "justified editorial rows",
            "canvas_dimensions": {"width": CANVAS_WIDTH, "height": canvas_height},
            "row_sizes": list(ROW_SIZES),
            "outer_gutter": OUTER_GUTTER,
            "cell_gap": GAP,
            "background_rgb": list(BACKGROUND),
            "image_treatment": "full-source, aspect-ratio-preserving resize; no cropping, distortion, recoloring, or stylization",
        },
        "assets": {"rgb_montage": str(rgb_output.relative_to(REPOSITORY_ROOT)), "condition_montage": str(control_output.relative_to(REPOSITORY_ROOT))},
        "samples": samples,
    }
    PROVENANCE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {rgb_output}")
    print(f"Wrote {control_output}")
    print(f"Wrote {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
