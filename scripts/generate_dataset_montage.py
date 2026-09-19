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

# Ordered left-to-right, top-to-bottom.  These are sourced from the existing
# source-diversity review candidates, then independently checked against both
# frozen train manifests below.  Categories are domains, not quality ratings.
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
    ("painting_humanart_1000000002893", "comic illustration"),
    ("painting_humanart_2000000000974", "painting / science-fiction illustration"),
    ("real_human_humanart_17000000001852", "real human / dance photography"),
    ("real_human_humanart_17000000000973", "real human / stage performance"),
    ("real_human_humanart_15000000002388", "real human / action photography"),
    ("real_human_humanart_15000000001681", "real human / equestrian photography"),
    ("real_human_humanart_15000000001590", "real human / urban photography"),
    ("sculpture_humanart_14000000003822", "sculpture / public monument"),
    ("sculpture_humanart_14000000000666", "sculpture / figurative artwork"),
    ("sculpture_humanart_14000000003911", "sculpture / public monument"),
    ("sculpture_humanart_14000000000820", "sculpture / outdoor artwork"),
    ("sculpture_humanart_14000000004138", "sculpture / equestrian artwork"),
)

COLUMNS = 6
ROWS = 4
CELL_SIZE = 320
GAP = 8
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


def _center_square_crop(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return (left, top, left + side, top + side)


def _render_cell(path: Path, crop_box: tuple[int, int, int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return image.crop(crop_box).resize((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)


def _render_montage(paths_and_crops: list[tuple[Path, tuple[int, int, int, int]]]) -> Image.Image:
    width = COLUMNS * CELL_SIZE + (COLUMNS + 1) * GAP
    height = ROWS * CELL_SIZE + (ROWS + 1) * GAP
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    for order, (path, crop_box) in enumerate(paths_and_crops):
        x = GAP + (order % COLUMNS) * (CELL_SIZE + GAP)
        y = GAP + (order // COLUMNS) * (CELL_SIZE + GAP)
        canvas.paste(_render_cell(path, crop_box), (x, y))
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(SELECTION) != COLUMNS * ROWS:
        raise ValueError("Selection must exactly fill the declared montage grid")
    index = DatasetIndex.discover(DATASET_ROOT)
    snapshot_train = _manifest_stems(DATASET_ROOT / "manifests/train.jsonl")
    project_train = _manifest_stems(PROJECT_TRAIN_MANIFEST)
    pose_metadata = _pose_metadata()

    rgb_cells: list[tuple[Path, tuple[int, int, int, int]]] = []
    control_cells: list[tuple[Path, tuple[int, int, int, int]]] = []
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
        crop_box = _center_square_crop(*rgb_size)
        rgb_cells.append((rgb_path, crop_box))
        control_cells.append((control_path, crop_box))
        sample: dict[str, Any] = {
            "order": order,
            "stem": stem,
            "split": "train",
            "source_rgb_path": str(rgb_path),
            "pose_condition_path": str(control_path),
            "source_dimensions": {"width": rgb_size[0], "height": rgb_size[1]},
            "pose_condition_dimensions": {"width": control_size[0], "height": control_size[1]},
            "displayed_dimensions": {"width": CELL_SIZE, "height": CELL_SIZE},
            "source_crop_box_xyxy": {"left": crop_box[0], "top": crop_box[1], "right": crop_box[2], "bottom": crop_box[3]},
            "crop_policy": "center square crop, identically applied to the exact paired RGB and existing pose render; no source artifacts were changed",
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
    _render_montage(rgb_cells).save(rgb_output, optimize=True)
    _render_montage(control_cells).save(control_output, optimize=True)
    payload = {
        "schema_version": 1,
        "description": "Representative exact paired RGB and existing pose-condition renders from the frozen PoseBridge training split, in shared left-to-right/top-to-bottom order.",
        "dataset_root": str(DATASET_ROOT),
        "frozen_train_manifests_verified": [str(DATASET_ROOT / "manifests/train.jsonl"), str(PROJECT_TRAIN_MANIFEST)],
        "layout": {"columns": COLUMNS, "rows": ROWS, "cell_dimensions": {"width": CELL_SIZE, "height": CELL_SIZE}, "cell_gap": GAP, "background_rgb": list(BACKGROUND)},
        "assets": {"rgb_montage": str(rgb_output.relative_to(REPOSITORY_ROOT)), "condition_montage": str(control_output.relative_to(REPOSITORY_ROOT))},
        "samples": samples,
    }
    PROVENANCE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {rgb_output}")
    print(f"Wrote {control_output}")
    print(f"Wrote {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
