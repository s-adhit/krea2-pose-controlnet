"""Read-only forensic diagnosis for authoritative pose coordinates and rasters.

This intentionally reads the authoritative JSONL directly rather than through
the fail-closed sidecar loader: its purpose is to explain records which the
loader correctly rejects.  It never edits annotations, data, or shards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.control_reconstruction import render_record
from pose_controlnet.dataset_index import DatasetIndex
from pose_controlnet.pose_targets import POSEBRIDGE_BODY_RENDERER


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
DEFAULT_STEMS = (
    "coco_100098_193288", "coco_299468_426600", "coco_156320_crowd",
    "painting_humanart_10000000000555", "real_human_humanart_15000000000201",
    "sculpture_humanart_14000000000243",
)
INVALID_STEMS = (
    "painting_humanart_2000000000804", "sculpture_humanart_14000000001208",
)
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def rows_by_stem(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("stem") in wanted:
                found[row["stem"]] = row
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Missing authoritative rows: {sorted(missing)}")
    return found


def shard_geometry(root: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            stem = sample.get("stem")
            if stem in wanted:
                found[stem] = {
                    "split": sample["split"], "source_size": sample["source_size"],
                    "resized_size": sample["resized_size"], "crop_box": sample["crop_box"],
                    "bucket": sample["bucket"], "shard": str(path),
                }
        if len(found) == len(wanted):
            break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Missing shard samples: {sorted(missing)}")
    return found


def boundary_amounts(x: float, y: float, width: int, height: int) -> dict[str, dict[str, float]]:
    raw = {
        "left": max(0.0, -x), "right": max(0.0, x - width),
        "top": max(0.0, -y), "bottom": max(0.0, y - height),
    }
    return {
        edge: {"pixels": amount, "percent_of_axis": amount / (width if edge in {"left", "right"} else height) * 100.0}
        for edge, amount in raw.items() if amount > 0.0
    }


def bbox_diagnostic(bbox: Any, width: int, height: int) -> dict[str, Any] | None:
    if bbox is None:
        return None
    x, y, box_width, box_height = map(float, bbox)
    edges = boundary_amounts(x, y, width, height)
    # A bbox's right/bottom edge is x+w/y+h, not its origin.
    right = max(0.0, x + box_width - width)
    bottom = max(0.0, y + box_height - height)
    if right:
        edges["right"] = {"pixels": right, "percent_of_axis": right / width * 100.0}
    if bottom:
        edges["bottom"] = {"pixels": bottom, "percent_of_axis": bottom / height * 100.0}
    return {"xywh": [x, y, box_width, box_height], "extends_outside_source_canvas": bool(edges), "outside_amounts": edges}


def invalid_diagnostic(row: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    width, height = int(row["source_width"]), int(row["source_height"])
    people = []
    invalid = []
    for person_number, person in enumerate(row["people"]):
        joints = []
        for index, (x, y, visibility) in enumerate(person["keypoints_coco17"]):
            in_bounds = 0.0 <= x <= width and 0.0 <= y <= height
            joint = {"joint_index": index, "joint_name": COCO17_NAMES[index], "x": x, "y": y, "visibility": visibility, "in_bounds": in_bounds}
            if visibility > 0 and not in_bounds:
                joint["outside"] = boundary_amounts(x, y, width, height)
                invalid.append({"person_number": person_number, "annotation_id": person["annotation_id"], **joint})
            joints.append(joint)
        people.append({"person_number": person_number, "annotation_id": person["annotation_id"], "bbox": bbox_diagnostic(person.get("bbox_xywh"), width, height), "joints": joints})
    return {
        "stem": row["stem"], "authoritative_source_image_name": row["source_image_name"],
        "authoritative_source_size": [width, height], "active_shard_source_size": geometry["source_size"],
        "resized_size": geometry["resized_size"], "crop_box": geometry["crop_box"],
        "bucket_final_training_size": geometry["bucket"], "split": geometry["split"],
        "source_size_matches_active_shard": [width, height] == geometry["source_size"],
        "people": people, "invalid_visible_joints": invalid,
    }


def record_for_render(row: Mapping[str, Any], geometry: Mapping[str, Any], *, frame: str, clip: bool) -> dict[str, Any]:
    sw, sh = map(float, geometry["source_size"])
    if frame == "source":
        bucket = [int(sw), int(sh)]
        project = lambda x, y: (x, y)
    elif frame == "final":
        rw, rh = map(float, geometry["resized_size"])
        left, top, _, _ = map(float, geometry["crop_box"])
        bucket = list(map(int, geometry["bucket"]))
        project = lambda x, y: (x * rw / sw - left, y * rh / sh - top)
    else:
        raise ValueError(frame)
    people = []
    for person in row["people"]:
        points = []
        for x, y, visibility in person["keypoints_coco17"]:
            x, y = project(float(x), float(y))
            if clip:
                x = min(max(x, 0.0), bucket[0] - 1.0)
                y = min(max(y, 0.0), bucket[1] - 1.0)
            points.append([x, y, visibility])
        people.append({"keypoints_training": points})
    return {"stem": row["stem"], "bucket": bucket, "people": people, "renderer": dict(POSEBRIDGE_BODY_RENDERER)}


def foreground_metrics(actual: Image.Image, reconstructed: Image.Image) -> dict[str, Any]:
    a, b = np.asarray(actual.convert("RGB")), np.asarray(reconstructed.convert("RGB"))
    result: dict[str, Any] = {"size": list(actual.size), "mae": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), "thresholds": {}}
    for threshold in (0, 1, 10, 64, 128, 254):
        ma, mb = np.any(a > threshold, axis=2), np.any(b > threshold, axis=2)
        intersection, union = int((ma & mb).sum()), int((ma | mb).sum())
        if ma.any() and mb.any():
            da, db = distance_transform_edt(~ma), distance_transform_edt(~mb)
            distances = np.concatenate((db[ma], da[mb]))
            mean_distance, p95_distance = float(distances.mean()), float(np.quantile(distances, 0.95))
        else:
            mean_distance, p95_distance = None, None
        result["thresholds"][str(threshold)] = {
            "iou": 1.0 if union == 0 else intersection / union,
            "actual_foreground_pixels": int(ma.sum()), "reconstructed_foreground_pixels": int(mb.sum()),
            "symmetric_mean_foreground_distance_px": mean_distance,
            "symmetric_p95_foreground_distance_px": p95_distance,
        }
    return result


def panel(images: list[tuple[str, Image.Image]], destination: Path) -> None:
    cell = (320, 320); label_height = 22
    sheet = Image.new("RGB", (cell[0] * len(images), cell[1] + label_height), "black")
    draw = ImageDraw.Draw(sheet)
    for number, (label, image) in enumerate(images):
        copy = image.copy(); copy.thumbnail((cell[0], cell[1]))
        x, y = number * cell[0] + (cell[0] - copy.width) // 2, (cell[1] - copy.height) // 2
        sheet.paste(copy, (x, y)); draw.text((number * cell[0] + 4, cell[1] + 4), label, fill="white")
    sheet.save(destination)


def reconstruction_diagnostic(row: Mapping[str, Any], geometry: Mapping[str, Any], control_path: Path, output: Path) -> dict[str, Any]:
    actual_source = Image.open(control_path).convert("RGB")
    source_available = actual_source.size == tuple(geometry["source_size"])
    source_reconstruction = render_record(record_for_render(row, geometry, frame="source", clip=False))
    actual_final = actual_source.resize(tuple(geometry["resized_size"]), LANCZOS).crop(tuple(geometry["crop_box"]))
    # This is the vector-coordinate path under test: no source raster is used.
    final_reconstruction = render_record(record_for_render(row, geometry, frame="final", clip=False))
    # This control condition isolates the historical source rasterization from
    # the resize/crop geometry: same vector source skeleton, same PIL path.
    vector_source_preprocessed = source_reconstruction.resize(tuple(geometry["resized_size"]), LANCZOS).crop(tuple(geometry["crop_box"]))
    difference = ImageChops.difference(actual_final, final_reconstruction)
    panel([
        ("actual control source", actual_source), ("vector source skeleton", source_reconstruction),
        ("actual after training preprocess", actual_final), ("vector final-frame render", final_reconstruction),
        ("absolute difference", difference),
    ], output / f"{row['stem']}.png")
    return {
        "stem": row["stem"], "source_image_name": row["source_image_name"], "control_path": str(control_path),
        "native_stored_control_size": list(actual_source.size), "active_source_size": geometry["source_size"],
        "resized_size": geometry["resized_size"], "crop_box": geometry["crop_box"],
        "final_training_control_size": geometry["bucket"], "source_control_available": source_available,
        "source_frame_actual_vs_vector": foreground_metrics(actual_source, source_reconstruction) if source_available else None,
        "final_frame_actual_preprocessed_vs_vector_final": foreground_metrics(actual_final, final_reconstruction),
        "final_frame_actual_preprocessed_vs_vector_source_preprocessed": foreground_metrics(actual_final, vector_source_preprocessed),
        "artifact": str(output / f"{row['stem']}.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-jsonl", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stems", nargs="*", default=list(DEFAULT_STEMS), choices=DEFAULT_STEMS,
                        help="Requested reconstruction stems (default: all six).")
    args = parser.parse_args()
    reconstruction_stems = tuple(args.stems)
    wanted = set(reconstruction_stems) | set(INVALID_STEMS)
    rows = rows_by_stem(args.authoritative_jsonl, wanted)
    geometry = shard_geometry(args.latent_root, wanted)
    index = DatasetIndex.discover(args.dataset_root)
    if args.output_dir.exists():
        raise ValueError(f"Output must not exist: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    result = {
        "invalid_joint_diagnostics": [invalid_diagnostic(rows[stem], geometry[stem]) for stem in INVALID_STEMS],
        "reconstruction": [reconstruction_diagnostic(rows[stem], geometry[stem], index.control_by_stem[stem], args.output_dir) for stem in reconstruction_stems],
        "preprocessing": {
            "implementation": "pose_controlnet.paired_preprocessing._apply_geometry", "library": "Pillow (PIL)",
            "resize": "Image.resize(resized_size, Image.Resampling.LANCZOS)",
            "resized_size_rounding": "Python round(source_dimension * scale), ties-to-even", "crop": "Image.crop((left, top, left + bucket_width, top + bucket_height))",
            "crop_origin": "(resized_size - bucket) // 2 per axis (floor integer division)",
            "shared_rgb_control": "preprocess_pair computes one geometry after requiring RGB/control native dimensions to match, then applies _apply_geometry to both",
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(args.output_dir / "report.json"), "artifacts": len(reconstruction_stems)}, indent=2))


if __name__ == "__main__":
    main()
