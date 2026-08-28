"""Fail-closed, versioned authoritative pose-target sidecars.

This module deliberately keeps target provenance separate from both the raster
conditioning image and the (future) frozen reward estimator.  It never runs a
pose detector.  Annotated sources are read from their original annotation
files; Danbooru is read only from an exported historical DWPose result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from pose_controlnet.reference_pose import COCO_17, load_coco_annotations, parse_coco_stem


POSE_TARGET_SIDECAR_VERSION = 1
RECORDS_NAME = "records.jsonl"
METADATA_NAME = "metadata.json"
COMMON_BODY_17 = COCO_17
# COCO index -> OpenPose/DWPose body-18 index.  The DWPose neck is deliberately
# omitted: it is synthetic and must never become a reward target.
OPENPOSE18_TO_COCO17 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)


class PoseTargetError(ValueError):
    """Raised when authoritative target provenance is incomplete or ambiguous."""


def source_for_stem(stem: str) -> str:
    if stem.startswith("coco_"):
        return "coco"
    if stem.startswith("painting_humanart_"):
        return "humanart_painting"
    if stem.startswith("real_human_humanart_"):
        return "humanart_real_human"
    if stem.startswith("sculpture_humanart_"):
        return "humanart_sculpture"
    if stem.startswith("danbooru_"):
        return "danbooru"
    raise PoseTargetError(f"Unrecognised PoseBridge source stem: {stem!r}")


def common_body_mapping(schema: str) -> dict[str, Any]:
    """Return the explicit source-to-common physical-body mapping.

    All returned mappings have exactly the 17 physical COCO body joints.  No
    face, hand, foot-extra, or synthetic-neck target can enter the reward.
    """
    if schema == "coco17":
        indices = tuple(range(17))
    elif schema == "openpose18":
        indices = OPENPOSE18_TO_COCO17
    else:
        raise PoseTargetError(f"Unsupported reward joint schema: {schema!r}")
    return {"common_schema": "coco17_body", "common_joints": list(COMMON_BODY_17), "source_indices": list(indices)}


def transform_person(
    person: Mapping[str, Any], *, source_size: Iterable[int], resized_size: Iterable[int],
    crop_box: Iterable[int], bucket: Iterable[int],
) -> dict[str, Any]:
    """Map points and xywh box through persisted resize/crop geometry.

    `keypoints_training` is clipped to the training canvas and carries an
    `in_frame` mask.  Its `reward_visible` mask is authoritative visibility /
    confidence AND in-frame, so cropped-away joints cannot become targets.
    Raw source coordinates and source visibility remain untouched.
    """
    sw, sh = _size(source_size, "source_size"); rw, rh = _size(resized_size, "resized_size")
    bw, bh = _size(bucket, "bucket"); left, top, right, bottom = _crop(crop_box)
    asserted_source_size = person.get("_authoritative_source_size")
    if asserted_source_size is not None and tuple(asserted_source_size) != (sw, sh):
        raise PoseTargetError(
            f"Authoritative source size {tuple(asserted_source_size)} does not match persisted shard geometry {(sw, sh)}"
        )
    if right - left != bw or bottom - top != bh:
        raise PoseTargetError("crop_box dimensions must equal bucket")
    raw = _keypoints(person.get("keypoints_source"))
    sx, sy = rw / sw, rh / sh
    training, in_frame, reward_visible = [], [], []
    for x, y, score in raw:
        mapped_x, mapped_y = x * sx - left, y * sy - top
        present = score > 0
        inside = present and 0.0 <= mapped_x <= bw - 1 and 0.0 <= mapped_y <= bh - 1
        training.append([min(max(mapped_x, 0.0), bw - 1), min(max(mapped_y, 0.0), bh - 1), score])
        in_frame.append(inside)
        reward_visible.append(inside and present)
    bbox = person.get("bbox_xywh")
    training_box = _transform_box(bbox, sx=sx, sy=sy, left=left, top=top, width=bw, height=bh) if bbox is not None else None
    return {
        "person_id": person.get("person_id"), "annotation_id": person.get("annotation_id"),
        "bbox_source_xywh": bbox, "keypoints_source": raw,
        "visibility_or_confidence_source": [point[2] for point in raw],
        "keypoints_training": training, "keypoints_training_in_frame": in_frame,
        "reward_visible_mask": reward_visible, "bbox_training_xywh": training_box,
    }


def build_sidecar_records(
    geometry_by_stem: Mapping[str, Mapping[str, Any]], source_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all records, failing before publication if any source is unresolved."""
    expected = Counter(source_for_stem(stem) for stem in geometry_by_stem)
    missing = sorted(set(expected) - set(source_specs))
    if missing:
        raise PoseTargetError(f"No authoritative source specification for: {', '.join(missing)}")
    loaders = {name: _load_source(name, source_specs[name]) for name in expected}
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for stem, geometry in sorted(geometry_by_stem.items()):
        source = source_for_stem(stem)
        spec, lookup = loaders[source]
        people = lookup(stem)
        if people is None:
            unresolved.append(stem)
            continue
        transformed = [transform_person(person, source_size=geometry["source_size"], resized_size=geometry["resized_size"], crop_box=geometry["crop_box"], bucket=geometry["bucket"]) for person in people]
        records.append({
            "schema_version": POSE_TARGET_SIDECAR_VERSION, "stem": stem, "source": source,
            "target_provenance": spec["target_provenance"],
            "annotation_source": spec["annotation_source"],
            "source_size": list(geometry["source_size"]), "resized_size": list(geometry["resized_size"]),
            "crop_box": list(geometry["crop_box"]), "bucket": list(geometry["bucket"]),
            "joint_schema": spec["joint_schema"], "common_body_mapping": common_body_mapping(spec["joint_schema"]),
            "person_grouping": "one record per source person; image-level list preserves source grouping",
            "people": transformed, "renderer": spec["provenance_metadata"]["renderer"],
            "provenance_metadata": spec["provenance_metadata"],
        })
    if unresolved:
        raise PoseTargetError(f"Authoritative targets missing for {len(unresolved)} samples (examples: {unresolved[:8]})")
    return records, {"expected_counts": dict(sorted(expected.items())), "records": len(records)}


def write_sidecar(records: Iterable[Mapping[str, Any]], output_dir: str | Path, *, build_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish an immutable sidecar directory after deterministic hashing."""
    destination = Path(output_dir)
    if destination.exists():
        raise PoseTargetError(f"Refusing to overwrite existing sidecar: {destination}")
    ordered = sorted(records, key=lambda row: str(row["stem"]))
    content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in ordered)
    digest = hashlib.sha256(content.encode()).hexdigest()
    parent = destination.parent; parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        (temporary / RECORDS_NAME).write_text(content, encoding="utf-8")
        metadata = {"schema_version": POSE_TARGET_SIDECAR_VERSION, "read_only": True, "records_file": RECORDS_NAME, "records_sha256": digest, "record_count": len(ordered), **dict(build_metadata)}
        (temporary / METADATA_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if destination.exists():
            raise PoseTargetError(f"Refusing to overwrite existing sidecar: {destination}")
        os.replace(temporary, destination)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink()
        temporary.rmdir()
        raise
    return metadata


def load_sidecar(sidecar_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(sidecar_dir)
    metadata = json.loads((root / METADATA_NAME).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != POSE_TARGET_SIDECAR_VERSION or metadata.get("read_only") is not True:
        raise PoseTargetError(f"Unsupported or mutable sidecar: {root}")
    raw = (root / RECORDS_NAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata.get("records_sha256"):
        raise PoseTargetError(f"Sidecar digest mismatch: {root}")
    records = [json.loads(line) for line in raw.decode().splitlines()]
    if len(records) != metadata.get("record_count") or len({row.get("stem") for row in records}) != len(records):
        raise PoseTargetError(f"Sidecar record membership invalid: {root}")
    return metadata, records


def _load_source(name: str, supplied: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    spec = dict(supplied)
    required = ("target_provenance", "annotation_source", "joint_schema", "provenance_metadata", "format")
    absent = [key for key in required if key not in spec]
    if absent:
        raise PoseTargetError(f"{name}: source specification missing {absent}")
    expected_provenance = "dwpose_pseudolabel" if name == "danbooru" else "original_annotation"
    if spec["target_provenance"] != expected_provenance:
        raise PoseTargetError(f"{name}: requires target_provenance={expected_provenance!r}")
    if not isinstance(spec["provenance_metadata"], Mapping) or not isinstance(spec["provenance_metadata"].get("renderer"), Mapping):
        raise PoseTargetError(f"{name}: provenance_metadata.renderer must identify the exact historical renderer")
    if name == "danbooru":
        required_dwpose = ("detector", "pose_checkpoint", "pose_checkpoint_sha256", "thresholds", "body_joint_mapping", "renderer")
        absent_dwpose = [key for key in required_dwpose if key not in spec["provenance_metadata"]]
        if absent_dwpose:
            raise PoseTargetError(f"danbooru: historical DWPose provenance missing {absent_dwpose}")
    if spec["format"] == "coco_keypoints":
        paths = spec.get("annotation_paths")
        if not isinstance(paths, list) or not paths:
            raise PoseTargetError(f"{name}: coco_keypoints requires non-empty annotation_paths")
        images, by_image, by_annotation, hashes = load_coco_annotations(paths)
        spec["provenance_metadata"] = dict(spec["provenance_metadata"]) | {"annotation_sha256": hashes}
        return spec, _coco_lookup(name, spec, images, by_image, by_annotation)
    if spec["format"] == "historical_dwpose_jsonl":
        path = Path(spec.get("pseudolabel_path", ""))
        if not path.is_file():
            raise PoseTargetError(f"danbooru: historical DWPose export missing: {path}")
        rows = {row["stem"]: row for row in _jsonl(path)}
        return spec, lambda stem: _dwpose_people(rows.get(stem), spec) if stem in rows else None
    raise PoseTargetError(f"{name}: unsupported annotation format {spec['format']!r}")


def _coco_lookup(name: str, spec: Mapping[str, Any], images: Mapping[int, Any], by_image: Mapping[int, list[Any]], by_annotation: Mapping[int, Any]):
    pattern = spec.get("stem_image_id_regex")
    compiled = re.compile(pattern) if isinstance(pattern, str) else None
    def lookup(stem: str) -> list[dict[str, Any]] | None:
        if name == "coco":
            image_id, annotation_id = parse_coco_stem(stem)
        else:
            if compiled is None:
                raise PoseTargetError(f"{name}: requires stem_image_id_regex for authoritative image join")
            match = compiled.fullmatch(stem)
            if match is None or not match.groupdict().get("image_id"):
                raise PoseTargetError(f"{name}: stem does not match image-id mapping: {stem}")
            image_id, annotation_id = int(match["image_id"]), None
        image = images.get(image_id)
        if image is None:
            return None
        candidates = [by_annotation.get(annotation_id)] if annotation_id is not None else by_image.get(image_id, [])
        people = []
        for annotation in candidates:
            if not annotation or annotation.get("image_id") != image_id or annotation.get("category_id") != 1 or annotation.get("iscrowd", 0):
                continue
            keypoints = annotation.get("keypoints")
            if not isinstance(keypoints, list) or len(keypoints) != 51 or int(annotation.get("num_keypoints", 0)) < 1:
                continue
            people.append({"person_id": annotation["id"], "annotation_id": annotation["id"], "bbox_xywh": annotation.get("bbox"), "keypoints_source": [keypoints[index:index + 3] for index in range(0, 51, 3)], "_authoritative_source_size": [image["width"], image["height"]]})
        return people if annotation_id is None or people else None
    return lookup


def _dwpose_people(row: Mapping[str, Any], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    if row.get("source_size") is None or not isinstance(row.get("people"), list):
        raise PoseTargetError("danbooru: historical export must retain source_size and people")
    people = []
    for person in row["people"]:
        points = person.get("body_keypoints")
        if not isinstance(points, list) or len(points) != 18:
            raise PoseTargetError("danbooru: expected 18 historical DWPose body keypoints per person")
        ordered = [points[index] for index in OPENPOSE18_TO_COCO17]
        people.append({"person_id": person.get("person_id"), "annotation_id": person.get("person_id"), "bbox_xywh": person.get("bbox_xywh"), "keypoints_source": ordered, "_authoritative_source_size": row["source_size"]})
    return people


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keypoints(points: Any) -> list[list[float]]:
    if not isinstance(points, list) or len(points) != 17:
        raise PoseTargetError("Expected exactly 17 source body keypoints")
    result = []
    for point in points:
        if not isinstance(point, list) or len(point) != 3 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in point):
            raise PoseTargetError("Invalid source keypoint")
        result.append([float(value) for value in point])
    return result


def _transform_box(box: Any, *, sx: float, sy: float, left: int, top: int, width: int, height: int) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        raise PoseTargetError("bbox_xywh must be a four-number list")
    x, y, w, h = map(float, box); x0, y0 = x * sx - left, y * sy - top; x1, y1 = (x + w) * sx - left, (y + h) * sy - top
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(width - 1.0, x1), min(height - 1.0, y1)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _size(values: Iterable[int], name: str) -> tuple[int, int]:
    result = tuple(values)
    if len(result) != 2 or not all(isinstance(value, int) and value > 0 for value in result):
        raise PoseTargetError(f"Invalid {name}: {result!r}")
    return result[0], result[1]


def _crop(values: Iterable[int]) -> tuple[int, int, int, int]:
    result = tuple(values)
    if len(result) != 4 or not all(isinstance(value, int) for value in result) or result[2] <= result[0] or result[3] <= result[1]:
        raise PoseTargetError(f"Invalid crop_box: {result!r}")
    return result  # type: ignore[return-value]
