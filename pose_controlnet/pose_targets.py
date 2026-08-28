"""Versioned, partially-covered authoritative pose-target sidecars.

This module deliberately keeps target provenance separate from both the raster
conditioning image and the frozen reward estimator.  It never runs a pose
detector or derives joints from a control raster.  A source may explicitly be
unavailable for pose reward; only a source claiming an authoritative target is
fail-closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from pose_controlnet.reference_pose import COCO_17, load_coco_annotations, parse_coco_stem


POSE_TARGET_SIDECAR_VERSION = 2
RECORDS_NAME = "records.jsonl"
METADATA_NAME = "metadata.json"
COMMON_BODY_17 = COCO_17
class PoseTargetError(ValueError):
    """Raised when authoritative target provenance is incomplete or ambiguous."""


class _AuthoritativePeople(list[dict[str, Any]]):
    """A source-grouped people list retaining image dimensions even when empty."""

    def __init__(
        self, people: Iterable[dict[str, Any]], source_size: Iterable[int] | None = None,
        source_image_id: str | int | None = None,
    ):
        super().__init__(people)
        self.source_size = tuple(source_size) if source_size is not None else None
        self.source_image_id = source_image_id


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
    if schema != "coco17":
        raise PoseTargetError(f"Unsupported reward joint schema: {schema!r}")
    indices = tuple(range(17))
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
        if score < 0:
            raise PoseTargetError("Authoritative keypoint visibility/confidence must be non-negative")
        mapped_x, mapped_y = x * sx - left, y * sy - top
        present = score > 0
        if present and not (0.0 <= x < sw and 0.0 <= y < sh):
            raise PoseTargetError("Visible authoritative keypoint lies outside its source image")
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
    """Build all records, failing only for claimed-but-invalid targets."""
    expected = Counter(source_for_stem(stem) for stem in geometry_by_stem)
    missing = sorted(set(expected) - set(source_specs))
    if missing:
        raise PoseTargetError(f"No authoritative source specification for: {', '.join(missing)}")
    specs = {name: _validate_source_spec(name, source_specs[name]) for name in expected}
    loaders = {name: _load_source(name, specs[name]) for name in expected if specs[name]["pose_reward_available"]}
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for stem, geometry in sorted(geometry_by_stem.items()):
        source = source_for_stem(stem)
        spec = specs[source]
        if not spec["pose_reward_available"]:
            records.append(_unavailable_record(stem, source, geometry))
            continue
        _, lookup = loaders[source]
        people = lookup(stem)
        if people is None:
            unresolved.append(stem)
            continue
        authoritative_source_size = getattr(people, "source_size", None)
        if authoritative_source_size is not None and authoritative_source_size != tuple(geometry["source_size"]):
            raise PoseTargetError(
                f"Authoritative source size {authoritative_source_size} does not match persisted shard geometry {tuple(geometry['source_size'])}"
            )
        transformed = [transform_person(person, source_size=geometry["source_size"], resized_size=geometry["resized_size"], crop_box=geometry["crop_box"], bucket=geometry["bucket"]) for person in people]
        records.append({
            "schema_version": POSE_TARGET_SIDECAR_VERSION, "stem": stem, "source": source,
            "pose_reward_available": True,
            "target_provenance": spec["target_provenance"],
            "annotation_source": spec["annotation_source"],
            "source_image_id": getattr(people, "source_image_id", None),
            "source_size": list(geometry["source_size"]), "resized_size": list(geometry["resized_size"]),
            "crop_box": list(geometry["crop_box"]), "bucket": list(geometry["bucket"]),
            "joint_schema": spec["joint_schema"], "common_body_mapping": common_body_mapping(spec["joint_schema"]),
            "person_grouping": "one record per source person; image-level list preserves source grouping",
            "people": transformed, "renderer": spec["provenance_metadata"]["renderer"],
            "provenance_metadata": spec["provenance_metadata"],
        })
    if unresolved:
        raise PoseTargetError(f"Claimed authoritative targets missing for {len(unresolved)} samples (examples: {unresolved[:8]})")
    return records, {"expected_counts": dict(sorted(expected.items())), "records": len(records), "coverage": coverage_summary(records)}


def coverage_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return stable authoritative-target coverage counts for all source stems."""
    counts: dict[str, Counter[str]] = {}
    for record in records:
        source = str(record["source"])
        bucket = counts.setdefault(source, Counter())
        bucket["total"] += 1
        bucket["available" if record.get("pose_reward_available") is True else "unavailable"] += 1
    sources = {
        source: _coverage_row(counts[source]["total"], counts[source]["available"])
        for source in sorted(counts)
    }
    total = sum(row["total"] for row in sources.values())
    available = sum(row["available"] for row in sources.values())
    return {"total": _coverage_row(total, available), "sources": sources}


def pose_reward_target_for_stem(records_by_stem: Mapping[str, Mapping[str, Any]], stem: str) -> Mapping[str, Any] | None:
    """Training-facing lookup: return a target record or ``None`` when unavailable."""
    try:
        record = records_by_stem[stem]
    except KeyError as exc:
        raise PoseTargetError(f"No sidecar record for training stem: {stem}") from exc
    _validate_record(record)
    return record if record["pose_reward_available"] else None


def _coverage_row(total: int, available: int) -> dict[str, Any]:
    unavailable = total - available
    return {
        "total": total, "available": available, "unavailable": unavailable,
        "available_percent": 0.0 if total == 0 else 100.0 * available / total,
        "unavailable_percent": 0.0 if total == 0 else 100.0 * unavailable / total,
    }


def _unavailable_record(stem: str, source: str, geometry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": POSE_TARGET_SIDECAR_VERSION, "stem": stem, "source": source,
        "pose_reward_available": False, "target_provenance": "unavailable",
        "annotation_source": None, "source_size": list(geometry["source_size"]),
        "resized_size": list(geometry["resized_size"]), "crop_box": list(geometry["crop_box"]),
        "bucket": list(geometry["bucket"]), "people": None,
    }


def write_sidecar(records: Iterable[Mapping[str, Any]], output_dir: str | Path, *, build_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish an immutable sidecar directory after deterministic hashing."""
    destination = Path(output_dir)
    if destination.exists():
        raise PoseTargetError(f"Refusing to overwrite existing sidecar: {destination}")
    ordered = sorted(records, key=lambda row: str(row["stem"]))
    for record in ordered:
        _validate_record(record)
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
    for record in records:
        _validate_record(record)
    return metadata, records


def _validate_source_spec(name: str, supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one source's explicit coverage decision before any target load."""
    spec = dict(supplied)
    required = ("pose_reward_available", "target_provenance", "format")
    absent = [key for key in required if key not in spec]
    if absent:
        raise PoseTargetError(f"{name}: source specification missing {absent}")
    if not isinstance(spec["pose_reward_available"], bool):
        raise PoseTargetError(f"{name}: pose_reward_available must be boolean")
    if not spec["pose_reward_available"]:
        if spec["target_provenance"] != "unavailable" or spec["format"] != "unavailable":
            raise PoseTargetError(f"{name}: unavailable source requires target_provenance='unavailable' and format='unavailable'")
        return spec
    required_available = ("annotation_source", "joint_schema", "provenance_metadata")
    absent_available = [key for key in required_available if key not in spec]
    if absent_available:
        raise PoseTargetError(f"{name}: available source specification missing {absent_available}")
    if spec["target_provenance"] != "original_annotation":
        raise PoseTargetError(f"{name}: available source requires target_provenance='original_annotation'")
    if spec["joint_schema"] != "coco17":
        raise PoseTargetError(f"{name}: available source requires joint_schema='coco17'")
    if not isinstance(spec["provenance_metadata"], Mapping) or not isinstance(spec["provenance_metadata"].get("renderer"), Mapping):
        raise PoseTargetError(f"{name}: provenance_metadata.renderer must identify the exact historical renderer")
    return spec


def _load_source(name: str, spec: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Load only an available authoritative source; no detector fallback exists."""
    if spec["format"] == "coco_keypoints":
        if name != "coco":
            raise PoseTargetError(f"{name}: coco_keypoints is reserved for original COCO annotations")
        paths = spec.get("annotation_paths")
        if not isinstance(paths, list) or not paths:
            raise PoseTargetError(f"{name}: coco_keypoints requires non-empty annotation_paths")
        images, by_image, by_annotation, hashes = load_coco_annotations(paths)
        spec["provenance_metadata"] = dict(spec["provenance_metadata"]) | {"annotation_sha256": hashes}
        return spec, _coco_lookup(name, spec, images, by_image, by_annotation)
    if spec["format"] == "humanart_pose_adapter_jsonl":
        if not name.startswith("humanart_"):
            raise PoseTargetError(f"{name}: humanart_pose_adapter_jsonl is reserved for Human-Art sources")
        path = Path(spec.get("adapter_path", ""))
        if not path.is_file():
            raise PoseTargetError(f"{name}: Human-Art adapter export missing: {path}")
        rows = _adapter_rows(path)
        spec["provenance_metadata"] = dict(spec["provenance_metadata"]) | {
            "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        return spec, _humanart_adapter_lookup(name, rows)
    raise PoseTargetError(f"{name}: unsupported annotation format {spec['format']!r}")


def _coco_lookup(name: str, spec: Mapping[str, Any], images: Mapping[int, Any], by_image: Mapping[int, list[Any]], by_annotation: Mapping[int, Any]):
    def lookup(stem: str) -> list[dict[str, Any]] | None:
        image_id, annotation_id = parse_coco_stem(stem)
        image = images.get(image_id)
        if image is None:
            return None
        candidates = [by_annotation.get(annotation_id)] if annotation_id is not None else by_image.get(image_id, [])
        people = []
        for annotation in candidates:
            if not annotation or annotation.get("image_id") != image_id or annotation.get("category_id") != 1 or annotation.get("iscrowd", 0):
                continue
            keypoints = annotation.get("keypoints")
            if int(annotation.get("num_keypoints", 0)) < 1:
                continue
            if not isinstance(keypoints, list) or len(keypoints) != 51:
                raise PoseTargetError(f"coco: malformed keypoints for annotation {annotation.get('id')}")
            people.append({"person_id": annotation["id"], "annotation_id": annotation["id"], "bbox_xywh": annotation.get("bbox"), "keypoints_source": [keypoints[index:index + 3] for index in range(0, 51, 3)], "_authoritative_source_size": [image["width"], image["height"]]})
        if annotation_id is not None and not people:
            return None
        return _AuthoritativePeople(people, [image["width"], image["height"]], image_id)
    return lookup


def _adapter_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Read the canonical Human-Art import adapter, not a raw Human-Art file."""
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PoseTargetError(f"Human-Art adapter line {line_number}: invalid JSON") from exc
        stem = row.get("stem")
        if not isinstance(stem, str) or not stem:
            raise PoseTargetError(f"Human-Art adapter line {line_number}: missing stem")
        if stem in rows:
            raise PoseTargetError(f"Human-Art adapter has duplicate stem: {stem}")
        rows[stem] = row
    return rows


def _humanart_adapter_lookup(name: str, rows: Mapping[str, Mapping[str, Any]]):
    def lookup(stem: str) -> list[dict[str, Any]] | None:
        row = rows.get(stem)
        if row is None:
            return None
        declared_source = row.get("source")
        if declared_source is not None and declared_source != name:
            raise PoseTargetError(f"{stem}: Human-Art adapter source {declared_source!r} != {name!r}")
        source_size = row.get("source_size")
        if source_size is not None:
            _size(source_size, f"{stem}.source_size")
        people = row.get("people")
        if not isinstance(people, list):
            raise PoseTargetError(f"{stem}: Human-Art adapter requires a people list")
        normalized = []
        for index, person in enumerate(people):
            if not isinstance(person, Mapping):
                raise PoseTargetError(f"{stem}: Human-Art person {index} is not an object")
            points = person.get("keypoints")
            _keypoints(points)
            item = {
                "person_id": person.get("person_id"), "annotation_id": person.get("annotation_id", person.get("person_id")),
                "bbox_xywh": person.get("bbox_xywh"), "keypoints_source": points,
            }
            if source_size is not None:
                item["_authoritative_source_size"] = source_size
            normalized.append(item)
        source_image_id = row.get("source_image_id")
        if source_image_id is not None and not isinstance(source_image_id, (str, int)):
            raise PoseTargetError(f"{stem}: source_image_id must be a string or integer when present")
        return _AuthoritativePeople(normalized, source_size, source_image_id)
    return lookup


def _validate_record(record: Mapping[str, Any]) -> None:
    """Validate the availability branch consumed by training-facing readers."""
    available = record.get("pose_reward_available")
    if not isinstance(available, bool):
        raise PoseTargetError(f"{record.get('stem')}: pose_reward_available must be boolean")
    if available:
        required = ("stem", "source", "annotation_source", "joint_schema", "common_body_mapping", "people", "renderer")
        missing = [key for key in required if key not in record]
        if missing or record.get("target_provenance") != "original_annotation" or record.get("joint_schema") != "coco17":
            raise PoseTargetError(f"{record.get('stem')}: malformed available pose target")
        if not isinstance(record["people"], list):
            raise PoseTargetError(f"{record.get('stem')}: available pose target lacks people list")
        return
    if record.get("target_provenance") != "unavailable" or record.get("people") is not None:
        raise PoseTargetError(f"{record.get('stem')}: malformed unavailable pose target")


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
    x, y, w, h = map(float, box)
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w < 0 or h < 0:
        raise PoseTargetError("bbox_xywh must contain finite coordinates with non-negative width/height")
    x0, y0 = x * sx - left, y * sy - top; x1, y1 = (x + w) * sx - left, (y + h) * sy - top
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
