"""Authoritative, read-only reference-pose sidecars for evaluation.

This module intentionally accepts annotations only from their original source.
It never reads or derives joints from a rendered control raster.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


COCO_17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
_COCO_STEM = re.compile(r"^coco_(?P<image_id>\d+)_(?P<annotation_id>\d+|crowd)$")

# This is the exact OpenPose-style unified skeleton used by the source control
# renderer.  Index 1 is the synthesized neck, so it intentionally has no COCO
# identity and is never a PCK joint.
COCO_TO_UNIFIED = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
UNIFIED_LIMBS = (
    (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 1),
    (0, 14), (0, 15),
    (14, 16), (15, 17),
)
MIN_LIMBS = 5
# The renderer's inclusion test is evaluated after neck synthesis.  These are
# the torso/head anchors from its unified-18 representation.
CORE_UNIFIED_JOINTS = (0, 1, 2, 5, 8, 11)


class ReferencePoseError(ValueError):
    """Raised when source provenance cannot be proved exactly."""


def renderer_joint_states(keypoints: Iterable[Iterable[float]]) -> tuple[list[dict[str, Any]], list[list[float]], int]:
    """Return analytic source/render/PCK state for each authoritative COCO joint.

    Rendering is determined only from the original renderer topology: a unified
    endpoint is represented iff it belongs to a limb whose two endpoints have
    visibility greater than zero.  This must not be inferred from raster pixels.
    """
    coco = [list(map(float, point)) for point in keypoints]
    if len(coco) != 17 or any(len(point) != 3 for point in coco):
        raise ReferencePoseError("Expected 17 COCO [x, y, visibility] joints")
    unified = [[0.0, 0.0, 0.0] for _ in range(18)]
    for coco_index, unified_index in enumerate(COCO_TO_UNIFIED):
        unified[unified_index] = coco[coco_index].copy()
    # The original renderer synthesizes a neck only when both shoulders exist.
    left_shoulder, right_shoulder = coco[5], coco[6]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0:
        unified[1] = [
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            min(left_shoulder[2], right_shoulder[2]),
        ]
    rendered = set()
    visible_limb_count = 0
    for first, second in UNIFIED_LIMBS:
        if unified[first][2] > 0 and unified[second][2] > 0:
            visible_limb_count += 1
            rendered.update((first, second))
    states = []
    for coco_index, unified_index in enumerate(COCO_TO_UNIFIED):
        source_visible = coco[coco_index][2] > 0
        rendered_in_control = unified_index in rendered
        states.append({
            "coco_index": coco_index,
            "coco_joint": COCO_17[coco_index],
            "unified_index": unified_index,
            "source_visible": source_visible,
            "rendered_in_control": rendered_in_control,
            "pck_eligible": source_visible and rendered_in_control,
        })
    return states, unified, visible_limb_count


def has_core_visibility(unified_keypoints: Iterable[Iterable[float]]) -> bool:
    """Match the renderer's post-neck-synthesis core-visibility predicate."""
    points = list(unified_keypoints)
    return any(float(points[index][2]) > 0 for index in CORE_UNIFIED_JOINTS)


def renderer_includes_person(keypoints: Iterable[Iterable[float]]) -> tuple[bool, dict[str, Any]]:
    """Recompute the Human-Art/crowd renderer qualification from source joints."""
    states, unified, visible_limb_count = renderer_joint_states(keypoints)
    core_visible = has_core_visibility(unified)
    return core_visible and visible_limb_count >= MIN_LIMBS, {
        "has_core_visibility": core_visible,
        "visible_limb_count": visible_limb_count,
        "minimum_limb_count": MIN_LIMBS,
        "joint_states": states,
    }


def pck_person_from_source(keypoints: Iterable[Iterable[float]]) -> dict[str, Any]:
    """Convert authoritative COCO joints to a PCK person without dropping provenance."""
    points = [list(map(float, point)) for point in keypoints]
    states, _, _ = renderer_joint_states(points)
    # Visibility is the PCK eligibility mask.  Coordinates remain authoritative
    # source/bucket coordinates even when a joint is not eligible.
    pck_points = [point[:2] + [1.0 if state["pck_eligible"] else 0.0]
                  for point, state in zip(points, states)]
    return {"keypoints": pck_points, "joint_states": states, "keypoints_authoritative": points}


def reference_person_from_sidecar(
    person: Mapping[str, Any], *, source_size: tuple[int, int],
    resized_size: tuple[int, int], crop_box: tuple[int, int, int, int],
    requires_renderer_qualification: bool,
) -> dict[str, Any]:
    """Construct a transformed PCK reference while retaining raw COCO provenance."""
    source_keypoints = person.get("keypoints")
    if not isinstance(source_keypoints, list):
        raise ReferencePoseError("Sidecar person lacks authoritative keypoints")
    flattened = [value for joint in source_keypoints for value in joint]
    bucket_keypoints = transform_keypoints(
        flattened, source_size=source_size, resized_size=resized_size, crop_box=crop_box,
    )
    included, details = renderer_includes_person(source_keypoints)
    if not requires_renderer_qualification:
        included = True
    pck = pck_person_from_source(bucket_keypoints)
    return {
        "person_id": person.get("annotation_id"),
        "annotation_id": person.get("annotation_id"),
        "reference_rendered": included,
        "renderer_qualification_recomputed": requires_renderer_qualification,
        "renderer": details,
        "keypoints_source": [list(map(float, joint)) for joint in source_keypoints],
        "keypoints_bucket": bucket_keypoints,
        **pck,
    }


def reference_people_from_sidecar(
    record: Mapping[str, Any], *, source_size: tuple[int, int],
    resized_size: tuple[int, int], crop_box: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Reproduce the renderer's available person-inclusion behavior.

    Human-Art sidecar construction already guaranteed removal of `iscrowd == 1`
    and `num_keypoints == 0`; those unavailable source fields are not invented.
    Human-Art and COCO crowd then recompute the renderer's core/limb filter.
    A COCO single stem is the explicitly requested annotation only.
    """
    source = record.get("source")
    mode = record.get("mode")
    requires_qualification = source == "humanart" or (source == "coco" and mode == "crowd")
    return [reference_person_from_sidecar(
        person, source_size=source_size, resized_size=resized_size, crop_box=crop_box,
        requires_renderer_qualification=requires_qualification,
    ) for person in record.get("people", [])]


def parse_coco_stem(stem: str) -> tuple[int, int | None]:
    """Return the COCO image id and optional person annotation id."""
    match = _COCO_STEM.fullmatch(stem)
    if match is None:
        raise ReferencePoseError(f"Not a supported COCO PoseBridge stem: {stem!r}")
    annotation = match["annotation_id"]
    return int(match["image_id"]), None if annotation == "crowd" else int(annotation)


def transform_keypoints(
    keypoints: Iterable[float], *, source_size: tuple[int, int],
    resized_size: tuple[int, int], crop_box: tuple[int, int, int, int],
) -> list[list[float]]:
    """Map COCO x/y/v triples through the persisted resize and crop geometry."""
    values = list(keypoints)
    if len(values) != 51:
        raise ReferencePoseError(f"Expected 17 COCO x/y/v triples, got {len(values)} values")
    source_width, source_height = source_size
    resized_width, resized_height = resized_size
    left, top, right, bottom = crop_box
    if source_width < 1 or source_height < 1 or resized_width < 1 or resized_height < 1:
        raise ReferencePoseError("Geometry dimensions must be positive")
    if right <= left or bottom <= top:
        raise ReferencePoseError("Geometry crop box must have positive area")
    scale_x, scale_y = resized_width / source_width, resized_height / source_height
    result = []
    for offset in range(0, 51, 3):
        x, y, visibility = values[offset:offset + 3]
        # COCO uses v=0 for unlabelled joints; keep the value but never assign
        # a fabricated coordinate to it.
        if int(visibility) == 0:
            result.append([float(x), float(y), float(visibility)])
        else:
            result.append([float(x) * scale_x - left, float(y) * scale_y - top, float(visibility)])
    return result


def load_coco_annotations(paths: Iterable[str | Path]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]], dict[str, str]]:
    """Load official COCO keypoint JSON files, rejecting ambiguous duplicate IDs."""
    images: dict[int, dict[str, Any]] = {}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_annotation: dict[int, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for supplied_path in paths:
        path = Path(supplied_path)
        if not path.is_file():
            raise ReferencePoseError(f"COCO annotation JSON is missing: {path}")
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReferencePoseError(f"Invalid JSON: {path}") from exc
        for image in payload.get("images", []):
            image_id = image.get("id")
            if not isinstance(image_id, int) or not isinstance(image.get("width"), int) or not isinstance(image.get("height"), int):
                raise ReferencePoseError(f"Malformed COCO image record in {path}")
            previous = images.get(image_id)
            if previous is not None and previous != image:
                raise ReferencePoseError(f"Ambiguous duplicate COCO image id {image_id}")
            images[image_id] = image
        for annotation in payload.get("annotations", []):
            annotation_id, image_id = annotation.get("id"), annotation.get("image_id")
            if not isinstance(annotation_id, int) or not isinstance(image_id, int):
                raise ReferencePoseError(f"Malformed COCO annotation record in {path}")
            previous = by_annotation.get(annotation_id)
            if previous is not None and previous != annotation:
                raise ReferencePoseError(f"Ambiguous duplicate COCO annotation id {annotation_id}")
            by_annotation[annotation_id] = annotation
            by_image[image_id].append(annotation)
    return images, by_image, by_annotation, hashes


def build_coco_reference_records(samples: Iterable[Mapping[str, Any]], annotation_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Build per-stem references, validating IDs and original image dimensions."""
    images, by_image, by_annotation, annotation_hashes = load_coco_annotations(annotation_paths)
    records = []
    for sample in samples:
        stem = sample.get("stem")
        if not isinstance(stem, str) or not stem.startswith("coco_"):
            continue
        image_id, required_annotation_id = parse_coco_stem(stem)
        image = images.get(image_id)
        if image is None:
            raise ReferencePoseError(f"{stem}: source COCO image id {image_id} was not found")
        source_size = tuple(sample.get("source_size", ()))
        resized_size = tuple(sample.get("resized_size", ()))
        crop_box = tuple(sample.get("crop_box", ()))
        bucket = tuple(sample.get("bucket", ()))
        if source_size != (image["width"], image["height"]):
            raise ReferencePoseError(f"{stem}: annotation size {(image['width'], image['height'])} does not match shard source_size {source_size}")
        if len(resized_size) != 2 or len(crop_box) != 4 or len(bucket) != 2:
            raise ReferencePoseError(f"{stem}: shard lacks complete persisted paired geometry")
        candidates = by_image.get(image_id, []) if required_annotation_id is None else [by_annotation.get(required_annotation_id)]
        people = []
        for annotation in candidates:
            if annotation is None or annotation.get("image_id") != image_id:
                raise ReferencePoseError(f"{stem}: annotation id {required_annotation_id} is not attached to image {image_id}")
            if annotation.get("category_id") != 1 or annotation.get("iscrowd", 0) != 0:
                continue
            keypoints = annotation.get("keypoints")
            if not isinstance(keypoints, list) or len(keypoints) != 51 or int(annotation.get("num_keypoints", 0)) < 1:
                continue
            people.append({
                "person_id": annotation["id"], "annotation_id": annotation["id"],
                "bbox_xywh": annotation.get("bbox"), "area": annotation.get("area"),
                "keypoints_source": [list(map(float, keypoints[index:index + 3])) for index in range(0, 51, 3)],
                "keypoints_bucket": transform_keypoints(keypoints, source_size=source_size, resized_size=resized_size, crop_box=crop_box),
            })
        if required_annotation_id is not None and not people:
            raise ReferencePoseError(f"{stem}: required COCO person annotation {required_annotation_id} has no usable 17-keypoint record")
        records.append({
            "stem": stem, "source": "coco", "source_image_id": image_id,
            "source_dimensions": list(source_size), "joint_schema": {"name": "coco_17", "joints": list(COCO_17)},
            "people": people,
            "geometry": {"source_size": list(source_size), "resized_size": list(resized_size), "crop_box": list(crop_box), "bucket": list(bucket), "coordinate_transform": "x_bucket=x*(resized_width/source_width)-crop_left; y_bucket=y*(resized_height/source_height)-crop_top"},
            "provenance": {"dataset": "MS COCO 2017 person keypoints", "annotation_sha256": annotation_hashes},
        })
    return records


def write_reference_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path) -> str:
    """Atomically write a deterministic immutable sidecar and return its hash."""
    path = Path(output)
    ordered = sorted(records, key=lambda record: str(record["stem"]))
    content = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in ordered)
    digest = hashlib.sha256(content.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest
