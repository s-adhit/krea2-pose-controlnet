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


POSE_TARGET_SIDECAR_VERSION = 3
RECORDS_NAME = "records.jsonl"
METADATA_NAME = "metadata.json"
COMMON_BODY_17 = COCO_17

# The checked-in v1 export is the sole numerical source for the available
# targets.  It already contains the original COCO / Human-Art annotations, so
# sidecar construction must never reopen either large upstream annotation set.
AUTHORITATIVE_EXPORT_SCHEMA_VERSION = 1
AUTHORITATIVE_EXPORT_FORMAT = "pose_targets_authoritative_v1_jsonl"

# Versioned, reviewed exception policy for defects in the *original* Human-Art
# numerical annotations.  This is deliberately an allow-list with exact raw
# values, rather than a geometry tolerance or a silent repair.  The affected
# coordinates remain authoritative reconstruction inputs, but they can never
# become pose-reward targets.  Any addition, removal, or alteration fails the
# active-dataset build/audit until it is reviewed and this policy is revised.
AUTHORITATIVE_SOURCE_OOB_POLICY = {
    "policy_version": "humanart_original_source_oob_v1",
    "scope": "active authoritative export only",
    "rationale": "Verified original Human-Art numerical annotation defects; raw values are retained for historical raster reconstruction.",
    "reviewed_anomalies": (
        ("painting_humanart_2000000000804", "2000000007651", 9, 704.8835, 622.4527, 1024, 589),
        ("painting_humanart_2000000000804", "2000000007651", 13, 579.3224, 835.9159, 1024, 589),
        ("painting_humanart_2000000000804", "2000000007651", 14, 465.2948, 848.2689, 1024, 589),
        ("painting_humanart_2000000000804", "2000000007651", 15, 810.484, 1195.846, 1024, 589),
        ("painting_humanart_2000000000804", "2000000007651", 16, 480.6118, 1140.7945, 1024, 589),
        ("sculpture_humanart_14000000001208", "14000000088574", 15, 1982.027, 3177.6055, 4128, 3096),
        ("sculpture_humanart_14000000001208", "14000000088574", 16, 1461.6133, 3164.4104, 4128, 3096),
    ),
}

# This describes the actual historical PoseBridge body raster, not the reward
# target.  In particular, the renderer-only neck is absent from COCO-17.
POSEBRIDGE_BODY_RENDERER = {
    "validated_historical_renderer": True,
    "identifier": "posebridge_body_renderer_v1",
    "topology": "openpose_body18",
    "coordinate_mapping": "coco17_to_unified18_with_renderer_only_neck",
    "line_width": 3,
    "endpoint_radius": 4,
    "endpoint_rgb": [255, 255, 255],
    "body_colors": "openpose_18_rainbow_rgb",
    "hands": False,
}

AUTHORITATIVE_DIAGNOSTIC_STEMS = frozenset({
    "coco_156320_crowd", "coco_299468_426600", "coco_379542_449327", "coco_64240_crowd",
    "painting_humanart_10000000000555", "painting_humanart_10000000000838",
    "painting_humanart_1000000000218", "painting_humanart_1000000001225",
    "painting_humanart_1000000002653", "painting_humanart_6000000000319",
    "painting_humanart_6000000002434", "painting_humanart_9000000000608",
    "painting_humanart_9000000002722", "real_human_humanart_15000000000201",
    "real_human_humanart_15000000002196", "real_human_humanart_17000000000288",
    "real_human_humanart_17000000001552", "real_human_humanart_17000000001695",
    "sculpture_humanart_14000000000243", "sculpture_humanart_14000000001143",
    "sculpture_humanart_14000000004479",
})


class PoseTargetError(ValueError):
    """Raised when authoritative target provenance is incomplete or ambiguous."""


class _AuthoritativePeople(list[dict[str, Any]]):
    """A source-grouped people list retaining image dimensions even when empty."""

    def __init__(
        self, people: Iterable[dict[str, Any]], source_size: Iterable[int] | None = None,
        source_image_id: str | int | None = None, **metadata: Any,
    ):
        super().__init__(people)
        self.source_size = tuple(source_size) if source_size is not None else None
        self.source_image_id = source_image_id
        for key, value in metadata.items():
            setattr(self, key, value)


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

    `keypoints_training` remains the clipped compatibility/raster field, but
    every joint also gets an explicit provenance object.  The latter retains
    the raw source coordinate and its unclipped geometric training coordinate
    so historical reconstruction and future reward code are separate,
    auditable consumers of the same annotation.
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
    training, in_frame, reward_visible, joint_provenance = [], [], [], []
    for x, y, score in raw:
        if score < 0:
            raise PoseTargetError("Authoritative keypoint visibility/confidence must be non-negative")
        mapped_x, mapped_y = x * sx - left, y * sy - top
        present = score > 0
        source_in_bounds = 0.0 <= x <= sw and 0.0 <= y <= sh
        final_in_frame = 0.0 <= mapped_x <= bw - 1 and 0.0 <= mapped_y <= bh - 1
        if not present:
            invalid_reason = "source_visibility_or_confidence_zero"
        elif not source_in_bounds:
            invalid_reason = "source_coordinate_out_of_bounds"
        elif not final_in_frame:
            invalid_reason = "transformed_coordinate_out_of_frame"
        else:
            invalid_reason = None
        reward_valid = invalid_reason is None
        training.append([min(max(mapped_x, 0.0), bw - 1), min(max(mapped_y, 0.0), bh - 1), score])
        in_frame.append(final_in_frame)
        reward_visible.append(reward_valid)
        joint_provenance.append({
            "source_coordinate": [x, y],
            "source_visibility_confidence": score,
            "source_in_bounds": source_in_bounds,
            "training_coordinate": [mapped_x, mapped_y],
            "final_in_frame": final_in_frame,
            "reward_joint_valid": reward_valid,
            "reward_invalid_reason": invalid_reason,
        })
    bbox = person.get("bbox_xywh")
    training_box = _transform_box(bbox, sx=sx, sy=sy, left=left, top=top, width=bw, height=bh) if bbox is not None else None
    return {
        "person_id": person.get("person_id"), "annotation_id": person.get("annotation_id"),
        "bbox_source_xywh": bbox, "keypoints_source": raw,
        "visibility_or_confidence_source": [point[2] for point in raw],
        "keypoints_training": training, "keypoints_training_in_frame": in_frame,
        "reward_visible_mask": reward_visible, "joint_provenance": joint_provenance,
        "bbox_training_xywh": training_box,
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


def load_authoritative_export(path: str | Path) -> tuple[dict[str, _AuthoritativePeople], dict[str, Any]]:
    """Load and validate the checked-in original-annotation export once.

    The returned lookup is deliberately keyed only by the final PoseBridge
    stem.  Duplicate records, including records outside the active manifests,
    are an ambiguity and fail the build.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise PoseTargetError(f"Authoritative export missing: {artifact}")
    rows: dict[str, _AuthoritativePeople] = {}
    raw = artifact.read_bytes()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PoseTargetError(f"Authoritative export line {line_number}: invalid JSON") from exc
        stem, people = _authoritative_export_row(row, line_number)
        if stem in rows:
            raise PoseTargetError(f"Authoritative export has duplicate stem: {stem}")
        rows[stem] = people
    if not rows:
        raise PoseTargetError("Authoritative export contains no records")
    return rows, {
        "format": AUTHORITATIVE_EXPORT_FORMAT,
        "schema_version": AUTHORITATIVE_EXPORT_SCHEMA_VERSION,
        "path": str(artifact.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(rows),
    }


def build_authoritative_sidecar_records(
    geometry_by_stem: Mapping[str, Mapping[str, Any]], *, authoritative_jsonl: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build v3 records from the authoritative export and persisted shard geometry."""
    export, export_metadata = load_authoritative_export(authoritative_jsonl)
    expected = Counter(source_for_stem(stem) for stem in geometry_by_stem)
    source_oob = authoritative_source_oob_report(export, active_stems=geometry_by_stem)
    if source_oob["status"] != "PASS":
        raise PoseTargetError(
            "Authoritative visible source-coordinate out-of-bounds contract failed: "
            f"unexpected={source_oob['unexpected_count']}, missing_reviewed={source_oob['missing_reviewed_count']}, "
            f"altered_reviewed={source_oob['altered_reviewed_count']}"
        )
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for stem, geometry in sorted(geometry_by_stem.items()):
        source = source_for_stem(stem)
        if source == "danbooru":
            records.append(_unavailable_record(stem, source, geometry))
            continue
        people = export.get(stem)
        if people is None:
            unresolved.append(stem)
            continue
        row_source = getattr(people, "export_source", None)
        if row_source != ("coco" if source == "coco" else "humanart"):
            raise PoseTargetError(f"{stem}: authoritative export source {row_source!r} disagrees with active source {source!r}")
        if people.source_size != tuple(geometry["source_size"]):
            raise PoseTargetError(
                f"{stem}: authoritative source size {people.source_size} does not match persisted shard geometry {tuple(geometry['source_size'])}"
            )
        transformed = [transform_person(
            person, source_size=geometry["source_size"], resized_size=geometry["resized_size"],
            crop_box=geometry["crop_box"], bucket=geometry["bucket"],
        ) for person in people]
        records.append({
            "schema_version": POSE_TARGET_SIDECAR_VERSION, "stem": stem, "source": source,
            "pose_reward_available": True, "target_provenance": "original_annotation",
            "annotation_source": AUTHORITATIVE_EXPORT_FORMAT,
            "source_image_id": people.source_image_id,
            "source_image_name": getattr(people, "source_image_name", None),
            "source_annotation_split": getattr(people, "source_annotation_split", None),
            "source_size": list(geometry["source_size"]), "resized_size": list(geometry["resized_size"]),
            "crop_box": list(geometry["crop_box"]), "bucket": list(geometry["bucket"]),
            "geometry_transform": "x_final=clip(x_source*resized_width/source_width-crop_left,0,bucket_width-1); y_final=clip(y_source*resized_height/source_height-crop_top,0,bucket_height-1)",
            "bbox_clip_convention": "xywh is transformed then intersected with the inclusive final pixel canvas [0,width-1] x [0,height-1]",
            "joint_schema": "coco17", "common_body_mapping": common_body_mapping("coco17"),
            "person_grouping": "original image-level people list preserved; renderer-only neck is never a target joint",
            "consumer_semantics": {
                "historical_reconstruction": {
                    "keypoint_field": "keypoints_source",
                    "coordinate_space": "raw_authoritative_source",
                    "source_oob_behavior": "preserve raw coordinates; historical rasterization clips off-canvas geometry naturally",
                },
                "pose_reward": {
                    "joint_field": "joint_provenance",
                    "eligibility_field": "reward_joint_valid",
                    "source_oob_behavior": "visible source-out-of-bounds joints are excluded with source_coordinate_out_of_bounds",
                },
            },
            "people": transformed, "renderer": dict(POSEBRIDGE_BODY_RENDERER),
            "provenance_metadata": {"authoritative_export": export_metadata, **getattr(people, "export_metadata", {})},
        })
    if unresolved:
        raise PoseTargetError(f"Claimed authoritative targets missing for {len(unresolved)} active samples (examples: {unresolved[:8]})")
    diagnostic = diagnostic_coverage({record["stem"]: record for record in records}) if AUTHORITATIVE_DIAGNOSTIC_STEMS <= set(geometry_by_stem) else {
        "expected_annotated_stems": len(AUTHORITATIVE_DIAGNOSTIC_STEMS), "status": "NOT_APPLICABLE_PARTIAL_GEOMETRY",
    }
    summary = {
        "expected_counts": dict(sorted(expected.items())), "records": len(records),
        "coverage": coverage_summary(records), "authoritative_export": export_metadata,
        "inactive_authoritative_records": len(set(export) - set(geometry_by_stem)),
        "source_oob_contract": source_oob,
        "diagnostic_coverage": diagnostic,
    }
    if summary["diagnostic_coverage"]["status"] == "FAIL":
        raise PoseTargetError(f"Authoritative diagnostic stems unresolved: {summary['diagnostic_coverage']}")
    return records, summary


def _authoritative_export_row(row: Any, line_number: int) -> tuple[str, _AuthoritativePeople]:
    """Normalize exactly one checked-in export row without altering its meaning."""
    if not isinstance(row, Mapping):
        raise PoseTargetError(f"Authoritative export line {line_number}: record is not an object")
    stem = row.get("stem")
    if not isinstance(stem, str) or not stem:
        raise PoseTargetError(f"Authoritative export line {line_number}: missing stem")
    if row.get("schema_version") != AUTHORITATIVE_EXPORT_SCHEMA_VERSION:
        raise PoseTargetError(f"{stem}: unsupported authoritative export schema version")
    if row.get("final_file_name") != f"{stem}.jpg":
        raise PoseTargetError(f"{stem}: final_file_name must be the exact final PoseBridge JPG name")
    source = row.get("source")
    if source not in {"coco", "humanart"}:
        raise PoseTargetError(f"{stem}: unsupported authoritative export source {source!r}")
    expected_source = source_for_stem(stem)
    if expected_source == "danbooru" or (source == "coco") != (expected_source == "coco"):
        raise PoseTargetError(f"{stem}: export source does not match final PoseBridge stem")
    if source == "humanart":
        medium = row.get("medium")
        expected_medium = expected_source.removeprefix("humanart_")
        if medium != expected_medium:
            raise PoseTargetError(f"{stem}: authoritative Human-Art medium {medium!r} != stem medium {expected_medium!r}")
    if row.get("target_provenance") != "original_annotation" or row.get("pose_reward_available") is not True:
        raise PoseTargetError(f"{stem}: authoritative export record does not claim original available annotations")
    if row.get("joint_schema") != "coco17" or tuple(row.get("joint_names", ())) != tuple(COCO_17):
        raise PoseTargetError(f"{stem}: authoritative export must contain named COCO-17 joints")
    source_size = (row.get("source_width"), row.get("source_height"))
    _size(source_size, f"{stem}.source_size")
    source_image_id = row.get("source_image_id")
    if not isinstance(source_image_id, (str, int)):
        raise PoseTargetError(f"{stem}: source_image_id must be a string or integer")
    if not isinstance(row.get("source_image_name"), str) or not row["source_image_name"]:
        raise PoseTargetError(f"{stem}: source_image_name is required")
    if not isinstance(row.get("source_annotation_split"), str) or not row["source_annotation_split"]:
        raise PoseTargetError(f"{stem}: source_annotation_split is required")
    people_payload = row.get("people")
    if not isinstance(people_payload, list):
        raise PoseTargetError(f"{stem}: people must be a list")
    people: list[dict[str, Any]] = []
    for person_index, person in enumerate(people_payload):
        if not isinstance(person, Mapping):
            raise PoseTargetError(f"{stem}: person {person_index} is not an object")
        annotation_id = person.get("annotation_id")
        if not isinstance(annotation_id, (str, int)):
            raise PoseTargetError(f"{stem}: person {person_index} lacks original annotation_id")
        points = _keypoints(person.get("keypoints_coco17"))
        bbox = person.get("bbox_xywh")
        if bbox is not None:
            _transform_box(bbox, sx=1.0, sy=1.0, left=0, top=0, width=source_size[0], height=source_size[1])
        declared_visible = person.get("num_visible_keypoints")
        if not isinstance(declared_visible, int) or declared_visible != sum(point[2] > 0 for point in points):
            raise PoseTargetError(f"{stem}: person {person_index} has inconsistent num_visible_keypoints")
        people.append({
            "person_id": annotation_id, "annotation_id": annotation_id, "bbox_xywh": bbox,
            "keypoints_source": points, "_authoritative_source_size": list(source_size),
        })
    return stem, _AuthoritativePeople(
        people, source_size, source_image_id, export_source=source,
        source_image_name=row["source_image_name"], source_annotation_split=row["source_annotation_split"],
        export_metadata={
            "source": source, "medium": row.get("medium"), "sample_type": row.get("sample_type"),
            "source_schema_version": row["schema_version"],
        },
    )


def authoritative_source_oob_report(
    export: Mapping[str, _AuthoritativePeople], *, active_stems: Mapping[str, Any] | Iterable[str],
) -> dict[str, Any]:
    """Audit visible source-OOB joints against the exact reviewed policy.

    This is intentionally evaluated only for active stems.  Inactive export
    rows are not silently certified as part of the dataset used for training.
    """
    active = set(active_stems)
    events: list[dict[str, Any]] = []
    for stem in sorted(active & set(export)):
        people = export[stem]
        width, height = _size(people.source_size or (), f"{stem}.source_size")
        for person in people:
            annotation_id = person.get("annotation_id")
            for joint_index, (x, y, visibility) in enumerate(_keypoints(person.get("keypoints_source"))):
                if visibility > 0 and not (0.0 <= x <= width and 0.0 <= y <= height):
                    events.append({
                        "stem": stem, "annotation_id": annotation_id,
                        "joint_index": joint_index, "joint_name": COCO_17[joint_index],
                        "raw_coordinate": [x, y], "source_size": [width, height],
                        "overshoot": {
                            "left": max(0.0, -x), "top": max(0.0, -y),
                            "right": max(0.0, x - width), "bottom": max(0.0, y - height),
                        },
                    })
    all_expected = {
        (stem, annotation_id, joint_index): (x, y, width, height)
        for stem, annotation_id, joint_index, x, y, width, height
        in AUTHORITATIVE_SOURCE_OOB_POLICY["reviewed_anomalies"]
    }
    # Small fixture/partial-geometry builds remain useful in tests.  The exact
    # reviewed-set requirement applies once the active dataset includes this
    # versioned policy's scope; unknown OOB joints still fail in either mode.
    contract_applicable = bool({key[0] for key in all_expected} & active)
    expected = all_expected if contract_applicable else {}
    actual = {(event["stem"], str(event["annotation_id"]), event["joint_index"]): event for event in events}
    unexpected = [event for key, event in sorted(actual.items()) if key not in expected]
    missing = [
        {"stem": stem, "annotation_id": annotation_id, "joint_index": joint_index, "joint_name": COCO_17[joint_index]}
        for stem, annotation_id, joint_index in sorted(set(expected) - set(actual))
    ]
    altered = []
    for key in sorted(set(expected) & set(actual)):
        event = actual[key]; x, y, width, height = expected[key]
        if tuple(event["raw_coordinate"]) != (x, y) or tuple(event["source_size"]) != (width, height):
            altered.append({"expected": {"raw_coordinate": [x, y], "source_size": [width, height]}, "actual": event})
    affected_stems = sorted({event["stem"] for event in events})
    return {
        "policy": {key: value for key, value in AUTHORITATIVE_SOURCE_OOB_POLICY.items() if key != "reviewed_anomalies"},
        "contract_applicable": contract_applicable,
        "expected_visible_source_oob_joint_count": len(expected),
        "visible_source_oob_joint_count": len(events),
        "affected_stems": affected_stems,
        "events": events,
        "unexpected_events": unexpected, "unexpected_count": len(unexpected),
        "missing_reviewed_events": missing, "missing_reviewed_count": len(missing),
        "altered_reviewed_events": altered, "altered_reviewed_count": len(altered),
        "status": "PASS" if not unexpected and not missing and not altered else "FAIL",
    }


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


def diagnostic_coverage(records_by_stem: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Fail-closed result for the explicitly reviewed 21 annotation stems."""
    missing = sorted(stem for stem in AUTHORITATIVE_DIAGNOSTIC_STEMS if stem not in records_by_stem)
    unavailable = sorted(stem for stem in AUTHORITATIVE_DIAGNOSTIC_STEMS
                         if stem in records_by_stem and records_by_stem[stem].get("pose_reward_available") is not True)
    return {"expected_annotated_stems": len(AUTHORITATIVE_DIAGNOSTIC_STEMS), "resolved": len(AUTHORITATIVE_DIAGNOSTIC_STEMS) - len(missing) - len(unavailable),
            "missing": missing, "unavailable": unavailable, "status": "PASS" if not missing and not unavailable else "FAIL"}


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
        for person_index, person in enumerate(record["people"]):
            joints = person.get("joint_provenance")
            if not isinstance(joints, list) or len(joints) != 17:
                raise PoseTargetError(f"{record.get('stem')}: person {person_index} lacks 17 joint provenance entries")
            for joint_index, joint in enumerate(joints):
                required_joint_fields = (
                    "source_coordinate", "source_visibility_confidence", "source_in_bounds",
                    "training_coordinate", "final_in_frame", "reward_joint_valid", "reward_invalid_reason",
                )
                if not isinstance(joint, Mapping) or any(field not in joint for field in required_joint_fields):
                    raise PoseTargetError(f"{record.get('stem')}: person {person_index} joint {joint_index} has incomplete provenance")
                if joint["reward_joint_valid"] and joint["reward_invalid_reason"] is not None:
                    raise PoseTargetError(f"{record.get('stem')}: person {person_index} joint {joint_index} has contradictory reward provenance")
                if not joint["reward_joint_valid"] and not isinstance(joint["reward_invalid_reason"], str):
                    raise PoseTargetError(f"{record.get('stem')}: person {person_index} joint {joint_index} lacks reward invalid reason")
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
