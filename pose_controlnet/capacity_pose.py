"""Read-only projection of the exact Mixed-32 source-space pose sidecar.

The capacity reference sidecar deliberately contains only immutable source
coordinates so it can be reused for native evaluation and alternate-resolution
training.  This module projects those coordinates through one already-verified
paired training geometry into the existing fixed-box Keypoint R-CNN loss input.
It never writes the sidecar, latent cache, checkpoint, or model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from pose_controlnet.pose_targets import source_for_stem, transform_person
from pose_controlnet.reference_pose import load_exact_capacity_reference_sidecar


def _geometry(sample: Mapping[str, Any], *, stem: str) -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    for field, size in (("source_size", 2), ("resized_size", 2), ("crop_box", 4), ("bucket", 2)):
        value = sample.get(field)
        if not isinstance(value, (list, tuple)) or len(value) != size or any(not isinstance(item, int) for item in value):
            raise ValueError(f"{stem}: alternate 768 geometry is incomplete")
        values[field] = list(value)
    return values


def usable_capacity_pose_record(record: Mapping[str, Any]) -> bool:
    """Whether an available record has at least one valid fixed-box joint."""
    if record.get("pose_reward_available") is not True or not isinstance(record.get("people"), list):
        return False
    for person in record["people"]:
        xywh, joints = person.get("bbox_training_xywh"), person.get("joint_provenance")
        if (isinstance(xywh, list) and len(xywh) == 4 and xywh[2] > 0 and xywh[3] > 0
                and isinstance(joints, list) and any(joint.get("reward_joint_valid") is True for joint in joints)):
            return True
    return False


def load_capacity_pose_records(*, sidecar: str | Path, experiment_name: str,
                               data: Any, stems: Iterable[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the exact immutable Mixed-32 JSONL and reproject it for ``data``.

    The six Danbooru rows remain present for exact identity, but are explicitly
    unavailable to the numerical reward.  Every eligible row must retain a
    usable authoritative target after the exact paired 768 transform.
    """
    ordered = tuple(stems)
    samples = {data[index]["stem"]: data[index] for index in range(len(data))}
    if tuple(samples) != ordered:
        raise ValueError("Selected capacity dataset no longer preserves the exact immutable Mixed-32 order")
    geometry = {stem: _geometry(samples[stem], stem=stem) for stem in ordered}
    metadata, source_rows = load_exact_capacity_reference_sidecar(
        sidecar, experiment_name=experiment_name, expected_stems=ordered, geometry_by_stem=geometry,
    )
    projected: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        stem = source["stem"]
        domain = source_for_stem(stem)
        row_geometry = geometry[stem]
        if domain == "danbooru":
            if source.get("pose_scoring_available") is not False or source.get("people") is not None:
                raise ValueError(f"{stem}: Danbooru must remain unavailable for numerical pose reward")
            projected[stem] = {
                "stem": stem, "source": domain, "pose_reward_available": False,
                "sidecar_status": "unavailable", **row_geometry,
            }
            continue
        if source.get("pose_scoring_available") is not True or not isinstance(source.get("people"), list):
            raise ValueError(f"{stem}: eligible authoritative pose target is unavailable")
        if source.get("source_size") != row_geometry["source_size"]:
            raise ValueError(f"{stem}: authoritative source dimensions disagree with 768 training geometry")
        people = []
        for person in source["people"]:
            people.append(transform_person(
                {"person_id": person.get("person_id"), "annotation_id": person.get("annotation_id"),
                 "bbox_xywh": person.get("bbox_source_xywh"), "keypoints_source": person.get("keypoints_source")},
                **row_geometry,
            ))
        row = {"stem": stem, "source": domain, "pose_reward_available": True,
               "sidecar_status": "available", "people": people, **row_geometry}
        if not usable_capacity_pose_record(row):
            raise ValueError(f"{stem}: no valid authoritative fixed-box pose target remains after 768 paired geometry")
        projected[stem] = row
    if tuple(projected) != ordered:
        raise AssertionError("Capacity pose sidecar projection escaped exact manifest order")
    return metadata, projected
