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


CAPACITY_REFERENCE_FORMAT_VERSION = 1
EXACT_MANIFEST_REFERENCE_FORMAT_VERSION = 1
EXACT_MANIFEST_REFERENCE_KIND = "exact_manifest_authoritative_pose_v1"


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
            "schema_version": CAPACITY_REFERENCE_FORMAT_VERSION,
            "stem": stem, "source": "coco", "status": "available",
            "mode": "crowd" if required_annotation_id is None else "single", "source_image_id": image_id,
            "source_size": list(source_size), "resized_size": list(resized_size), "crop_box": list(crop_box), "bucket": list(bucket),
            "source_dimensions": list(source_size), "joint_schema": {"name": "coco_17", "joints": list(COCO_17)},
            "person_grouping": "one record per official COCO person; image-level list preserves COCO grouping",
            "people": [{**person, "keypoints": person["keypoints_source"]} for person in people],
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


def _capacity_geometry(sample: Mapping[str, Any], *, stem: str) -> dict[str, list[int]]:
    """Return persisted paired geometry without recomputing or normalising it."""
    fields = (("source_size", 2), ("resized_size", 2), ("crop_box", 4), ("bucket", 2))
    geometry: dict[str, list[int]] = {}
    for field, length in fields:
        value = sample.get(field)
        if not isinstance(value, (list, tuple)) or len(value) != length or any(not isinstance(item, int) for item in value):
            raise ReferencePoseError(f"{stem}: verified latent record lacks compatible persisted {field}")
        geometry[field] = list(value)
    source_width, source_height = geometry["source_size"]
    resized_width, resized_height = geometry["resized_size"]
    left, top, right, bottom = geometry["crop_box"]
    bucket_width, bucket_height = geometry["bucket"]
    if min(source_width, source_height, resized_width, resized_height, bucket_width, bucket_height) < 1:
        raise ReferencePoseError(f"{stem}: persisted geometry has non-positive dimensions")
    if right - left != bucket_width or bottom - top != bucket_height or left < 0 or top < 0 or right > resized_width or bottom > resized_height:
        raise ReferencePoseError(f"{stem}: persisted crop geometry is incompatible")
    return geometry


def resolve_exact_capacity_latent_samples(*, experiment_name: str, latent_root: str | Path,
                                          manifest_path: str | Path | None = None) -> tuple[tuple[str, ...], list[dict[str, Any]], list[str]]:
    """Resolve only one exact capacity manifest from direct verified ``train-*.pt`` shards.

    This deliberately does not consult a shard-set parent, cached text-conditioning
    archives, checkpoints, or any recursive descendant.  A duplicate requested
    stem is ambiguous even if its payload happens to compare equal.
    """
    # Import locally: overfit_capacity imports pose_targets, which imports this module.
    from pose_controlnet.overfit_capacity import OVERFIT_SAMPLE_COUNT, experiment, manifest_stems

    spec = experiment(experiment_name)
    selected_manifest = Path(manifest_path) if manifest_path is not None else spec.manifest
    stems = manifest_stems(selected_manifest)
    if len(stems) != OVERFIT_SAMPLE_COUNT or len(set(stems)) != OVERFIT_SAMPLE_COUNT:
        raise ReferencePoseError(f"{experiment_name}: manifest is not exactly {OVERFIT_SAMPLE_COUNT} unique samples")
    if spec.source != "coco" or any(not stem.startswith("coco_") for stem in stems):
        raise ReferencePoseError(f"{experiment_name}: authoritative COCO reference construction requires a COCO-only manifest")

    root = Path(latent_root)
    shards = sorted(path for path in root.glob("train-*.pt") if path.is_file())
    if not shards:
        raise ReferencePoseError(f"No direct verified train-*.pt latent shards under explicit root: {root}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production environment always includes torch
        raise ReferencePoseError("PyTorch is required to read verified latent shards") from exc

    wanted, resolved = set(stems), {}
    used: list[str] = []
    for shard in shards:
        try:
            payload = torch.load(shard, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ReferencePoseError(f"Unreadable verified latent shard: {shard}") from exc
        if not isinstance(payload, dict) or payload.get("format_version") != 1 or payload.get("split") != "train" or not isinstance(payload.get("samples"), list):
            raise ReferencePoseError(f"Invalid verified v1 train latent shard: {shard}")
        shard_used = False
        for sample in payload["samples"]:
            if not isinstance(sample, Mapping):
                raise ReferencePoseError(f"Malformed sample in verified latent shard: {shard}")
            stem = sample.get("stem")
            if stem not in wanted:
                continue
            if stem in resolved:
                raise ReferencePoseError(f"Ambiguous requested stem {stem!r} appears in multiple latent records")
            geometry = _capacity_geometry(sample, stem=stem)
            # The copy contains only persisted metadata; latent tensors and captions
            # cannot become accidental geometry/reference sources downstream.
            resolved[stem] = {"stem": stem, **geometry}
            shard_used = True
        if shard_used:
            used.append(str(shard.resolve()))
    missing = sorted(wanted - set(resolved))
    if missing:
        raise ReferencePoseError(f"Requested capacity manifest stems missing from verified train latent shards: {missing[:8]}")
    samples = [resolved[stem] for stem in stems]
    return stems, samples, used


def _capacity_metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".metadata.json")


def _manifest_domain(stem: str) -> tuple[str, str]:
    """Return scorer source and exact source domain for a known PoseBridge stem."""
    if stem.startswith("coco_"):
        return "coco", "coco"
    if stem.startswith("painting_humanart_"):
        return "humanart", "humanart_painting"
    if stem.startswith("real_human_humanart_"):
        return "humanart", "humanart_real_human"
    if stem.startswith("sculpture_humanart_"):
        return "humanart", "humanart_sculpture"
    if stem.startswith("danbooru_"):
        return "danbooru", "danbooru"
    raise ReferencePoseError(f"Unsupported capacity-manifest stem: {stem!r}")


def _exact_manifest_stems(manifest_path: str | Path) -> tuple[str, ...]:
    """Read one immutable capacity manifest without deriving any path layout."""
    path = Path(manifest_path)
    if not path.is_file():
        raise ReferencePoseError(f"Exact capacity manifest is missing: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ReferencePoseError(f"Exact capacity manifest is invalid JSONL: {path}") from exc
    stems = []
    for index, row in enumerate(rows, 1):
        file_name = row.get("file_name") if isinstance(row, Mapping) else None
        if not isinstance(file_name, str) or Path(file_name).name != file_name or Path(file_name).suffix.lower() != ".jpg":
            raise ReferencePoseError(f"Exact capacity manifest row {index} has an invalid bare .jpg file_name")
        stems.append(Path(file_name).stem)
    if len(stems) != 32 or len(set(stems)) != len(stems):
        raise ReferencePoseError("Exact capacity manifest must contain exactly 32 unique stems")
    return tuple(stems)


def _source_people_for_exact_manifest(record: Mapping[str, Any], *, stem: str) -> list[dict[str, Any]]:
    """Copy only source-space target data; never carry old training geometry forward."""
    people = record.get("people")
    if not isinstance(people, list):
        raise ReferencePoseError(f"{stem}: authoritative target has no people list")
    copied: list[dict[str, Any]] = []
    for person_index, person in enumerate(people):
        if not isinstance(person, Mapping):
            raise ReferencePoseError(f"{stem}: authoritative person {person_index} is malformed")
        points = person.get("keypoints_source")
        if not isinstance(points, list) or len(points) != 17 or any(
            not isinstance(point, list) or len(point) != 3 or not all(isinstance(value, (int, float)) for value in point)
            for point in points
        ):
            raise ReferencePoseError(f"{stem}: authoritative person {person_index} lacks 17 source-space keypoints")
        source_points = [[float(value) for value in point] for point in points]
        copied.append({
            "person_id": person.get("person_id"), "annotation_id": person.get("annotation_id"),
            "bbox_source_xywh": person.get("bbox_source_xywh"),
            "keypoints": source_points, "keypoints_source": source_points,
            "source_visibility_or_confidence": [point[2] for point in source_points],
            "source_visible_mask": [point[2] > 0 for point in source_points],
        })
    return copied


def build_exact_manifest_reference_records(*, manifest_path: str | Path,
                                           authoritative_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
                                           authoritative_metadata: Mapping[str, Any],
                                           compatible_experiments: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build source-space-only references for one exact 32-stem manifest.

    This is intentionally independent of latent shards and of training
    resolution.  Native evaluation geometry is supplied later by the persisted
    generation result, immediately before PCK scoring.
    """
    manifest = Path(manifest_path)
    stems = _exact_manifest_stems(manifest)
    source_rows = list(authoritative_records.values()) if isinstance(authoritative_records, Mapping) else list(authoritative_records)
    supplied_records: dict[str, Mapping[str, Any]] = {}
    for source in source_rows:
        stem = source.get("stem") if isinstance(source, Mapping) else None
        if not isinstance(stem, str):
            raise ReferencePoseError("Authoritative target lookup contains a record without a stem")
        if stem in supplied_records:
            raise ReferencePoseError(f"Authoritative target lookup contains duplicate stem: {stem}")
        supplied_records[stem] = source
    expected_sha = authoritative_metadata.get("records_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ReferencePoseError("Authoritative target metadata lacks records_sha256")
    output: list[dict[str, Any]] = []
    available, unavailable = 0, 0
    for stem in stems:
        scorer_source, domain = _manifest_domain(stem)
        if domain == "danbooru":
            output.append({
                "schema_version": EXACT_MANIFEST_REFERENCE_FORMAT_VERSION, "stem": stem,
                "source": scorer_source, "source_domain": domain, "status": "unavailable",
                "pose_scoring_available": False, "target_provenance": "unavailable",
                "reason": "authoritative_numerical_pose_target_unavailable", "people": None,
            })
            unavailable += 1
            continue
        source = supplied_records.get(stem)
        if source is None:
            raise ReferencePoseError(f"{stem}: eligible manifest stem is missing from authoritative pose targets")
        if source.get("stem") != stem or source.get("pose_reward_available") is not True:
            raise ReferencePoseError(f"{stem}: authoritative pose target is mismatched or unavailable")
        authoritative_domain = "coco" if scorer_source == "coco" else domain
        if source.get("source") != authoritative_domain:
            raise ReferencePoseError(f"{stem}: authoritative pose target source does not match manifest domain")
        source_size = source.get("source_size")
        if not isinstance(source_size, list) or len(source_size) != 2 or any(not isinstance(value, int) or value < 1 for value in source_size):
            raise ReferencePoseError(f"{stem}: authoritative pose target lacks valid source dimensions")
        people = _source_people_for_exact_manifest(source, stem=stem)
        # Preserve immutable numerical provenance without carrying any prior
        # bucket/crop coordinates that could leak a training resolution.
        provenance = {
            "authoritative_records_sha256": expected_sha,
            "authoritative_records_file": authoritative_metadata.get("records_file"),
            "source_record_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "target_provenance": source.get("target_provenance"),
            "annotation_source": source.get("annotation_source"),
            "source_image_id": source.get("source_image_id"),
            "source_image_name": source.get("source_image_name"),
            "source_annotation_split": source.get("source_annotation_split"),
            "provenance_metadata": source.get("provenance_metadata"),
            "renderer": source.get("renderer"),
        }
        output.append({
            "schema_version": EXACT_MANIFEST_REFERENCE_FORMAT_VERSION, "stem": stem,
            "source": scorer_source, "source_domain": domain, "status": "available",
            "pose_scoring_available": True, "target_provenance": "original_annotation",
            "source_size": list(source_size), "joint_schema": source.get("joint_schema"),
            "mode": "crowd" if source.get("sample_type") == "crowd" else "single",
            "people": people, "provenance": provenance,
        })
        available += 1
    identities = tuple(compatible_experiments)
    if not identities or len(set(identities)) != len(identities):
        raise ReferencePoseError("Compatible experiment identities must be a non-empty unique list")
    metadata = {
        "format_version": EXACT_MANIFEST_REFERENCE_FORMAT_VERSION,
        "sidecar_kind": EXACT_MANIFEST_REFERENCE_KIND, "read_only": True,
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "manifest_stems": list(stems), "compatible_experiments": list(identities),
        "authoritative_source": {
            "path": authoritative_metadata.get("source_path"),
            "records_file": authoritative_metadata.get("records_file"),
            "records_sha256": expected_sha,
            "schema_version": authoritative_metadata.get("schema_version"),
        },
        "coverage": {"total": len(stems), "eligible_available": available, "explicitly_unavailable": unavailable},
    }
    return output, metadata


def write_exact_manifest_reference_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path,
                                         *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish a non-overwritable sidecar in exact manifest order."""
    path = Path(output); metadata_path = _capacity_metadata_path(path)
    if path.exists() or metadata_path.exists():
        raise ReferencePoseError(f"Refusing to overwrite immutable capacity reference sidecar: {path}")
    ordered = [dict(record) for record in records]
    stems = [record.get("stem") for record in ordered]
    if len(stems) != len(set(stems)) or any(not isinstance(stem, str) for stem in stems):
        raise ReferencePoseError("Exact-manifest reference output has duplicate or invalid stems")
    if list(metadata.get("manifest_stems", ())) != stems:
        raise ReferencePoseError("Exact-manifest reference output does not preserve manifest stem order")
    content = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in ordered)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return _write_capacity_reference_files(path, metadata_path, content, {
        "records_file": path.name, "records_sha256": digest, "record_count": len(ordered), **dict(metadata),
    })


def _write_capacity_reference_files(path: Path, metadata_path: Path, content: str,
                                    published: Mapping[str, Any]) -> dict[str, Any]:
    """Write both immutable files together after all caller validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        metadata_temporary = temporary_path.with_name(temporary_path.name + ".metadata")
        metadata_temporary.write_text(json.dumps(dict(published), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if path.exists() or metadata_path.exists():
            raise ReferencePoseError(f"Refusing to overwrite immutable capacity reference sidecar: {path}")
        os.replace(temporary_path, path); os.replace(metadata_temporary, metadata_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        metadata_temporary = locals().get("metadata_temporary")
        if isinstance(metadata_temporary, Path) and metadata_temporary.exists():
            metadata_temporary.unlink()
    return dict(published)


def write_capacity_reference_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically create a non-overwritable exact-manifest reference sidecar."""
    path = Path(output)
    metadata_path = _capacity_metadata_path(path)
    if path.exists() or metadata_path.exists():
        raise ReferencePoseError(f"Refusing to overwrite immutable capacity reference sidecar: {path}")
    ordered = sorted((dict(record) for record in records), key=lambda record: str(record.get("stem")))
    stems = [record.get("stem") for record in ordered]
    if len(stems) != len(set(stems)) or any(not isinstance(stem, str) for stem in stems):
        raise ReferencePoseError("Capacity reference output has duplicate or invalid stems")
    content = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in ordered)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    published = {
        "format_version": CAPACITY_REFERENCE_FORMAT_VERSION, "read_only": True,
        "records_file": path.name, "records_sha256": digest, "record_count": len(ordered), **dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        metadata_temporary = temporary_path.with_name(temporary_path.name + ".metadata")
        metadata_temporary.write_text(json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if path.exists() or metadata_path.exists():
            raise ReferencePoseError(f"Refusing to overwrite immutable capacity reference sidecar: {path}")
        os.replace(temporary_path, path); os.replace(metadata_temporary, metadata_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        metadata_temporary = locals().get("metadata_temporary")
        if isinstance(metadata_temporary, Path) and metadata_temporary.exists():
            metadata_temporary.unlink()
    return published


def build_exact_coco_capacity_reference_sidecar(*, experiment_name: str, latent_root: str | Path,
                                                annotation_paths: Iterable[str | Path], output: str | Path,
                                                manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Build one exact COCO capacity sidecar from official labels and persisted geometry."""
    selected_manifest = Path(manifest_path) if manifest_path is not None else None
    stems, samples, shards = resolve_exact_capacity_latent_samples(
        experiment_name=experiment_name, latent_root=latent_root, manifest_path=selected_manifest,
    )
    annotations = tuple(Path(path) for path in annotation_paths)
    if not annotations:
        raise ReferencePoseError("At least one official COCO person_keypoints annotation JSON is required")
    permitted_names = {"person_keypoints_train2017.json", "person_keypoints_val2017.json"}
    if any(path.name not in permitted_names for path in annotations):
        raise ReferencePoseError("Capacity references require official COCO person_keypoints_train2017.json and/or person_keypoints_val2017.json annotations")
    records = build_coco_reference_records(samples, annotations)
    by_stem = {record["stem"]: record for record in records}
    if len(records) != len(stems) or set(by_stem) != set(stems):
        raise ReferencePoseError("Exact capacity reference output has missing or unexpected stems")
    for stem, sample in zip(stems, samples):
        record = by_stem[stem]
        for field in ("source_size", "resized_size", "crop_box", "bucket"):
            if record[field] != sample[field]:
                raise ReferencePoseError(f"{stem}: authoritative record did not preserve persisted {field}")
        record["experiment"] = experiment_name
        record["manifest_stem_order"] = list(stems)
    source_manifest = selected_manifest
    if source_manifest is None:
        from pose_controlnet.overfit_capacity import experiment
        source_manifest = experiment(experiment_name).manifest
    manifest_bytes = source_manifest.read_bytes()
    annotation_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in annotations}
    return write_capacity_reference_jsonl(records, output, metadata={
        "experiment": experiment_name, "source": "coco", "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "stems": list(stems),
        "latent_root": str(Path(latent_root).resolve()), "latent_shards": shards,
        "official_annotation_paths": [str(path.resolve()) for path in annotations], "official_annotation_sha256": annotation_hashes,
        "output_record_count": len(records), "people_count": sum(len(record["people"]) for record in records),
    })


def load_exact_capacity_reference_sidecar(path: str | Path, *, experiment_name: str,
                                          expected_stems: Iterable[str],
                                          geometry_by_stem: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load an exact immutable capacity sidecar and reject partial coverage."""
    sidecar = Path(path)
    metadata_path = _capacity_metadata_path(sidecar)
    if not sidecar.is_file() or not metadata_path.is_file():
        raise ReferencePoseError(f"Explicit immutable capacity reference sidecar and metadata are required: {sidecar}")
    try:
        raw = sidecar.read_bytes(); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferencePoseError(f"Unreadable capacity reference sidecar: {sidecar}") from exc
    expected = tuple(expected_stems)
    if len(expected) != len(set(expected)):
        raise ReferencePoseError("Expected capacity manifest stems are not unique")
    if metadata.get("read_only") is not True:
        raise ReferencePoseError("Capacity reference sidecar metadata has an unsupported schema")
    if metadata.get("records_sha256") != hashlib.sha256(raw).hexdigest() or metadata.get("record_count") != len(records):
        raise ReferencePoseError("Capacity reference sidecar integrity check failed")
    stems = [record.get("stem") for record in records]
    if len(stems) != len(set(stems)) or set(stems) != set(expected) or len(records) != len(expected):
        raise ReferencePoseError("Capacity reference sidecar does not exactly cover the requested manifest stems")
    generic = metadata.get("sidecar_kind") == EXACT_MANIFEST_REFERENCE_KIND
    if generic:
        if metadata.get("format_version") != EXACT_MANIFEST_REFERENCE_FORMAT_VERSION:
            raise ReferencePoseError("Exact-manifest reference sidecar metadata has an unsupported schema")
        if experiment_name not in metadata.get("compatible_experiments", []):
            raise ReferencePoseError("Exact-manifest reference sidecar is not declared compatible with the requested experiment")
        if metadata.get("manifest_stems") != list(expected) or stems != list(expected):
            raise ReferencePoseError("Exact-manifest reference sidecar manifest provenance is not the exact requested stem order")
        source = metadata.get("authoritative_source")
        if not isinstance(source, Mapping) or not isinstance(source.get("records_sha256"), str):
            raise ReferencePoseError("Exact-manifest reference sidecar lacks authoritative source SHA provenance")
        for record in records:
            stem = record["stem"]
            expected_source, domain = _manifest_domain(stem)
            if (record.get("schema_version") != EXACT_MANIFEST_REFERENCE_FORMAT_VERSION
                    or record.get("source") != expected_source or record.get("source_domain") != domain):
                raise ReferencePoseError(f"{stem}: exact-manifest reference record provenance is inconsistent")
            available = record.get("status") == "available"
            if available != (record.get("pose_scoring_available") is True):
                raise ReferencePoseError(f"{stem}: exact-manifest reference availability is inconsistent")
            if domain == "danbooru":
                if available or record.get("target_provenance") != "unavailable" or record.get("people") is not None:
                    raise ReferencePoseError(f"{stem}: Danbooru must remain explicitly unavailable for numerical pose scoring")
                continue
            if not available or record.get("target_provenance") != "original_annotation" or not isinstance(record.get("people"), list):
                raise ReferencePoseError(f"{stem}: eligible exact-manifest reference record is unavailable or malformed")
            if geometry_by_stem is not None:
                actual = geometry_by_stem.get(stem)
                if actual is None or list(actual.get("source_size", ())) != record.get("source_size"):
                    raise ReferencePoseError(f"{stem}: source dimensions cannot be reconciled with persisted native generation metadata")
        return metadata, records

    if metadata.get("format_version") != CAPACITY_REFERENCE_FORMAT_VERSION:
        raise ReferencePoseError("Capacity reference sidecar metadata has an unsupported schema")
    if metadata.get("experiment") != experiment_name or metadata.get("source") != "coco":
        raise ReferencePoseError("Capacity reference sidecar provenance is inconsistent with the requested COCO experiment")
    if metadata.get("stems") != list(expected):
        raise ReferencePoseError("Capacity reference sidecar manifest provenance is not the exact requested stem order")
    for record in records:
        stem = record["stem"]
        if record.get("schema_version") != CAPACITY_REFERENCE_FORMAT_VERSION or record.get("experiment") != experiment_name or record.get("source") != "coco" or record.get("status") != "available":
            raise ReferencePoseError(f"{stem}: capacity reference record provenance is inconsistent")
        if not isinstance(record.get("people"), list):
            raise ReferencePoseError(f"{stem}: capacity reference record has no official COCO people")
        if geometry_by_stem is not None:
            actual = geometry_by_stem.get(stem)
            if actual is None:
                raise ReferencePoseError(f"{stem}: persisted generation geometry is unavailable")
            for field in ("source_size", "resized_size", "crop_box", "bucket"):
                if list(actual.get(field, ())) != record.get(field):
                    raise ReferencePoseError(f"{stem}: reference geometry cannot be reconciled with persisted generation metadata")
    return metadata, records
