"""Frozen five-pose Krea-2 Turbo control-scale sweep for candidate mix-025.

This evaluation-only runner is deliberately separate from both canonical
``inference.py`` and historical benchmarks.  It keeps the final-val Turbo
runtime and scoring mechanics, but permits a local 0.0 control multiplier for
the explicitly scoped internal zero-control reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from einops import rearrange
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.diffusion import forward_pose_control, patchify_and_position
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, save_image
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, turbo_metadata, turbo_schedule, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val
from scripts import prompting_guide_study as guide


SWEEP_SPEC = Path("docs/evaluation/control-scale-sweep/mix-025-native-v1.json")
# Updated together with the immutable JSON spec.  A mismatch is a hard error.
SWEEP_SPEC_SHA256 = "af82b4c0b0290a852cdd3d7b6b917603e3e9298aa51dfa2ace737d41a28d91a6"
OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/control-scale-sweep/mix-025-native-v1")
SCALES = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
POSE_CLASSES = ("simple_single", "dynamic_airborne", "inversion", "seated_crouched", "multi_person")
STEMS = (
    "sculpture_humanart_14000000003803",
    "coco_49731_461706",
    "real_human_humanart_15000000000521",
    "real_human_humanart_15000000000477",
    "real_human_humanart_17000000002207",
)
RUNTIME = turbo_metadata()
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
GENERATION_COUNT = len(SCALES) * len(STEMS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required control-scale sweep JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Control-scale sweep JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Control-scale sweep JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _scale_label(scale: float) -> str:
    if scale not in SCALES:
        raise ValueError(f"Control scale must be exactly one of {SCALES}, got {scale}")
    return f"{scale:.2f}".replace(".", "p")


def _validate_locks(spec: Mapping[str, Any]) -> None:
    conditions = spec.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(STEMS):
        raise ValueError("Frozen sweep spec must contain exactly five conditions")
    if tuple(row.get("stem") for row in conditions) != STEMS or tuple(row.get("pose_class") for row in conditions) != POSE_CLASSES:
        raise ValueError("Frozen sweep spec pose ordering or representative classes drifted")
    required = {"stem", "pose_class", "prompt", "sampling_seed", "control_sha256", "bucket"}
    if any(set(row) != required or not isinstance(row["prompt"], str) or not row["prompt"].strip()
           or not isinstance(row["sampling_seed"], int) or not isinstance(row["control_sha256"], str)
           or len(row["control_sha256"]) != 64 or not isinstance(row["bucket"], list) or len(row["bucket"]) != 2
           for row in conditions):
        raise ValueError("Frozen sweep conditions are malformed")
    if tuple(spec.get("control_scales", ())) != SCALES or spec.get("generation_count") != GENERATION_COUNT:
        raise ValueError("Frozen sweep must be exactly five poses x seven ordered scales")
    if spec.get("candidate") != "mix-025" or spec.get("candidate_kind") != "trainable_tensor_interpolation":
        raise ValueError("Frozen sweep is locked to candidate mix-025")
    if spec.get("geometry") != NATIVE_GEOMETRY:
        raise ValueError("Frozen sweep is locked to native/aspect-preserving geometry")
    if spec.get("runtime") != RUNTIME:
        raise ValueError("Frozen sweep violates the locked Krea-2 Turbo 8-step CFG-0 mu=1.15 runtime")
    if spec.get("turbo_checkpoint") != "/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors":
        raise ValueError("Frozen sweep Turbo checkpoint provenance drifted")
    if spec.get("clip_model_id") != "openai/clip-vit-base-patch32":
        raise ValueError("Frozen sweep CLIP metric provenance drifted")
    interpolation = spec.get("checkpoint_interpolation")
    if not isinstance(interpolation, dict) or interpolation.get("candidate_id") != "mix-025" or interpolation.get("alpha") != .25:
        raise ValueError("Frozen sweep mix-025 interpolation provenance is malformed")
    if spec.get("final_val_spec_sha256") != final_val.FINAL_SPEC_SHA256:
        raise ValueError("Frozen sweep final-val provenance drifted")


def load_sweep_spec(path: str | Path = SWEEP_SPEC) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != SWEEP_SPEC_SHA256:
        raise ValueError(f"Frozen control-scale sweep SHA-256 mismatch: {source}")
    spec = _read_json(source)
    _validate_locks(spec)
    return spec


def _expected_pairs(spec: Mapping[str, Any]) -> list[tuple[str, float]]:
    return [(row["stem"], scale) for scale in SCALES for row in spec["conditions"]]


def _candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    return guide._candidate_contract(candidate, checkpoint)


def _inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], PreparedLatentShardDataset, dict[str, Path], dict[str, Any], Path | None, dict[str, Any], Path]:
    spec = load_sweep_spec(args.spec)
    if args.turbo_ckpt != spec["turbo_checkpoint"]:
        raise ValueError("Control-scale sweep refuses a Turbo checkpoint different from its frozen provenance")
    if args.clip_model_id != spec["clip_model_id"]:
        raise ValueError("Control-scale sweep refuses a CLIP metric different from its frozen provenance")
    prompt_rows = guide.load_study_rows(spec["prompt_source"]["path"])
    inherited = {row["stem"]: row["prompt"] for row in prompt_rows if row["mode"] == spec["prompt_source"]["mode"]}
    if any(inherited.get(row["stem"]) != row["prompt"] for row in spec["conditions"]):
        raise ValueError("Frozen sweep prompt mapping drifted from its pinned prompting-study source")
    final_spec, final_digest = final_val.load_final_spec(args.final_spec)
    if final_digest != spec["final_val_spec_sha256"]:
        raise ValueError("Current final-val spec does not match the frozen control-scale sweep")
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, final_spec)
    metadata = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or metadata.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Control-scale sweep requires --dataset-root or latent shards.json.dataset_root")
    controls = final_val.resolve_final_controls(dataset_root, list(STEMS))
    if tuple(controls) != STEMS:
        raise ValueError("Control-scale sweep controls do not resolve in the frozen stem order")
    for condition in spec["conditions"]:
        stem, sample = condition["stem"], _sample_by_stem(dataset, condition["stem"])
        bucket = [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8]
        if bucket != condition["bucket"] or final_spec["per_stem_seeds"][stem]["sampling"] != condition["sampling_seed"]:
            raise ValueError(f"Frozen native geometry or final-val seed drifted for {stem}")
        if _sha256(controls[stem]) != condition["control_sha256"]:
            raise ValueError(f"Frozen pose-control SHA-256 drifted for {stem}")
    candidate, checkpoint, training = final_val.resolve_candidate(args.candidate)
    if candidate.get("label") != "mix-025":
        raise ValueError("Control-scale sweep is locked to candidate mix-025")
    endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
    final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    candidate_contract = _candidate_contract(candidate, checkpoint)
    if candidate_contract.get("checkpoint_interpolation") != spec["checkpoint_interpolation"]:
        raise ValueError("Resolved mix-025 candidate provenance drifted from frozen sweep spec")
    contract = {
        "kind": spec["kind"], "frozen_spec": str(Path(args.spec)), "frozen_spec_sha256": SWEEP_SPEC_SHA256,
        "candidate": spec["candidate"], "candidate_kind": spec["candidate_kind"],
        "checkpoint_interpolation": spec["checkpoint_interpolation"], "runtime": spec["runtime"],
        "geometry": spec["geometry"], "control_scales": list(SCALES), "generation_count": GENERATION_COUNT,
        "conditions": spec["conditions"], "final_val_spec_sha256": final_digest,
        "prompt_source": spec["prompt_source"], "turbo_checkpoint": spec["turbo_checkpoint"],
        "clip_model_id": spec["clip_model_id"], "zero_control_note": spec["zero_control_note"],
    }
    output = Path(args.output_root)
    provenance = output / "control_scale_provenance.json"
    if provenance.exists() and _read_json(provenance) != contract:
        raise ValueError(f"Existing output has conflicting immutable control-scale provenance: {provenance}")
    if not provenance.exists():
        _write(provenance, contract)
    return spec, contract, dataset, controls, candidate, checkpoint, training, output


def _directory(output: Path, stem: str, scale: float) -> Path:
    return output / "generations" / stem / f"control_scale_{_scale_label(scale)}"


def _image_path(output: Path, stem: str, scale: float, candidate: Mapping[str, Any]) -> Path:
    return _directory(output, stem, scale) / final_val._image_name(candidate)


def _copy_control(source: Path, target: Path) -> None:
    if target.exists() and _sha256(target) != _sha256(source):
        raise ValueError(f"Existing pose control conflicts with authoritative final-val control: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _generation_metadata(condition: Mapping[str, Any], scale: float, contract: Mapping[str, Any], control: Path) -> dict[str, Any]:
    return {
        "stem": condition["stem"], "pose_class": condition["pose_class"], "prompt": condition["prompt"],
        "seed": condition["sampling_seed"], "control_scale": scale, "control_path": str(control),
        "control_sha256": condition["control_sha256"], "bucket": condition["bucket"],
        "geometry": NATIVE_GEOMETRY, "frozen_spec_sha256": SWEEP_SPEC_SHA256,
        "final_val_spec_sha256": contract["final_val_spec_sha256"], "candidate": "mix-025",
        "runtime": contract["runtime"], "checkpoint_interpolation": contract["checkpoint_interpolation"],
    }


def _generation_status(output: Path, spec: Mapping[str, Any], candidate: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    payload_path = output / "generation_results.json"
    payload = _read_json(payload_path) if payload_path.exists() else None
    observed, recorded = [], []
    for stem, scale in _expected_pairs(spec):
        condition = next(row for row in spec["conditions"] if row["stem"] == stem)
        directory, image = _directory(output, stem, scale), _image_path(output, stem, scale, candidate)
        control, metadata_path = output / "controls" / f"{stem}.png", directory / "metadata.json"
        if image.is_file():
            try:
                with Image.open(image) as opened: opened.verify()
                with Image.open(control) as opened: opened.verify()
                metadata = _read_json(metadata_path); expected = _generation_metadata(condition, scale, contract, control)
                if any(metadata.get(key) != value for key, value in expected.items() if key != "control_path") or not isinstance(metadata.get("control_path"), str):
                    raise ValueError("generation metadata contract mismatch")
                if _sha256(control) != condition["control_sha256"]:
                    raise ValueError("generation control hash mismatch")
            except Exception as exc:
                raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if directory.exists() or control.exists():
                raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
        if payload is not None:
            recorded.append(payload.get("generated_artifacts", {}).get(stem, {}).get(_scale_label(scale)) == str(image.relative_to(output)))
    if not any(observed) and payload is None:
        return "missing"
    expected_artifacts = {stem: {_scale_label(scale): str(_image_path(output, stem, scale, candidate).relative_to(output)) for scale in SCALES} for stem in STEMS}
    if (all(observed) and payload is not None and all(recorded) and all(payload.get(key) == value for key, value in contract.items())
            and payload.get("generated_artifacts") == expected_artifacts):
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


@torch.inference_mode()
def _sample_at_control_scale(model: Any, vae_decode, sample: Mapping[str, Any], device: torch.device, seed: int, scale: float):
    """Locked Turbo sampler with the sweep's sole extension: an allowed zero scale."""
    if scale not in SCALES:
        raise ValueError("Control-scale sweep received an unpinned scale")
    latent = sample["latent"][None].to(device)
    control_latent = sample["control"][None].to(device=device, dtype=torch.bfloat16)
    if not torch.isfinite(control_latent).all() or control_latent.abs().max().item() == 0.0:
        raise ValueError("Turbo sampling requires finite, non-empty source pose control latents")
    noise = torch.randn(latent.shape, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(seed))
    text, text_mask = sample["context"][None].to(device=device, dtype=torch.bfloat16), sample["mask"][None].to(device=device, dtype=torch.bool)
    patch = model.config.patch
    image, pos, mask = patchify_and_position(noise, text.shape[1], patch, text_mask)
    # Do not call the generic positive-only helper: scale 0.0 is intentional,
    # local to this frozen evaluation, and produces the internal reference.
    control, _, _ = patchify_and_position(control_latent * scale, text.shape[1], patch, text_mask)
    schedule = turbo_schedule(image_sequence_length=image.shape[1], steps=8, mu=1.15)
    for current, previous in zip(schedule[:-1], schedule[1:]):
        timestep = torch.full((1,), current, dtype=image.dtype, device=device)
        velocity = forward_pose_control(model, image, control, text, timestep, pos, mask, gradient_checkpointing_blocks=0)
        image = image + (previous - current) * velocity
    height, width = latent.shape[-2:]
    decoded = rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=height // patch, w=width // patch)
    pixels = vae_decode(decoded.to(torch.bfloat16))
    return ((pixels.clamp(-1, 1) * .5 + .5) * 255.0)[0].permute(1, 2, 0).float().cpu().byte().numpy()


def preflight(args: argparse.Namespace) -> None:
    spec, contract, dataset, controls, _, _, training, output = _inputs(args)
    _write(output / "checkpoint_preflight.json", {**contract, "dataset_sample_count": len(dataset),
           "control_paths_resolved": {stem: str(path) for stem, path in controls.items()}, "training_metadata": training,
           "source_rgb_fallback_permitted": False, "generation_plan": _expected_pairs(spec)})
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run control-scale generation from the GH200 host shell with CUDA visible")
    spec, contract, dataset, controls, candidate, checkpoint, training, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract) == "complete":
        print(json.dumps({"already_complete": "mix-025", "generation_count": GENERATION_COUNT})); return
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint)
    final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    vae, conditioner = load_krea_vae("cuda"), guide.PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    for condition in spec["conditions"]:
        stem, sample = condition["stem"], dict(_sample_by_stem(dataset, condition["stem"]))
        bucket = [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8]
        if bucket != condition["bucket"]:
            raise ValueError(f"Frozen native geometry changed for {stem}")
        context, mask = guide._conditioning(conditioner, condition["prompt"], sample)
        sample.update({"prompt": condition["prompt"], "context": context, "mask": mask})
        _copy_control(controls[stem], output / "controls" / f"{stem}.png")
        for scale in SCALES:
            directory = _directory(output, stem, scale); metadata = _generation_metadata(condition, scale, contract, controls[stem])
            metadata_path = directory / "metadata.json"
            if metadata_path.exists() and _read_json(metadata_path) != metadata:
                raise ValueError(f"Existing generation metadata conflicts with frozen sweep: {metadata_path}")
            if not metadata_path.exists(): _write(metadata_path, metadata)
            pixels = _sample_at_control_scale(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                               torch.device("cuda"), condition["sampling_seed"], scale)
            save_image(pixels, _image_path(output, stem, scale, candidate))
    artifacts = {stem: {_scale_label(scale): str(_image_path(output, stem, scale, candidate).relative_to(output)) for scale in SCALES} for stem in STEMS}
    _write(output / "generation_results.json", {**contract, "generated_artifacts": artifacts,
           "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None),
           "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training, "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _sidecar(path: str | Path, stems: list[str]) -> tuple[dict[str, Any], str]:
    final_spec, _ = final_val.load_final_spec(final_val.FINAL_SPEC)
    sidecar, digest = final_val._load_final_sidecar(path, list(final_spec["stems"]))
    records = {str(record["stem"]): record for record in sidecar["records"]}
    if set(stems) - set(records):
        raise ValueError("Authoritative final-val sidecar is missing a frozen control-scale sweep stem")
    return {"records": [records[stem] for stem in stems]}, digest


def _score_records(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    records, expected = payload.get("per_generation"), _expected_pairs(spec)
    if (not isinstance(records, list) or len(records) != GENERATION_COUNT
            or [(record.get("stem"), record.get("control_scale")) for record in records] != expected):
        raise ValueError("Control-scale score artifact is incomplete or not in frozen pose/scale order")
    return records


def _compact_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return guide._compact_group(records)


def _aggregate_tables(records: list[dict[str, Any]], scale_results: list[dict[str, Any]], spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if tuple(item.get("control_scale") for item in scale_results) != SCALES:
        raise ValueError("Control-scale aggregate results are not in exact frozen scale order")
    by_scale: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scale[record["control_scale"]].append(record); by_class[record["pose_class"]].append(record)
    if tuple(by_scale) != SCALES or any(len(by_scale[scale]) != len(STEMS) for scale in SCALES):
        raise ValueError("Scored records do not form the exact five-pose x seven-scale matrix")
    summary_by_scale = []
    for result in scale_results:
        pose, clip = result["pose"], _compact_group(by_scale[result["control_scale"]])["clip"]
        summary_by_scale.append({"control_scale": result["control_scale"], "generation_count": len(by_scale[result["control_scale"]]),
                                 "pck": pose, "clip": clip,
                                 "matched_people": pose["matched_people"], "detection_coverage": pose["detection_coverage"],
                                 "generated_person_count": pose["predicted_people"], "unmatched_predicted_people": pose["unmatched_predicted_people"]})
    rows_by_class = []
    for condition in spec["conditions"]:
        pose_class = condition["pose_class"]
        values = sorted(by_class[pose_class], key=lambda item: item["control_scale"])
        if len(values) != len(SCALES) or tuple(item["control_scale"] for item in values) != SCALES:
            raise ValueError(f"Pose class {pose_class} lacks the exact seven-scale sweep")
        rows_by_class.append({"pose_class": pose_class, "stem": condition["stem"], "by_control_scale": [
            {"control_scale": value["control_scale"], **_compact_group([value]),
             "matched_people": value["pck"].get("matched_people"), "detection_coverage": value["pck"].get("detection_coverage"),
             "generated_person_count": value["pck"].get("predicted_people")} for value in values]})
    return ({"group_by": "control_scale", "rows": summary_by_scale}, {"group_by": "pose_class_then_control_scale", "rows": rows_by_class})


def score(args: argparse.Namespace) -> None:
    spec, contract, dataset, _, candidate, _, _, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract) != "complete":
        raise FileNotFoundError("Control-scale scoring requires all 35 validated generations")
    if not args.reference_sidecar:
        raise ValueError("Control-scale PCK requires the authoritative final-val --reference-sidecar")
    sidecar, sidecar_digest = _sidecar(args.reference_sidecar, list(STEMS))
    existing = output / "pck_clip_results.json"
    if existing.exists():
        previous = _read_json(existing)
        if (any(previous.get(key) != value for key, value in contract.items()) or previous.get("reference_sidecar_sha256") != sidecar_digest
                or previous.get("clip_model") != args.clip_model_id or previous.get("confidence_threshold") != .5):
            raise ValueError("Existing control-scale score artifact conflicts with frozen provenance")
    geometry = {stem: turbo_scoring_geometry(_sample_by_stem(dataset, stem)) for stem in STEMS}
    sidecar_by_stem = {record["stem"]: record for record in sidecar["records"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    per_generation, scale_results = [], []
    for scale in SCALES:
        image_for = lambda stem, current=scale: _image_path(output, stem, current, candidate)
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for, detector=detector, confidence_threshold=.5, require_images=True)
        per_image = {row["stem"]: row for row in pose["per_image"]}
        for condition in spec["conditions"]:
            stem, image = condition["stem"], image_for(condition["stem"])
            pck = per_image.get(stem, {"stem": stem, "reference_available": False, "reason": "authoritative_reference_pose_unavailable"})
            per_generation.append({"stem": stem, "pose_class": condition["pose_class"], "prompt": condition["prompt"], "seed": condition["sampling_seed"],
                                   "control_scale": scale, "image": str(image.relative_to(output)), "pck": pck,
                                   "clip_cosine_similarity": _clip_score(clip, processor, device, condition["prompt"], image)})
        scale_results.append({"control_scale": scale, "pose": pose})
    by_scale, by_class = _aggregate_tables(per_generation, scale_results, spec)
    _write(existing, {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "reference_sidecar_sha256": sidecar_digest,
           "clip_model": args.clip_model_id, "confidence_threshold": .5, "per_generation": per_generation,
           "aggregate_by_control_scale": by_scale, "aggregate_by_pose_class": by_class, "scale_results": scale_results})
    print(existing)


def _validated_scores(output: Path, contract: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = _read_json(output / "pck_clip_results.json")
    if any(scores.get(key) != value for key, value in contract.items()):
        raise ValueError("Control-scale score artifact conflicts with frozen provenance")
    return scores, _score_records(scores, spec)


def report(args: argparse.Namespace) -> None:
    spec, contract, _, _, candidate, _, training, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract) != "complete":
        raise FileNotFoundError("Control-scale report requires all 35 validated generations")
    scores, records = _validated_scores(output, contract, spec)
    by_scale, by_class = _aggregate_tables(records, scores.get("scale_results", []), spec)
    if scores.get("aggregate_by_control_scale") != by_scale or scores.get("aggregate_by_pose_class") != by_class:
        raise ValueError("Control-scale aggregate score tables do not match individual scored generations")
    _write(output / "metrics_by_control_scale.json", {**contract, **by_scale})
    _write(output / "metrics_by_pose_class.json", {**contract, **by_class})
    labels = ("pose control", *(f"{scale:.2f}" for scale in SCALES))
    rows = []
    for condition in spec["conditions"]:
        stem = condition["stem"]
        paths = [output / "controls" / f"{stem}.png", *(_image_path(output, stem, scale, candidate) for scale in SCALES)]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"Control-scale grid requires the control and all seven generations for {stem}")
        make_contact_sheet([(stem, paths)], output / "per_pose_scale_grids" / f"{stem}.png", thumbnail_width=220, thumbnail_height=220, column_labels=labels)
        rows.append((stem, paths))
    make_contact_sheet(rows, output / "control_scale_contact_sheet.png", thumbnail_width=180, thumbnail_height=180, column_labels=labels)
    _write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "generation_count": GENERATION_COUNT,
           "score_artifact": "pck_clip_results.json", "metrics_by_control_scale": "metrics_by_control_scale.json",
           "metrics_by_pose_class": "metrics_by_pose_class.json", "contact_sheet": "control_scale_contact_sheet.png",
           "per_pose_scale_grids": "per_pose_scale_grids", "source_rgb_fallback_used": False})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    spec, contract, _, _, candidate, _, _, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract) != "complete":
        raise FileNotFoundError("Control-scale summary requires all 35 validated generations")
    scores, records = _validated_scores(output, contract, spec)
    by_scale, _ = _aggregate_tables(records, scores.get("scale_results", []), spec)
    compact = {"candidate": candidate["label"], "generation_count": GENERATION_COUNT, "control_scales": list(SCALES),
               "by_control_scale": [{"control_scale": row["control_scale"], "pck_005": row["pck"]["pck_005"],
                                     "pck_010": row["pck"]["pck_010"], "pck_020": row["pck"]["pck_020"],
                                     "clip_mean": row["clip"]["mean_cosine_similarity"], "matched_people": row["matched_people"],
                                     "detection_coverage": row["detection_coverage"], "generated_person_count": row["generated_person_count"]}
                                    for row in by_scale["rows"]], "frozen_spec_sha256": SWEEP_SPEC_SHA256}
    _write(output / "compact_summary.json", compact)
    print(json.dumps(compact, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "generate", "score", "report", "summary"))
    parser.add_argument("--candidate", choices=("mix-025",), default="mix-025")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--spec", default=str(SWEEP_SPEC))
    parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC))
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    parser.add_argument("--dataset-root")
    parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
    parser.add_argument("--reference-sidecar", help="required only for score")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return parser


def main() -> None:
    args = parser().parse_args()
    {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__":
    main()
