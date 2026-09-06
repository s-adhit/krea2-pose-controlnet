"""Frozen 12-condition hard-pose and multi-person stress benchmark for mix-025.

This is a single-candidate stress suite.  It deliberately does not compare
checkpoints, mutate canonical inference, or use an RGB/reference fallback.
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
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, save_image
from pose_controlnet.post1500_evaluation import _pool_pose
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_metadata, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val


SPEC = Path("docs/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1.json")
SPEC_SHA256 = "ed2fe5021aadce7bb32a6c2cdbd13573f9631054377715498ca6c486b04572a7"
OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1")
COUNT = 12
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
RUNTIME = {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0, "mu": 1.15, "mu_resolution_dependent": False}
SINGLE_CATEGORIES = ("seated", "crouched", "lying_floor", "overhead_arms", "strong_torso_rotation", "strong_foreshortening", "inversion", "airborne")
MULTI_CATEGORIES = ("two_person_interaction", "overlapping_two_person_pose", "three_to_four_person_interaction", "crowd_dense_group")
LOW_STRICT_PCK = .30
LOW_COARSE_PCK = .50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required hard-pose benchmark JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Hard-pose benchmark JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Hard-pose benchmark JSON must be an object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_locks(spec: Mapping[str, Any]) -> None:
    if (spec.get("candidate"), spec.get("control_scale"), spec.get("geometry"), spec.get("runtime"), spec.get("generation_count")) != ("mix-025", 1.0, NATIVE_GEOMETRY, RUNTIME, COUNT):
        raise ValueError("Hard-pose benchmark violates mix-025, control-scale-1.0, native-geometry, or Turbo runtime lock")
    source = spec.get("frozen_sources")
    if not isinstance(source, dict) or source.get("final_spec") != {"path": str(final_val.FINAL_SPEC), "sha256": final_val.FINAL_SPEC_SHA256}:
        raise ValueError("Hard-pose benchmark final-val frozen-spec provenance drifted")
    if source.get("prompts") != {"path": "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl", "sha256": "23d448d573a2ffd20adfd73fa88f34ebc08df280a051cb0931d9ecdcc1231ceb"}:
        raise ValueError("Hard-pose benchmark frozen prompt provenance drifted")
    sidecar = source.get("authoritative_pose_sidecar")
    if sidecar != {"path": "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3", "metadata_sha256": "57aac981a75231e3e0c4a6713bf9758db233c59e2b2cd71f7ea0a97a7ce7676e", "records_sha256": "3cc4defc282cb11e956ec06517eff4e8369622d4c0b3b567ab2247efb4a499a7"}:
        raise ValueError("Hard-pose benchmark requires the canonical authoritative final-val v3 pose sidecar")
    conditions = spec.get("conditions")
    required = {"order", "condition_id", "hard_pose_category", "pose_class", "person_group", "expected_reference_people", "stem", "source_domain", "prompt", "seed", "native_bucket"}
    if not isinstance(conditions, list) or len(conditions) != COUNT or any(set(row) != required for row in conditions if isinstance(row, dict)):
        raise ValueError("Hard-pose benchmark must contain exactly twelve complete frozen conditions")
    if [row["order"] for row in conditions] != list(range(1, COUNT + 1)) or len({row["stem"] for row in conditions}) != COUNT:
        raise ValueError("Hard-pose benchmark condition order or stem uniqueness drifted")
    control_hashes = spec.get("control_sha256")
    if not isinstance(control_hashes, dict) or set(control_hashes) != {row["stem"] for row in conditions} or any(not isinstance(value, str) or len(value) != 64 for value in control_hashes.values()):
        raise ValueError("Hard-pose benchmark must pin every frozen control SHA-256")
    expected = SINGLE_CATEGORIES + MULTI_CATEGORIES
    if tuple(row["condition_id"].split("_", 1)[1] for row in conditions) != expected:
        raise ValueError("Hard-pose benchmark category coverage/order drifted")
    if any(row["person_group"] != ("single-person" if index < 8 else "multi-person") for index, row in enumerate(conditions)):
        raise ValueError("Hard-pose benchmark single/multi-person partition drifted")
    if any(row["hard_pose_category"] != ("single_person_hard_pose" if index < 8 else "multi_person_stress") for index, row in enumerate(conditions)):
        raise ValueError("Hard-pose benchmark hard-pose category partition drifted")
    if any(not isinstance(row["prompt"], str) or not row["prompt"].strip() or not isinstance(row["seed"], int) or not isinstance(row["native_bucket"], list) or len(row["native_bucket"]) != 2 for row in conditions):
        raise ValueError("Hard-pose benchmark has incomplete frozen prompt, seed, or native bucket identity")


def _validate_historical_artifacts(spec: Mapping[str, Any]) -> None:
    for relative, expected in spec["historical_artifact_sha256"].items():
        if _sha256(Path(relative)) != expected:
            raise ValueError(f"Historical artifact changed; hard-pose benchmark refuses to proceed: {relative}")
    source = spec["frozen_sources"]
    expected = {source["final_spec"]["path"]: source["final_spec"]["sha256"], source["prompts"]["path"]: source["prompts"]["sha256"],
                f"{source['authoritative_pose_sidecar']['path']}/metadata.json": source["authoritative_pose_sidecar"]["metadata_sha256"],
                f"{source['authoritative_pose_sidecar']['path']}/records.jsonl": source["authoritative_pose_sidecar"]["records_sha256"]}
    for relative, digest in expected.items():
        if _sha256(Path(relative)) != digest:
            raise ValueError(f"Frozen hard-pose source artifact changed: {relative}")


def load_spec(path: str | Path = SPEC) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != SPEC_SHA256:
        raise ValueError(f"Frozen hard-pose spec SHA-256 mismatch: {source}")
    spec = _read_json(source); _validate_locks(spec); _validate_historical_artifacts(spec)
    return spec


def _prompt_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 48 or len({row.get("stem") for row in rows}) != 48:
        raise ValueError("Frozen final-val prompt source is incomplete")
    return {str(row["stem"]): row for row in rows}


def _authoritative_records(sidecar_path: Path, full_stems: list[str], conditions: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    # The shared loader rejects diagnostic or hand-written sidecars and checks
    # the full final-val frozen population before this suite subsets it.
    sidecar, digest = final_val._load_final_sidecar(sidecar_path, full_stems)
    records = {str(row["stem"]): row for row in sidecar["records"]}
    selected = []
    for condition in conditions:
        record = records.get(condition["stem"])
        if not isinstance(record, dict) or record.get("status") != "available":
            raise ValueError(f"Hard-pose condition lacks authoritative pose: {condition['stem']}")
        if len(record.get("people", [])) != condition["expected_reference_people"]:
            raise ValueError(f"Hard-pose authoritative person-count drifted: {condition['stem']}")
        source = "coco" if condition["source_domain"] == "coco" else "humanart"
        if record.get("source") != source:
            raise ValueError(f"Hard-pose authoritative source-domain drifted: {condition['stem']}")
        selected.append(record)
    return {"records": selected}, digest


def _inputs(args: argparse.Namespace):
    spec = load_spec(args.spec); conditions = list(spec["conditions"])
    final_spec, final_digest = final_val.load_final_spec(args.final_spec)
    prompt_rows = _prompt_rows(Path(spec["frozen_sources"]["prompts"]["path"]))
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, final_spec)
    shards = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shards.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Hard-pose benchmark requires --dataset-root or latent shards.json.dataset_root")
    stems = [row["stem"] for row in conditions]
    controls = final_val.resolve_final_controls(dataset_root, stems)
    geometries, control_hashes = {}, {}
    for condition in conditions:
        stem, sample = condition["stem"], _sample_by_stem(dataset, condition["stem"])
        prompt = prompt_rows.get(stem, {}).get("text")
        if prompt != condition["prompt"] or sample.get("prompt") != condition["prompt"] or final_spec["per_stem_seeds"][stem]["sampling"] != condition["seed"]:
            raise ValueError(f"Hard-pose frozen prompt/seed identity drifted: {stem}")
        geometry = turbo_scoring_geometry(sample)
        if geometry.get("bucket") != condition["native_bucket"]:
            raise ValueError(f"Hard-pose native cached-latent bucket drifted: {stem}")
        observed_control_hash = _sha256(controls[stem])
        if observed_control_hash != spec["control_sha256"][stem]:
            raise ValueError(f"Hard-pose frozen authoritative control SHA-256 drifted: {stem}")
        geometries[stem], control_hashes[stem] = geometry, observed_control_hash
    selected_sidecar, sidecar_digest = _authoritative_records(Path(spec["frozen_sources"]["authoritative_pose_sidecar"]["path"]), list(final_spec["stems"]), conditions)
    candidate, checkpoint, training = final_val.resolve_candidate("mix-025")
    if candidate.get("label") != "mix-025" or candidate.get("kind") != "trainable_tensor_interpolation":
        raise ValueError("Hard-pose benchmark is locked to mix-025 interpolation")
    endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
    final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    contract = {"kind": spec["kind"], "frozen_spec": str(Path(args.spec)), "frozen_spec_sha256": SPEC_SHA256, "candidate": "mix-025", "candidate_kind": candidate["kind"], "checkpoint_interpolation": candidate["interpolation"], "runtime": RUNTIME, "control_scale": 1.0, "geometry": NATIVE_GEOMETRY, "final_val_spec_sha256": final_digest, "authoritative_pose_sidecar": spec["frozen_sources"]["authoritative_pose_sidecar"], "authoritative_pose_records_sha256": sidecar_digest, "conditions": conditions, "control_sha256": control_hashes, "geometry_by_stem": geometries, "historical_artifact_sha256": spec["historical_artifact_sha256"]}
    output, provenance = Path(args.output_root), Path(args.output_root) / "hard_pose_provenance.json"
    if provenance.exists() and _read_json(provenance) != contract:
        raise ValueError(f"Existing output has conflicting immutable hard-pose provenance: {provenance}")
    if not provenance.exists():
        if getattr(args, "action", None) != "preflight":
            raise FileNotFoundError("Hard-pose benchmark requires preflight provenance before later stages")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Hard-pose preflight refuses a non-empty output root without immutable provenance")
        _write(provenance, contract)
    return spec, contract, dataset, controls, geometries, selected_sidecar, candidate, checkpoint, training, output


def _directory(output: Path, condition: Mapping[str, Any]) -> Path:
    return output / "generations" / str(condition["condition_id"])


def _image_path(output: Path, condition: Mapping[str, Any]) -> Path:
    return _directory(output, condition) / "mix-025.png"


def _metadata(condition: Mapping[str, Any], contract: Mapping[str, Any], control: Path) -> dict[str, Any]:
    return {**condition, "candidate": "mix-025", "control_scale": 1.0, "geometry": NATIVE_GEOMETRY, "control_path": str(control), "control_sha256": contract["control_sha256"][condition["stem"]], "frozen_spec_sha256": SPEC_SHA256, "final_val_spec_sha256": contract["final_val_spec_sha256"], "runtime": RUNTIME, "checkpoint_interpolation": contract["checkpoint_interpolation"]}


def _generation_status(output: Path, contract: Mapping[str, Any]) -> str:
    payload_path = output / "generation_results.json"; payload = _read_json(payload_path) if payload_path.exists() else None
    observed = []
    for condition in contract["conditions"]:
        image, directory, control = _image_path(output, condition), _directory(output, condition), output / "controls" / f"{condition['stem']}.png"
        metadata_path = directory / "metadata.json"
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    if list(opened.size) != condition["native_bucket"]: raise ValueError("output dimensions changed")
                    opened.verify()
                if _read_json(metadata_path) != _metadata(condition, contract, control) or _sha256(control) != contract["control_sha256"][condition["stem"]]:
                    raise ValueError("metadata/control identity changed")
            except Exception as exc:
                raise ValueError(f"Hard-pose generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if directory.exists() or control.exists():
                raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
    expected = {row["condition_id"]: str(_image_path(output, row).relative_to(output)) for row in contract["conditions"]}
    if not any(observed) and payload is None:
        if (output / "generations").exists() or (output / "controls").exists():
            raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
        return "missing"
    if all(observed) and payload is not None and all(payload.get(key) == value for key, value in contract.items()) and payload.get("generated_artifacts") == expected:
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


def preflight(args: argparse.Namespace) -> None:
    _, contract, dataset, controls, _, _, _, _, training, output = _inputs(args)
    _write(output / "checkpoint_preflight.json", {**contract, "dataset_sample_count": len(dataset), "control_paths_resolved": {stem: str(path) for stem, path in controls.items()}, "training_metadata": training, "generation_count": COUNT, "source_rgb_fallback_permitted": False})
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("Run hard-pose generation from the GH200 host shell with CUDA visible")
    _, contract, dataset, controls, _, _, candidate, checkpoint, training, output = _inputs(args)
    if _generation_status(output, contract) == "complete":
        print(json.dumps({"already_complete": True, "generation_count": COUNT})); return
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint); final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    vae = load_krea_vae("cuda")
    for condition in contract["conditions"]:
        stem, directory = condition["stem"], _directory(output, condition); source, target = controls[stem], output / "controls" / f"{stem}.png"
        if _sha256(source) != contract["control_sha256"][stem]: raise ValueError(f"Authoritative control hash changed: {stem}")
        if target.exists() and _sha256(target) != contract["control_sha256"][stem]: raise ValueError(f"Existing control conflicts: {target}")
        if not target.exists(): target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
        metadata = _metadata(condition, contract, target); metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata: raise ValueError(f"Existing metadata conflicts: {metadata_path}")
        if not metadata_path.exists(): _write(metadata_path, metadata)
        image = _image_path(output, condition)
        if image.exists(): continue
        sample = dict(_sample_by_stem(dataset, stem))
        pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample, torch.device("cuda"), condition["seed"], steps=8, guidance=0.0, mu=1.15, control_scale=1.0)
        save_image(pixels, image)
    artifacts = {row["condition_id"]: str(_image_path(output, row).relative_to(output)) for row in contract["conditions"]}
    _write(output / "generation_results.json", {**contract, "generated_artifacts": artifacts, "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training, "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _failure_flags(pck: Mapping[str, Any]) -> list[str]:
    if not pck.get("reference_available"): return []
    flags = []
    if int(pck.get("unmatched_reference_people", 0)) > 0: flags.append("missed_reference_person")
    if int(pck.get("unmatched_predicted_people", 0)) > 0: flags.append("extra_predicted_people")
    if pck.get("pck_005") is not None and float(pck["pck_005"]) < LOW_STRICT_PCK: flags.append("low_strict_pck")
    if pck.get("pck_020") is not None and float(pck["pck_020"]) < LOW_COARSE_PCK: flags.append("low_coarse_pck")
    return flags


def _compact_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row["pck"] for row in records if row["pck"].get("reference_available")]
    pose = _pool_pose(available) if available else {"evaluable_sample_count": 0, "pck_005": None, "pck_010": None, "pck_020": None}
    clips = aggregate([float(row["clip_cosine_similarity"]) for row in records])
    return {"generation_count": len(records), "pck": pose, "clip": {"mean_cosine_similarity": clips["mean"], "median_cosine_similarity": clips["median"], "std_cosine_similarity": clips["std"], "sample_count": clips["sample_count"]}, "pck_unavailable_count": len(records) - len(available)}


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records: groups[str(row[key])].append(row)
    return {"group_by": key, "rows": [{key: value, **_compact_group(rows)} for value, rows in groups.items()]}


def _score_records(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    if not isinstance(records, list) or len(records) != COUNT or [row.get("condition_id") for row in records] != [row["condition_id"] for row in contract["conditions"]]:
        raise ValueError("Hard-pose score artifact is incomplete or not in frozen condition order")
    return records


def score(args: argparse.Namespace) -> None:
    _, contract, dataset, _, geometries, sidecar, _, _, _, output = _inputs(args)
    if _generation_status(output, contract) != "complete": raise FileNotFoundError("Hard-pose scoring requires all twelve validated generations")
    if args.reference_sidecar != contract["authoritative_pose_sidecar"]["path"]: raise ValueError("Hard-pose scoring requires the locked canonical authoritative pose sidecar")
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    from pose_controlnet.post1500_evaluation import score_authoritative_pck
    by_stem = {row["stem"]: row for row in sidecar["records"]}; rows = []
    for condition in contract["conditions"]:
        stem, image = condition["stem"], _image_path(output, condition)
        result = score_authoritative_pck(sidecar={"records": [by_stem[stem]]}, geometry_by_stem={stem: geometries[stem]}, image_for=lambda _: image, detector=detector, confidence_threshold=.5, require_images=True)
        per_image = result["per_image"][0] if result["per_image"] else {"stem": stem, "reference_available": False, "reason": result["unavailable"][0]["reason"]}
        rows.append({**condition, "image": str(image.relative_to(output)), "pck": per_image, "clip_cosine_similarity": _clip_score(clip, processor, device, condition["prompt"], image), "observable_failure_flags": _failure_flags(per_image)})
    by_pose, by_group, by_category = _aggregate(rows, "pose_class"), _aggregate(rows, "person_group"), _aggregate(rows, "hard_pose_category")
    _write(output / "pck_clip_results.json", {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "clip_model": args.clip_model_id, "confidence_threshold": .5, "per_generation": rows, "aggregate_by_pose_class": by_pose, "aggregate_by_person_group": by_group, "aggregate_by_hard_pose_category": by_category})
    print(output / "pck_clip_results.json")


def _validated_scores(output: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(output / "pck_clip_results.json")
    if any(payload.get(key) != value for key, value in contract.items()): raise ValueError("Hard-pose score artifact conflicts with frozen provenance")
    return payload, _score_records(payload, contract)


def _failure_taxonomy(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("missed_reference_person", "extra_predicted_people", "low_strict_pck", "low_coarse_pck")
    return {"kind": "measured_observable_failure_taxonomy", "thresholds": {"low_strict_pck_below": LOW_STRICT_PCK, "low_coarse_pck_below": LOW_COARSE_PCK}, "counts": {label: sum(label in row["observable_failure_flags"] for row in records) for label in labels}, "conditions": {label: [row["condition_id"] for row in records if label in row["observable_failure_flags"]] for label in labels}, "note": "Only detector/PCK-derived observable fields are labeled; no subjective failure labels are inferred."}


def report(args: argparse.Namespace) -> None:
    _, contract, _, _, _, _, _, _, _, output = _inputs(args)
    if _generation_status(output, contract) != "complete": raise FileNotFoundError("Hard-pose report requires all twelve validated generations")
    scores, rows = _validated_scores(output, contract)
    by_pose, by_group, by_category = _aggregate(rows, "pose_class"), _aggregate(rows, "person_group"), _aggregate(rows, "hard_pose_category")
    if (scores.get("aggregate_by_pose_class"), scores.get("aggregate_by_person_group"), scores.get("aggregate_by_hard_pose_category")) != (by_pose, by_group, by_category): raise ValueError("Hard-pose aggregate tables do not match per-image results")
    _write(output / "metrics_by_pose_class.json", {**contract, **by_pose})
    _write(output / "metrics_by_person_group.json", {**contract, **by_group, "aggregate_by_hard_pose_category": by_category})
    all_rows = [(row["condition_id"], [output / "controls" / f"{row['stem']}.png", _image_path(output, row)]) for row in contract["conditions"]]
    if any(not path.is_file() for _, paths in all_rows for path in paths): raise FileNotFoundError("Hard-pose report requires every control and generated image")
    labels = ("pose control", "generation")
    make_contact_sheet(all_rows, output / "hard_pose_contact_sheet.png", thumbnail_width=260, thumbnail_height=220, column_labels=labels)
    make_contact_sheet(all_rows[:8], output / "single_person_hard_pose_grid.png", thumbnail_width=280, thumbnail_height=230, column_labels=labels)
    make_contact_sheet(all_rows[8:], output / "multi_person_stress_grid.png", thumbnail_width=280, thumbnail_height=230, column_labels=labels)
    for row, paths in all_rows: make_contact_sheet([(row, paths)], output / "per_condition_grids" / f"{row}.png", thumbnail_width=420, thumbnail_height=360, column_labels=labels)
    _write(output / "evaluation_summary.json", {**contract, "generation_count": COUNT, "score_artifact": "pck_clip_results.json", "metrics": {"by_pose_class": "metrics_by_pose_class.json", "by_person_group": "metrics_by_person_group.json"}, "failure_taxonomy": _failure_taxonomy(rows), "contact_sheets": {"all": "hard_pose_contact_sheet.png", "single_person": "single_person_hard_pose_grid.png", "multi_person": "multi_person_stress_grid.png", "per_condition": "per_condition_grids/*.png"}, "source_rgb_fallback_used": False})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    _, contract, _, _, _, _, _, _, _, output = _inputs(args)
    if _generation_status(output, contract) != "complete": raise FileNotFoundError("Hard-pose summary requires all twelve validated generations")
    _, rows = _validated_scores(output, contract); by_pose, by_group, by_category = _aggregate(rows, "pose_class"), _aggregate(rows, "person_group"), _aggregate(rows, "hard_pose_category")
    compact = {"frozen_spec_sha256": SPEC_SHA256, "candidate": "mix-025", "generation_count": COUNT, "runtime": RUNTIME, "control_scale": 1.0, "geometry": NATIVE_GEOMETRY, "by_pose_class": by_pose["rows"], "by_person_group": by_group["rows"], "by_hard_pose_category": by_category["rows"], "failure_taxonomy": _failure_taxonomy(rows)}
    _write(output / "compact_summary.json", compact); print(output / "compact_summary.json")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "generate", "score", "report", "summary")); parser.add_argument("--output-root", default=str(OUTPUT_ROOT)); parser.add_argument("--spec", default=str(SPEC)); parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC)); parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents"); parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning"); parser.add_argument("--dataset-root"); parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors"); parser.add_argument("--reference-sidecar", default="docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3"); parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return parser


def main() -> None:
    args = parser().parse_args(); {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__": main()
