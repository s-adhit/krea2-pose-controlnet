"""Frozen hands/wrists Turbo benchmark; generation is intentionally GH200-only."""
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
from pose_controlnet.post1500_evaluation import _pool_pose, score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val
from scripts import prompting_guide_study as guide
from scripts import turbo_baseline_benchmark as baseline


SPEC = Path("docs/evaluation/hands/hands-fingers-turbo-v1.json")
SPEC_SHA256 = "d61bacf17c7390b1f8e625d2c986c0098ab74638449c3cea9b7b5d67d6ff6fce"
OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1")
CANDIDATES = ("turbo-base", "parent-4000", "mix-025", "finish-control-a4300")
PROMPT_MODES = ("hand_neutral", "hand_compatible", "hand_conflicting")
SCALES = (0.75, 1.0, 1.5)
CONDITION_COUNT, PRIMARY_COUNT, SCALE_COUNT, TOTAL_COUNT = 6, 72, 18, 90
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
RUNTIME = {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0, "mu": 1.15, "mu_resolution_dependent": False}
SIDECAR = "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required hand benchmark JSON is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Hand benchmark JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_historical(spec: Mapping[str, Any]) -> None:
    for relative, digest in spec["historical_artifact_sha256"].items():
        if _sha256(Path(relative)) != digest:
            raise ValueError(f"Historical artifact changed; hand benchmark refuses to proceed: {relative}")
    sources = spec["frozen_sources"]
    expected = {
        sources["final_spec"]["path"]: sources["final_spec"]["sha256"],
        sources["source_prompts"]["path"]: sources["source_prompts"]["sha256"],
        sources["turbo_baseline_spec"]["path"]: sources["turbo_baseline_spec"]["sha256"],
        f"{sources['authoritative_pose_sidecar']['path']}/metadata.json": sources["authoritative_pose_sidecar"]["metadata_sha256"],
        f"{sources['authoritative_pose_sidecar']['path']}/records.jsonl": sources["authoritative_pose_sidecar"]["records_sha256"],
    }
    for relative, digest in expected.items():
        if _sha256(Path(relative)) != digest:
            raise ValueError(f"Frozen hand benchmark source drifted: {relative}")


def _validate_locks(spec: Mapping[str, Any]) -> None:
    if (spec.get("format_version"), spec.get("kind"), spec.get("candidate_order"), spec.get("prompt_mode_order")) != (1, "hands_fingers_turbo_frozen_pose_control_comparison", list(CANDIDATES), list(PROMPT_MODES)):
        raise ValueError("Hand benchmark candidate or prompt-mode ordering drifted")
    if (spec.get("primary_generation_count"), spec.get("control_scale_generation_count"), spec.get("generation_count")) != (PRIMARY_COUNT, SCALE_COUNT, TOTAL_COUNT):
        raise ValueError("Hand benchmark generation-count contract drifted")
    if spec.get("geometry") != NATIVE_GEOMETRY or spec.get("runtime") != RUNTIME or spec.get("primary_control_scale") != 1.0:
        raise ValueError("Hand benchmark native geometry or locked Turbo runtime/control scale drifted")
    scale = spec.get("control_scale_substudy")
    if scale != {"candidate": "mix-025", "prompt_mode": "hand_neutral", "scales": list(SCALES), "generation_count": SCALE_COUNT}:
        raise ValueError("Hand benchmark mix-025 control-scale sub-study drifted")
    required = {"order", "condition_id", "selection_class", "stem", "seed", "native_bucket", "control_sha256", "source_prompt"}
    conditions = spec.get("conditions")
    if (not isinstance(conditions, list) or len(conditions) != CONDITION_COUNT or any(set(row) != required for row in conditions)
            or [row["order"] for row in conditions] != list(range(1, CONDITION_COUNT + 1))
            or len({row["stem"] for row in conditions}) != CONDITION_COUNT):
        raise ValueError("Hand benchmark must contain exactly six complete frozen conditions")
    expected_classes = ("simple_standing", "arms_raised_overhead", "bent_elbow_heavy", "foreshortened_arm_hand", "inversion_unusual_orientation", "two_person_interaction_visible_hands")
    if tuple(row["selection_class"] for row in conditions) != expected_classes:
        raise ValueError("Hand benchmark required hand/wrist condition coverage drifted")
    if any(not isinstance(row["seed"], int) or not isinstance(row["native_bucket"], list) or len(row["native_bucket"]) != 2 or len(row["control_sha256"]) != 64 or not row["source_prompt"].strip() for row in conditions):
        raise ValueError("Hand benchmark has incomplete frozen condition identity")
    modes = spec.get("prompt_modes")
    if not isinstance(modes, dict) or list(modes) != list(PROMPT_MODES) or modes.get("hand_neutral") != {"append": ""} or modes.get("hand_compatible") != {"append": ", natural relaxed hands and natural fingers"} or modes.get("hand_conflicting") != {"append": ", both arms folded tightly across the chest with hands hidden in pockets"}:
        raise ValueError("Hand benchmark frozen prompt wording drifted")
    sources = spec.get("frozen_sources")
    if not isinstance(sources, dict) or sources.get("final_spec") != {"path": str(final_val.FINAL_SPEC), "sha256": final_val.FINAL_SPEC_SHA256} or sources.get("authoritative_pose_sidecar", {}).get("path") != SIDECAR:
        raise ValueError("Hand benchmark final-val or authoritative-sidecar provenance drifted")
    baseline_spec = baseline.load_spec()
    if sources.get("turbo_baseline_spec") != {"path": str(baseline.SPEC), "sha256": baseline.SPEC_SHA256}:
        raise ValueError("Hand benchmark Turbo-baseline provenance drifted")
    if [row["id"] for row in baseline_spec["candidates"]] != list(CANDIDATES):
        raise ValueError("Hand benchmark requires the frozen four-candidate Turbo baseline contract")


def load_spec(path: str | Path = SPEC) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != SPEC_SHA256:
        raise ValueError(f"Frozen hand benchmark spec SHA-256 mismatch: {source}")
    spec = _read(source); _validate_locks(spec); _validate_historical(spec)
    return spec


def _source_prompts(path: Path) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 48 or len({row.get("stem") for row in rows}) != 48:
        raise ValueError("Frozen final-val source prompt file is incomplete")
    return {str(row["stem"]): str(row["text"]) for row in rows}


def _prompt(condition: Mapping[str, Any], mode: str, spec: Mapping[str, Any]) -> str:
    return str(condition["source_prompt"]) + str(spec["prompt_modes"][mode]["append"])


def _rows(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = [{"study": "primary", "condition": condition, "prompt_mode": mode, "candidate": candidate, "control_scale": 1.0, "prompt": _prompt(condition, mode, spec)}
               for condition in spec["conditions"] for mode in PROMPT_MODES for candidate in CANDIDATES]
    scales = [{"study": "control_scale", "condition": condition, "prompt_mode": "hand_neutral", "candidate": "mix-025", "control_scale": scale, "prompt": _prompt(condition, "hand_neutral", spec)}
              for condition in spec["conditions"] for scale in SCALES]
    if (len(primary), len(scales), len(primary) + len(scales)) != (PRIMARY_COUNT, SCALE_COUNT, TOTAL_COUNT):
        raise AssertionError("Frozen hand benchmark row count is invalid")
    return primary + scales


def _candidate_contract(spec: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    return baseline._candidate_config(_read(Path(spec["frozen_sources"]["turbo_baseline_spec"]["path"])), candidate)


def _sidecar_records(full_stems: list[str], conditions: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    sidecar, digest = final_val._load_final_sidecar(Path(SIDECAR), full_stems)
    raw_records = [json.loads(line) for line in (Path(SIDECAR) / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    raw_by_stem = {str(row["stem"]): row for row in raw_records}
    by_stem = {str(row["stem"]): row for row in sidecar["records"]}
    selected, selected_raw = [], []
    for condition in conditions:
        record = by_stem.get(condition["stem"])
        raw = raw_by_stem.get(condition["stem"])
        if not isinstance(record, dict) or not isinstance(raw, dict) or record.get("status") != "available":
            raise ValueError(f"Hand benchmark condition lacks authoritative pose: {condition['stem']}")
        wrists = _wrist_regions(raw)
        if not any("coordinate" in wrist for wrist in wrists):
            raise ValueError(f"Frozen hand benchmark condition has no authoritative in-frame wrist: {condition['stem']}")
        selected.append(record); selected_raw.append(raw)
    return {"records": selected, "raw_records": selected_raw}, digest


def _wrist_regions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    regions = []
    for person_index, person in enumerate(record.get("people", [])):
        joints = person.get("joint_provenance", [])
        for joint_index, side in ((9, "left_wrist"), (10, "right_wrist")):
            joint = joints[joint_index] if len(joints) > joint_index else None
            if isinstance(joint, Mapping) and joint.get("final_in_frame") and isinstance(joint.get("training_coordinate"), list) and len(joint["training_coordinate"]) == 2:
                regions.append({"person_index": person_index, "person_id": person.get("person_id"), "joint_index": joint_index, "joint": side, "coordinate": [float(joint["training_coordinate"][0]), float(joint["training_coordinate"][1])]})
            else:
                regions.append({"person_index": person_index, "person_id": person.get("person_id"), "joint_index": joint_index, "joint": side, "unavailable_reason": "authoritative_wrist_not_in_frame_or_not_available"})
    return regions


def _inputs(args: argparse.Namespace):
    spec = load_spec(args.spec)
    if args.turbo_ckpt != "/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors" or args.clip_model_id != "openai/clip-vit-base-patch32":
        raise ValueError("Hand benchmark refuses Turbo checkpoint or CLIP provenance override")
    final_spec, final_digest = final_val.load_final_spec(args.final_spec)
    conditions = list(spec["conditions"]); prompt_source = _source_prompts(Path(spec["frozen_sources"]["source_prompts"]["path"]))
    for condition in conditions:
        stem = condition["stem"]
        if prompt_source.get(stem) != condition["source_prompt"] or final_spec["per_stem_seeds"].get(stem, {}).get("sampling") != condition["seed"]:
            raise ValueError(f"Frozen hand source-caption prompt or seed drifted: {stem}")
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, final_spec)
    shard_metadata = _read(Path(args.latent_root) / "shards.json"); dataset_root = args.dataset_root or shard_metadata.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Hand benchmark requires --dataset-root or latent shards.json.dataset_root")
    stems = [row["stem"] for row in conditions]; controls = final_val.resolve_final_controls(dataset_root, stems)
    geometry, hashes = {}, {}
    for condition in conditions:
        stem, sample = condition["stem"], _sample_by_stem(dataset, condition["stem"])
        current = turbo_scoring_geometry(sample)
        if current.get("bucket") != condition["native_bucket"]:
            raise ValueError(f"Frozen native bucket drifted: {stem}")
        digest = _sha256(controls[stem])
        if digest != condition["control_sha256"]:
            raise ValueError(f"Frozen authoritative control SHA-256 drifted: {stem}")
        geometry[stem], hashes[stem] = current, digest
    sidecar, sidecar_digest = _sidecar_records(list(final_spec["stems"]), conditions)
    candidates: dict[str, tuple[dict[str, Any] | None, Path | None, dict[str, Any]]] = {"turbo-base": (None, None, {"candidate_kind": "unmodified_krea2_turbo", "pose_lora_control_adapter_state": "absent"})}
    for candidate in CANDIDATES[1:]:
        candidates[candidate] = baseline._resolve_controlled_candidate(candidate)
    contract = {"kind": spec["kind"], "frozen_spec": str(Path(args.spec)), "frozen_spec_sha256": SPEC_SHA256, "candidate_order": list(CANDIDATES), "prompt_mode_order": list(PROMPT_MODES), "primary_generation_count": PRIMARY_COUNT, "control_scale_generation_count": SCALE_COUNT, "generation_count": TOTAL_COUNT, "runtime": RUNTIME, "geometry": NATIVE_GEOMETRY, "primary_control_scale": 1.0, "control_scale_substudy": spec["control_scale_substudy"], "final_val_spec_sha256": final_digest, "authoritative_pose_sidecar": spec["frozen_sources"]["authoritative_pose_sidecar"], "authoritative_pose_records_sha256": sidecar_digest, "turbo_checkpoint": {"path": args.turbo_ckpt, "sha256": _sha256(Path(args.turbo_ckpt))}, "conditions": conditions, "control_sha256": hashes, "candidate_contracts": {candidate: _candidate_contract(spec, candidate) for candidate in CANDIDATES}, "historical_artifact_sha256": spec["historical_artifact_sha256"]}
    output = Path(args.output_root); provenance = output / "hand_benchmark_provenance.json"
    if provenance.exists() and _read(provenance) != contract:
        raise ValueError(f"Existing hand benchmark output has conflicting immutable provenance: {provenance}")
    if not provenance.exists():
        if getattr(args, "action", None) != "preflight":
            raise FileNotFoundError("Hand benchmark requires successful preflight provenance before later stages")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Hand benchmark preflight refuses non-empty output root without immutable provenance")
        _write(provenance, contract)
    return spec, contract, dataset, controls, geometry, sidecar, candidates, output


def _path(output: Path, row: Mapping[str, Any]) -> Path:
    suffix = row["candidate"] if row["study"] == "primary" else f"mix-025_scale-{row['control_scale']:.2f}"
    return output / "generations" / row["condition"]["condition_id"] / row["prompt_mode"] / f"{suffix}.png"


def _metadata(output: Path, row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    condition, candidate = row["condition"], row["candidate"]
    return {"study": row["study"], "condition_id": condition["condition_id"], "selection_class": condition["selection_class"], "stem": condition["stem"], "candidate": candidate, "candidate_contract": contract["candidate_contracts"][candidate], "prompt_mode": row["prompt_mode"], "prompt": row["prompt"], "seed": condition["seed"], "control_scale": row["control_scale"] if candidate != "turbo-base" else None, "control_applied": candidate != "turbo-base", "pose_lora_control_adapter_loaded": candidate != "turbo-base", "control_path": str(output / "controls" / f"{condition['stem']}.png"), "control_sha256": contract["control_sha256"][condition["stem"]], "native_bucket": condition["native_bucket"], "geometry": NATIVE_GEOMETRY, "runtime": RUNTIME, "frozen_spec_sha256": SPEC_SHA256, "turbo_checkpoint": contract["turbo_checkpoint"]}


def _expected_artifacts(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    return [str(_path(output, row).relative_to(output)) for row in rows]


def _generation_status(output: Path, contract: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    payload_path = output / "generation_results.json"; payload = _read(payload_path) if payload_path.exists() else None; observed = []
    for row in rows:
        image, metadata_path = _path(output, row), _path(output, row).with_suffix(".json")
        control = output / "controls" / f"{row['condition']['stem']}.png"
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    if list(opened.size) != row["condition"]["native_bucket"]: raise ValueError("native dimensions changed")
                    opened.verify()
                metadata = _read(metadata_path); expected = _metadata(output, row, contract)
                if any(metadata.get(key) != value for key, value in expected.items() if key != "control_path") or not isinstance(metadata.get("control_path"), str) or _sha256(control) != expected["control_sha256"]:
                    raise ValueError("metadata/control identity drifted")
            except Exception as exc:
                raise ValueError(f"Hand generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if image.parent.exists() or control.exists():
                raise ValueError("Existing hand generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
    if not any(observed) and payload is None:
        if (output / "generations").exists() or (output / "controls").exists():
            raise ValueError("Existing hand generation output is incomplete or inconsistent; refusing to overwrite it")
        return "missing"
    if all(observed) and payload is not None and all(payload.get(key) == value for key, value in contract.items()) and payload.get("generated_artifacts") == _expected_artifacts(output, rows):
        return "complete"
    raise ValueError("Existing hand generation output is incomplete or inconsistent; refusing to overwrite it")


def _copy_control(source: Path, target: Path, expected: str) -> None:
    if _sha256(source) != expected: raise ValueError(f"Authoritative control hash changed: {source}")
    if target.exists() and _sha256(target) != expected: raise ValueError(f"Existing control conflicts: {target}")
    if not target.exists(): target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())


def preflight(args: argparse.Namespace) -> None:
    spec, contract, dataset, controls, geometry, sidecar, candidates, output = _inputs(args); rows = _rows(spec)
    _write(output / "checkpoint_preflight.json", {**contract, "dataset_sample_count": len(dataset), "control_paths_resolved": {stem: str(path) for stem, path in controls.items()}, "geometry_by_stem": geometry, "wrist_regions": {row["stem"]: _wrist_regions(next(record for record in sidecar["raw_records"] if record["stem"] == row["stem"])) for row in contract["conditions"]}, "resolved_candidates": {candidate: ("unmodified Turbo; no Pose-LoRA/control state" if candidate == "turbo-base" else candidates[candidate][2]) for candidate in CANDIDATES}, "generation_plan": [{key: row[key] for key in ("study", "prompt_mode", "candidate", "control_scale", "prompt")} | {"condition_id": row["condition"]["condition_id"], "seed": row["condition"]["seed"]} for row in rows], "source_rgb_fallback_permitted": False})
    print(output / "checkpoint_preflight.json")


def _conditioned_sample(conditioner, sample: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    context, mask = guide._conditioning(conditioner, prompt, sample)
    return {**sample, "prompt": prompt, "context": context, "mask": mask}


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("Run hand benchmark generation from the GH200 host shell with CUDA visible")
    spec, contract, dataset, controls, _, _, candidates, output = _inputs(args); rows = _rows(spec)
    if _generation_status(output, contract, rows) == "complete": print(json.dumps({"already_complete": True, "generation_count": TOTAL_COUNT})); return
    vae, conditioner, device = load_krea_vae("cuda"), guide.PoseTextConditioner(device="cuda", dtype=torch.bfloat16), torch.device("cuda")
    samples = {condition["stem"]: dict(_sample_by_stem(dataset, condition["stem"])) for condition in contract["conditions"]}; compatibility: dict[str, Any] = {}; base_report = None
    for candidate_id in CANDIDATES:
        model = None
        if candidate_id == "turbo-base":
            model = baseline.build_unmodified_turbo_base_model(args.turbo_ckpt, "cuda"); base_report = getattr(model, "_krea_checkpoint_report", None)
        else:
            model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval(); candidate, checkpoint, _ = candidates[candidate_id]; assert candidate is not None
            state = final_val.candidate_trainable_state(candidate, checkpoint); final_val.load_trainable_state_dict(model, state)
            compatibility[candidate_id] = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, state))
        for row in [item for item in rows if item["candidate"] == candidate_id]:
            condition, image = row["condition"], _path(output, row); control = output / "controls" / f"{condition['stem']}.png"; _copy_control(controls[condition["stem"]], control, contract["control_sha256"][condition["stem"]])
            metadata, metadata_path = _metadata(output, row, contract), image.with_suffix(".json")
            if metadata_path.exists() and _read(metadata_path) != metadata: raise ValueError(f"Existing hand generation metadata conflicts: {metadata_path}")
            if image.exists(): continue
            if not metadata_path.exists(): _write(metadata_path, metadata)
            sample = _conditioned_sample(conditioner, samples[condition["stem"]], row["prompt"])
            if candidate_id == "turbo-base":
                pixels = baseline.sample_unmodified_turbo_base_image(model, lambda latent: decode_normalized_latents(vae, latent), sample, device, condition["seed"])
            else:
                pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample, device, condition["seed"], steps=8, guidance=0.0, mu=1.15, control_scale=row["control_scale"])
            save_image(pixels, image)
        del model; torch.cuda.empty_cache()
    _write(output / "generation_results.json", {**contract, "generated_artifacts": _expected_artifacts(output, rows), "raw_to_turbo_control_compatibility": compatibility, "turbo_base_checkpoint_report": base_report, "turbo_base_load": {"model_surgery": "none", "pose_lora_control_adapter_loaded": False}, "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _compact(records: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row["pck"] for row in records if row["pck"].get("reference_available")]
    pose = _pool_pose(available) if available else {"evaluable_sample_count": 0, "pck_005": None, "pck_010": None, "pck_020": None, "matched_people": 0, "predicted_people": 0, "unmatched_reference_people": 0, "unmatched_predicted_people": 0, "detection_coverage": None}
    clips = aggregate([float(row["clip_cosine_similarity"]) for row in records])
    return {"generation_count": len(records), "pck": pose, "clip": {"mean_cosine_similarity": clips["mean"], "median_cosine_similarity": clips["median"], "std_cosine_similarity": clips["std"], "sample_count": clips["sample_count"]}, "matched_people": pose["matched_people"], "predicted_people": pose["predicted_people"], "unmatched_reference_people": pose["unmatched_reference_people"], "unmatched_predicted_people": pose["unmatched_predicted_people"], "detection_coverage": pose["detection_coverage"], "pck_unavailable_count": len(records) - len(available)}


def _aggregate(records: list[dict[str, Any]], key: str, expected: tuple[Any, ...]) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in records: groups[row[key]].append(row)
    if set(groups) != set(expected): raise ValueError(f"Hand benchmark aggregate {key} membership drifted")
    return {"group_by": key, "rows": [{key: value, **_compact(groups[value])} for value in expected]}


def _score_records(payload: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    expected = [(row["study"], row["condition"]["condition_id"], row["prompt_mode"], row["candidate"], row["control_scale"]) for row in rows]
    actual = [(row.get("study"), row.get("condition_id"), row.get("prompt_mode"), row.get("candidate"), row.get("control_scale")) for row in records] if isinstance(records, list) else []
    if len(actual) != TOTAL_COUNT or actual != expected: raise ValueError("Hand benchmark score artifact is incomplete or not in frozen row order")
    return records


def score(args: argparse.Namespace) -> None:
    spec, contract, _, _, geometry, sidecar, _, output = _inputs(args); rows = _rows(spec)
    if _generation_status(output, contract, rows) != "complete": raise FileNotFoundError("Hand scoring requires all 90 validated generations")
    if args.reference_sidecar != SIDECAR: raise ValueError("Hand scoring requires the locked authoritative final-val v3 sidecar")
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5); processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    by_stem = {record["stem"]: record for record in sidecar["records"]}; results = []
    for row in rows:
        condition, image = row["condition"], _path(output, row); stem = condition["stem"]
        pck_result = score_authoritative_pck(sidecar={"records": [by_stem[stem]]}, geometry_by_stem={stem: geometry[stem]}, image_for=lambda _: image, detector=detector, confidence_threshold=.5, require_images=True)
        pck = pck_result["per_image"][0] if pck_result["per_image"] else {"stem": stem, "reference_available": False, "reason": pck_result["unavailable"][0]["reason"]}
        results.append({"study": row["study"], "condition_id": condition["condition_id"], "selection_class": condition["selection_class"], "stem": stem, "prompt_mode": row["prompt_mode"], "prompt": row["prompt"], "candidate": row["candidate"], "control_scale": row["control_scale"], "seed": condition["seed"], "image": str(image.relative_to(output)), "pck": pck, "clip_cosine_similarity": _clip_score(clip, processor, device, row["prompt"], image)})
    primary = [row for row in results if row["study"] == "primary"]; scales = [row for row in results if row["study"] == "control_scale"]
    by_candidate = _aggregate(primary, "candidate", CANDIDATES); by_prompt = _aggregate(primary, "prompt_mode", PROMPT_MODES)
    pairs = tuple(f"{candidate}__{mode}" for candidate in CANDIDATES for mode in PROMPT_MODES)
    for row in primary: row["candidate_prompt_mode"] = f"{row['candidate']}__{row['prompt_mode']}"
    by_pair = _aggregate(primary, "candidate_prompt_mode", pairs); by_scale = _aggregate(scales, "control_scale", SCALES)
    _write(output / "pck_clip_results.json", {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "clip_model": args.clip_model_id, "confidence_threshold": .5, "per_generation": results, "aggregate_by_candidate": by_candidate, "aggregate_by_prompt_mode": by_prompt, "aggregate_by_candidate_prompt_mode": by_pair, "aggregate_by_control_scale": by_scale})
    print(output / "pck_clip_results.json")


def _validated_scores(output: Path, contract: Mapping[str, Any], rows: list[dict[str, Any]]):
    payload = _read(output / "pck_clip_results.json")
    if any(payload.get(key) != value for key, value in contract.items()): raise ValueError("Hand score artifact conflicts with frozen provenance")
    return payload, _score_records(payload, rows)


def _crop(image: Path, coordinate: list[float], destination: Path) -> list[int]:
    with Image.open(image) as opened: source = opened.convert("RGB")
    width, height = source.size; radius = max(96, min(224, min(width, height) // 4)); x, y = coordinate
    left, top = max(0, min(width - 2 * radius, int(round(x - radius)))), max(0, min(height - 2 * radius, int(round(y - radius))))
    box = [left, top, min(width, left + 2 * radius), min(height, top + 2 * radius)]
    destination.parent.mkdir(parents=True, exist_ok=True); source.crop(tuple(box)).save(destination); return box


def _make_crops(output: Path, contract: Mapping[str, Any], sidecar: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = {record["stem"]: record for record in sidecar["raw_records"]}; result = {}
    for row in rows:
        image = _path(output, row); key = str(image.relative_to(output)); regions = []
        for wrist in _wrist_regions(records[row["condition"]["stem"]]):
            item = dict(wrist)
            if "coordinate" in wrist:
                filename = f"p{wrist['person_index']}_{wrist['joint']}.png"; target = output / "hand_crops" / image.relative_to(output).with_suffix("") / filename
                item.update({"crop": str(target.relative_to(output)), "crop_box": _crop(image, wrist["coordinate"], target), "crop_policy": "authoritative_wrist_centered_square_radius_clamped_96_to_224"})
            regions.append(item)
        result[key] = regions
    return result


def report(args: argparse.Namespace) -> None:
    spec, contract, _, _, _, sidecar, _, output = _inputs(args); rows = _rows(spec)
    if _generation_status(output, contract, rows) != "complete": raise FileNotFoundError("Hand report requires all 90 validated generations")
    scores, records = _validated_scores(output, contract, rows); primary = [row for row in records if row["study"] == "primary"]; scales = [row for row in records if row["study"] == "control_scale"]
    for row in primary: row["candidate_prompt_mode"] = f"{row['candidate']}__{row['prompt_mode']}"
    tables = {"candidate": _aggregate(primary, "candidate", CANDIDATES), "prompt": _aggregate(primary, "prompt_mode", PROMPT_MODES), "pair": _aggregate(primary, "candidate_prompt_mode", tuple(f"{candidate}__{mode}" for candidate in CANDIDATES for mode in PROMPT_MODES)), "scale": _aggregate(scales, "control_scale", SCALES)}
    if (scores.get("aggregate_by_candidate"), scores.get("aggregate_by_prompt_mode"), scores.get("aggregate_by_candidate_prompt_mode"), scores.get("aggregate_by_control_scale")) != (tables["candidate"], tables["prompt"], tables["pair"], tables["scale"]): raise ValueError("Hand aggregate tables do not match per-image results")
    _write(output / "metrics_by_candidate.json", {**contract, **tables["candidate"]}); _write(output / "metrics_by_prompt_mode.json", {**contract, **tables["prompt"]}); _write(output / "metrics_by_candidate_prompt_mode.json", {**contract, **tables["pair"]}); _write(output / "metrics_by_control_scale.json", {**contract, **tables["scale"]})
    main_rows = [(f"{row['condition']['condition_id']} | {row['prompt_mode']}", [output / "controls" / f"{row['condition']['stem']}.png", *[_path(output, {**row, "candidate": candidate, "study": "primary", "control_scale": 1.0}) for candidate in CANDIDATES]]) for row in [{"condition": c, "prompt_mode": m} for c in contract["conditions"] for m in PROMPT_MODES]]
    make_contact_sheet(main_rows, output / "hand_benchmark_contact_sheet.png", thumbnail_width=150, thumbnail_height=160, column_labels=("pose control", *CANDIDATES))
    crop_index = _make_crops(output, contract, sidecar, rows); _write(output / "hand_crop_regions.json", {"frozen_spec_sha256": SPEC_SHA256, "regions": crop_index})
    crop_rows = []
    for row in rows:
        label = f"{row['condition']['condition_id']} | {row['prompt_mode']} | {row['candidate']} | {row['control_scale']}"
        for item in crop_index[str(_path(output, row).relative_to(output))]:
            if "crop" in item: crop_rows.append((f"{label} | p{item['person_index']} {item['joint']}", [output / item["crop"]]))
    if crop_rows: make_contact_sheet(crop_rows, output / "hand_crop_contact_sheet.png", thumbnail_width=200, thumbnail_height=200, column_labels=("authoritative wrist crop",))
    for condition in contract["conditions"]:
        relevant = [row for row in rows if row["condition"]["stem"] == condition["stem"]]; wrist_count = len([item for item in crop_index[str(_path(output, relevant[0]).relative_to(output))] if "crop" in item])
        grid = []
        for row in relevant:
            crops = [output / item["crop"] for item in crop_index[str(_path(output, row).relative_to(output))] if "crop" in item]
            if len(crops) != wrist_count: raise ValueError("Hand crop availability drifted within frozen condition")
            grid.append((f"{row['candidate']} | {row['prompt_mode']} | scale {row['control_scale']}", crops))
        make_contact_sheet(grid, output / "per_condition_hand_grids" / f"{condition['condition_id']}.png", thumbnail_width=170, thumbnail_height=170, column_labels=tuple(f"wrist {index + 1}" for index in range(wrist_count)))
    scale_grid = [(row["condition"]["condition_id"] + f" | scale {row['control_scale']}", [output / item["crop"] for item in crop_index[str(_path(output, row).relative_to(output))] if "crop" in item][:1]) for row in [row for row in rows if row["study"] == "control_scale"]]
    make_contact_sheet(scale_grid, output / "control_scale_hand_grid.png", thumbnail_width=190, thumbnail_height=190, column_labels=("first authoritative wrist crop",))
    checklist = {"kind": "human_review_only_no_automatic_anatomy_classification", "fields": ["missing_hand", "duplicated_hand_fingers", "malformed_or_anatomically_implausible_fingers", "wrist_discontinuity", "arm_hand_pose_mismatch", "acceptable_hand"], "values": ["unreviewed", "yes", "no", "not_applicable"], "note": "Leave unreviewed until a human inspects the deterministic wrist-region grids; no field is auto-filled."}
    _write(output / "evaluation_summary.json", {**contract, "generation_count": TOTAL_COUNT, "score_artifact": "pck_clip_results.json", "metrics": {"by_candidate": "metrics_by_candidate.json", "by_prompt_mode": "metrics_by_prompt_mode.json", "by_candidate_prompt_mode": "metrics_by_candidate_prompt_mode.json", "by_control_scale": "metrics_by_control_scale.json"}, "human_review_checklist": checklist, "hand_crop_regions": "hand_crop_regions.json", "contact_sheets": {"main": "hand_benchmark_contact_sheet.png", "crops": "hand_crop_contact_sheet.png", "per_condition": "per_condition_hand_grids/*.png", "control_scale": "control_scale_hand_grid.png"}, "source_rgb_fallback_used": False})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    spec, contract, _, _, _, _, _, output = _inputs(args); rows = _rows(spec)
    if _generation_status(output, contract, rows) != "complete": raise FileNotFoundError("Hand summary requires all 90 validated generations")
    _, records = _validated_scores(output, contract, rows); primary = [row for row in records if row["study"] == "primary"]; scales = [row for row in records if row["study"] == "control_scale"]
    compact = {"frozen_spec_sha256": SPEC_SHA256, "generation_count": TOTAL_COUNT, "primary_generation_count": PRIMARY_COUNT, "control_scale_generation_count": SCALE_COUNT, "candidate_order": list(CANDIDATES), "prompt_mode_order": list(PROMPT_MODES), "runtime": RUNTIME, "geometry": NATIVE_GEOMETRY, "by_candidate": _aggregate(primary, "candidate", CANDIDATES)["rows"], "by_prompt_mode": _aggregate(primary, "prompt_mode", PROMPT_MODES)["rows"], "by_control_scale": _aggregate(scales, "control_scale", SCALES)["rows"], "human_review_required": True}
    _write(output / "compact_summary.json", compact); print(output / "compact_summary.json")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("preflight", "generate", "score", "report", "summary")); parser.add_argument("--output-root", default=str(OUTPUT_ROOT)); parser.add_argument("--spec", default=str(SPEC)); parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC)); parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents"); parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning"); parser.add_argument("--dataset-root"); parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors"); parser.add_argument("--reference-sidecar", default=SIDECAR); parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32"); return parser


def main() -> None:
    args = parser().parse_args(); {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__": main()
