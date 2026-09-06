"""Frozen native-cache versus dynamic-768 Krea-2 Turbo release ablation.

This evaluation-only entrypoint deliberately leaves canonical ``inference.py``
and all historical benchmark namespaces untouched.  Native mode consumes the
authoritative cached control latent; dynamic mode consumes only the same
authoritative pose-control raster, applies the shared 768 geometry policy, and
encodes that transformed control.  It never opens or substitutes an RGB input.
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
from pose_controlnet.paired_preprocessing import apply_resize_center_crop_geometry, choose_bucket, resize_center_crop_geometry
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator
from pose_controlnet.resolution_policy import RESOLUTION_768_BUCKETS
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_metadata, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, encode_preprocessed_image, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val
from scripts import prompting_guide_study as guide


ABLATION_SPEC = Path("docs/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1.json")
ABLATION_SPEC_SHA256 = "fe2d109e1e08c05007e38f605306ce1f9092794d0b7e0017a553e1fd3d6906cf"
OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1")
GEOMETRY_MODES = ("native_aspect_preserving_cached_latent_bucket", "dynamic_768_bucket")
CONDITION_CLASSES = ("near_square", "portrait", "landscape", "extreme_portrait", "multi_person")
STEMS = (
    "sculpture_humanart_14000000003803", "real_human_humanart_15000000000521",
    "real_human_humanart_15000000000477", "sculpture_humanart_14000000000288",
    "real_human_humanart_17000000002207",
)
GENERATION_COUNT = 10
RUNTIME = turbo_metadata()
PROMPT_SOURCE_SHA256 = "23d448d573a2ffd20adfd73fa88f34ebc08df280a051cb0931d9ecdcc1231ceb"
HISTORICAL_ARTIFACT_SHA256 = {
    "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json": final_val.FINAL_SPEC_SHA256,
    "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl": PROMPT_SOURCE_SHA256,
    "docs/evaluation/prompting-guide/prompting_study.jsonl": "4fae6d39ac7354d451ca13556d3a2a89e303691ceb30727b8561b13b494450df",
    "docs/evaluation/control-scale-sweep/mix-025-native-v1.json": "af82b4c0b0290a852cdd3d7b6b917603e3e9298aa51dfa2ace737d41a28d91a6",
    "inference.py": "60992bba7f628dbbdadfd1cb59c3b4f385a016de3ac20898591ef7f1f8c47a5b",
}


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
        raise FileNotFoundError(f"Required native/dynamic ablation JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Native/dynamic ablation JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Native/dynamic ablation JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_final_val_prompts(path: Path) -> dict[str, str]:
    if _sha256(path) != PROMPT_SOURCE_SHA256:
        raise ValueError("Frozen final-val prompt source SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {str(row["stem"]): str(row["text"]) for row in rows}


def _validate_historical_artifacts() -> None:
    for relative, expected in HISTORICAL_ARTIFACT_SHA256.items():
        if _sha256(Path(relative)) != expected:
            raise ValueError(f"Historical artifact changed; native/dynamic ablation refuses to proceed: {relative}")


def _validate_locks(spec: Mapping[str, Any]) -> None:
    conditions = spec.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(STEMS):
        raise ValueError("Frozen native/dynamic spec must contain exactly five conditions")
    if tuple(row.get("stem") for row in conditions) != STEMS or tuple(row.get("condition_class") for row in conditions) != CONDITION_CLASSES:
        raise ValueError("Frozen native/dynamic condition ordering/classes drifted")
    required = {"stem", "condition_class", "orientation", "source_size", "sampling_seed", "control_sha256", "native_bucket", "dynamic_768_bucket", "prompt"}
    if any(set(row) != required or not isinstance(row["prompt"], str) or not row["prompt"].strip()
           or not isinstance(row["sampling_seed"], int) or not isinstance(row["control_sha256"], str)
           or len(row["control_sha256"]) != 64 or any(not isinstance(row[key], list) or len(row[key]) != 2 for key in ("source_size", "native_bucket", "dynamic_768_bucket"))
           for row in conditions):
        raise ValueError("Frozen native/dynamic conditions are malformed")
    if tuple(spec.get("geometry_modes", ())) != GEOMETRY_MODES or spec.get("generation_count") != GENERATION_COUNT:
        raise ValueError("Frozen native/dynamic ablation must be exactly five conditions x two ordered geometry modes")
    if spec.get("candidate") != "mix-025" or spec.get("candidate_kind") != "trainable_tensor_interpolation":
        raise ValueError("Frozen native/dynamic ablation is locked to candidate mix-025")
    if spec.get("control_scale") != 1.0:
        raise ValueError("Frozen native/dynamic ablation is locked to control scale 1.0")
    if spec.get("runtime") != RUNTIME:
        raise ValueError("Frozen native/dynamic ablation violates the locked Krea-2 Turbo 8-step CFG-0 mu=1.15 runtime")
    if spec.get("final_val_spec_sha256") != final_val.FINAL_SPEC_SHA256:
        raise ValueError("Frozen native/dynamic final-val provenance drifted")
    if spec.get("clip_model_id") != "openai/clip-vit-base-patch32" or spec.get("turbo_checkpoint") != "/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors":
        raise ValueError("Frozen native/dynamic runtime metric/checkpoint provenance drifted")
    if spec.get("prompt_source") != {"path": "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl", "sha256": PROMPT_SOURCE_SHA256}:
        raise ValueError("Frozen native/dynamic prompt provenance drifted")
    interpolation = spec.get("checkpoint_interpolation")
    if not isinstance(interpolation, dict) or interpolation.get("candidate_id") != "mix-025" or interpolation.get("alpha") != .25:
        raise ValueError("Frozen native/dynamic mix-025 interpolation provenance is malformed")


def load_ablation_spec(path: str | Path = ABLATION_SPEC) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != ABLATION_SPEC_SHA256:
        raise ValueError(f"Frozen native/dynamic spec SHA-256 mismatch: {source}")
    spec = _read_json(source)
    _validate_locks(spec)
    prompts = _read_final_val_prompts(Path(spec["prompt_source"]["path"]))
    if any(prompts.get(row["stem"]) != row["prompt"] for row in spec["conditions"]):
        raise ValueError("Frozen native/dynamic prompt mapping drifted from final-val source")
    _validate_historical_artifacts()
    return spec


def _expected_pairs(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(row["stem"], mode) for row in spec["conditions"] for mode in GEOMETRY_MODES]


def _geometry_dict(source_size: tuple[int, int], bucket: tuple[int, int]) -> dict[str, list[int]]:
    geometry = resize_center_crop_geometry(source_size, bucket)
    return {"source_size": list(geometry.source_size), "resized_size": list(geometry.resized_size), "crop_box": list(geometry.crop_box), "bucket": list(geometry.bucket)}


def _framing_comparison(native: Mapping[str, Any], dynamic: Mapping[str, Any]) -> dict[str, Any]:
    def crop_fraction(geometry: Mapping[str, Any]) -> dict[str, float]:
        resized, crop = geometry["resized_size"], geometry["crop_box"]
        return {"horizontal": (resized[0] - (crop[2] - crop[0])) / resized[0], "vertical": (resized[1] - (crop[3] - crop[1])) / resized[1]}
    return {"native_geometry": native, "dynamic_768_geometry": dynamic,
            "native_crop_fraction_of_resized": crop_fraction(native), "dynamic_768_crop_fraction_of_resized": crop_fraction(dynamic),
            "bucket_changed": native["bucket"] != dynamic["bucket"]}


def _candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    return guide._candidate_contract(candidate, checkpoint)


def _inputs(args: argparse.Namespace):
    spec = load_ablation_spec(args.spec)
    if args.candidate != "mix-025" or args.turbo_ckpt != spec["turbo_checkpoint"] or args.clip_model_id != spec["clip_model_id"]:
        raise ValueError("Native/dynamic ablation refuses candidate/runtime/CLIP provenance drift")
    final_spec, final_digest = final_val.load_final_spec(args.final_spec)
    if final_digest != spec["final_val_spec_sha256"]:
        raise ValueError("Current final-val spec does not match frozen native/dynamic ablation")
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, final_spec)
    shards = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shards.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Native/dynamic ablation requires --dataset-root or latent shards.json.dataset_root")
    controls = final_val.resolve_final_controls(dataset_root, list(STEMS))
    geometries: dict[str, dict[str, dict[str, list[int]]]] = {}
    for row in spec["conditions"]:
        stem, sample, control = row["stem"], _sample_by_stem(dataset, row["stem"]), controls[row["stem"]]
        native = turbo_scoring_geometry(sample)
        if native["bucket"] != row["native_bucket"]:
            raise ValueError(f"Frozen native cached bucket drifted for {stem}")
        with Image.open(control) as image:
            control_size = image.size
        if list(control_size) != row["source_size"] or _sha256(control) != row["control_sha256"]:
            raise ValueError(f"Authoritative final-val control/source contract drifted for {stem}")
        dynamic_bucket = choose_bucket(control_size, RESOLUTION_768_BUCKETS)
        dynamic = _geometry_dict(control_size, dynamic_bucket)
        if dynamic["bucket"] != row["dynamic_768_bucket"]:
            raise ValueError(f"Frozen dynamic-768 bucket drifted for {stem}")
        if final_spec["per_stem_seeds"][stem]["sampling"] != row["sampling_seed"]:
            raise ValueError(f"Frozen final-val seed drifted for {stem}")
        geometries[stem] = {GEOMETRY_MODES[0]: native, GEOMETRY_MODES[1]: dynamic}
    candidate, checkpoint, training = final_val.resolve_candidate(args.candidate)
    endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
    final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    if candidate.get("label") != "mix-025" or _candidate_contract(candidate, checkpoint).get("checkpoint_interpolation") != spec["checkpoint_interpolation"]:
        raise ValueError("Resolved mix-025 candidate provenance drifted from frozen native/dynamic spec")
    contract = {"kind": spec["kind"], "frozen_spec": str(Path(args.spec)), "frozen_spec_sha256": ABLATION_SPEC_SHA256,
                "candidate": "mix-025", "candidate_kind": spec["candidate_kind"], "checkpoint_interpolation": spec["checkpoint_interpolation"],
                "runtime": spec["runtime"], "control_scale": 1.0, "geometry_modes": list(GEOMETRY_MODES), "generation_count": GENERATION_COUNT,
                "conditions": spec["conditions"], "final_val_spec_sha256": final_digest, "prompt_source": spec["prompt_source"],
                "turbo_checkpoint": spec["turbo_checkpoint"], "clip_model_id": spec["clip_model_id"],
                "expected_geometry_by_stem": geometries,
                "source_rgb_fallback_permitted": False, "historical_artifact_sha256": HISTORICAL_ARTIFACT_SHA256}
    output = Path(args.output_root)
    provenance = output / "native_dynamic_provenance.json"
    if provenance.exists() and _read_json(provenance) != contract:
        raise ValueError(f"Existing output has conflicting immutable native/dynamic provenance: {provenance}")
    if not provenance.exists():
        _write(provenance, contract)
    return spec, contract, dataset, controls, geometries, candidate, checkpoint, training, output


def _directory(output: Path, stem: str, mode: str) -> Path:
    return output / "generations" / stem / mode


def _image_path(output: Path, stem: str, mode: str, candidate: Mapping[str, Any]) -> Path:
    return _directory(output, stem, mode) / final_val._image_name(candidate)


def _copy_control(source: Path, target: Path) -> None:
    if target.exists() and _sha256(target) != _sha256(source):
        raise ValueError(f"Existing pose control conflicts with authoritative final-val control: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _generation_metadata(condition: Mapping[str, Any], mode: str, geometry: Mapping[str, Any], contract: Mapping[str, Any], control: Path) -> dict[str, Any]:
    return {"stem": condition["stem"], "condition_class": condition["condition_class"], "orientation": condition["orientation"], "prompt": condition["prompt"],
            "seed": condition["sampling_seed"], "control_scale": 1.0, "control_path": str(control), "control_sha256": condition["control_sha256"],
            "geometry_mode": mode, "geometry": geometry, "output_dimensions": geometry["bucket"], "output_bucket": geometry["bucket"],
            "dynamic_control_encode_seed": condition["sampling_seed"] if mode == GEOMETRY_MODES[1] else None,
            "frozen_spec_sha256": ABLATION_SPEC_SHA256, "final_val_spec_sha256": contract["final_val_spec_sha256"], "candidate": "mix-025",
            "runtime": contract["runtime"], "checkpoint_interpolation": contract["checkpoint_interpolation"]}


def _generation_status(output: Path, spec: Mapping[str, Any], candidate: Mapping[str, Any], contract: Mapping[str, Any], geometries: Mapping[str, Any]) -> str:
    payload_path = output / "generation_results.json"; payload = _read_json(payload_path) if payload_path.exists() else None; observed = []
    conditions = {row["stem"]: row for row in spec["conditions"]}
    for stem, mode in _expected_pairs(spec):
        image, metadata_path, control = _image_path(output, stem, mode, candidate), _directory(output, stem, mode) / "metadata.json", output / "controls" / f"{stem}.png"
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    if list(opened.size) != geometries[stem][mode]["bucket"]: raise ValueError("output dimensions differ from frozen bucket")
                    opened.verify()
                with Image.open(control) as opened: opened.verify()
                expected = _generation_metadata(conditions[stem], mode, geometries[stem][mode], contract, control)
                metadata = _read_json(metadata_path)
                if any(metadata.get(key) != value for key, value in expected.items() if key != "control_path") or not isinstance(metadata.get("control_path"), str) or _sha256(control) != conditions[stem]["control_sha256"]:
                    raise ValueError("generation metadata/control contract mismatch")
            except Exception as exc:
                raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if _directory(output, stem, mode).exists() or control.exists():
                raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
    expected_artifacts = {row["stem"]: {mode: str(_image_path(output, row["stem"], mode, candidate).relative_to(output)) for mode in GEOMETRY_MODES} for row in spec["conditions"]}
    if not any(observed) and payload is None: return "missing"
    if all(observed) and payload is not None and all(payload.get(key) == value for key, value in contract.items()) and payload.get("generated_artifacts") == expected_artifacts:
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


def _dynamic_sample(control: Path, geometry: Mapping[str, Any], vae: Any, seed: int, device: torch.device) -> dict[str, Any]:
    with Image.open(control) as source:
        prepared = apply_resize_center_crop_geometry(source.convert("RGB"), resize_center_crop_geometry(source.size, tuple(geometry["bucket"])))
    latent = encode_preprocessed_image(vae, prepared, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    if not torch.isfinite(latent).all() or latent.abs().max().item() == 0.0:
        raise ValueError("Dynamic-768 pose VAE encoding produced an empty or non-finite control latent")
    return {"latent": torch.zeros_like(latent), "control": latent}


def preflight(args: argparse.Namespace) -> None:
    spec, contract, dataset, controls, geometries, _, _, training, output = _inputs(args)
    framing = {row["stem"]: _framing_comparison(geometries[row["stem"]][GEOMETRY_MODES[0]], geometries[row["stem"]][GEOMETRY_MODES[1]]) for row in spec["conditions"]}
    _write(output / "checkpoint_preflight.json", {**contract, "dataset_sample_count": len(dataset), "control_paths_resolved": {stem: str(path) for stem, path in controls.items()},
           "geometry_by_stem": geometries, "framing_geometry_comparison": framing, "training_metadata": training, "generation_plan": _expected_pairs(spec)})
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("Run native/dynamic generation from the GH200 host shell with CUDA visible")
    spec, contract, dataset, controls, geometries, candidate, checkpoint, training, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract, geometries) == "complete":
        print(json.dumps({"already_complete": "mix-025", "generation_count": GENERATION_COUNT})); return
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint); final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    vae, conditioner, device = load_krea_vae("cuda"), guide.PoseTextConditioner(device="cuda", dtype=torch.bfloat16), torch.device("cuda")
    for condition in spec["conditions"]:
        stem, cached = condition["stem"], dict(_sample_by_stem(dataset, condition["stem"])); context, mask = guide._conditioning(conditioner, condition["prompt"], cached)
        _copy_control(controls[stem], output / "controls" / f"{stem}.png")
        for mode in GEOMETRY_MODES:
            directory, image, metadata_path = _directory(output, stem, mode), _image_path(output, stem, mode, candidate), _directory(output, stem, mode) / "metadata.json"
            metadata = _generation_metadata(condition, mode, geometries[stem][mode], contract, controls[stem])
            if metadata_path.exists() and _read_json(metadata_path) != metadata: raise ValueError(f"Existing generation metadata conflicts with frozen ablation: {metadata_path}")
            if image.exists(): continue
            if not metadata_path.exists(): _write(metadata_path, metadata)
            sample = dict(cached) if mode == GEOMETRY_MODES[0] else _dynamic_sample(controls[stem], geometries[stem][mode], vae, condition["sampling_seed"], device)
            sample.update({"context": context, "mask": mask})
            pixels = sample_turbo_pose_image(
                model, lambda latent: decode_normalized_latents(vae, latent), sample, device,
                condition["sampling_seed"], steps=8, guidance=0.0, mu=1.15, control_scale=1.0,
            )
            save_image(pixels, image)
    artifacts = {row["stem"]: {mode: str(_image_path(output, row["stem"], mode, candidate).relative_to(output)) for mode in GEOMETRY_MODES} for row in spec["conditions"]}
    _write(output / "generation_results.json", {**contract, "generated_artifacts": artifacts, "geometry_by_stem": geometries, "training_metadata": training,
           "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None), "raw_to_turbo_control_compatibility": compatibility, "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _sidecar(path: str | Path) -> tuple[dict[str, Any], str]:
    final_spec, _ = final_val.load_final_spec(final_val.FINAL_SPEC); sidecar, digest = final_val._load_final_sidecar(path, list(final_spec["stems"]))
    records = {str(record["stem"]): record for record in sidecar["records"]}
    if set(STEMS) - set(records): raise ValueError("Authoritative final-val pose sidecar is missing an ablation stem")
    return {"records": [records[stem] for stem in STEMS]}, digest


def _compact_group(rows: list[dict[str, Any]]) -> dict[str, Any]: return guide._compact_group(rows)


def _score_records(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    if not isinstance(records, list) or len(records) != GENERATION_COUNT or [(row.get("stem"), row.get("geometry_mode")) for row in records] != _expected_pairs(spec):
        raise ValueError("Native/dynamic score artifact is incomplete or not in frozen condition/geometry order")
    return records


def _aggregate_tables(records: list[dict[str, Any]], mode_results: Mapping[str, Any], spec: Mapping[str, Any]):
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list); by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records: by_mode[row["geometry_mode"]].append(row); by_condition[row["condition_class"]].append(row)
    if tuple(by_mode) != GEOMETRY_MODES or any(len(by_mode[mode]) != len(STEMS) for mode in GEOMETRY_MODES): raise ValueError("Scored records do not form the exact five-condition x two-geometry matrix")
    geometry_rows = []
    for mode in GEOMETRY_MODES:
        pose, compact = mode_results[mode], _compact_group(by_mode[mode])
        geometry_rows.append({"geometry_mode": mode, "generation_count": len(by_mode[mode]), "pck": pose, "clip": compact["clip"], "matched_people": pose["matched_people"], "detection_coverage": pose["detection_coverage"], "generated_person_count": pose["predicted_people"], "unmatched_predicted_people": pose["unmatched_predicted_people"]})
    condition_rows = []
    for condition in spec["conditions"]:
        values = by_condition[condition["condition_class"]]
        if len(values) != 2 or tuple(value["geometry_mode"] for value in values) != GEOMETRY_MODES: raise ValueError(f"Condition class lacks exact ordered geometry pair: {condition['condition_class']}")
        condition_rows.append({"condition_class": condition["condition_class"], "orientation": condition["orientation"], "stem": condition["stem"], "framing_geometry_comparison": values[0]["framing_geometry_comparison"], "by_geometry": [{"geometry_mode": value["geometry_mode"], **_compact_group([value]), "matched_people": value["pck"].get("matched_people"), "detection_coverage": value["pck"].get("detection_coverage"), "generated_person_count": value["pck"].get("predicted_people"), "output_dimensions": value["output_dimensions"]} for value in values]})
    return {"group_by": "geometry_mode", "rows": geometry_rows}, {"group_by": "condition_class", "rows": condition_rows}


def score(args: argparse.Namespace) -> None:
    spec, contract, _, _, geometries, candidate, _, _, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract, geometries) != "complete": raise FileNotFoundError("Native/dynamic scoring requires all 10 validated generations")
    if not args.reference_sidecar: raise ValueError("Native/dynamic PCK requires the authoritative final-val --reference-sidecar")
    sidecar, digest = _sidecar(args.reference_sidecar); existing = output / "pck_clip_results.json"
    if existing.exists():
        prior = _read_json(existing)
        if any(prior.get(key) != value for key, value in contract.items()) or prior.get("reference_sidecar_sha256") != digest or prior.get("clip_model") != args.clip_model_id: raise ValueError("Existing native/dynamic score artifact conflicts with frozen provenance")
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5); processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    per_generation, mode_results = [], {}
    conditions = {row["stem"]: row for row in spec["conditions"]}
    for mode in GEOMETRY_MODES:
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem={stem: geometries[stem][mode] for stem in STEMS}, image_for=lambda stem, current=mode: _image_path(output, stem, current, candidate), detector=detector, confidence_threshold=.5, require_images=True)
        per_image = {row["stem"]: row for row in pose["per_image"]}; mode_results[mode] = pose
        for condition in spec["conditions"]:
            stem, image = condition["stem"], _image_path(output, condition["stem"], mode, candidate)
            per_generation.append({"stem": stem, "condition_class": condition["condition_class"], "orientation": condition["orientation"], "prompt": condition["prompt"], "seed": condition["sampling_seed"], "geometry_mode": mode, "image": str(image.relative_to(output)), "output_dimensions": geometries[stem][mode]["bucket"], "output_bucket": geometries[stem][mode]["bucket"], "framing_geometry_comparison": _framing_comparison(geometries[stem][GEOMETRY_MODES[0]], geometries[stem][GEOMETRY_MODES[1]]), "pck": per_image.get(stem, {"stem": stem, "reference_available": False}), "clip_cosine_similarity": _clip_score(clip, processor, device, condition["prompt"], image)})
    # Reorder from scoring-by-mode to the frozen condition-major matrix.
    per_generation = sorted(per_generation, key=lambda row: _expected_pairs(spec).index((row["stem"], row["geometry_mode"])))
    by_geometry, by_condition = _aggregate_tables(per_generation, mode_results, spec)
    _write(existing, {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "reference_sidecar_sha256": digest, "clip_model": args.clip_model_id, "confidence_threshold": .5, "per_generation": per_generation, "mode_results": mode_results, "aggregate_by_geometry": by_geometry, "aggregate_by_condition": by_condition})
    print(existing)


def _validated_scores(output: Path, contract: Mapping[str, Any], spec: Mapping[str, Any]):
    scores = _read_json(output / "pck_clip_results.json")
    if any(scores.get(key) != value for key, value in contract.items()): raise ValueError("Native/dynamic score artifact conflicts with frozen provenance")
    return scores, _score_records(scores, spec)


def report(args: argparse.Namespace) -> None:
    spec, contract, _, _, geometries, candidate, _, training, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract, geometries) != "complete": raise FileNotFoundError("Native/dynamic report requires all 10 validated generations")
    scores, records = _validated_scores(output, contract, spec); by_geometry, by_condition = _aggregate_tables(records, scores["mode_results"], spec)
    if scores.get("aggregate_by_geometry") != by_geometry or scores.get("aggregate_by_condition") != by_condition: raise ValueError("Native/dynamic aggregate score tables do not match individual generations")
    _write(output / "metrics_by_geometry.json", {**contract, **by_geometry}); _write(output / "metrics_by_condition.json", {**contract, **by_condition})
    rows = []
    for condition in spec["conditions"]:
        paths = [output / "controls" / f"{condition['stem']}.png", *(_image_path(output, condition["stem"], mode, candidate) for mode in GEOMETRY_MODES)]
        if any(not path.is_file() for path in paths): raise FileNotFoundError(f"Native/dynamic grid requires pose control and both generations: {condition['stem']}")
        make_contact_sheet([(condition["stem"], paths)], output / "per_condition_grids" / f"{condition['stem']}.png", thumbnail_width=240, thumbnail_height=240, column_labels=("pose control", "native", "dynamic-768")); rows.append((condition["stem"], paths))
    make_contact_sheet(rows, output / "native_vs_dynamic768_contact_sheet.png", thumbnail_width=200, thumbnail_height=200, column_labels=("pose control", "native", "dynamic-768"))
    _write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "generation_count": GENERATION_COUNT, "score_artifact": "pck_clip_results.json", "metrics_by_geometry": "metrics_by_geometry.json", "metrics_by_condition": "metrics_by_condition.json", "contact_sheet": "native_vs_dynamic768_contact_sheet.png", "per_condition_grids": "per_condition_grids", "source_rgb_fallback_used": False})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    spec, contract, _, _, geometries, candidate, _, _, output = _inputs(args)
    if _generation_status(output, spec, candidate, contract, geometries) != "complete": raise FileNotFoundError("Native/dynamic summary requires all 10 validated generations")
    scores, records = _validated_scores(output, contract, spec); by_geometry, _ = _aggregate_tables(records, scores["mode_results"], spec)
    compact = {"candidate": candidate["label"], "control_scale": 1.0, "generation_count": GENERATION_COUNT, "geometry_modes": list(GEOMETRY_MODES), "frozen_spec_sha256": ABLATION_SPEC_SHA256,
               "by_geometry": [{"geometry_mode": row["geometry_mode"], "pck_005": row["pck"]["pck_005"], "pck_010": row["pck"]["pck_010"], "pck_020": row["pck"]["pck_020"], "clip_mean": row["clip"]["mean_cosine_similarity"], "matched_people": row["matched_people"], "detection_coverage": row["detection_coverage"], "generated_person_count": row["generated_person_count"]} for row in by_geometry["rows"]]}
    _write(output / "compact_summary.json", compact); print(json.dumps(compact, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("preflight", "generate", "score", "report", "summary")); parser.add_argument("--candidate", choices=("mix-025",), default="mix-025")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT)); parser.add_argument("--spec", default=str(ABLATION_SPEC)); parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC)); parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents"); parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning"); parser.add_argument("--dataset-root"); parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors"); parser.add_argument("--reference-sidecar", help="required only for score"); parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return parser


def main() -> None:
    args = parser().parse_args(); {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__": main()
