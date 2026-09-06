"""Frozen Krea-2 Turbo base-versus-Pose-Control final-val comparison.

The ``turbo-base`` candidate is deliberately built from the untouched Turbo
checkpoint and sampled through its native input projection.  It never creates
the expanded control input layer and never loads Pose-LoRA/control-adapter
state.  This evaluation-only entry point leaves ``inference.py``, training,
and historical benchmark roots untouched.
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
from safetensors.torch import load_file
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.diffusion import patchify_and_position
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, save_image
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, turbo_metadata, turbo_schedule, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val
from scripts import prompting_guide_study as guide


SPEC = Path("docs/evaluation/turbo-baseline/turbo-baseline-final-val-v1.json")
SPEC_SHA256 = "f0775c8aac404a1a4a3303ee8272d49d217d398c8379d7559d84c59288292bcc"
OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation/turbo-baseline/turbo-baseline-final-val-v1")
CANDIDATES = ("turbo-base", "parent-4000", "mix-025", "finish-control-a4300")
FINAL_COUNT = 48
GENERATION_COUNT = FINAL_COUNT * len(CANDIDATES)
RUNTIME = turbo_metadata()
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"


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
        raise FileNotFoundError(f"Required Turbo baseline JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Turbo baseline JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Turbo baseline JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_locks(spec: Mapping[str, Any]) -> None:
    if spec.get("candidate_order") != list(CANDIDATES):
        raise ValueError("Turbo baseline candidate ordering drifted")
    if spec.get("generation_count") != GENERATION_COUNT:
        raise ValueError("Turbo baseline must be exactly 48 conditions x four candidates = 192 generations")
    if spec.get("geometry") != NATIVE_GEOMETRY or spec.get("runtime") != RUNTIME:
        raise ValueError("Turbo baseline violates native geometry or locked Turbo 8-step CFG-0 mu=1.15 runtime")
    benchmark = spec.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("name") != "final_val_benchmark_48" or benchmark.get("count") != FINAL_COUNT:
        raise ValueError("Turbo baseline must use exactly the frozen final_val_benchmark_48")
    final_source = benchmark.get("final_spec")
    prompts = benchmark.get("prompt_source")
    sidecar = benchmark.get("pose_sidecar")
    if final_source != {"path": str(final_val.FINAL_SPEC), "sha256": final_val.FINAL_SPEC_SHA256}:
        raise ValueError("Turbo baseline final-val spec provenance drifted")
    if prompts != {"path": "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl", "sha256": "23d448d573a2ffd20adfd73fa88f34ebc08df280a051cb0931d9ecdcc1231ceb"}:
        raise ValueError("Turbo baseline frozen prompt provenance drifted")
    if not isinstance(sidecar, dict) or sidecar.get("path") != "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3":
        raise ValueError("Turbo baseline authoritative pose sidecar provenance drifted")
    turbo = spec.get("turbo_checkpoint")
    if turbo != {"path": "/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors", "identity_policy": "sha256_pinned_by_preflight_and_required_unchanged_for_all_later_stages"}:
        raise ValueError("Turbo baseline checkpoint identity policy drifted")
    candidates = spec.get("candidates")
    if not isinstance(candidates, list) or [row.get("id") for row in candidates if isinstance(row, dict)] != list(CANDIDATES):
        raise ValueError("Turbo baseline candidate definitions drifted")
    base, parent, mix, finish = candidates
    if base != {"id": "turbo-base", "kind": "unmodified_krea2_turbo", "pose_lora_control_adapter_state": "absent", "control_scale": None}:
        raise ValueError("turbo-base must be the unmodified Turbo model with no Pose-LoRA/control-adapter state")
    for row, expected_id, expected_hash, expected_step in (
        (parent, "parent-4000", "0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd", 4000),
        (finish, "finish-control-a4300", "17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff", 4300),
    ):
        if row.get("id") != expected_id or row.get("kind") != "pose_control_checkpoint" or row.get("sha256") != expected_hash or row.get("step") != expected_step or row.get("control_scale") != 1.0:
            raise ValueError(f"Turbo baseline {expected_id} checkpoint/control contract drifted")
    if (mix.get("id"), mix.get("kind"), mix.get("alpha"), mix.get("compute_dtype"), mix.get("formula"), mix.get("tensor_scope"), mix.get("endpoints"), mix.get("control_scale")) != (
        "mix-025", "trainable_tensor_interpolation", .25, "float32", "(1 - alpha) * parent-4000 + alpha * finish-control-a4300", "state['model'] trainable control/LoRA tensors only", ["parent-4000", "finish-control-a4300"], 1.0):
        raise ValueError("Turbo baseline mix-025 interpolation contract drifted")


def _validate_historical_artifacts(spec: Mapping[str, Any]) -> None:
    for relative, expected in spec["historical_artifact_sha256"].items():
        if _sha256(Path(relative)) != expected:
            raise ValueError(f"Historical artifact changed; Turbo baseline refuses to proceed: {relative}")
    benchmark = spec["benchmark"]
    expected = {
        benchmark["final_spec"]["path"]: benchmark["final_spec"]["sha256"],
        benchmark["prompt_source"]["path"]: benchmark["prompt_source"]["sha256"],
        f"{benchmark['pose_sidecar']['path']}/metadata.json": benchmark["pose_sidecar"]["metadata_sha256"],
        f"{benchmark['pose_sidecar']['path']}/records.jsonl": benchmark["pose_sidecar"]["records_sha256"],
    }
    for relative, digest in expected.items():
        if _sha256(Path(relative)) != digest:
            raise ValueError(f"Frozen final-val artifact changed; Turbo baseline refuses to proceed: {relative}")


def load_spec(path: str | Path = SPEC) -> dict[str, Any]:
    source = Path(path)
    if _sha256(source) != SPEC_SHA256:
        raise ValueError(f"Frozen Turbo baseline spec SHA-256 mismatch: {source}")
    spec = _read_json(source)
    _validate_locks(spec)
    _validate_historical_artifacts(spec)
    return spec


def _prompt_rows(path: Path, stems: list[str]) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mapped = {str(row.get("stem")): row for row in rows}
    if len(rows) != FINAL_COUNT or list(mapped) != stems or any(not isinstance(mapped[stem].get("text"), str) or not mapped[stem]["text"].strip() for stem in stems):
        raise ValueError("Frozen source-caption prompt order/content drifted from final-val benchmark")
    return mapped


def _expected_pairs(stems: list[str]) -> list[tuple[str, str]]:
    return [(stem, candidate) for stem in stems for candidate in CANDIDATES]


def _candidate_config(spec: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    candidate = next((dict(row) for row in spec["candidates"] if row["id"] == candidate_id), None)
    if candidate is None:
        raise ValueError(f"Unknown frozen Turbo baseline candidate: {candidate_id}")
    return candidate


def _resolve_controlled_candidate(candidate_id: str) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    if candidate_id not in {"parent-4000", "mix-025", "finish-control-a4300"}:
        raise ValueError("turbo-base has no Pose-LoRA/control adapter state to resolve")
    return final_val.resolve_candidate(candidate_id)


def _inputs(args: argparse.Namespace):
    spec = load_spec(args.spec)
    if args.turbo_ckpt != spec["turbo_checkpoint"]["path"] or args.clip_model_id != spec["clip_model_id"]:
        raise ValueError("Turbo baseline refuses Turbo checkpoint or CLIP metric provenance override")
    final_spec, final_digest = final_val.load_final_spec(args.final_spec)
    if final_digest != spec["benchmark"]["final_spec"]["sha256"]:
        raise ValueError("Frozen final-val spec/hash drifted")
    stems = list(final_spec["stems"])
    prompts = _prompt_rows(Path(spec["benchmark"]["prompt_source"]["path"]), stems)
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, final_spec)
    shards = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shards.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Turbo baseline requires --dataset-root or latent shards.json.dataset_root")
    controls = final_val.resolve_final_controls(dataset_root, stems)
    conditions, geometries = [], {}
    for stem in stems:
        sample = _sample_by_stem(dataset, stem)
        geometry = turbo_scoring_geometry(sample)
        control = controls[stem]
        with Image.open(control) as image:
            if image.size[0] < 1 or image.size[1] < 1:
                raise ValueError(f"Invalid pose control geometry: {stem}")
        conditions.append({"stem": stem, "prompt": prompts[stem]["text"], "seed": final_spec["per_stem_seeds"][stem]["sampling"],
                           "source": prompts[stem].get("source"), "orientation": prompts[stem].get("orientation"),
                           "control_sha256": _sha256(control), "control_source_size": list(Image.open(control).size), "geometry": geometry})
        geometries[stem] = geometry
    if len(conditions) != FINAL_COUNT or len({row["stem"] for row in conditions}) != FINAL_COUNT:
        raise ValueError("Turbo baseline did not resolve exactly 48 final-val conditions")
    turbo_hash = _sha256(Path(args.turbo_ckpt))
    candidates: dict[str, tuple[dict[str, Any] | None, Path | None, dict[str, Any]]] = {"turbo-base": (None, None, {"candidate_kind": "unmodified_krea2_turbo", "pose_lora_control_adapter_state": "absent"})}
    for candidate_id in CANDIDATES[1:]:
        candidates[candidate_id] = _resolve_controlled_candidate(candidate_id)
    contract = {
        "kind": spec["kind"], "frozen_spec": str(Path(args.spec)), "frozen_spec_sha256": SPEC_SHA256,
        "candidate_order": list(CANDIDATES), "generation_count": GENERATION_COUNT, "runtime": spec["runtime"],
        "geometry": NATIVE_GEOMETRY, "final_val_spec_sha256": final_digest, "prompt_source": spec["benchmark"]["prompt_source"],
        "pose_sidecar": spec["benchmark"]["pose_sidecar"], "clip_model_id": spec["clip_model_id"],
        "turbo_checkpoint": {"path": args.turbo_ckpt, "sha256": turbo_hash}, "conditions": conditions,
        "candidate_contracts": {candidate_id: _candidate_config(spec, candidate_id) for candidate_id in CANDIDATES},
        "historical_artifact_sha256": spec["historical_artifact_sha256"],
    }
    output = Path(args.output_root)
    provenance = output / "turbo_baseline_provenance.json"
    if provenance.exists() and _read_json(provenance) != contract:
        raise ValueError(f"Existing output has conflicting immutable Turbo baseline provenance: {provenance}")
    if not provenance.exists():
        if getattr(args, "action", None) != "preflight":
            raise FileNotFoundError("Turbo baseline requires a successful preflight to pin Turbo checkpoint identity before later stages")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Turbo baseline preflight refuses a non-empty output root without immutable provenance")
        _write(provenance, contract)
    return spec, contract, final_spec, dataset, controls, geometries, candidates, output


def _directory(output: Path, stem: str) -> Path:
    return output / "generations" / stem


def _image_path(output: Path, stem: str, candidate: str) -> Path:
    return _directory(output, stem) / f"{candidate}.png"


def _metadata_path(output: Path, stem: str, candidate: str) -> Path:
    return _directory(output, stem) / f"{candidate}.json"


def _copy_control(source: Path, target: Path, expected_hash: str) -> None:
    if _sha256(source) != expected_hash:
        raise ValueError(f"Authoritative control hash changed: {source}")
    if target.exists() and _sha256(target) != expected_hash:
        raise ValueError(f"Existing pose control conflicts with authoritative final-val control: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _generation_metadata(condition: Mapping[str, Any], candidate: str, contract: Mapping[str, Any], control: Path) -> dict[str, Any]:
    base = {"stem": condition["stem"], "candidate": candidate, "candidate_contract": contract["candidate_contracts"][candidate],
            "prompt": condition["prompt"], "seed": condition["seed"], "source": condition["source"], "orientation": condition["orientation"],
            "control_path": str(control), "control_sha256": condition["control_sha256"], "geometry": condition["geometry"],
            "output_dimensions": condition["geometry"]["bucket"], "frozen_spec_sha256": SPEC_SHA256,
            "final_val_spec_sha256": contract["final_val_spec_sha256"], "runtime": contract["runtime"], "turbo_checkpoint": contract["turbo_checkpoint"]}
    if candidate == "turbo-base":
        return {**base, "control_applied": False, "pose_lora_control_adapter_loaded": False, "control_scale": None}
    return {**base, "control_applied": True, "pose_lora_control_adapter_loaded": True, "control_scale": 1.0}


def _generation_status(output: Path, contract: Mapping[str, Any]) -> str:
    payload_path = output / "generation_results.json"; payload = _read_json(payload_path) if payload_path.exists() else None
    conditions = {row["stem"]: row for row in contract["conditions"]}; observed = []
    for stem, candidate in _expected_pairs([row["stem"] for row in contract["conditions"]]):
        image, metadata_path = _image_path(output, stem, candidate), _metadata_path(output, stem, candidate)
        control = output / "controls" / f"{stem}.png"
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    if list(opened.size) != conditions[stem]["geometry"]["bucket"]:
                        raise ValueError("output dimensions differ from frozen native geometry")
                    opened.verify()
                expected = _generation_metadata(conditions[stem], candidate, contract, control)
                metadata = _read_json(metadata_path)
                if any(metadata.get(key) != value for key, value in expected.items() if key != "control_path") or not isinstance(metadata.get("control_path"), str):
                    raise ValueError("generation metadata contract mismatch")
                if _sha256(control) != conditions[stem]["control_sha256"]:
                    raise ValueError("generation control hash mismatch")
            except Exception as exc:
                raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if _directory(output, stem).exists() and (metadata_path.exists() or image.exists()):
                raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
    expected_artifacts = {stem: {candidate: str(_image_path(output, stem, candidate).relative_to(output)) for candidate in CANDIDATES} for stem in conditions}
    if not any(observed) and payload is None:
        if (output / "generations").exists() or (output / "controls").exists():
            raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
        return "missing"
    if all(observed) and payload is not None and all(payload.get(key) == value for key, value in contract.items()) and payload.get("generated_artifacts") == expected_artifacts:
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


def build_unmodified_turbo_base_model(turbo_ckpt: str, device: str):
    """Load only official Turbo tensors into the native, non-control MMDiT."""
    from k2_lora import K2_RAW_CONFIG, inspect_krea_checkpoint
    from mmdit import SingleStreamDiT
    report = inspect_krea_checkpoint(turbo_ckpt, checkpoint_name="Turbo")
    with torch.device("meta"):
        model = SingleStreamDiT(K2_RAW_CONFIG)
    incompatible = model.load_state_dict(load_file(turbo_ckpt), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Unmodified Turbo strict checkpoint load failed")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False); model._krea_checkpoint_report = report
    if model.first.in_features != model.config.channels * model.config.patch ** 2 or any(".A" in key or ".B" in key for key in model.state_dict()):
        raise AssertionError("turbo-base unexpectedly has Pose-LoRA/control-adapter state")
    return model


@torch.inference_mode()
def sample_unmodified_turbo_base_image(model: Any, vae_decode, sample: Mapping[str, Any], device: torch.device, seed: int):
    """Official-style Turbo denoising with no control latent or adapter path."""
    latent = sample["latent"][None].to(device)
    noise = torch.randn(latent.shape, device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(seed))
    text, text_mask = sample["context"][None].to(device=device, dtype=torch.bfloat16), sample["mask"][None].to(device=device, dtype=torch.bool)
    image, pos, mask = patchify_and_position(noise, text.shape[1], model.config.patch, text_mask)
    for current, previous in zip(turbo_schedule(image_sequence_length=image.shape[1], steps=8, mu=1.15)[:-1], turbo_schedule(image_sequence_length=image.shape[1], steps=8, mu=1.15)[1:]):
        timestep = torch.full((1,), current, dtype=image.dtype, device=device)
        image = image + (previous - current) * model(image, text, timestep, pos, mask)
    height, width = latent.shape[-2:]; patch = model.config.patch
    decoded = rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=height // patch, w=width // patch)
    pixels = vae_decode(decoded.to(torch.bfloat16))
    return ((pixels.clamp(-1, 1) * .5 + .5) * 255.0)[0].permute(1, 2, 0).float().cpu().byte().numpy()


def preflight(args: argparse.Namespace) -> None:
    _, contract, _, dataset, controls, geometries, candidates, output = _inputs(args)
    resolved = {candidate: ("unmodified Turbo; no Pose-LoRA/control adapter" if candidate == "turbo-base" else candidates[candidate][2]) for candidate in CANDIDATES}
    _write(output / "checkpoint_preflight.json", {**contract, "dataset_sample_count": len(dataset), "control_paths_resolved": {stem: str(path) for stem, path in controls.items()}, "geometry_by_stem": geometries, "resolved_candidates": resolved, "generation_plan": _expected_pairs([row["stem"] for row in contract["conditions"]])})
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run Turbo baseline generation from the GH200 host shell with CUDA visible")
    _, contract, _, dataset, controls, _, candidates, output = _inputs(args)
    if _generation_status(output, contract) == "complete":
        print(json.dumps({"already_complete": True, "generation_count": GENERATION_COUNT})); return
    vae = load_krea_vae("cuda"); conditioner = guide.PoseTextConditioner(device="cuda", dtype=torch.bfloat16); device = torch.device("cuda")
    base_model = None; control_model = None; base_report = None; compatibility: dict[str, Any] = {}
    for candidate_id in CANDIDATES:
        if candidate_id == "turbo-base":
            base_model = build_unmodified_turbo_base_model(args.turbo_ckpt, "cuda")
            base_report = getattr(base_model, "_krea_checkpoint_report", None)
        else:
            if control_model is None:
                control_model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
            candidate, checkpoint, _ = candidates[candidate_id]
            assert candidate is not None
            state = final_val.candidate_trainable_state(candidate, checkpoint)
            final_val.load_trainable_state_dict(control_model, state)
            compatibility[candidate_id] = raw_to_turbo_control_compatibility(control_model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, state))
        for condition in contract["conditions"]:
            stem = condition["stem"]; directory = _directory(output, stem); image = _image_path(output, stem, candidate_id); metadata_path = _metadata_path(output, stem, candidate_id)
            _copy_control(controls[stem], output / "controls" / f"{stem}.png", condition["control_sha256"])
            metadata = _generation_metadata(condition, candidate_id, contract, controls[stem])
            if metadata_path.exists() and _read_json(metadata_path) != metadata:
                raise ValueError(f"Existing generation metadata conflicts with frozen Turbo baseline: {metadata_path}")
            if image.exists():
                continue
            if not metadata_path.exists(): _write(metadata_path, metadata)
            sample = dict(_sample_by_stem(dataset, stem)); context, mask = guide._conditioning(conditioner, condition["prompt"], sample); sample.update({"context": context, "mask": mask})
            if candidate_id == "turbo-base":
                assert base_model is not None
                pixels = sample_unmodified_turbo_base_image(base_model, lambda latent: decode_normalized_latents(vae, latent), sample, device, condition["seed"])
            else:
                from pose_controlnet.turbo_evaluation import sample_turbo_pose_image
                assert control_model is not None
                pixels = sample_turbo_pose_image(control_model, lambda latent: decode_normalized_latents(vae, latent), sample, device, condition["seed"], steps=8, guidance=0.0, mu=1.15, control_scale=1.0)
            save_image(pixels, image)
        if candidate_id == "turbo-base":
            del base_model; base_model = None; torch.cuda.empty_cache()
    artifacts = {row["stem"]: {candidate: str(_image_path(output, row["stem"], candidate).relative_to(output)) for candidate in CANDIDATES} for row in contract["conditions"]}
    _write(output / "generation_results.json", {**contract, "generated_artifacts": artifacts, "turbo_base_checkpoint_report": base_report, "turbo_base_load": {"model_surgery": "none", "pose_lora_control_adapter_loaded": False}, "raw_to_turbo_control_compatibility": compatibility, "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _sidecar(path: str | Path, stems: list[str]) -> tuple[dict[str, Any], str]:
    sidecar, digest = final_val._load_final_sidecar(path, stems)
    return sidecar, digest


def _score_records(payload: Mapping[str, Any], stems: list[str]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    if not isinstance(records, list) or len(records) != GENERATION_COUNT or [(row.get("stem"), row.get("candidate")) for row in records] != _expected_pairs(stems):
        raise ValueError("Turbo baseline score artifact is incomplete or not in frozen condition/candidate order")
    return records


def _compact(records: list[dict[str, Any]]) -> dict[str, Any]:
    return guide._compact_group(records)


def _aggregate_tables(records: list[dict[str, Any]], candidate_results: Mapping[str, Any], contract: Mapping[str, Any]):
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records: by_candidate[row["candidate"]].append(row)
    if tuple(by_candidate) != CANDIDATES or any(len(by_candidate[candidate]) != FINAL_COUNT for candidate in CANDIDATES):
        raise ValueError("Scored records do not form the exact 48-condition x four-candidate matrix")
    rows = []
    for candidate in CANDIDATES:
        pose, clip = candidate_results[candidate], _compact(by_candidate[candidate])["clip"]
        rows.append({"candidate": candidate, "generation_count": FINAL_COUNT, "pck": pose, "clip": clip,
                     "matched_people": pose["matched_people"], "detection_coverage": pose["detection_coverage"], "generated_person_count": pose["predicted_people"],
                     "unmatched_reference_people": pose["unmatched_reference_people"], "unmatched_predicted_people": pose["unmatched_predicted_people"],
                     "single_person": pose.get("single_person"), "multi_person": pose.get("multi_person"), "per_source": pose.get("per_source")})
    by_group, by_source = {}, {}
    for group in ("single-person", "multi-person"):
        by_group[group] = {candidate: _compact([row for row in by_candidate[candidate] if row["pck"].get("person_group") == group]) for candidate in CANDIDATES}
    for source in ("coco", "humanart"):
        by_source[source] = {candidate: _compact([row for row in by_candidate[candidate] if row["pck"].get("source") == source]) for candidate in CANDIDATES}
    return {"group_by": "candidate", "rows": rows}, {"group_by": "person_group", "rows": by_group}, {"group_by": "source", "rows": by_source}


def score(args: argparse.Namespace) -> None:
    _, contract, final_spec, dataset, _, geometries, _, output = _inputs(args)
    if _generation_status(output, contract) != "complete":
        raise FileNotFoundError("Turbo baseline scoring requires all 192 validated generations")
    if not args.reference_sidecar:
        raise ValueError("Turbo baseline PCK requires the immutable final-val --reference-sidecar")
    stems = list(final_spec["stems"]); sidecar, digest = _sidecar(args.reference_sidecar, stems); device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, .5); processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    conditions = {row["stem"]: row for row in contract["conditions"]}; per_generation, candidate_results = [], {}
    for candidate in CANDIDATES:
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometries, image_for=lambda stem, current=candidate: _image_path(output, stem, current), detector=detector, confidence_threshold=.5, require_images=True)
        candidate_results[candidate] = pose; per_image = {row["stem"]: row for row in pose["per_image"]}
        for stem in stems:
            condition = conditions[stem]; image = _image_path(output, stem, candidate)
            per_generation.append({"stem": stem, "candidate": candidate, "prompt": condition["prompt"], "seed": condition["seed"], "source": condition["source"], "orientation": condition["orientation"], "geometry": geometries[stem], "image": str(image.relative_to(output)), "pck": per_image[stem], "clip_cosine_similarity": _clip_score(clip, processor, device, condition["prompt"], image)})
    per_generation.sort(key=lambda row: _expected_pairs(stems).index((row["stem"], row["candidate"])))
    by_candidate, by_group, by_source = _aggregate_tables(per_generation, candidate_results, contract)
    _write(output / "pck_clip_results.json", {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "reference_sidecar_sha256": digest, "clip_model": args.clip_model_id, "confidence_threshold": .5, "per_generation": per_generation, "candidate_results": candidate_results, "aggregate_by_candidate": by_candidate, "aggregate_by_person_group": by_group, "aggregate_by_source": by_source})
    print(output / "pck_clip_results.json")


def _validated_scores(output: Path, contract: Mapping[str, Any], stems: list[str]):
    scores = _read_json(output / "pck_clip_results.json")
    if any(scores.get(key) != value for key, value in contract.items()):
        raise ValueError("Turbo baseline score artifact conflicts with frozen provenance")
    return scores, _score_records(scores, stems)


def report(args: argparse.Namespace) -> None:
    _, contract, final_spec, _, _, _, _, output = _inputs(args); stems = list(final_spec["stems"])
    if _generation_status(output, contract) != "complete": raise FileNotFoundError("Turbo baseline report requires all 192 validated generations")
    scores, records = _validated_scores(output, contract, stems); by_candidate, by_group, by_source = _aggregate_tables(records, scores["candidate_results"], contract)
    if scores.get("aggregate_by_candidate") != by_candidate or scores.get("aggregate_by_person_group") != by_group or scores.get("aggregate_by_source") != by_source:
        raise ValueError("Turbo baseline aggregate tables do not match individual generations")
    _write(output / "metrics_by_candidate.json", {**contract, **by_candidate, "aggregate_by_person_group": by_group, "aggregate_by_source": by_source})
    all_rows = []
    for stem in stems:
        paths = [output / "controls" / f"{stem}.png", *(_image_path(output, stem, candidate) for candidate in CANDIDATES)]
        if any(not path.is_file() for path in paths): raise FileNotFoundError(f"Turbo baseline contact sheet is incomplete: {stem}")
        all_rows.append((stem, paths))
    make_contact_sheet(all_rows, output / "turbo_baseline_contact_sheet.png", thumbnail_width=180, thumbnail_height=180, column_labels=("pose control", *CANDIDATES))
    for candidate in CANDIDATES:
        make_contact_sheet([(stem, [output / "controls" / f"{stem}.png", _image_path(output, stem, candidate)]) for stem in stems], output / f"{candidate}_contact_sheet.png", thumbnail_width=220, thumbnail_height=220, column_labels=("pose control", candidate))
    _write(output / "evaluation_summary.json", {**contract, "generation_count": GENERATION_COUNT, "score_artifact": "pck_clip_results.json", "metrics_by_candidate": "metrics_by_candidate.json", "contact_sheet": "turbo_baseline_contact_sheet.png", "candidate_contact_sheets": {candidate: f"{candidate}_contact_sheet.png" for candidate in CANDIDATES}})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    _, contract, final_spec, _, _, _, _, output = _inputs(args); stems = list(final_spec["stems"])
    if _generation_status(output, contract) != "complete": raise FileNotFoundError("Turbo baseline summary requires all 192 validated generations")
    scores, records = _validated_scores(output, contract, stems); by_candidate, _, _ = _aggregate_tables(records, scores["candidate_results"], contract)
    compact = {"frozen_spec_sha256": SPEC_SHA256, "generation_count": GENERATION_COUNT, "candidate_order": list(CANDIDATES), "runtime": RUNTIME, "rows": [{"candidate": row["candidate"], "pck_005": row["pck"]["pck_005"], "pck_010": row["pck"]["pck_010"], "pck_020": row["pck"]["pck_020"], "clip_mean": row["clip"]["mean_cosine_similarity"], "matched_people": row["matched_people"], "detection_coverage": row["detection_coverage"], "generated_person_count": row["generated_person_count"], "unmatched_reference_people": row["unmatched_reference_people"], "unmatched_predicted_people": row["unmatched_predicted_people"]} for row in by_candidate["rows"]]}
    _write(output / "compact_summary.json", compact); print(json.dumps(compact, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "generate", "score", "report", "summary")); parser.add_argument("--output-root", default=str(OUTPUT_ROOT)); parser.add_argument("--spec", default=str(SPEC)); parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC)); parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents"); parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning"); parser.add_argument("--dataset-root"); parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors"); parser.add_argument("--reference-sidecar"); parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return parser


def main() -> None:
    args = parser().parse_args(); {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__":
    main()
