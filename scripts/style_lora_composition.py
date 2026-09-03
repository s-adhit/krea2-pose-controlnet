"""Isolated Krea-2 Turbo Pose-LoRA + one Style-LoRA composition matrix.

This is evaluation-only.  It does not import or modify ``inference.py`` and
never merges Style-LoRA tensors into the mixed Pose-LoRA checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from pose_controlnet.style_lora import STYLE_LORA_SPECS, StyleLoRAAdapter, audit_style_lora, sha256
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val
from scripts import prompting_guide_study as guide


SPEC_FILE = Path("docs/evaluation/style-lora-composition/style_lora_composition_v1.jsonl")
SPEC_SHA256 = "cf3ac68a5500b5ab2938349b8eb74db1a6f711c9ee7f49c97e629beeccab52cb"
KIND = "isolated_turbo_pose_lora_plus_single_style_lora_v1"
POSE_CANDIDATE = "mix-025"
STYLE_ORDER = ("pose-only", "darkbrush", "rainywindow", "retroanime", "realism")
EXPECTED_CONDITIONS = ("simple_single", "dynamic_airborne", "inversion", "multi_person")
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required Style-LoRA composition artifact is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Style-LoRA composition JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Style-LoRA composition JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact JSON value used for immutable-provenance comparison.

    Audit dataclasses intentionally contain tuples (for example ``errors``),
    while JSON represents those as lists.  Comparing a live Python structure to
    a deserialized JSON artifact therefore creates false provenance drift.
    Canonicalization is also the boundary between the immutable experiment
    identity and stage-local artifact state.
    """
    try:
        result = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("Style-LoRA immutable provenance is not canonical JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Style-LoRA immutable provenance must be a JSON object")
    return result


def _provenance_sha256(provenance: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical_json(provenance), allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stage_payload(provenance: Mapping[str, Any], **stage_state: Any) -> dict[str, Any]:
    """Attach mutable stage output without changing immutable experiment identity."""
    return {"immutable_provenance": _canonical_json(provenance), **stage_state}


def _validate_or_write_provenance(output: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _canonical_json(provenance)
    path = output / "style_lora_provenance.json"
    if path.exists() and _read_json(path) != canonical:
        raise ValueError(f"Existing output has conflicting immutable Style-LoRA provenance: {path}")
    if not path.exists():
        _write(path, canonical)
    return canonical


def load_rows(path: str | Path = SPEC_FILE) -> list[dict[str, Any]]:
    source = Path(path)
    if sha256(source) != SPEC_SHA256:
        raise ValueError(f"Frozen Style-LoRA composition spec SHA-256 mismatch: {source}")
    try:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Frozen Style-LoRA composition spec is missing: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Frozen Style-LoRA composition spec contains invalid JSON: {source}") from exc
    fields = {"condition_id", "stem", "prompt", "style_ids", "style_trigger", "style_hashes"}
    if len(rows) != 4 or any(not isinstance(row, dict) or set(row) != fields for row in rows):
        raise ValueError("Style-LoRA composition spec must contain exactly four exact-schema rows")
    if tuple(row["condition_id"] for row in rows) != EXPECTED_CONDITIONS:
        raise ValueError("Style-LoRA composition spec condition order drifted")
    stems = [row["stem"] for row in rows]
    if len(stems) != len(set(stems)) or any(not isinstance(row["prompt"], str) or not row["prompt"].strip() for row in rows):
        raise ValueError("Style-LoRA composition spec has duplicate stems or empty prompts")
    expected_hashes = {style: spec["sha256"] for style, spec in STYLE_LORA_SPECS.items()}
    for row in rows:
        if tuple(row["style_ids"]) != STYLE_ORDER or row["style_trigger"] != "" or row["style_hashes"] != expected_hashes:
            raise ValueError("Style-LoRA composition spec has unsupported variants, trigger wording, or hashes")
    return rows


def _audits() -> dict[str, dict[str, Any]]:
    audits = {style: audit_style_lora(style).json() for style in STYLE_LORA_SPECS}
    failures = [style for style in ("darkbrush", "rainywindow", "retroanime") if not audits[style]["supported"]]
    if failures:
        raise ValueError(f"Official Krea Style-LoRA audit failed closed: {failures}")
    return audits


def _write_audits(output: Path, audits: Mapping[str, Mapping[str, Any]]) -> None:
    for style, audit in audits.items():
        _write(output / "audits" / f"{style}.json", audit)


def _variants(audits: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    variants = ["pose-only", "darkbrush", "rainywindow", "retroanime"]
    if audits["realism"].get("supported"):
        variants.append("realism")
    return tuple(variants)


def _candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    return guide._candidate_contract(candidate, checkpoint)


def _immutable_candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    """Keep pinned candidate identity, excluding host-specific checkpoint paths."""
    contract = _candidate_contract(candidate, checkpoint)
    interpolation = contract.get("checkpoint_interpolation")
    if not isinstance(interpolation, Mapping):
        return contract
    endpoints = interpolation.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("Style-LoRA interpolation lacks exactly two pinned endpoints")
    immutable_endpoints = []
    for endpoint in endpoints:
        if not isinstance(endpoint, Mapping):
            raise ValueError("Style-LoRA interpolation endpoint provenance is invalid")
        immutable_endpoints.append({key: endpoint.get(key) for key in ("candidate", "step", "sha256")})
    immutable_interpolation = {
        key: interpolation.get(key)
        for key in ("candidate_id", "alpha", "formula", "tensor_scope", "compute_dtype")
    }
    immutable_interpolation["endpoints"] = immutable_endpoints
    return {"candidate_kind": contract.get("candidate_kind"), "checkpoint_step": contract.get("checkpoint_step"),
            "checkpoint_interpolation": immutable_interpolation}


def _immutable_style_loras(audits: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select the adapter identity/mapping/scaling fields that must not drift."""
    fields = ("style_id", "sha256", "namespace", "supported", "tensor_count", "target_count", "rank", "dtype", "mapping", "scaling_rule")
    result: dict[str, dict[str, Any]] = {}
    for style in STYLE_LORA_SPECS:
        audit = audits.get(style)
        if not isinstance(audit, Mapping):
            raise ValueError(f"Style-LoRA audit is missing: {style}")
        result[style] = {field: audit.get(field) for field in fields}
    return result


def _immutable_provenance(*, rows: list[dict[str, Any]], variants: tuple[str, ...], seed_by_stem: Mapping[str, int],
                          control_hashes: Mapping[str, str], bucket_by_stem: Mapping[str, list[int]],
                          style_strength: float, audits: Mapping[str, Mapping[str, Any]], final_digest: str,
                          candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    if not math.isfinite(style_strength):
        raise ValueError(f"Style strength must be finite, got {style_strength!r}")
    return _canonical_json({
        "kind": KIND,
        "candidate": POSE_CANDIDATE,
        "frozen_spec": {"sha256": SPEC_SHA256, "record_count": len(rows)},
        "conditions": rows,
        "variants": list(variants),
        "sampling_seeds": dict(seed_by_stem),
        "control_sha256": dict(control_hashes),
        "buckets": dict(bucket_by_stem),
        "style_strength": style_strength,
        "style_loras": _immutable_style_loras(audits),
        "turbo": guide.TURBO,
        "geometry": NATIVE_GEOMETRY,
        "final_val_spec_sha256": final_digest,
        "source_rgb_fallback_permitted": False,
        **_immutable_candidate_contract(candidate, checkpoint),
    })


def _validate_runtime() -> None:
    guide._validate_locked_runtime_contract()
    if NATIVE_GEOMETRY != guide.NATIVE_GEOMETRY:
        raise ValueError("Style-LoRA composition violates native/aspect-preserving geometry")


def _inputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], PreparedLatentShardDataset, dict[str, Path], dict[str, Any], Path | None, dict[str, Any], Path, dict[str, dict[str, Any]]]:
    _validate_runtime()
    output = Path(args.output_root)
    audits = _audits()
    variants = _variants(audits)
    spec, final_digest = final_val.load_final_spec(args.final_spec)
    stems = [row["stem"] for row in rows]
    if any(stem not in spec["stems"] for stem in stems):
        raise ValueError("Every Style-LoRA composition stem must be a frozen final-val stem")
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, spec)
    shards = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shards.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Style-LoRA composition requires --dataset-root or latent shards.json.dataset_root")
    controls = final_val.resolve_final_controls(dataset_root, stems)
    candidate, checkpoint, training = final_val.resolve_candidate(args.candidate)
    if candidate.get("label") != POSE_CANDIDATE:
        raise ValueError("Style-LoRA composition is locked to mix-025")
    endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
    final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    seed_by_stem = {stem: int(spec["per_stem_seeds"][stem]["sampling"]) for stem in stems}
    control_hashes = {stem: sha256(path) for stem, path in controls.items()}
    bucket_by_stem = {stem: [int(_sample_by_stem(dataset, stem)["latent"].shape[-1] * 8),
                             int(_sample_by_stem(dataset, stem)["latent"].shape[-2] * 8)] for stem in stems}
    contract = _immutable_provenance(
        rows=rows, variants=variants, seed_by_stem=seed_by_stem, control_hashes=control_hashes,
        bucket_by_stem=bucket_by_stem, style_strength=float(args.style_strength), audits=audits,
        final_digest=final_digest, candidate=candidate, checkpoint=checkpoint,
    )
    _validate_or_write_provenance(output, contract)
    return contract, dataset, controls, candidate, checkpoint, training, output, audits


def _directory(output: Path, stem: str) -> Path:
    return output / "generations" / stem


def _image_name(style: str) -> str:
    return f"{style}.png"


def _copy_control(source: Path, target: Path) -> None:
    if target.exists() and sha256(target) != sha256(source):
        raise ValueError(f"Existing pose control conflicts with authoritative final-val control: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _metadata(row: Mapping[str, Any], style: str, contract: Mapping[str, Any], control: Path) -> dict[str, Any]:
    result = {"condition_id": row["condition_id"], "stem": row["stem"], "prompt": row["prompt"],
              "style_id": style, "style_strength": contract["style_strength"] if style != "pose-only" else 0.0,
              "style_trigger": row["style_trigger"], "seed": contract["sampling_seeds"][row["stem"]],
              "control_path": str(control), "control_sha256": contract["control_sha256"][row["stem"]], "bucket": contract["buckets"][row["stem"]],
              "geometry": NATIVE_GEOMETRY, "frozen_spec_sha256": SPEC_SHA256, "candidate": POSE_CANDIDATE,
              "source_rgb_fallback_used": False, **contract["turbo"],
              **{key: contract[key] for key in ("candidate_kind", "checkpoint_step", "checkpoint_interpolation", "checkpoint_sha256") if key in contract}}
    if style != "pose-only":
        audit = contract["style_loras"][style]
        result["style_lora"] = {"sha256": audit["sha256"], "mapping_target_count": audit["target_count"],
                                "scaling_rule": audit["scaling_rule"]}
    return result


def _generation_status(output: Path, rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> str:
    payload_path = output / "generation_results.json"; payload = _read_json(payload_path) if payload_path.exists() else None
    observed, recorded = [], []
    for row in rows:
        control = output / "controls" / f"{row['stem']}.png"
        for style in contract["variants"]:
            directory = _directory(output, row["stem"]); image = directory / _image_name(style); metadata_path = directory / f"{style}.json"
            if image.is_file():
                try:
                    with Image.open(image) as opened: opened.verify()
                    with Image.open(control) as opened: opened.verify()
                    metadata = _read_json(metadata_path)
                    expected = _metadata(row, style, contract, control)
                    if metadata != expected:
                        raise ValueError("generation metadata contract mismatch")
                except Exception as exc:
                    raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
                observed.append(True)
            else:
                if directory.exists() or control.exists():
                    raise ValueError("Existing Style-LoRA generation output is incomplete; refusing to overwrite")
                observed.append(False)
            if payload is not None:
                recorded.append(payload.get("generated_artifacts", {}).get(row["stem"], {}).get(style) == str(image.relative_to(output)))
    if not any(observed) and payload is None:
        return "missing"
    if (all(observed) and payload is not None and all(recorded)
            and payload.get("immutable_provenance") == contract):
        return "complete"
    raise ValueError("Existing Style-LoRA generation output is incomplete or inconsistent; refusing to overwrite")


def audit(args: argparse.Namespace) -> None:
    rows = load_rows(args.prompt_file)
    contract, _, _, _, _, _, output, audits = _inputs(args, rows)
    _write_audits(output, audits)
    _write(output / "style_lora_audit_summary.json", _stage_payload(
        contract, style_lora_audit_details=audits, variants_if_preflight_passes=list(_variants(audits)), rows=len(rows),
    ))
    print(output / "style_lora_audit_summary.json")


def preflight(args: argparse.Namespace) -> None:
    rows = load_rows(args.prompt_file)
    contract, dataset, controls, _, _, training, output, _ = _inputs(args, rows)
    _write(output / "checkpoint_preflight.json", _stage_payload(
        contract, dataset_sample_count=len(dataset), control_paths_resolved={stem: str(path) for stem, path in controls.items()},
        training_metadata=training, generation_count=len(rows) * len(contract["variants"]),
        text_conditioning="online frozen semantic prompts; source captions are never sampled",
    ))
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run Style-LoRA generation from the GH200 host shell with CUDA visible")
    rows = load_rows(args.prompt_file)
    contract, dataset, controls, candidate, checkpoint, training, output, audits = _inputs(args, rows)
    if _generation_status(output, rows, contract) == "complete":
        print(json.dumps({"already_complete": POSE_CANDIDATE, "generation_count": len(rows) * len(contract["variants"])})); return
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint)
    final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    vae, conditioner = load_krea_vae("cuda"), guide.PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    adapters: dict[str, StyleLoRAAdapter] = {}
    artifacts: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        sample = dict(_sample_by_stem(dataset, row["stem"])); bucket = [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8]
        control_output = output / "controls" / f"{row['stem']}.png"; _copy_control(controls[row["stem"]], control_output)
        context, mask = guide._conditioning(conditioner, row["prompt"], sample)
        generated_sample = dict(sample, prompt=row["prompt"], context=context, mask=mask)
        for style in contract["variants"]:
            directory = _directory(output, row["stem"]); metadata = _metadata(row, style, contract, control_output)
            metadata_path = directory / f"{style}.json"
            if metadata_path.exists() and _read_json(metadata_path) != metadata:
                raise ValueError(f"Existing Style-LoRA metadata conflicts with frozen contract: {metadata_path}")
            if not metadata_path.exists(): _write(metadata_path, metadata)
            if style == "pose-only":
                pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), generated_sample, torch.device("cuda"), metadata["seed"], control_scale=1.0)
            else:
                if style not in adapters:
                    adapters[style] = StyleLoRAAdapter.load(audit_style_lora(style), device="cuda")
                from pose_controlnet.style_lora import applied_style_lora
                # A fresh scope for every image prevents persistent or double application.
                with applied_style_lora(model, adapters[style], metadata["style_strength"]):
                    pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), generated_sample, torch.device("cuda"), metadata["seed"], control_scale=1.0)
            save_image(pixels, directory / _image_name(style)); artifacts[row["stem"]][style] = str((_directory(output, row["stem"]) / _image_name(style)).relative_to(output))
    _write(output / "generation_results.json", _stage_payload(
        contract, generated_artifacts=artifacts, raw_to_turbo_control_compatibility=compatibility,
        training_metadata=training, source_rgb_fallback_used=False,
    ))
    print(output / "generation_results.json")


def _sidecar(path: str | Path, stems: list[str]) -> tuple[dict[str, Any], str]:
    spec, _ = final_val.load_final_spec(final_val.FINAL_SPEC)
    full, digest = final_val._load_final_sidecar(path, list(spec["stems"]))
    by_stem = {str(row["stem"]): row for row in full["records"]}
    if any(stem not in by_stem for stem in stems):
        raise ValueError("Authoritative final-val sidecar lacks a Style-LoRA composition stem")
    return {"records": [by_stem[stem] for stem in stems]}, digest


def _score_provenance(contract: Mapping[str, Any], *, sidecar_sha256: str, clip_model_id: str) -> dict[str, Any]:
    """Immutable identity for the scoring method, distinct from run-local paths."""
    return _canonical_json({
        "experiment_provenance_sha256": _provenance_sha256(contract),
        "reference_sidecar_sha256": sidecar_sha256,
        "clip_model_id": clip_model_id,
        "confidence_threshold": 0.5,
    })


def _score_records(payload: Mapping[str, Any], rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    expected = [(row["stem"], style) for row in rows for style in variants]
    if not isinstance(records, list) or [(item.get("stem"), item.get("style_id")) for item in records] != expected:
        raise ValueError("Style-LoRA score artifact is incomplete or does not match the frozen matrix")
    return records


def score(args: argparse.Namespace) -> None:
    rows = load_rows(args.prompt_file)
    contract, dataset, _, _, _, _, output, _ = _inputs(args, rows)
    if _generation_status(output, rows, contract) != "complete":
        raise FileNotFoundError("Style-LoRA scoring requires the complete validated generation matrix")
    if not args.reference_sidecar:
        raise ValueError("Style-LoRA PCK requires the authoritative final-val --reference-sidecar")
    sidecar, sidecar_digest = _sidecar(args.reference_sidecar, [row["stem"] for row in rows]); by_stem = {row["stem"]: row for row in sidecar["records"]}
    score_provenance = _score_provenance(contract, sidecar_sha256=sidecar_digest, clip_model_id=args.clip_model_id)
    score_path = output / "pck_clip_results.json"
    if score_path.exists():
        existing = _read_json(score_path)
        if (existing.get("immutable_provenance") != contract
                or existing.get("scoring_provenance") != score_provenance):
            raise ValueError("Existing Style-LoRA score artifact conflicts with immutable provenance")
    geometry = {row["stem"]: turbo_scoring_geometry(_sample_by_stem(dataset, row["stem"])) for row in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    records = []
    for row in rows:
        for style in contract["variants"]:
            image = _directory(output, row["stem"]) / _image_name(style)
            result = score_authoritative_pck(sidecar={"records": [by_stem[row["stem"]]]}, geometry_by_stem={row["stem"]: geometry[row["stem"]]}, image_for=lambda _: image, detector=detector, confidence_threshold=.5, require_images=True)
            pck = result["per_image"][0] if result["per_image"] else {"stem": row["stem"], "reference_available": False, "reason": result["unavailable"][0]["reason"]}
            records.append({"condition_id": row["condition_id"], "stem": row["stem"], "style_id": style, "prompt": row["prompt"], "seed": contract["sampling_seeds"][row["stem"]],
                            "image": str(image.relative_to(output)), "pck": pck, "clip_cosine_similarity": _clip_score(clip, processor, device, row["prompt"], image)})
    _write(score_path, _stage_payload(
        contract, scoring_provenance=score_provenance, reference_sidecar=str(Path(args.reference_sidecar).resolve()),
        per_generation=records,
    ))
    print(output / "pck_clip_results.json")


def _compact(records: list[dict[str, Any]]) -> dict[str, Any]:
    pck = [record["pck"] for record in records if record["pck"].get("reference_available")]
    pose = _pool_pose(pck) if pck else {"evaluable_sample_count": 0, "pck_005": None, "pck_010": None, "pck_020": None}
    clips = aggregate([float(record["clip_cosine_similarity"]) for record in records])
    return {"generation_count": len(records), "pck": pose, "pck_unavailable_count": len(records) - len(pck),
            "clip": {"mean_cosine_similarity": clips["mean"], "median_cosine_similarity": clips["median"], "std_cosine_similarity": clips["std"], "sample_count": clips["sample_count"]},
            "style_fidelity_metric": "not_claimed; first matrix requires qualitative review"}


def _validated_scores(output: Path, contract: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = _read_json(output / "pck_clip_results.json")
    if scores.get("immutable_provenance") != contract:
        raise ValueError("Style-LoRA score artifact conflicts with immutable provenance")
    score_provenance = scores.get("scoring_provenance")
    if (not isinstance(score_provenance, Mapping)
            or score_provenance.get("experiment_provenance_sha256") != _provenance_sha256(contract)
            or not isinstance(score_provenance.get("reference_sidecar_sha256"), str)
            or not isinstance(score_provenance.get("clip_model_id"), str)
            or score_provenance.get("confidence_threshold") != 0.5):
        raise ValueError("Style-LoRA score artifact lacks immutable scoring provenance")
    return _score_records(scores, rows, contract["variants"])


def report(args: argparse.Namespace) -> None:
    rows = load_rows(args.prompt_file)
    contract, _, _, _, _, training, output, _ = _inputs(args, rows)
    if _generation_status(output, rows, contract) != "complete":
        raise FileNotFoundError("Style-LoRA report requires the complete validated generation matrix")
    records = _validated_scores(output, contract, rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records: grouped[record["style_id"]].append(record)
    if tuple(grouped) != tuple(contract["variants"]) or any(len(grouped[style]) != len(rows) for style in grouped):
        raise ValueError("Style-LoRA score records do not form the required four-pose matrix")
    metrics = {"group_by": "style_id", "rows": [{"style_id": style, **_compact(grouped[style])} for style in contract["variants"]]}
    _write(output / "metrics_by_style.json", _stage_payload(contract, **metrics))
    for row in rows:
        paths = [output / "controls" / f"{row['stem']}.png"] + [_directory(output, row["stem"]) / _image_name(style) for style in contract["variants"]]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"Style-LoRA comparison grid requires control and all variants: {row['stem']}")
        make_contact_sheet([(row["condition_id"], paths)], output / "comparison_grids" / f"{row['condition_id']}.png", thumbnail_width=220, thumbnail_height=220,
                           column_labels=("pose control",) + tuple(contract["variants"]))
    aggregate_rows = [(row["condition_id"], [output / "controls" / f"{row['stem']}.png"] + [_directory(output, row["stem"]) / _image_name(style) for style in contract["variants"]]) for row in rows]
    make_contact_sheet(aggregate_rows, output / "style_lora_contact_sheet.png", thumbnail_width=180, thumbnail_height=180, column_labels=("pose control",) + tuple(contract["variants"]))
    _write(output / "evaluation_summary.json", _stage_payload(
        contract, training_metadata=training, generation_count=len(records), score_artifact="pck_clip_results.json",
        compact_metrics="metrics_by_style.json", comparison_grids={row["condition_id"]: str(Path("comparison_grids") / f"{row['condition_id']}.png") for row in rows},
        aggregate_contact_sheet="style_lora_contact_sheet.png", source_rgb_fallback_used=False,
    ))
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    rows = load_rows(args.prompt_file)
    contract, _, _, _, _, _, output, _ = _inputs(args, rows)
    if _generation_status(output, rows, contract) != "complete":
        raise FileNotFoundError("Style-LoRA summary requires the complete validated generation matrix")
    records = _validated_scores(output, contract, rows)
    print(json.dumps({"candidate": POSE_CANDIDATE, "style_strength": contract["style_strength"], "generation_count": len(records),
                      "by_style": [{"style_id": style, **_compact([record for record in records if record["style_id"] == style])} for style in contract["variants"]]}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "preflight", "generate", "score", "report", "summary"))
    parser.add_argument("--candidate", choices=(POSE_CANDIDATE,), default=POSE_CANDIDATE)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--style-strength", type=float, default=1.0)
    parser.add_argument("--prompt-file", default=str(SPEC_FILE))
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
    {"audit": audit, "preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__":
    main()
