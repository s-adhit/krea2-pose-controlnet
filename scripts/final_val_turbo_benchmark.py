"""Frozen 48-item final-validation Krea-2 Turbo evaluation.

This is deliberately separate from ``turbo_benchmark.py``: that script's
24-item diagnostic contract is historical and must remain byte-for-byte
strict.  This entry point evaluates selected real checkpoints and allowlisted
trainable-tensor interpolations against the immutable final-val benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.dataset_index import validate_posebridge_snapshot
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, make_evaluation_spec, save_image
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.pose_targets import load_sidecar, pck_records_from_v3
from pose_controlnet.turbo_evaluation import (
    controlled_branch_metadata, raw_to_turbo_control_compatibility,
    sample_turbo_pose_image, turbo_metadata, turbo_scoring_geometry,
)
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae


FINAL_SPEC = Path("docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json")
FINAL_SPEC_SHA256 = "93a5254e57fa208263f6188573e0760ffedd954bf3b3b3425109ea0178957cd0"
FINAL_COUNT = 48
FINAL_TURBO = {**turbo_metadata(), "control_scale": 1.0}
CANDIDATES = {
    "parent-4000": {
        "checkpoint_root": "/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000",
        "step": 4000,
        "label": "parent-4000",
        "sha256": "0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd",
        "production_provenance": {
            "format": 1,
            "run_name": "pose-control-production-cooldown-3000-to5000",
            "max_steps": 5000,
        },
    },
    "finish-control-a4300": {
        "checkpoint_root": "/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-finish-control-4000-to4500",
        "step": 4300,
        "label": "finish-control-a4300",
        "sha256": "17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff",
        "production_provenance": {
            "format": 1,
            "run_name": "pose-control-finish-control-4000-to4500",
            "max_steps": 4500,
        },
    },
    "mix-025": {
        "kind": "trainable_tensor_interpolation",
        "label": "mix-025",
        "alpha": 0.25,
        "endpoints": ("parent-4000", "finish-control-a4300"),
    },
    "mix-050": {
        "kind": "trainable_tensor_interpolation",
        "label": "mix-050",
        "alpha": 0.50,
        "endpoints": ("parent-4000", "finish-control-a4300"),
    },
    "mix-075": {
        "kind": "trainable_tensor_interpolation",
        "label": "mix-075",
        "alpha": 0.75,
        "endpoints": ("parent-4000", "finish-control-a4300"),
    },
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
        raise FileNotFoundError(f"Required final-val JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Final-val JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Final-val JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_final_spec(path: str | Path, *, expected_sha256: str | None = FINAL_SPEC_SHA256) -> tuple[dict[str, Any], str]:
    """Load the pinned immutable spec and reject all structural drift."""
    source = Path(path)
    digest = _sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Final-val benchmark spec SHA-256 mismatch: {source}")
    spec = _read_json(source)
    stems = spec.get("stems")
    if (spec.get("format_version"), spec.get("kind"), spec.get("split"), spec.get("seed")) != (1, "final_val_turbo_fixed_pose", "val", 420300):
        raise ValueError("Final-val benchmark spec has an incompatible identity contract")
    if not isinstance(stems, list) or len(stems) != FINAL_COUNT or len(stems) != len(set(stems)) or not all(isinstance(stem, str) and stem for stem in stems):
        raise ValueError("Final-val benchmark spec must contain exactly 48 unique frozen stems")
    if spec.get("turbo") != FINAL_TURBO:
        raise ValueError("Final-val benchmark spec violates the locked 8-step CFG-0 mu=1.15 control-scale-1.0 Turbo contract")
    benchmark = spec.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("name") != "final_val_benchmark_48":
        raise ValueError("Final-val benchmark spec lacks final_val_benchmark_48 provenance")
    if benchmark.get("source_counts") != {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8}:
        raise ValueError("Final-val benchmark source quotas differ from the frozen contract")
    if benchmark.get("orientation_counts") != {"landscape": 16, "near_square": 17, "portrait": 15}:
        raise ValueError("Final-val benchmark orientation counts differ from the frozen contract")
    required = ("per_stem_seeds", "sample_identities")
    if any(not isinstance(spec.get(key), dict) or set(spec[key]) != set(stems) for key in required):
        raise ValueError("Final-val benchmark spec lacks complete frozen seed/cache identities")
    return spec, digest


def validate_cached_contract(dataset: PreparedLatentShardDataset, spec: Mapping[str, Any]) -> None:
    """Recompute the shared cache identity calculation before any evaluation."""
    stems = list(spec["stems"])
    observed = make_evaluation_spec(dataset, split="val", count=FINAL_COUNT, seed=420300,
                                    kind="final_val_turbo_fixed_pose", stems=stems)
    for key in ("kind", "split", "seed", "stems", "per_stem_seeds", "sample_identities"):
        if observed[key] != spec.get(key):
            raise ValueError(f"Final-val cache identity/seed contract mismatch for {key}")


def resolve_final_controls(dataset_root: str | Path, stems: list[str]) -> dict[str, Path]:
    """Resolve controls through the read-only physical DatasetIndex validation."""
    snapshot = validate_posebridge_snapshot(dataset_root)
    controls = {record.stem: record.control_path for record in snapshot.records_by_split["val"]}
    missing = [stem for stem in stems if stem not in controls]
    if missing:
        raise ValueError(f"Final-val DatasetIndex cannot resolve frozen control stems: {missing[:3]}")
    return {stem: controls[stem] for stem in stems}


def candidate_checkpoint(candidate: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Resolve exactly one allowlisted real controlled checkpoint, never a base adapter."""
    if candidate not in CANDIDATES:
        raise ValueError(f"Final-val supports only {', '.join(CANDIDATES)}")
    selected = dict(CANDIDATES[candidate])
    if selected.get("kind") == "trainable_tensor_interpolation":
        raise ValueError("Interpolated final-val candidates do not have a standalone checkpoint")
    root = Path(selected["checkpoint_root"])
    checkpoint = root / f"step_{selected['step']:06d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required exact final-val checkpoint is missing: {checkpoint}")
    state = load_training_state(checkpoint)
    if state.get("global_step") != selected["step"]:
        raise ValueError(f"Final-val checkpoint filename/embedded step mismatch: {checkpoint}")
    observed_sha256 = _sha256(checkpoint)
    if observed_sha256 != selected["sha256"]:
        raise ValueError(f"Final-val checkpoint SHA-256 mismatch: {checkpoint}")
    gate = state.get("gate_e")
    if gate is not None:
        # Keep the historical diagnostic provenance contract strict whenever a
        # checkpoint actually carries that schema.
        metadata = controlled_branch_metadata([(selected["step"], checkpoint)])
    else:
        # These two final-val candidates predate gate_e.  Their full-training
        # state has already been schema-validated by load_training_state; bind
        # its project-owned production provenance to this exact pinned file.
        production = state.get("production_pose_control")
        expected = selected["production_provenance"]
        if (not isinstance(production, Mapping)
                or any(production.get(key) != value for key, value in expected.items())
                or production.get("current_step") != selected["step"]):
            raise ValueError(f"Final-val checkpoint lacks compatible production provenance: {checkpoint}")
        metadata = {
            "validation": "pinned_final_val_legacy_production_metadata",
            "checkpoint_sha256": observed_sha256,
            "production_pose_control": production,
        }
    return selected, checkpoint, metadata


def _endpoint_provenance(candidate: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    return {"candidate": candidate["label"], "path": str(checkpoint),
            "step": candidate["step"], "sha256": _sha256(checkpoint)}


def resolve_candidate(candidate: str) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    """Resolve a pinned real checkpoint or one explicit in-memory interpolation."""
    if candidate not in CANDIDATES:
        raise ValueError(f"Final-val supports only {', '.join(CANDIDATES)}")
    selected = dict(CANDIDATES[candidate])
    if selected.get("kind") != "trainable_tensor_interpolation":
        return candidate_checkpoint(candidate)
    endpoint_ids = selected.get("endpoints")
    alpha = selected.get("alpha")
    if (not isinstance(endpoint_ids, tuple) or len(endpoint_ids) != 2
            or not isinstance(alpha, float) or not 0.0 < alpha < 1.0):
        raise ValueError(f"Invalid final-val interpolation definition: {candidate}")
    parent, parent_path, parent_metadata = candidate_checkpoint(endpoint_ids[0])
    finish, finish_path, finish_metadata = candidate_checkpoint(endpoint_ids[1])
    endpoints = [_endpoint_provenance(parent, parent_path), _endpoint_provenance(finish, finish_path)]
    selected["interpolation"] = {
        "candidate_id": candidate,
        "alpha": alpha,
        "formula": "(1 - alpha) * parent-4000 + alpha * finish-control-a4300",
        "endpoints": endpoints,
        "tensor_scope": "state['model'] trainable control/LoRA tensors only",
        "compute_dtype": "float32",
    }
    metadata = {
        "validation": "interpolated_trainable_model_tensors_only",
        "candidate_kind": selected["kind"],
        "interpolation": selected["interpolation"],
        "endpoint_training_metadata": {
            parent["label"]: parent_metadata,
            finish["label"]: finish_metadata,
        },
    }
    return selected, None, metadata


def validate_interpolation_trainable_state(parent: Mapping[str, Any], finish: Mapping[str, Any]) -> None:
    """Prove the two trainable-state mappings are safe to interpolate."""
    parent_keys, finish_keys = set(parent), set(finish)
    if parent_keys != finish_keys:
        raise ValueError(
            "Final-val interpolation requires exact matching trainable keys: "
            f"missing={sorted(parent_keys - finish_keys)[:5]}, "
            f"unexpected={sorted(finish_keys - parent_keys)[:5]}"
        )
    for key in sorted(parent_keys):
        parent_tensor, finish_tensor = parent[key], finish[key]
        if not isinstance(parent_tensor, torch.Tensor) or not isinstance(finish_tensor, torch.Tensor):
            raise ValueError(f"Final-val interpolation requires tensors only: {key}")
        if parent_tensor.shape != finish_tensor.shape:
            raise ValueError(f"Final-val interpolation tensor shape mismatch for {key}: "
                             f"{tuple(parent_tensor.shape)} != {tuple(finish_tensor.shape)}")
        if not (parent_tensor.is_floating_point() and finish_tensor.is_floating_point()):
            raise ValueError(f"Final-val interpolation requires floating trainable tensor: {key}")


def interpolate_trainable_state(parent: Mapping[str, Any], finish: Mapping[str, Any], alpha: float) -> dict[str, torch.Tensor]:
    """Blend only matching trainable model tensors, in FP32, back to parent dtype."""
    if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
        raise ValueError(f"Interpolation alpha must be strictly between zero and one, got {alpha!r}")
    validate_interpolation_trainable_state(parent, finish)
    blended: dict[str, torch.Tensor] = {}
    for key in sorted(parent):
        parent_tensor, finish_tensor = parent[key], finish[key]
        blended[key] = (
            parent_tensor.detach().to(device="cpu", dtype=torch.float32) * (1.0 - alpha)
            + finish_tensor.detach().to(device="cpu", dtype=torch.float32) * alpha
        ).to(dtype=parent_tensor.dtype)
    return blended


def candidate_trainable_state(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, torch.Tensor]:
    """Return the evaluation model state, never optimizer/scheduler/RNG metadata."""
    if candidate.get("kind") != "trainable_tensor_interpolation":
        if checkpoint is None:
            raise ValueError("Real final-val candidate is missing its checkpoint")
        state = load_training_state(checkpoint)
        model = state.get("model")
        if not isinstance(model, Mapping):
            raise ValueError(f"Final-val checkpoint lacks trainable model state: {checkpoint}")
        return dict(model)
    interpolation = candidate.get("interpolation")
    if not isinstance(interpolation, Mapping):
        raise ValueError("Interpolated final-val candidate lacks resolved endpoint provenance")
    endpoints = interpolation.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("Interpolated final-val candidate lacks exactly two endpoints")
    endpoint_states = _interpolation_endpoint_models(interpolation)
    return interpolate_trainable_state(endpoint_states[0], endpoint_states[1], float(interpolation["alpha"]))


def _interpolation_endpoint_models(interpolation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    endpoints = interpolation.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("Interpolated final-val candidate lacks exactly two endpoints")
    endpoint_states: list[Mapping[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, Mapping):
            raise ValueError("Interpolated final-val endpoint provenance is invalid")
        path = Path(str(endpoint.get("path", "")))
        if _sha256(path) != endpoint.get("sha256"):
            raise ValueError(f"Interpolation endpoint SHA-256 changed after preflight: {path}")
        state = load_training_state(path)
        if state.get("global_step") != endpoint.get("step"):
            raise ValueError(f"Interpolation endpoint filename/embedded step mismatch: {path}")
        model = state.get("model")
        if not isinstance(model, Mapping):
            raise ValueError(f"Interpolation endpoint lacks trainable model state: {path}")
        endpoint_states.append(model)
    return endpoint_states


def candidate_raw_to_turbo_state(candidate: Mapping[str, Any], checkpoint: Path | None,
                                 trainable_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Keep Raw/Turbo validation provenance separate from the interpolated tensors."""
    if not _is_interpolation(candidate):
        if checkpoint is None:
            raise ValueError("Real final-val candidate is missing its checkpoint")
        return load_training_state(checkpoint)
    endpoints = candidate["interpolation"]["endpoints"]
    states = [load_training_state(Path(endpoint["path"])) for endpoint in endpoints]
    raw_checkpoints = [state.get("config", {}).get("raw_ckpt") for state in states]
    if not all(isinstance(raw, str) and raw for raw in raw_checkpoints) or raw_checkpoints[0] != raw_checkpoints[1]:
        raise ValueError("Interpolation endpoints do not share exact Krea-2 Raw provenance")
    return {"config": {"raw_ckpt": raw_checkpoints[0]}, "model": dict(trainable_state)}


def _is_interpolation(candidate: Mapping[str, Any]) -> bool:
    return candidate.get("kind") == "trainable_tensor_interpolation"


def _image_name(candidate: Mapping[str, Any]) -> str:
    return f"{candidate['label']}.png" if _is_interpolation(candidate) else f"step_{candidate['step']:06d}.png"


def _output_contract(spec: Mapping[str, Any], spec_sha256: str, candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    contract = {"kind": "final_val_turbo_fixed_pose", "final_spec_sha256": spec_sha256,
                "stems": list(spec["stems"]), "candidate": candidate["label"], "turbo": FINAL_TURBO}
    if _is_interpolation(candidate):
        return {**contract, "candidate_kind": candidate["kind"], "checkpoint_step": None,
                "checkpoint_interpolation": candidate["interpolation"]}
    if checkpoint is None:
        raise ValueError("Real final-val candidate is missing its checkpoint")
    return {**contract, "checkpoint_step": candidate["step"], "checkpoint_sha256": _sha256(checkpoint)}


def _validate_or_write_output(output: Path, contract: Mapping[str, Any]) -> None:
    path = output / "final_val_provenance.json"
    if path.exists():
        if _read_json(path) != contract:
            raise ValueError(f"Existing final-val output has conflicting immutable provenance: {path}")
    else:
        _write(path, contract)


def _sample_metadata(stem: str, sample: Mapping[str, Any], control: Path, spec_sha256: str,
                     candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    metadata = {"stem": stem, "prompt": sample["prompt"], "control_path": str(control),
                "seed": int(sample["_final_seed"]), "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8],
                "candidate": candidate["label"], "final_spec_sha256": spec_sha256,
                "control_scale": 1.0, **turbo_metadata()}
    if _is_interpolation(candidate):
        return {**metadata, "candidate_kind": candidate["kind"], "checkpoint_step": None,
                "checkpoint_interpolation": candidate["interpolation"]}
    if checkpoint is None:
        raise ValueError("Real final-val candidate is missing its checkpoint")
    return {**metadata, "checkpoint_step": candidate["step"], "checkpoint_sha256": _sha256(checkpoint)}


def _generation_status(output: Path, stems: list[str], candidate: Mapping[str, Any], spec_sha256: str) -> str:
    payload_path = output / "generation_results.json"
    payload = _read_json(payload_path) if payload_path.exists() else None
    observed, recorded = [], []
    for stem in stems:
        directory = output / "fixed_pose" / stem
        image = directory / _image_name(candidate)
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    opened.verify()
                metadata = _read_json(directory / "metadata.json")
                if (metadata.get("stem") != stem or metadata.get("candidate") != candidate["label"]
                        or metadata.get("checkpoint_step") != (None if _is_interpolation(candidate) else candidate["step"])
                        or metadata.get("final_spec_sha256") != spec_sha256 or metadata.get("control_scale") != 1.0
                        or any(metadata.get(key) != value for key, value in turbo_metadata().items())
                        or (_is_interpolation(candidate) and metadata.get("checkpoint_interpolation") != candidate.get("interpolation"))):
                    raise ValueError("metadata contract mismatch")
            except Exception as exc:
                raise ValueError(f"Final-val generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            observed.append(False)
        if payload is not None:
            generated = (payload.get("generated_artifacts", {}).get(stem) == [_image_name(candidate)]
                         if _is_interpolation(candidate)
                         else payload.get("generated_steps", {}).get(stem) == [candidate["step"]])
            recorded.append(generated)
    if not any(observed) and (payload is None or not any(recorded)):
        return "missing"
    if all(observed) and payload is not None and all(recorded) and payload.get("stems") == stems and payload.get("candidate") == candidate["label"] and payload.get("final_spec_sha256") == spec_sha256 and payload.get("turbo") == FINAL_TURBO:
        return "complete"
    raise ValueError("Existing final-val generation output is incomplete or inconsistent; refusing to overwrite it")


def _qualitative_image_paths(output: Path, stem: str, candidate: Mapping[str, Any]) -> list[Path]:
    """Return the conditioning control and this candidate's generated image.

    Qualitative reports must never resolve a dataset RGB/reference image.  The
    generated image is deliberately addressed only through the validated
    candidate-specific output root.
    """
    directory = output / "fixed_pose" / stem
    control = directory / "control.png"
    generated = directory / _image_name(candidate)
    if not control.is_file():
        raise FileNotFoundError(f"Final-val report is missing pose control for {stem}: {control}")
    if not generated.is_file():
        raise FileNotFoundError(f"Final-val report is missing generated output for {stem}: {generated}")
    return [control, generated]


def _inputs(args) -> tuple[dict[str, Any], str, PreparedLatentShardDataset, dict[str, Path], dict[str, Any], Path | None, dict[str, Any], Path]:
    spec, spec_sha256 = load_final_spec(args.final_spec)
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    validate_cached_contract(dataset, spec)
    shards = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shards.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Final-val requires --dataset-root or latent shards.json.dataset_root")
    controls = resolve_final_controls(dataset_root, list(spec["stems"]))
    candidate, checkpoint, training_metadata = resolve_candidate(args.candidate)
    return spec, spec_sha256, dataset, controls, candidate, checkpoint, training_metadata, Path(args.output_root)


def preflight(args) -> None:
    spec, digest, dataset, controls, candidate, checkpoint, training, output = _inputs(args)
    if _is_interpolation(candidate):
        endpoint_models = _interpolation_endpoint_models(candidate["interpolation"])
        validate_interpolation_trainable_state(endpoint_models[0], endpoint_models[1])
    contract = _output_contract(spec, digest, candidate, checkpoint); _validate_or_write_output(output, contract)
    preflight_metadata = {"local_checkpoint": str(checkpoint)} if checkpoint is not None else {
        "interpolation_endpoints": candidate["interpolation"]["endpoints"],
        "interpolation_alpha": candidate["interpolation"]["alpha"],
        "interpolation_candidate_id": candidate["interpolation"]["candidate_id"],
    }
    _write(output / "checkpoint_preflight.json", {**contract, "sample_count": len(dataset),
           "final_sample_count": len(spec["stems"]), "control_paths_resolved": len(controls),
           **preflight_metadata, "training_metadata": training})
    print(output / "checkpoint_preflight.json")


def generate(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run final-val Turbo generation from the GH200 host shell with CUDA visible")
    spec, digest, dataset, controls, candidate, checkpoint, training, output = _inputs(args)
    contract = _output_contract(spec, digest, candidate, checkpoint); _validate_or_write_output(output, contract)
    if _generation_status(output, list(spec["stems"]), candidate, digest) == "complete":
        print(json.dumps({"already_complete": candidate["label"]})); return
    model = build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval(); vae = load_krea_vae("cuda")
    trainable_state = candidate_trainable_state(candidate, checkpoint)
    load_trainable_state_dict(model, trainable_state)
    compatibility_state = candidate_raw_to_turbo_state(candidate, checkpoint, trainable_state)
    compatibility = raw_to_turbo_control_compatibility(model, compatibility_state)
    for stem in spec["stems"]:
        sample = dict(_sample_by_stem(dataset, stem)); sample["_final_seed"] = spec["per_stem_seeds"][stem]["sampling"]
        directory = output / "fixed_pose" / stem; directory.mkdir(parents=True, exist_ok=True)
        source_control, target = controls[stem], directory / "control.png"
        if target.exists() and target.read_bytes() != source_control.read_bytes():
            raise ValueError(f"Existing final-val control conflicts with DatasetIndex resolution: {target}")
        if not target.exists(): target.write_bytes(source_control.read_bytes())
        metadata = _sample_metadata(stem, sample, source_control, digest, candidate, checkpoint)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing final-val metadata conflicts with frozen contract: {metadata_path}")
        if not metadata_path.exists(): _write(metadata_path, metadata)
        pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                         torch.device("cuda"), metadata["seed"], control_scale=1.0)
        save_image(pixels, directory / _image_name(candidate))
    generation_artifacts = ({"generated_artifacts": {stem: [_image_name(candidate)] for stem in spec["stems"]}}
                            if _is_interpolation(candidate)
                            else {"generated_steps": {stem: [candidate["step"]] for stem in spec["stems"]}})
    _write(output / "generation_results.json", {**contract, "stems": list(spec["stems"]),
           **generation_artifacts,
           "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None),
           "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training})
    print(output / "generation_results.json")


def _load_final_sidecar(path: str | Path, stems: list[str]) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.name == "diagnostic_reference_pose.json":
        raise ValueError("The historical diagnostic pose sidecar is not valid for final-val scoring")
    if source.is_dir():
        metadata, records = load_sidecar(source)
        if metadata.get("sidecar_kind") != "final_val_benchmark_48_authoritative_pose_targets_v3":
            raise ValueError("Final-val pose sidecar is not the canonical immutable v3 sidecar")
        frozen = metadata.get("frozen_stems")
        selection = metadata.get("frozen_selection")
        spec = metadata.get("frozen_spec")
        export = metadata.get("authoritative_source_pose_export")
        if frozen != stems or not isinstance(selection, dict) or selection.get("sha256") != "23d448d573a2ffd20adfd73fa88f34ebc08df280a051cb0931d9ecdcc1231ceb" or not isinstance(spec, dict) or spec.get("sha256") != FINAL_SPEC_SHA256 or not isinstance(export, dict) or not isinstance(export.get("sha256"), str):
            raise ValueError("Final-val pose sidecar provenance does not match the frozen benchmark")
        sidecar = {"records": pck_records_from_v3(records)}
        digest = str(metadata.get("records_sha256", ""))
    else:
        sidecar = _read_json(source); digest = _sha256(source)
    records = sidecar.get("records")
    if not isinstance(records, list) or [record.get("stem") if isinstance(record, dict) else None for record in records] != stems:
        raise ValueError("Final-val pose sidecar must contain exactly the frozen 48 stems in order")
    if any(record.get("source") not in {"coco", "humanart"} for record in records):
        raise ValueError("Final-val pose sidecar records must use scorer sources 'coco' or 'humanart'")
    return sidecar, digest


def score(args) -> None:
    spec, digest, dataset, _, candidate, checkpoint, _, output = _inputs(args)
    contract = _output_contract(spec, digest, candidate, checkpoint); _validate_or_write_output(output, contract)
    if _generation_status(output, list(spec["stems"]), candidate, digest) != "complete":
        raise FileNotFoundError("Final-val scoring requires the complete 48-image generation set")
    if not args.reference_sidecar:
        raise ValueError("Final-val PCK scoring requires an explicit immutable --reference-sidecar; no diagnostic sidecar fallback exists")
    sidecar, sidecar_digest = _load_final_sidecar(args.reference_sidecar, list(spec["stems"]))
    geometry = {stem: turbo_scoring_geometry(_sample_by_stem(dataset, stem)) for stem in spec["stems"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    image_for = lambda stem: output / "fixed_pose" / stem / _image_name(candidate)
    pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for, detector=detector, confidence_threshold=.5, require_images=True)
    from scripts.turbo_benchmark import _clip_score  # shared frozen CLIP scoring implementation
    clip_rows = [{"stem": stem, "cosine_similarity": _clip_score(clip, processor, device, _read_json(output / "fixed_pose" / stem / "metadata.json")["prompt"], image_for(stem))} for stem in spec["stems"]]
    values = aggregate([row["cosine_similarity"] for row in clip_rows])
    row = {"checkpoint_step": None if _is_interpolation(candidate) else candidate["step"], "candidate": candidate["label"],
           "candidate_kind": candidate.get("kind", "real_checkpoint"), "pose": pose,
           "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"], "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_sample": clip_rows}}
    _write(output / "pck_clip_results.json", {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "reference_sidecar_sha256": sidecar_digest,
           "clip_model": args.clip_model_id, "confidence_threshold": .5, "checkpoints": [row]})
    print(output / "pck_clip_results.json")


def report(args) -> None:
    spec, digest, _, _, candidate, checkpoint, training, output = _inputs(args)
    contract = _output_contract(spec, digest, candidate, checkpoint); _validate_or_write_output(output, contract)
    if _generation_status(output, list(spec["stems"]), candidate, digest) != "complete":
        raise FileNotFoundError("Final-val report requires the complete validated 48-image generation set")
    score_payload = _read_json(output / "pck_clip_results.json")
    if any(score_payload.get(key) != value for key, value in contract.items()):
        raise ValueError("Final-val score artifact conflicts with frozen output provenance")
    rows = score_payload.get("checkpoints")
    expected_step = None if _is_interpolation(candidate) else candidate["step"]
    if (not isinstance(rows, list) or len(rows) != 1 or rows[0].get("candidate") != candidate["label"]
            or rows[0].get("checkpoint_step") != expected_step):
        raise ValueError("Final-val report requires exactly one scored selected checkpoint")
    row = rows[0]; grid_rows = []
    for stem in spec["stems"]:
        grid_rows.append((stem, _qualitative_image_paths(output, stem, candidate)))
    labels = ("pose control", f"generated output ({candidate['label']})")
    make_contact_sheet(grid_rows[:4], output / "checkpoint_selection_grid.png", thumbnail_width=180, thumbnail_height=180, column_labels=labels)
    make_contact_sheet(grid_rows, output / "full_contact_sheet.png", thumbnail_width=320, thumbnail_height=320, column_labels=labels)
    _write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "checkpoints": rows,
           "benchmark": spec["benchmark"], "reference_sidecar": score_payload.get("reference_sidecar"),
           "qualitative_grids": {"checkpoint_selection": "checkpoint_selection_grid.png", "full_contact_sheet": "full_contact_sheet.png"},
           "comparison_kind": ("interpolated-trainable-state final-val evaluation" if _is_interpolation(candidate)
                               else "separate real-checkpoint final-val evaluation"),
           "base_or_zero_adapter_included": False,
           "checkpoint_interpolation_included": _is_interpolation(candidate), "production_winner_declared": False})
    print(output / "evaluation_summary.json")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "generate", "score", "report"), help="staged final-val action")
    parser.add_argument("--candidate", required=True, choices=tuple(CANDIDATES), help="one allowlisted real or interpolated controlled candidate")
    parser.add_argument("--output-root", required=True, help="new or matching candidate-specific final-val output directory")
    parser.add_argument("--final-spec", default=str(FINAL_SPEC), help="immutable frozen final-val spec (pinned SHA-256 by default)")
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    parser.add_argument("--dataset-root", help="PoseBridge HF snapshot; defaults to latent shards.json.dataset_root")
    parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
    parser.add_argument("--reference-sidecar", help="required only for score; immutable exact 48-stem authoritative pose sidecar")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return parser


def main() -> None:
    args = parser().parse_args(); {"preflight": preflight, "generate": generate, "score": score, "report": report}[args.action](args)


if __name__ == "__main__":
    main()
