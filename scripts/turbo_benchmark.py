"""Generic, staged, evaluation-only Krea-2 Turbo Pose-ControlNet benchmark.

Experiments are selected by a JSON spec and exact checkpoint subset.  This
module intentionally contains no experiment-specific run names, roots, or
checkpoint lists; changing an experiment never requires editing this source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.dataset_index import validate_posebridge_snapshot
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, make_evaluation_spec, save_image
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate, clip_feature_tensor, cosine_from_embeddings, prepare_clip_scoring_inputs
from pose_controlnet.keypoint_critic import (
    FixedBoxKeypointRCNNCritic,
    normalized_coordinate_distances,
    normalized_coordinate_huber,
    soft_coordinates,
)
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.turbo_evaluation import (
    TurboExperiment,
    assert_turbo_diagnostic_contract,
    assert_exact_diagnostic_stems,
    controlled_branch_metadata,
    discover_turbo_checkpoint_steps,
    exact_direct_local_turbo_checkpoints,
    exact_local_turbo_checkpoints,
    load_turbo_experiment_spec,
    normalize_turbo_steps,
    raw_to_turbo_control_compatibility,
    sample_turbo_pose_image,
    turbo_metadata,
    turbo_scoring_geometry,
    turbo_experiment_from_payload,
    validate_controlled_experiment_metadata,
)
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required Turbo evaluation JSON is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Turbo evaluation JSON must be an object: {path}")
    return value


def _manifest_stems(path: Path) -> tuple[str, ...]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Canonical diagnostic manifest is missing: {path}") from None
    try:
        return tuple(Path(record["file_name"]).stem for record in records)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Canonical diagnostic manifest has malformed records: {path}") from exc


def _config(args) -> TurboExperiment:
    cached = getattr(args, "_turbo_config", None)
    if cached is not None:
        return cached
    if getattr(args, "dynamic_experiment", False):
        config = _dynamic_experiment_config(args)
        args._turbo_config = config
        return config
    config = load_turbo_experiment_spec(args.spec, overrides={
        "checkpoint_root": args.checkpoint_root,
        "hf_repo_id": args.hf_repo_id,
        "hf_namespace": args.hf_namespace,
        "output_root": args.output_root,
    })
    args._turbo_config = config
    return config


def _parse_expected_sha256(values: Iterable[str]) -> dict[str, str]:
    """Parse exact ``STEP=SHA256`` provenance supplied at the CLI boundary."""
    parsed: dict[str, str] = {}
    for value in values:
        step, separator, digest = value.partition("=")
        if not separator or not step.isdigit() or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("--expected-sha256 values must be STEP=<lowercase-64-hex-sha256>")
        step = str(int(step))
        if step in parsed:
            raise ValueError(f"--expected-sha256 repeats step {step}")
        parsed[step] = digest
    return parsed


def _dynamic_experiment_config(args) -> TurboExperiment:
    """Resolve the controlled-branch CLI into the ordinary audited spec model."""
    expected_sha256 = _parse_expected_sha256(args.expected_sha256 or ())
    steps = normalize_turbo_steps(args.steps)
    if any(int(step) not in steps for step in expected_sha256):
        raise ValueError("--expected-sha256 includes a step that was not requested")
    if bool(args.baseline_output_root) != (args.baseline_step is not None):
        raise ValueError("--baseline-output-root and --baseline-step must be supplied together")
    try:
        args.checkpoint_label_template.format(step=steps[0])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("--checkpoint-label-template must be a valid format string using {step}") from exc
    payload: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "checkpoint_root": args.checkpoint_root,
        "hf_repo_id": args.hf_repo_id or "",
        "hf_namespace": f"{args.experiment_name}/full/",
        "output_root": args.output_root,
        "steps": list(steps),
        "checkpoint_validation": {"mode": "direct_local", "expected_sha256": expected_sha256},
        "labels": {"checkpoint_template": args.checkpoint_label_template},
        "turbo_contract": {**turbo_metadata(), "control_scale": 1.0},
        "diagnostics": {
            "canonical_manifest": args.diagnostic_manifest,
            "canonical_reference_spec": args.canonical_reference_spec,
            "expected_count": 24,
            "seed": 420200,
        },
        "paths": {
            "latent_root": args.latent_root,
            "text_conditioning_root": args.text_conditioning_root,
            "turbo_ckpt": args.turbo_ckpt,
            "reference_sidecar": args.reference_sidecar,
            "clip_model_id": args.clip_model_id,
        },
    }
    if args.baseline_output_root:
        payload["baseline"] = {
            "output_root": args.baseline_output_root,
            "checkpoint_step": args.baseline_step,
            "label": args.baseline_label or f"baseline {args.baseline_step}",
        }
    config = turbo_experiment_from_payload(payload)
    checkpoints = exact_direct_local_turbo_checkpoints(
        checkpoint_root=config.checkpoint_root, steps=config.steps or (),
        expected_sha256=config.checkpoint_validation.get("expected_sha256"),
    )
    metadata = controlled_branch_metadata(checkpoints)
    validate_controlled_experiment_metadata(config.checkpoint_root, metadata)
    # Keep checkpoint-varying counters and hashes nested, rather than copying
    # one final-step value into every candidate's report row.
    payload["training_metadata"] = metadata
    return turbo_experiment_from_payload(payload)


def _path(config: TurboExperiment, key: str, args, *, required: bool = True) -> Path | None:
    override = getattr(args, key.replace("-", "_"), None)
    value = override if override is not None else config.paths.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Turbo experiment spec requires paths.{key}")
    return Path(value)


def _steps(args, config: TurboExperiment, *, required: bool) -> tuple[int, ...]:
    if args.all_checkpoints:
        return discover_turbo_checkpoint_steps(config.checkpoint_root)
    if args.steps is not None:
        return normalize_turbo_steps(args.steps)
    if config.steps is not None:
        return config.steps
    if required:
        raise ValueError("Specify exactly one of --steps or --all-checkpoints")
    return ()


def _checkpoints(args, config: TurboExperiment, output: Path, *, required: bool) -> list[tuple[int, Path]]:
    steps = _steps(args, config, required=required)
    if not steps:
        return []
    if config.checkpoint_validation["mode"] == "direct_local":
        return exact_direct_local_turbo_checkpoints(
            checkpoint_root=config.checkpoint_root, steps=steps,
            expected_sha256=config.checkpoint_validation.get("expected_sha256"),
        )
    return exact_local_turbo_checkpoints(
        checkpoint_root=config.checkpoint_root,
        hf_repo_id=config.hf_repo_id,
        hf_namespace=config.hf_namespace,
        marker_download_dir=output / "hf-marker-cache",
        steps=steps,
    )


def _dataset_and_spec(args, config: TurboExperiment | None = None) -> tuple[PreparedLatentShardDataset, tuple[str, ...], dict[str, Any]]:
    """Build generic evaluator diagnostics, with a narrow legacy helper mode.

    Historical read-only diagnostic scripts import this helper.  Their mode is
    retained solely for source compatibility; the generic benchmark always
    passes a spec and never derives experiment paths or checkpoint steps here.
    """
    if config is None:
        dataset = PreparedLatentShardDataset(args.latent_root, "diagnostic_val", text_conditioning_root=args.text_conditioning_root)
        stems = assert_exact_diagnostic_stems(_manifest_stems(Path(args.diagnostic_manifest)), (record[3] for record in dataset.records))
        spec = make_evaluation_spec(dataset, split="diagnostic_val", count=len(stems), seed=getattr(args, "seed", 420200), kind="turbo_fixed_pose", stems=list(stems))
        spec["turbo"] = turbo_metadata()
        return dataset, stems, spec
    latent_root = _path(config, "latent_root", args)
    text_root = _path(config, "text_conditioning_root", args)
    manifest = Path(getattr(args, "diagnostic_manifest", None) or config.diagnostics["canonical_manifest"])
    dataset = PreparedLatentShardDataset(latent_root, "diagnostic_val", text_conditioning_root=text_root)
    stems = assert_exact_diagnostic_stems(
        _manifest_stems(manifest), (record[3] for record in dataset.records),
        expected_count=config.diagnostics["expected_count"],
    )
    spec = make_evaluation_spec(dataset, split="diagnostic_val", count=len(stems), seed=int(config.diagnostics.get("seed", 420200)),
                                kind="turbo_fixed_pose", stems=list(stems))
    spec["turbo"] = turbo_metadata()
    spec["control_scale"] = 1.0
    spec["experiment_name"] = config.experiment_name
    spec["resolved_experiment"] = {
        "checkpoint_root": str(config.checkpoint_root), "steps": list(config.steps or ()),
        "checkpoint_validation": dict(config.checkpoint_validation),
        "training_metadata": dict(config.training_metadata),
        "baseline": None if config.baseline is None else dict(config.baseline),
        "labels": dict(config.labels),
    }
    canonical_path = config.diagnostics.get("canonical_reference_spec")
    if not isinstance(canonical_path, str) or not canonical_path:
        raise ValueError("Turbo experiment spec requires diagnostics.canonical_reference_spec")
    canonical = _read_json(Path(canonical_path))
    assert_turbo_diagnostic_contract(spec, canonical, branch_name=config.experiment_name)
    return dataset, stems, spec


def _validate_output_spec(output: Path, spec: dict[str, Any], config: TurboExperiment) -> None:
    destination = output / "turbo_spec.json"
    if not destination.is_file():
        return
    existing = _read_json(destination)
    canonical = {key: spec[key] for key in ("kind", "seed", "stems", "per_stem_seeds", "sample_identities", "turbo")}
    if any(existing.get(key) != value for key, value in canonical.items()):
        raise ValueError(f"Existing evaluation output has a conflicting diagnostic/Turbo contract: {destination}")
    if existing.get("experiment_name") not in (None, config.experiment_name):
        raise ValueError(f"Existing evaluation output belongs to another experiment: {destination}")


def _write_spec_once(output: Path, spec: dict[str, Any], config: TurboExperiment) -> None:
    _validate_output_spec(output, spec, config)
    if not (output / "turbo_spec.json").is_file():
        _write(output / "turbo_spec.json", spec)


def _expected_stem_metadata(stem: str, sample: dict[str, Any], control_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "stem": stem,
        "prompt": sample["prompt"],
        "control_path": str(control_path),
        "seed": spec["per_stem_seeds"][stem]["sampling"],
        "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8],
        "control_scale": 1.0,
        **turbo_metadata(),
    }


def _generation_status(output: Path, stems: Iterable[str], step: int) -> str:
    """Return missing/complete or fail closed on a partial/corrupt result."""
    generation_path = output / "generation_results.json"
    payload = _read_json(generation_path) if generation_path.is_file() else None
    stems = tuple(stems)
    if payload is not None:
        if payload.get("metadata") != turbo_metadata() or payload.get("control_scale", 1.0) != 1.0:
            raise ValueError(f"Existing generation result has a conflicting Turbo contract: {generation_path}")
        if payload.get("stems") not in (None, list(stems)):
            raise ValueError(f"Existing generation result has a conflicting diagnostic order: {generation_path}")
    observed = []
    recorded = []
    for stem in stems:
        directory = output / "fixed_pose" / stem
        image = directory / f"step_{step:06d}.png"
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    opened.verify()
                metadata = _read_json(directory / "metadata.json")
                if metadata.get("stem") != stem or metadata.get("control_scale", 1.0) != 1.0 or any(metadata.get(key) != value for key, value in turbo_metadata().items()):
                    raise ValueError("metadata contract mismatch")
            except Exception as exc:
                raise ValueError(f"Existing generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            observed.append(False)
        if payload is not None:
            steps = payload.get("generated_steps", {}).get(stem, [])
            recorded.append(isinstance(steps, list) and step in steps)
    if not any(observed) and (payload is None or not any(recorded)):
        return "missing"
    if all(observed) and payload is not None and all(recorded):
        return "complete"
    raise ValueError(f"Existing generation output for step {step} is incomplete or inconsistent; refusing to overwrite it")


def _merged_generation_steps(output: Path, stems: Iterable[str], generated: dict[str, list[int]]) -> dict[str, list[int]]:
    """Retain every previously complete result while recording only new files."""
    path = output / "generation_results.json"
    previous = _read_json(path).get("generated_steps", {}) if path.is_file() else {}
    if not isinstance(previous, dict):
        raise ValueError(f"Malformed generation result index: {path}")
    merged: dict[str, list[int]] = {}
    for stem in stems:
        old_steps = previous.get(stem, [])
        if not isinstance(old_steps, list) or any(not isinstance(step, int) for step in old_steps):
            raise ValueError(f"Malformed generation step index for diagnostic stem {stem}")
        merged[stem] = sorted(set(old_steps) | set(generated.get(stem, [])))
    unknown = set(previous) - set(stems)
    if unknown:
        raise ValueError(f"Generation result index has unknown diagnostic stems: {sorted(unknown)[:3]}")
    return merged


def _existing_scored_rows(output: Path, spec: dict[str, Any], config: TurboExperiment) -> dict[int, dict[str, Any]]:
    path = output / "pck_clip_results.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    if payload.get("metadata") != turbo_metadata() or payload.get("control_scale", 1.0) != 1.0:
        raise ValueError(f"Existing score file has a conflicting Turbo contract: {path}")
    if payload.get("experiment_name") not in (None, config.experiment_name):
        raise ValueError(f"Existing score file belongs to another experiment: {path}")
    rows = payload.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError(f"Malformed scored checkpoint list: {path}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("checkpoint_step") if isinstance(row, dict) else None
        if not isinstance(step, int) or step in result:
            raise ValueError(f"Malformed or duplicate scored checkpoint step in {path}")
        if _generation_status(output, spec["stems"], step) != "complete":
            raise ValueError(f"Scored checkpoint step {step} lacks a complete generation result")
        result[step] = row
    return result


def _missing_generation_checkpoints(directory: Path, checkpoints: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
    """Compatibility helper: existing per-stem images are never overwritten."""
    return [(step, checkpoint) for step, checkpoint in checkpoints if not (directory / f"step_{step:06d}.png").is_file()]


def _clip_score(clip, processor, device: str, prompt: str, image_path: Path) -> float:
    with Image.open(image_path) as image:
        inputs = prepare_clip_scoring_inputs(processor, prompt, image.convert("RGB"), clip.config.text_config.max_position_embeddings).to(device)
    with torch.inference_mode():
        image_features = clip_feature_tensor(clip.get_image_features(pixel_values=inputs.pixel_values))
        text_features = clip_feature_tensor(clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask))
    return float(cosine_from_embeddings(image_features.float().cpu().numpy(), text_features.float().cpu().numpy())[0])


def _baseline_row(config: TurboExperiment, spec: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    if config.baseline is None:
        return None, None
    root_value, step = config.baseline.get("output_root"), config.baseline.get("checkpoint_step")
    if not isinstance(root_value, str) or not isinstance(step, int):
        raise ValueError("Turbo baseline requires output_root and checkpoint_step")
    root = Path(root_value)
    baseline_spec = _read_json(root / "turbo_spec.json")
    assert_turbo_diagnostic_contract(spec, baseline_spec, branch_name="configured baseline")
    rows = _read_json(root / "pck_clip_results.json").get("checkpoints")
    matches = [row for row in rows if isinstance(row, dict) and row.get("checkpoint_step") == step] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise ValueError("Configured baseline must contain exactly one completed scored checkpoint")
    return matches[0], root


def preflight(args) -> None:
    config, output = _config(args), _config(args).output_root
    dataset, stems, spec = _dataset_and_spec(args, config)
    _validate_output_spec(output, spec, config)
    baseline, baseline_root = _baseline_row(config, spec)
    checkpoints = _checkpoints(args, config, output, required=True)
    _write_spec_once(output, spec, config)
    checkpoint_entries = []
    for step, path in checkpoints:
        entry = {"checkpoint_step": step, "local_checkpoint": str(path)}
        if config.checkpoint_validation["mode"] == "hf_completion_marker":
            entry.update({"remote_checkpoint": f"{config.hf_namespace}step_{step:06d}.pt",
                          "remote_completion_marker": f"{config.hf_namespace}step_{step:06d}.pt.complete.json"})
        elif str(step) in config.checkpoint_validation.get("expected_sha256", {}):
            entry["expected_sha256"] = config.checkpoint_validation["expected_sha256"][str(step)]
        checkpoint_entries.append(entry)
    _write(output / "checkpoint_preflight.json", {
        "experiment_name": config.experiment_name, "metadata": turbo_metadata(), "control_scale": 1.0,
        "diagnostic_sample_count": len(dataset), "stems": list(stems),
        "local_checkpoint_root": str(config.checkpoint_root), "hf_repo_id": config.hf_repo_id,
        "hf_namespace": config.hf_namespace,
        "checkpoint_validation": config.checkpoint_validation,
        "training_metadata": config.training_metadata,
        "reused_baseline": None if baseline is None else {"checkpoint_step": baseline["checkpoint_step"], "source": str(baseline_root / "pck_clip_results.json"), "regenerated": False},
        "checkpoints": checkpoint_entries,
    })
    print(output / "checkpoint_preflight.json")


def generate(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run Turbo generation from the GH200 host shell with CUDA visible")
    config, output = _config(args), _config(args).output_root
    dataset, stems, spec = _dataset_and_spec(args, config)
    _write_spec_once(output, spec, config)
    checkpoints = _checkpoints(args, config, output, required=True)
    statuses = {step: _generation_status(output, stems, step) for step, _ in checkpoints}
    pending = [(step, checkpoint) for step, checkpoint in checkpoints if statuses[step] == "missing"]
    if not pending:
        print(json.dumps({"already_complete": [step for step, _ in checkpoints]}))
        return
    latent_root = _path(config, "latent_root", args)
    shard_metadata = _read_json(latent_root / "shards.json")
    snapshot = validate_posebridge_snapshot(getattr(args, "dataset_root", None) or shard_metadata["dataset_root"])
    controls = {record.stem: record.control_path for record in snapshot.records_by_split["diagnostic_val"]}
    if set(controls) != set(stems):
        raise ValueError("Diagnostic controls differ from the canonical diagnostic manifest")
    model = build_turbo_pose_model(_path(config, "turbo_ckpt", args), 64, 64, "cuda").eval()
    vae = load_krea_vae("cuda")
    generated: dict[str, list[int]] = {}
    compatibility: dict[str, dict[str, Any]] = {}
    for stem in stems:
        sample, directory = dict(_sample_by_stem(dataset, stem)), output / "fixed_pose" / stem
        directory.mkdir(parents=True, exist_ok=True)
        control_target = directory / "control.png"
        source_control = Path(controls[stem])
        if control_target.exists() and control_target.read_bytes() != source_control.read_bytes():
            raise ValueError(f"Existing control artifact conflicts with canonical control: {control_target}")
        if not control_target.exists():
            control_target.write_bytes(source_control.read_bytes())
        metadata = _expected_stem_metadata(stem, sample, source_control, spec)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing metadata conflicts with immutable Turbo contract: {metadata_path}")
        if not metadata_path.exists():
            _write(metadata_path, metadata)
        for step, checkpoint in pending:
            state = load_training_state(checkpoint)
            if state["global_step"] != step:
                raise ValueError(f"Checkpoint identity mismatch for step {step}")
            compatibility[str(step)] = raw_to_turbo_control_compatibility(model, state)
            load_trainable_state_dict(model, state["model"])
            pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                             torch.device("cuda"), metadata["seed"], control_scale=1.0)
            save_image(pixels, directory / f"step_{step:06d}.png")
            generated.setdefault(stem, []).append(step)
    _write(output / "generation_results.json", {
        "experiment_name": config.experiment_name, "metadata": turbo_metadata(), "control_scale": 1.0,
        "stems": list(stems), "checkpoints": sorted(step for step, _ in checkpoints),
        "generated_steps": _merged_generation_steps(output, stems, generated),
        "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None),
        "raw_to_turbo_control_compatibility": compatibility,
    })
    print(output / "generation_results.json")


def score(args) -> None:
    config, output = _config(args), _config(args).output_root
    dataset, stems, spec = _dataset_and_spec(args, config)
    _write_spec_once(output, spec, config)
    checkpoints = _checkpoints(args, config, output, required=True)
    rows = _existing_scored_rows(output, spec, config)
    pending = [step for step, _ in checkpoints if step not in rows]
    for step in pending:
        if _generation_status(output, stems, step) != "complete":
            raise FileNotFoundError(f"Cannot score step {step}: complete generation artifacts are required")
    if not pending:
        print(json.dumps({"already_complete": [step for step, _ in checkpoints]}))
        return
    sidecar = _read_json(_path(config, "reference_sidecar", args))
    geometry = {stem: turbo_scoring_geometry(_sample_by_stem(dataset, stem)) for stem in stems}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(getattr(args, "clip_model_id", None) or config.paths.get("clip_model_id", "openai/clip-vit-base-patch32"))
    clip = CLIPModel.from_pretrained(getattr(args, "clip_model_id", None) or config.paths.get("clip_model_id", "openai/clip-vit-base-patch32")).to(device).eval()
    unavailable = [{"stem": record["stem"], "pose_metric_status": "unavailable", "pose_metric_reason": "authoritative_reference_pose_unavailable", "pck_005": None, "pck_010": None, "pck_020": None}
                   for record in sidecar["records"] if record.get("status") != "available"]
    for step in pending:
        image_for = lambda stem, current=step: output / "fixed_pose" / stem / f"step_{current:06d}.png"
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for,
                                       detector=detector, confidence_threshold=.5, require_images=True)
        clip_rows = [{"stem": stem, "source": next(row.get("source") for row in sidecar["records"] if row["stem"] == stem),
                      "cosine_similarity": _clip_score(clip, processor, device, _read_json(output / "fixed_pose" / stem / "metadata.json")["prompt"], image_for(stem))}
                     for stem in stems]
        values = aggregate([row["cosine_similarity"] for row in clip_rows])
        rows[step] = {"checkpoint_step": step, "pose": pose, "pose_metric_unavailable_samples": unavailable,
                      "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"],
                               "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_sample": clip_rows}}
    _write(output / "pck_clip_results.json", {"experiment_name": config.experiment_name, "metadata": turbo_metadata(), "control_scale": 1.0,
           "clip_model": getattr(args, "clip_model_id", None) or config.paths.get("clip_model_id", "openai/clip-vit-base-patch32"),
           "confidence_threshold": .5, "checkpoints": [rows[step] for step in sorted(rows)]})
    print(output / "pck_clip_results.json")


def _summary_row(row: dict[str, Any], config: TurboExperiment, *, label: str,
                 training_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    pose, clip = row["pose"], row["clip"]
    metadata = dict(config.training_metadata if training_metadata is None else training_metadata)
    per_checkpoint = metadata.pop("per_checkpoint", {})
    checkpoint_metadata = per_checkpoint.get(str(row["checkpoint_step"]), {}) if isinstance(per_checkpoint, dict) else {}
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("training_metadata.per_checkpoint must map steps to metadata objects")
    return {"label": label, "checkpoint_step": row["checkpoint_step"], **metadata,
            **checkpoint_metadata,
            "control_scale": 1.0, **turbo_metadata(), "clip_mean_cosine_similarity": clip["mean_cosine_similarity"],
            "detection_coverage": pose["detection_coverage"], "joint_coverage": pose["joint_evaluation_coverage"],
            "pck_005": pose["pck_005"], "pck_010": pose["pck_010"], "pck_020": pose["pck_020"],
            "coco_pck": {key: pose["per_source"]["COCO"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "human_art_pck": {key: pose["per_source"]["Human-Art"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "multi_person_pck": {key: pose["multi_person"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "single_person_pck": {key: pose["single_person"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "matched_people": pose["matched_people"], "unmatched_reference_people": pose["unmatched_reference_people"],
            "predicted_people": pose["predicted_people"], "unmatched_predicted_people": pose["unmatched_predicted_people"]}


def _deltas(row: dict[str, Any], baseline: dict[str, Any], config: TurboExperiment) -> dict[str, float]:
    candidate = _summary_row(row, config, label="candidate")
    reference = _summary_row(baseline, config, label="baseline")
    keys = ("clip_mean_cosine_similarity", "detection_coverage", "joint_coverage", "pck_005", "pck_010", "pck_020", "matched_people", "unmatched_reference_people", "predicted_people", "unmatched_predicted_people")
    deltas = {key: candidate[key] - reference[key] for key in keys}
    for group in ("coco_pck", "human_art_pck", "single_person_pck", "multi_person_pck"):
        for threshold in ("pck_005", "pck_010", "pck_020"):
            deltas[f"{group}_{threshold}"] = candidate[group][threshold] - reference[group][threshold]
    return deltas


def _reference_image(config: TurboExperiment, stem: str) -> Path | None:
    reference = config.diagnostics.get("reference_images")
    if reference is None:
        return None
    if not isinstance(reference, dict) or not isinstance(reference.get("root"), str):
        raise ValueError("diagnostics.reference_images must provide root when configured")
    return Path(reference["root"]) / f"{stem}{reference.get('extension', '.png')}"


def _ranking(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Emit independently sortable metrics; this deliberately names no winner."""
    values = list(rows)
    metrics = {
        "pck_005": lambda row: row["pose"]["pck_005"],
        "pck_010": lambda row: row["pose"]["pck_010"],
        "pck_020": lambda row: row["pose"]["pck_020"],
        "clip_mean_cosine_similarity": lambda row: row["clip"]["mean_cosine_similarity"],
    }
    return {
        name: [{"rank": index + 1, "checkpoint_step": row["checkpoint_step"], "value": value(row)}
               for index, row in enumerate(sorted(values, key=lambda row: (-value(row), row["checkpoint_step"]))) ]
        for name, value in metrics.items()
    }


def report(args) -> None:
    config, output = _config(args), _config(args).output_root
    _, stems, spec = _dataset_and_spec(args, config)
    _validate_output_spec(output, spec, config)
    rows = _existing_scored_rows(output, spec, config)
    if not rows:
        raise FileNotFoundError("No completed scored checkpoints were discovered under the configured evaluation root")
    baseline, baseline_root = _baseline_row(config, spec)
    completed = [rows[step] for step in sorted(rows)]
    # Normalize persisted score order as part of report construction.
    score_path = output / "pck_clip_results.json"
    score_payload = _read_json(score_path)
    score_payload["checkpoints"] = completed
    _write(score_path, score_payload)
    baseline_visual_missing = []
    if baseline is not None:
        baseline_visual_missing = [
            baseline_root / "fixed_pose" / stem / f"step_{baseline['checkpoint_step']:06d}.png"
            for stem in stems
            if not (baseline_root / "fixed_pose" / stem / f"step_{baseline['checkpoint_step']:06d}.png").is_file()
        ]
    include_baseline_visuals = baseline is not None and not baseline_visual_missing
    labels = ["control"]
    include_reference = config.diagnostics.get("reference_images") is not None
    if include_reference:
        labels.append(str(config.labels.get("reference", "reference")))
    if include_baseline_visuals:
        labels.append(str(config.baseline.get("label", f"baseline {baseline['checkpoint_step']}")))
    labels.extend(str(config.labels.get("checkpoint_template", "checkpoint {step}")).format(step=row["checkpoint_step"]) for row in completed)
    grid_rows = []
    for stem in stems:
        paths = [output / "fixed_pose" / stem / "control.png"]
        if include_reference:
            reference = _reference_image(config, stem)
            if reference is None or not reference.is_file():
                raise FileNotFoundError(f"Configured reference artifact is missing: {reference}")
            paths.append(reference)
        if include_baseline_visuals:
            baseline_image = baseline_root / "fixed_pose" / stem / f"step_{baseline['checkpoint_step']:06d}.png"
            paths.append(baseline_image)
        paths.extend(output / "fixed_pose" / stem / f"step_{row['checkpoint_step']:06d}.png" for row in completed)
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Incomplete Turbo comparison row: {stem}")
        grid_rows.append((stem, paths))
    selection_name = f"{config.experiment_name}_checkpoint_selection_grid.png"
    contact_name = f"{config.experiment_name}_full_contact_sheet.png"
    make_contact_sheet(grid_rows[:min(4, len(grid_rows))], output / selection_name, thumbnail_width=180, thumbnail_height=180, column_labels=tuple(labels))
    make_contact_sheet(grid_rows, output / contact_name, thumbnail_width=320, thumbnail_height=320, column_labels=tuple(labels))
    comparison = ([] if baseline is None else [_summary_row(
        baseline, config, label=str(config.baseline.get("label", f"baseline {baseline['checkpoint_step']}")),
        training_metadata={},
    )])
    comparison.extend(_summary_row(row, config, label=str(config.labels.get("checkpoint_template", "checkpoint {step}")).format(step=row["checkpoint_step"])) for row in completed)
    _write(output / "evaluation_summary.json", {"experiment_name": config.experiment_name, "metadata": turbo_metadata(), "control_scale": 1.0,
           "training_metadata": config.training_metadata,
           "baseline": None if baseline is None else {"checkpoint_step": baseline["checkpoint_step"], "source": str(baseline_root / "pck_clip_results.json"), "regenerated": False, "result": baseline},
           "baseline_visual_artifacts_available": include_baseline_visuals,
           "baseline_visual_artifacts_missing_count": len(baseline_visual_missing),
           "comparison": comparison, "checkpoints": completed,
           "deltas_vs_baseline": {} if baseline is None else {str(row["checkpoint_step"]): _deltas(row, baseline, config) for row in completed},
           "spec_sha256": hashlib.sha256((output / "turbo_spec.json").read_bytes()).hexdigest(),
           "qualitative_grids": {"checkpoint_selection": selection_name, "full_contact_sheet": contact_name},
           "production_winner_declared": False})
    ranked_rows = ([] if baseline is None else [baseline]) + completed
    _write(output / "checkpoint_ranking.json", {"experiment_name": config.experiment_name,
           "includes_baseline": baseline is not None, "ranking": _ranking(ranked_rows),
           "composite_winner_declared": False})
    print(json.dumps(comparison, indent=2))
    print(output / "evaluation_summary.json")


def _alignment_rows(path: Path, *, expected_stems: tuple[str, ...], expected_steps: tuple[int, ...],
                    experiment_name: str | None, require_exact_steps: bool = True) -> dict[int, dict[str, Any]]:
    """Read a completed score file without re-running any external metric."""
    payload = _read_json(path / "pck_clip_results.json")
    if payload.get("metadata") != turbo_metadata() or payload.get("control_scale", 1.0) != 1.0:
        raise ValueError(f"Critic alignment requires matching Turbo provenance: {path / 'pck_clip_results.json'}")
    if experiment_name is not None and payload.get("experiment_name") not in (None, experiment_name):
        raise ValueError(f"Critic alignment score provenance names another experiment: {path / 'pck_clip_results.json'}")
    rows = payload.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError(f"Critic alignment score file has no checkpoint list: {path / 'pck_clip_results.json'}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("checkpoint_step") if isinstance(row, dict) else None
        if not isinstance(step, int) or step in result:
            raise ValueError(f"Critic alignment score file has malformed or duplicate checkpoint steps: {path}")
        result[step] = row
    if (tuple(sorted(result)) != expected_steps if require_exact_steps else not set(expected_steps) <= set(result)):
        raise ValueError(f"Critic alignment requested checkpoints differ from scored checkpoints at {path}")
    generation = _read_json(path / "generation_results.json")
    if generation.get("metadata") != turbo_metadata() or generation.get("control_scale", 1.0) != 1.0:
        raise ValueError(f"Critic alignment requires matching generated-image provenance: {path / 'generation_results.json'}")
    if generation.get("stems") != list(expected_stems):
        raise ValueError(f"Critic alignment generated-image stems differ from the Turbo spec at {path}")
    generated_steps = generation.get("checkpoints")
    if not isinstance(generated_steps, list) or (tuple(sorted(generated_steps)) != expected_steps if require_exact_steps else not set(expected_steps) <= set(generated_steps)):
        raise ValueError(f"Critic alignment requested checkpoints differ from generated checkpoints at {path}")
    spec = _read_json(path / "turbo_spec.json")
    if spec.get("kind") != "turbo_fixed_pose" or spec.get("turbo") != turbo_metadata() or tuple(spec.get("stems", ())) != expected_stems:
        raise ValueError(f"Critic alignment Turbo spec provenance is inconsistent: {path / 'turbo_spec.json'}")
    for step in expected_steps:
        if _generation_status(path, expected_stems, step) != "complete":
            raise ValueError(f"Critic alignment requires a complete generated image set for step {step}")
    return result


def _critic_alignment_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Prove the persisted branch metadata selects the exact critic contract."""
    resolved = spec.get("resolved_experiment")
    training = resolved.get("training_metadata") if isinstance(resolved, dict) else None
    if not isinstance(training, dict):
        raise ValueError("Critic alignment cannot prove the training pose-reward configuration from turbo_spec.json")
    if training.get("pose_loss") != "normalized_coordinate_huber":
        raise ValueError("Critic alignment requires normalized_coordinate_huber training provenance")
    if training.get("pose_loss_temperature") != 1.0:
        raise ValueError("Critic alignment requires pose_loss_temperature=1.0 provenance")
    return {
        "critic_identifier": FixedBoxKeypointRCNNCritic.identifier,
        "pose_loss": "normalized_coordinate_huber",
        "heatmap_input": "raw_logits_spatial_softmax",
        "temperature": 1.0,
        "coordinate_mapping": "authoritative_fixed_roi_cell_center",
        "coordinate_normalization": "both_prediction_and_target_inside_same_roi",
        "huber_delta": 1.0,
        "valid_mask": "reward_joint_valid_only",
        "phase1_unavailable": "excluded",
    }


def _alignment_target_tensors(record: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build fixed ROI tensors from the immutable training sidecar only."""
    if record.get("pose_reward_available") is not True:
        return None
    boxes, targets, valid = [], [], []
    people = record.get("people")
    if not isinstance(people, list):
        raise ValueError(f"{record.get('stem')}: available pose sidecar has no people list")
    for person in people:
        xywh, joints = person.get("bbox_training_xywh"), person.get("joint_provenance")
        if not isinstance(xywh, list) or len(xywh) != 4 or not isinstance(joints, list) or len(joints) != 17:
            raise ValueError(f"{record.get('stem')}: incomplete fixed ROI sidecar geometry")
        x, y, width, height = map(float, xywh)
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
            continue
        try:
            coordinates = [joint["training_coordinate"] for joint in joints]
            joint_valid = [bool(joint["reward_joint_valid"]) for joint in joints]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{record.get('stem')}: reward_joint_valid provenance is unavailable") from exc
        boxes.append((x, y, x + width, y + height)); targets.append(coordinates); valid.append(joint_valid)
    if not boxes or not any(any(row) for row in valid):
        return None
    return (
        torch.tensor(boxes, device=device, dtype=torch.float32),
        torch.tensor(targets, device=device, dtype=torch.float32),
        torch.tensor(valid, device=device, dtype=torch.bool),
    )


def _alignment_rgb(path: Path, expected_bucket: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(expected_bucket, list) or len(expected_bucket) != 2 or not all(isinstance(value, int) and value > 0 for value in expected_bucket):
        raise ValueError(f"Critic alignment cannot validate sidecar bucket geometry for {path}")
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if image.size != tuple(expected_bucket):
                raise ValueError(f"Critic alignment image geometry disagrees with sidecar bucket: {path}")
            pixels = np.asarray(image, dtype=np.float32)
    except OSError as exc:
        raise ValueError(f"Critic alignment generated image is unreadable: {path}") from exc
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(255.0).to(device)


def _alignment_sample_metrics(critic: FixedBoxKeypointRCNNCritic, rgb: torch.Tensor, boxes: torch.Tensor,
                              targets: torch.Tensor, valid: torch.Tensor) -> tuple[float, float, int]:
    """Evaluate the same raw-logit fixed-box normalized-Huber objective as training."""
    with torch.inference_mode():
        heatmaps = critic(rgb, [boxes])
        coordinates = soft_coordinates(heatmaps.logits, heatmaps.boxes_training, temperature=1.0)
        loss = normalized_coordinate_huber(coordinates, targets, heatmaps.boxes_training, valid, delta=1.0)
        distances = normalized_coordinate_distances(coordinates, targets, heatmaps.boxes_training, valid)
        count = int(valid.sum().item())
        if count < 1 or not torch.isfinite(loss) or not torch.isfinite(distances[valid]).all():
            raise ValueError("Critic alignment encountered an invalid normalized-coordinate metric")
        error = distances[valid].mean()
    return float(loss.item()), float(error.item()), count


def _alignment_statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    ordered = sorted(values); middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"mean": sum(values) / len(values), "median": median}


def _alignment_delta(value: float | int | None, baseline: float | int | None) -> dict[str, float | None]:
    if value is None or baseline is None:
        return {"absolute": None, "percent": None}
    absolute = float(value) - float(baseline)
    return {"absolute": absolute, "percent": None if float(baseline) == 0 else 100.0 * absolute / float(baseline)}


def _alignment_aggregate(step: int, label: str, samples: list[dict[str, Any]], external: dict[str, Any]) -> dict[str, Any]:
    losses, errors = [float(row["critic_loss"]) for row in samples], [float(row["normalized_coordinate_error"]) for row in samples]
    pose, clip = external.get("pose"), external.get("clip")
    if not isinstance(pose, dict) or not isinstance(clip, dict):
        raise ValueError(f"Critic alignment external score row is incomplete for step {step}")
    return {
        "checkpoint_step": step, "checkpoint_label": label,
        "eligible_sample_count": len(samples), "total_valid_joints": sum(int(row["valid_joint_count"]) for row in samples),
        "critic_loss": _alignment_statistics(losses), "normalized_coordinate_error": _alignment_statistics(errors),
        "pck_005": pose.get("pck_005"), "pck_010": pose.get("pck_010"), "pck_020": pose.get("pck_020"),
        "clip_mean_cosine_similarity": clip.get("mean_cosine_similarity"),
        "detection_coverage": pose.get("detection_coverage"), "joint_coverage": pose.get("joint_evaluation_coverage"),
    }


def _alignment_pearson(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean, second_mean = sum(first) / len(first), sum(second) / len(second)
    first_delta, second_delta = [value - first_mean for value in first], [value - second_mean for value in second]
    denominator = math.sqrt(sum(value * value for value in first_delta) * sum(value * value for value in second_delta))
    return None if denominator == 0 else sum(left * right for left, right in zip(first_delta, second_delta)) / denominator


def _alignment_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index]); ranks = [0.0] * len(values); index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2
        for member in order[index:end]: ranks[member] = rank
        index = end
    return ranks


def _alignment_correlations(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"observation_count": len(aggregates), "status": "diagnostic_descriptive_not_statistically_conclusive",
                              "expected_direction": "negative: lower internal critic error is better while higher external PCK is better"}
    for metric, value_key in (("critic_loss", "critic_loss"), ("normalized_coordinate_error", "normalized_coordinate_error")):
        errors = [float(row[value_key]["mean"]) for row in aggregates]
        result[metric] = {}
        for pck in ("pck_005", "pck_010", "pck_020"):
            values = [row[pck] for row in aggregates]
            if any(not isinstance(value, (float, int)) for value in values):
                result[metric][pck] = {"pearson": None, "spearman": None}
                continue
            pck_values = [float(value) for value in values]
            result[metric][pck] = {"pearson": _alignment_pearson(errors, pck_values),
                                   "spearman": _alignment_pearson(_alignment_ranks(errors), _alignment_ranks(pck_values))}
    return result


def _alignment_external_by_stem(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pose = row.get("pose", {})
    entries = pose.get("per_image") if isinstance(pose, dict) else None
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        if isinstance(item, dict) and isinstance(item.get("stem"), str):
            if item["stem"] in result:
                raise ValueError(f"Critic alignment external per-image metrics duplicate {item['stem']}")
            result[item["stem"]] = item
    return result


def _alignment_report(summary: dict[str, Any]) -> str:
    rows = summary["checkpoints"]
    lines = ["# Critic-alignment diagnostic", "", "Internal metrics are lower-is-better; external PCK is higher-is-better.",
             "Correlations are descriptive only (five checkpoint observations), not statistically conclusive.", "",
             "| checkpoint | critic loss mean | normalized coordinate error mean | PCK@.05 | PCK@.10 | PCK@.20 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['checkpoint_label']} ({row['checkpoint_step']}) | {row['critic_loss']['mean']:.7f} | {row['normalized_coordinate_error']['mean']:.7f} | {row['pck_005']:.6f} | {row['pck_010']:.6f} | {row['pck_020']:.6f} |")
    lines.extend(["", "## Descriptive correlations", ""])
    correlations = summary["correlations"]
    for metric in ("critic_loss", "normalized_coordinate_error"):
        for pck in ("pck_005", "pck_010", "pck_020"):
            values = correlations[metric][pck]
            pearson, spearman = values["pearson"], values["spearman"]
            lines.append(f"- {metric} vs {pck}: Pearson={pearson!r}; Spearman={spearman!r}.")
    lines.extend(["", "Interpretation criteria:", "", "- A. Internal critic improves and PCK improves: aligned / promising.", "- B. Internal critic improves and PCK worsens: reward misalignment.", "- C. Internal critic does not improve: auxiliary optimization/gradient effectiveness problem.", "", "A well-aligned critic generally has negative internal-error vs PCK correlation."])
    return "\n".join(lines) + "\n"


def critic_alignment(args) -> None:
    """Read existing Turbo images and score only the frozen fixed-box critic."""
    root = Path(args.output_root)
    spec = _read_json(root / "turbo_spec.json")
    if spec.get("kind") != "turbo_fixed_pose" or spec.get("turbo") != turbo_metadata() or spec.get("control_scale", 1.0) != 1.0:
        raise ValueError("Critic alignment requires a matching turbo_fixed_pose evaluation provenance")
    experiment_name = spec.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("Critic alignment requires an experiment_name in turbo_spec.json")
    if args.experiment_name is not None and args.experiment_name != experiment_name:
        raise ValueError("Critic alignment --experiment-name disagrees with turbo_spec.json")
    stems = tuple(spec.get("stems", ()))
    if not stems or len(set(stems)) != len(stems) or not all(isinstance(stem, str) and stem for stem in stems):
        raise ValueError("Critic alignment Turbo spec has an invalid diagnostic stem list")
    resolved = spec.get("resolved_experiment")
    configured_steps = resolved.get("steps") if isinstance(resolved, dict) else None
    if not isinstance(configured_steps, list):
        raise ValueError("Critic alignment cannot derive branch checkpoint steps from turbo_spec.json")
    expected_steps = normalize_turbo_steps(args.steps if args.steps is not None else configured_steps)
    if tuple(configured_steps) != expected_steps:
        raise ValueError("Critic alignment requested checkpoint set disagrees with resolved experiment provenance")
    contract = _critic_alignment_contract(spec)
    branch_rows = _alignment_rows(root, expected_stems=stems, expected_steps=expected_steps, experiment_name=experiment_name)
    baseline = resolved.get("baseline") if isinstance(resolved, dict) else None
    if not isinstance(baseline, dict):
        raise ValueError("Critic alignment requires baseline provenance in turbo_spec.json")
    baseline_root = Path(args.baseline_output_root) if args.baseline_output_root else Path(baseline.get("output_root", ""))
    baseline_step = args.baseline_step if args.baseline_step is not None else baseline.get("checkpoint_step")
    if not isinstance(baseline_step, int) or not str(baseline_root):
        raise ValueError("Critic alignment baseline provenance is incomplete")
    baseline_spec = _read_json(baseline_root / "turbo_spec.json")
    assert_turbo_diagnostic_contract(spec, baseline_spec, branch_name="critic-alignment baseline")
    baseline_rows = _alignment_rows(baseline_root, expected_stems=stems, expected_steps=(baseline_step,), experiment_name=None,
                                    require_exact_steps=False)
    sidecar_metadata, sidecar_records = load_sidecar(args.sidecar)
    if args.expected_sidecar_records_sha256 and sidecar_metadata.get("records_sha256") != args.expected_sidecar_records_sha256:
        raise ValueError("Critic alignment sidecar records SHA-256 mismatch")
    by_stem = {record.get("stem"): record for record in sidecar_records}
    missing_targets = [stem for stem in stems if stem not in by_stem]
    if missing_targets:
        raise ValueError(f"Critic alignment sidecar is missing diagnostic stems: {missing_targets[:3]}")
    # Every artifact and sidecar invariant is checked before model weights are constructed.
    device = torch.device(args.device)
    targets = {stem: _alignment_target_tensors(by_stem[stem], device) for stem in stems}
    eligible_stems = tuple(stem for stem in stems if targets[stem] is not None)
    if not eligible_stems:
        raise ValueError("Critic alignment has no Phase-1 eligible pose targets")
    all_roots = (("baseline", baseline_root, baseline_step), *(("branch", root, step) for step in expected_steps))
    for _, artifact_root, step in all_roots:
        for stem in stems:
            image = artifact_root / "fixed_pose" / stem / f"step_{step:06d}.png"
            if not image.is_file():
                raise FileNotFoundError(f"Critic alignment requires existing generated image: {image}")
            if targets[stem] is not None:
                _alignment_rgb(image, by_stem[stem].get("bucket"), device=torch.device("cpu"))
    critic = FixedBoxKeypointRCNNCritic().to(device).eval()
    checkpoint_items = [(baseline_step, str(baseline.get("label", f"baseline {baseline_step}")), baseline_root, baseline_rows[baseline_step])]
    checkpoint_items.extend((step, str(resolved.get("labels", {}).get("checkpoint_template", "checkpoint {step}")).format(step=step), root, branch_rows[step]) for step in expected_steps)
    samples: list[dict[str, Any]] = []; aggregates: list[dict[str, Any]] = []
    for step, label, artifact_root, external in checkpoint_items:
        external_by_stem = _alignment_external_by_stem(external)
        checkpoint_samples = []
        for stem in eligible_stems:
            target = targets[stem]
            assert target is not None
            boxes, coordinates, valid = target
            image = artifact_root / "fixed_pose" / stem / f"step_{step:06d}.png"
            loss, error, count = _alignment_sample_metrics(critic, _alignment_rgb(image, by_stem[stem]["bucket"], device), boxes, coordinates, valid)
            item = {"checkpoint_step": step, "checkpoint_label": label, "stem": stem, "source": by_stem[stem].get("source"),
                    "critic_loss": loss, "normalized_coordinate_error": error, "valid_joint_count": count}
            if stem in external_by_stem:
                item["external"] = external_by_stem[stem]
            checkpoint_samples.append(item); samples.append(item)
        aggregates.append(_alignment_aggregate(step, label, checkpoint_samples, external))
    baseline_aggregate = aggregates[0]
    delta_keys = ("critic_loss", "normalized_coordinate_error", "total_valid_joints", "pck_005", "pck_010", "pck_020", "clip_mean_cosine_similarity", "detection_coverage", "joint_coverage")
    deltas = {str(row["checkpoint_step"]): {
        key: _alignment_delta(row[key]["mean"] if key in ("critic_loss", "normalized_coordinate_error") else row[key],
                              baseline_aggregate[key]["mean"] if key in ("critic_loss", "normalized_coordinate_error") else baseline_aggregate[key])
        for key in delta_keys
    } for row in aggregates[1:]}
    summary = {"format_version": 1, "experiment_name": experiment_name, "read_only": True, "critic_contract": contract,
               "sidecar": {"path": str(args.sidecar), "records_sha256": sidecar_metadata.get("records_sha256")},
               "baseline": {"checkpoint_step": baseline_step, "checkpoint_label": baseline_aggregate["checkpoint_label"], "output_root": str(baseline_root)},
               "phase1_excluded_stems": [stem for stem in stems if stem not in eligible_stems], "checkpoints": aggregates,
               "deltas_vs_baseline": deltas, "correlations": _alignment_correlations(aggregates)}
    _write(root / "critic_alignment_samples.json", {"format_version": 1, "experiment_name": experiment_name, "samples": samples})
    _write(root / "critic_alignment_summary.json", summary)
    report_path = root / "critic_alignment_report.md"; report_path.write_text(_alignment_report(summary), encoding="utf-8")
    print(root / "critic_alignment_summary.json")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--spec", required=True, help="JSON Turbo experiment specification")
    common.add_argument("--checkpoint-root"); common.add_argument("--hf-repo-id"); common.add_argument("--hf-namespace"); common.add_argument("--output-root")
    common.add_argument("--latent-root"); common.add_argument("--text-conditioning-root"); common.add_argument("--turbo-ckpt")
    common.add_argument("--diagnostic-manifest"); common.add_argument("--dataset-root"); common.add_argument("--reference-sidecar"); common.add_argument("--clip-model-id")
    selection = common.add_mutually_exclusive_group()
    selection.add_argument("--steps", type=int, nargs="+", help="exact checkpoint steps to process")
    selection.add_argument("--all-checkpoints", action="store_true", help="discover direct step_XXXXXX.pt files under checkpoint_root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, function in (("preflight", preflight), ("generate", generate), ("score", score), ("report", report)):
        item = sub.add_parser(name, parents=[common]); item.set_defaults(function=function)
    experiment = sub.add_parser("experiment", help="run the staged generic controlled-branch evaluation")
    experiment.add_argument("--checkpoint-root", required=True)
    experiment.add_argument("--steps", type=int, nargs="+", required=True, help="exact branch checkpoint steps")
    experiment.add_argument("--output-root", required=True)
    experiment.add_argument("--experiment-name", required=True)
    experiment.add_argument("--checkpoint-label-template", default="checkpoint {step}")
    experiment.add_argument("--expected-sha256", nargs="*", default=[], metavar="STEP=SHA256")
    experiment.add_argument("--baseline-output-root"); experiment.add_argument("--baseline-step", type=int); experiment.add_argument("--baseline-label")
    experiment.add_argument("--hf-repo-id", default=None, help="recorded provenance only for direct-local evaluation")
    experiment.add_argument("--diagnostic-manifest", default="data/manifests/diagnostic_val.jsonl")
    experiment.add_argument("--canonical-reference-spec", default="docs/evaluation/turbo-8step-cfg0/turbo_spec.json")
    experiment.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    experiment.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    experiment.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
    experiment.add_argument("--reference-sidecar", default="data/manifests/diagnostic_reference_pose.json")
    experiment.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    experiment.add_argument("--dataset-root", default=None)
    experiment.add_argument("--skip-existing", action="store_true", help="reuse only complete validated artifacts (the default behavior)")
    experiment.set_defaults(function=lambda args: (preflight(args), generate(args), score(args), report(args)), dynamic_experiment=True,
                            spec=None, hf_namespace=None, all_checkpoints=False)
    alignment = sub.add_parser("critic-alignment", help="score existing Turbo images with the frozen normalized-coordinate critic")
    alignment.add_argument("--output-root", required=True, type=Path, help="completed branch evaluation directory")
    alignment.add_argument("--sidecar", required=True, type=Path, help="immutable training pose-target sidecar directory")
    alignment.add_argument("--steps", type=int, nargs="+", help="must exactly equal the branch generated/scored checkpoint set")
    alignment.add_argument("--baseline-output-root", type=Path, help="override baseline output root recorded in turbo_spec.json")
    alignment.add_argument("--baseline-step", type=int, help="override baseline checkpoint step recorded in turbo_spec.json")
    alignment.add_argument("--experiment-name", help="optional exact provenance assertion")
    alignment.add_argument("--expected-sidecar-records-sha256", help="optional exact immutable sidecar digest assertion")
    alignment.add_argument("--device", default="cuda", help="critic device; CPU is supported for focused diagnostics")
    alignment.set_defaults(function=critic_alignment)
    return parser


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
