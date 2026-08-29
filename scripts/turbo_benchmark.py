"""Generic, staged, evaluation-only Krea-2 Turbo Pose-ControlNet benchmark.

Experiments are selected by a JSON spec and exact checkpoint subset.  This
module intentionally contains no experiment-specific run names, roots, or
checkpoint lists; changing an experiment never requires editing this source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

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
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate, clip_feature_tensor, cosine_from_embeddings, prepare_clip_scoring_inputs
from pose_controlnet.turbo_evaluation import (
    TurboExperiment,
    assert_turbo_diagnostic_contract,
    assert_exact_diagnostic_stems,
    discover_turbo_checkpoint_steps,
    exact_direct_local_turbo_checkpoints,
    exact_local_turbo_checkpoints,
    load_turbo_experiment_spec,
    normalize_turbo_steps,
    raw_to_turbo_control_compatibility,
    sample_turbo_pose_image,
    turbo_metadata,
    turbo_scoring_geometry,
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
    return load_turbo_experiment_spec(args.spec, overrides={
        "checkpoint_root": args.checkpoint_root,
        "hf_repo_id": args.hf_repo_id,
        "hf_namespace": args.hf_namespace,
        "output_root": args.output_root,
    })


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
    metadata = config.training_metadata if training_metadata is None else training_metadata
    return {"label": label, "checkpoint_step": row["checkpoint_step"], **metadata,
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
    make_contact_sheet(grid_rows[:min(4, len(grid_rows))], output / "turbo_checkpoint_selection_grid.png", thumbnail_width=180, thumbnail_height=180, column_labels=tuple(labels))
    make_contact_sheet(grid_rows, output / "turbo_full_contact_sheet.png", thumbnail_width=320, thumbnail_height=320, column_labels=tuple(labels))
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
           "qualitative_grids": {"checkpoint_selection": "turbo_checkpoint_selection_grid.png", "full_contact_sheet": "turbo_full_contact_sheet.png"},
           "production_winner_declared": False})
    print(json.dumps(comparison, indent=2))
    print(output / "evaluation_summary.json")


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
    return parser


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
