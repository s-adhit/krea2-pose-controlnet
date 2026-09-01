"""Turbo capacity evaluation with an explicit no-generation score-only path.

``--stage score-only`` consumes only existing PNGs and persisted metadata.  It
does not open checkpoints, construct a training/model/sampler, or regenerate
any image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.capacity_resolution import (
    native_resolution_provenance, preprocess_native_evaluation_pair,
)
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet
from pose_controlnet.overfit_capacity import (
    NATIVE_RESOLUTION_POLICY, OVERFIT_STEPS, SelectedLatentShardDataset,
    canonical_resolution_policy, validate_manifest,
)
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.reference_pose import ReferencePoseError, load_exact_capacity_reference_sidecar


LEGACY_NATIVE_EXPERIMENT = "overfit32-mixed-r64-mse"
LEGACY_NATIVE_CHECKPOINT_ROOT = Path(
    "/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-mixed-r64-mse"
)
_LEGACY_EMPTY_RESOLUTION_VALUES = (None, "none")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment", required=True); value.add_argument("--stage", choices=("generate", "score", "score-only", "report", "all"), default="all")
    value.add_argument("--checkpoint-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints")
    value.add_argument("--output-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation")
    value.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    value.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    value.add_argument("--dataset-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_hf")
    value.add_argument("--reference-sidecar", type=Path, help="Required immutable exact-manifest authoritative pose reference JSONL for score-only scoring.")
    value.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
    value.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return value


def _inputs(args) -> tuple[Path, Path, tuple[str, ...], SelectedLatentShardDataset]:
    stems = validate_manifest("overfit32-mixed-r64-mse") if args.experiment.startswith("overfit32-mixed-r64-") else validate_manifest(args.experiment)
    checkpoints = Path(args.checkpoint_root) / args.experiment; output = Path(args.output_root) / args.experiment
    if checkpoints == output or checkpoints in output.parents or output in checkpoints.parents: raise ValueError("Checkpoint and evaluation output namespaces must be disjoint")
    selected = SelectedLatentShardDataset(PreparedLatentShardDataset(args.latent_root, "train", text_conditioning_root=args.text_conditioning_root), stems)
    return checkpoints, output, stems, selected


def _metadata_resolution_values(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return every recorded training-resolution field without inferring one."""
    values = {"training_resolution": metadata.get("training_resolution"),
              "resolution_policy": metadata.get("resolution_policy")}
    scientific = metadata.get("scientific_config")
    values["scientific_config.resolution"] = scientific.get("resolution") if isinstance(scientific, dict) else None
    return values


def _is_legacy_empty_resolution(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "none")


def _legacy_checkpoint_set(checkpoints: Path) -> None:
    """Require the historic run's complete, and only complete, checkpoint schedule."""
    observed = {path.name for path in checkpoints.glob("step_*.pt")}
    expected = {f"step_{step:06d}.pt" for step in OVERFIT_STEPS}
    if observed != expected:
        raise ValueError("Legacy native compatibility requires the exact checkpoint set 0,50,100,200,300,400,500")


def _legacy_generated_provenance(*, generated: dict[str, Any] | None, experiment_name: str,
                                 stems: tuple[str, ...], native_geometry: dict[str, Any]) -> None:
    """Require regenerated native metadata before score/report consumes images."""
    if not isinstance(generated, dict):
        raise ValueError("Reporting/scoring legacy native compatibility requires persisted regenerated native generation metadata")
    if (generated.get("experiment") != experiment_name or generated.get("training_set_overfit") is not True
            or generated.get("stems") != list(stems)
            or generated.get("steps") != list(OVERFIT_STEPS)
            or generated.get("training_resolution") != NATIVE_RESOLUTION_POLICY
            or generated.get("evaluation_resolution") != NATIVE_RESOLUTION_POLICY):
        raise ValueError("Reporting/scoring legacy native compatibility requires exact regenerated native generation evidence")
    generation_provenance = generated.get("evaluation_provenance")
    if not isinstance(generation_provenance, dict) or (
            generation_provenance.get("training_resolution") != NATIVE_RESOLUTION_POLICY
            or generation_provenance.get("evaluation_resolution") != NATIVE_RESOLUTION_POLICY
            or generation_provenance.get("training_resolution_source") != "legacy_native_compatibility"):
        raise ValueError("Reporting/scoring legacy native compatibility requires explicit regenerated native provenance")
    if any(generated.get(key) is not None for key in ("resolution_manifest", "resolution_cache", "alternate_resolution_cache")) \
            or any(generation_provenance.get(key) is not None for key in ("resolution_manifest", "resolution_cache", "alternate_resolution_cache")):
        raise ValueError("Legacy native compatibility refuses alternate-resolution generation provenance")
    geometry = _native_generation_geometry(generated, stems)
    expected_geometry = {
        stem: {field: native_geometry["samples"][stem][field]
               for field in ("source_size", "resized_size", "crop_box", "bucket")}
        for stem in stems
    }
    if geometry != expected_geometry:
        raise ValueError("Legacy native compatibility requires generation geometry to match verified native latent geometry")


def _legacy_native_compatibility(*, checkpoints: Path, metadata: dict[str, Any], experiment_name: str,
                                 stems: tuple[str, ...], native_geometry: dict[str, Any],
                                 generated: dict[str, Any] | None,
                                 require_persisted_generation: bool) -> bool:
    """Prove the one pre-resolution-axis native run; reject every other ambiguity."""
    if experiment_name != LEGACY_NATIVE_EXPERIMENT:
        return False
    if checkpoints.resolve() != LEGACY_NATIVE_CHECKPOINT_ROOT.resolve():
        raise ValueError("Legacy native compatibility requires the expected legacy checkpoint root")
    if stems != validate_manifest(LEGACY_NATIVE_EXPERIMENT):
        raise ValueError("Legacy native compatibility requires the exact immutable Mixed-32 stem order")
    declared = _metadata_resolution_values(metadata)
    if not all(_is_legacy_empty_resolution(value) for value in declared.values()):
        raise ValueError("Legacy native compatibility refuses contradictory checkpoint resolution provenance")
    checkpoint_steps = metadata.get("checkpoint_steps")
    if checkpoint_steps is not None and checkpoint_steps != list(OVERFIT_STEPS):
        raise ValueError("Legacy native compatibility refuses a non-canonical checkpoint schedule")
    _legacy_checkpoint_set(checkpoints)
    if metadata.get("resolution_manifest") is not None or metadata.get("resolution_cache_root") is not None:
        raise ValueError("Legacy native compatibility refuses alternate-resolution checkpoint provenance")
    cache = checkpoints.parent.parent / "resolution_cache" / experiment_name
    if (checkpoints / "resolution_manifest.json").exists() or cache.exists():
        raise ValueError("Legacy native compatibility refuses an alternate-resolution cache or provenance")
    if require_persisted_generation:
        _legacy_generated_provenance(
            generated=generated, experiment_name=experiment_name, stems=stems, native_geometry=native_geometry,
        )
    return True


def _training_resolution(checkpoints: Path, *, experiment_name: str | None = None,
                         stems: tuple[str, ...] | None = None, native_geometry: dict[str, Any] | None = None,
                         generated: dict[str, Any] | None = None,
                         require_persisted_generation: bool = True) -> tuple[str, str | None]:
    """Read training provenance only; evaluation policy is independently fixed."""
    try:
        metadata = json.loads((checkpoints / "experiment_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Evaluation requires readable training provenance: {checkpoints / 'experiment_metadata.json'}") from exc
    if experiment_name == LEGACY_NATIVE_EXPERIMENT:
        # This identity is intentionally not a general legacy alias.  Its sole
        # accepted resolution evidence is the pre-axis absent/``none`` form.
        declared = _metadata_resolution_values(metadata)
        if not all(_is_legacy_empty_resolution(value) for value in declared.values()):
            raise ValueError("Legacy native compatibility refuses contradictory checkpoint resolution provenance")
        if stems is not None and native_geometry is not None and _legacy_native_compatibility(
                checkpoints=checkpoints, metadata=metadata, experiment_name=experiment_name,
                stems=stems, native_geometry=native_geometry, generated=generated,
                require_persisted_generation=require_persisted_generation):
            return NATIVE_RESOLUTION_POLICY, "legacy_native_compatibility"
        raise ValueError("Legacy native compatibility evidence is incomplete")
    value = metadata.get("training_resolution")
    if value is None:
        scientific = metadata.get("scientific_config")
        value = scientific.get("resolution") if isinstance(scientific, dict) else metadata.get("resolution_policy")
    try:
        return canonical_resolution_policy(value), None
    except (TypeError, ValueError) as exc:
        raise ValueError("Training provenance does not declare a supported native or 768 resolution") from exc


def _evaluation_provenance(checkpoints: Path, data: SelectedLatentShardDataset,
                           stems: tuple[str, ...], *, experiment_name: str | None = None,
                           generated: dict[str, Any] | None = None,
                           require_persisted_generation: bool = True) -> dict[str, Any]:
    native_geometry = native_resolution_provenance(data, stems)
    training_resolution, training_resolution_source = _training_resolution(
        checkpoints, experiment_name=experiment_name, stems=stems,
        native_geometry=native_geometry, generated=generated,
        require_persisted_generation=require_persisted_generation,
    )
    value = {
        "training_resolution": training_resolution,
        "evaluation_resolution": NATIVE_RESOLUTION_POLICY,
        "native_geometry": native_geometry,
    }
    if training_resolution_source is not None:
        value["training_resolution_source"] = training_resolution_source
    return value


def _archive_command(output: Path) -> str:
    archived = output.with_name(output.name + ".partial-768-eval-archive")
    return f"mv -- {output} {archived}"


def _persisted_generation_metadata(output: Path) -> dict[str, Any] | None:
    """Read-only evidence for the narrow legacy resolver; no artifact is created."""
    path = output / "generation_results.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing generation metadata is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Existing generation metadata must be a JSON object: {path}")
    return value


def _verify_existing_generation(output: Path, *, experiment_name: str, stems: tuple[str, ...],
                                provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed before reporting or scoring; never create or replace a PNG."""
    path = output / "generation_results.json"
    try:
        generated = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Reporting/scoring requires compatible native generation metadata; "
            f"archive incomplete artifacts first if present: {_archive_command(output)}"
        ) from exc
    if generated.get("experiment") != experiment_name or generated.get("training_set_overfit") is not True:
        raise ValueError("Generation metadata does not identify the requested training-set capacity experiment")
    if generated.get("stems") != list(stems) or generated.get("steps") != list(OVERFIT_STEPS):
        raise ValueError("Generation metadata is not the exact immutable 32-sample/checkpoint set")
    if provenance is not None and provenance.get("training_resolution_source") == "legacy_native_compatibility":
        _legacy_generated_provenance(
            generated=generated, experiment_name=experiment_name, stems=stems,
            native_geometry=provenance["native_geometry"],
        )
    generation_provenance = generated.get("evaluation_provenance")
    if provenance is not None and (generated.get("training_resolution") != provenance["training_resolution"]
            or generated.get("evaluation_resolution") != NATIVE_RESOLUTION_POLICY
            or not isinstance(generation_provenance, dict)
            or generation_provenance.get("native_geometry") != provenance["native_geometry"]):
        raise ValueError(
            "Existing evaluation metadata is incompatible with required native evaluation; "
            f"archive it first: {_archive_command(output)}"
        )
    missing = [str(path) for stem in stems for path in (
        output / "training_set" / stem / "control.png",
        output / "training_set" / stem / "target.png",
        *(output / "training_set" / stem / f"step_{step:06d}.png" for step in OVERFIT_STEPS),
    ) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Reporting/scoring refuses incomplete existing generation set (first missing: {missing[0]}); "
            f"archive it before a clean retry: {_archive_command(output)}"
        )
    return generated


def _recover_native_evaluation_pairs(data: SelectedLatentShardDataset, stems: tuple[str, ...],
                                     physical: dict[str, Any]) -> dict[str, Any]:
    """Preflight exact paired native RGB/control geometry before any evaluation write."""
    if set(stems) - set(physical):
        raise ValueError("Selected training samples cannot be resolved in the authoritative dataset snapshot")
    samples = {data[index]["stem"]: data[index] for index in range(len(data))}
    if len(samples) != len(data) or set(stems) - set(samples):
        raise ValueError("Selected training samples lack exact persisted native latent records")
    pairs: dict[str, Any] = {}
    for stem in stems:
        sample = samples[stem]
        pair = preprocess_native_evaluation_pair(physical[stem], sample)
        bucket = tuple(sample.get("bucket") or ())
        if pair.rgb.size != bucket or pair.control.size != bucket:
            raise ValueError(f"{stem}: paired native RGB/control geometry is not recoverable exactly")
        pairs[stem] = pair
    return pairs


def _generation_results(*, experiment_name: str, stems: tuple[str, ...], provenance: dict[str, Any],
                        turbo: dict[str, Any], compatibility: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable provenance record written only after successful generation."""
    return {
        "experiment": experiment_name, "training_set_overfit": True, "stems": list(stems),
        "steps": list(OVERFIT_STEPS), "training_resolution": provenance["training_resolution"],
        "evaluation_resolution": NATIVE_RESOLUTION_POLICY, "evaluation_provenance": provenance,
        "turbo": turbo, "raw_to_turbo_control_compatibility": compatibility,
    }


def _native_generation_geometry(generated: dict[str, Any], stems: tuple[str, ...]) -> dict[str, dict[str, list[int]]]:
    """Use the geometry written by native generation, never a training cache."""
    samples = generated.get("evaluation_provenance", {}).get("native_geometry", {}).get("samples")
    if not isinstance(samples, dict) or set(samples) != set(stems):
        raise ValueError("Generation metadata lacks exact persisted native geometry for the immutable manifest")
    result: dict[str, dict[str, list[int]]] = {}
    for stem in stems:
        geometry = samples[stem]
        if not isinstance(geometry, dict):
            raise ValueError(f"{stem}: persisted native generation geometry is malformed")
        fields = (("source_size", 2), ("resized_size", 2), ("crop_box", 4), ("bucket", 2))
        copied: dict[str, list[int]] = {}
        for field, length in fields:
            value = geometry.get(field)
            if not isinstance(value, list) or len(value) != length or any(not isinstance(item, int) for item in value):
                raise ValueError(f"{stem}: persisted native generation geometry lacks valid {field}")
            copied[field] = list(value)
        source_width, source_height = copied["source_size"]
        resized_width, resized_height = copied["resized_size"]
        left, top, right, bottom = copied["crop_box"]
        bucket_width, bucket_height = copied["bucket"]
        if (min(source_width, source_height, resized_width, resized_height, bucket_width, bucket_height) < 1
                or right - left != bucket_width or bottom - top != bucket_height
                or left < 0 or top < 0 or right > resized_width or bottom > resized_height):
            raise ValueError(f"{stem}: persisted native generation geometry is incompatible")
        result[stem] = copied
    return result


def _states(root: Path) -> list[tuple[int, Path]]:
    from pose_controlnet.checkpointing import load_training_state
    values = []
    for step in OVERFIT_STEPS:
        path = root / f"step_{step:06d}.pt"
        if not path.is_file(): raise FileNotFoundError(f"Missing required capacity checkpoint: {path}")
        state = load_training_state(path)
        if state.get("global_step") != step or state.get("overfit_capacity", {}).get("fresh_lora_checkpoint_loaded") is not False:
            raise ValueError(f"Capacity checkpoint is not the expected fresh run: {path}")
        values.append((step, path))
    return values


def generate(args) -> None:
    # Keep all generation-only imports local: score-only must not instantiate a
    # Raw/Turbo model, VAE, sampler, or checkpoint reader.
    from pose_controlnet.checkpointing import load_training_state
    from pose_controlnet.dataset_index import validate_posebridge_snapshot
    from pose_controlnet.evaluation import save_image
    from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
    from pose_controlnet.overfit_capacity import deterministic_seed
    from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_metadata
    from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
    checkpoints, output, stems, data = _inputs(args)
    provenance = _evaluation_provenance(
        checkpoints, data, stems, experiment_name=args.experiment,
        require_persisted_generation=False,
    )
    if output.exists() and any(output.iterdir()):
        if (output / "generation_results.json").is_file():
            _verify_existing_generation(output, experiment_name=args.experiment, stems=stems, provenance=provenance)
            return
        raise FileExistsError(
            "Refusing to mix a native evaluation with pre-existing incomplete artifacts; "
            f"archive them first: {_archive_command(output)}"
        )
    snapshot = validate_posebridge_snapshot(args.dataset_root)
    physical = {record.stem: record for record in snapshot.records_by_split["train"]}
    pairs = _recover_native_evaluation_pairs(data, stems, physical)
    if not torch.cuda.is_available(): raise RuntimeError("Run Turbo generation only from the GH200 host shell with CUDA visible")
    states = _states(checkpoints)
    model = build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval(); vae = load_krea_vae("cuda"); compatibility = {}
    for stem in stems:
        sample, record, directory = _sample_by_stem(data, stem), physical[stem], output / "training_set" / stem
        directory.mkdir(parents=True, exist_ok=True)
        pair = pairs[stem]
        for name, image in (("control.png", pair.control), ("target.png", pair.rgb)):
            destination = directory / name
            if destination.exists(): raise FileExistsError(f"Refusing to replace an existing native evaluation artifact: {destination}")
            image.save(destination)
        _write(directory / "metadata.json", {"stem": stem, "prompt": sample["prompt"], "seed": deterministic_seed(stem), "training_set_overfit": True,
            "target_rgb": str(record.rgb_path), "pose_control": str(record.control_path), "training_resolution": provenance["training_resolution"],
            "evaluation_resolution": NATIVE_RESOLUTION_POLICY, "native_geometry": provenance["native_geometry"]["samples"][stem], "turbo": turbo_metadata()})
        for step, path in states:
            destination = directory / f"step_{step:06d}.png"
            if destination.exists(): continue
            state = load_training_state(path); compatibility[str(step)] = raw_to_turbo_control_compatibility(model, state)
            load_trainable_state_dict(model, state["model"])
            pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample, torch.device("cuda"), deterministic_seed(stem))
            save_image(pixels, destination)
    _write(output / "generation_results.json", _generation_results(
        experiment_name=args.experiment, stems=stems, provenance=provenance,
        turbo=turbo_metadata(), compatibility=compatibility,
    ))


def _update_score_summary(output: Path, metrics: dict[str, Any]) -> None:
    """Refresh reports without touching qualitative grids or generated PNGs."""
    existing: dict[str, Any] = {}
    summary = output / "overfit_summary.json"
    if summary.is_file():
        try:
            existing = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing overfit summary is malformed: {summary}") from exc
    if "qualitative_grids" in existing:
        metrics = {**metrics, "qualitative_grids": existing["qualitative_grids"]}
    _write(summary, metrics)


def _read_existing_summary(output: Path) -> dict[str, Any]:
    """Return a valid prior summary so report-only work never discards provenance."""
    path = output / "overfit_summary.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing overfit summary is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Existing overfit summary must be a JSON object: {path}")
    return value


def _read_existing_metrics(output: Path) -> dict[str, Any] | None:
    """Read scores only when they already exist; report never computes them."""
    path = output / "training_set_overfit_metrics.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing overfit metrics are malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Existing overfit metrics must be a JSON object: {path}")
    return value


def score(args) -> None:
    checkpoints, output, stems, data = _inputs(args)
    if args.reference_sidecar is None:
        raise ValueError("Score-only capacity evaluation requires --reference-sidecar; diagnostic_reference_pose.json is never a fallback")
    provenance = _evaluation_provenance(
        checkpoints, data, stems, experiment_name=args.experiment,
        generated=_persisted_generation_metadata(output),
    )
    generated = _verify_existing_generation(output, experiment_name=args.experiment, stems=stems, provenance=provenance)
    geometry = _native_generation_geometry(generated, stems)
    metadata, records = load_exact_capacity_reference_sidecar(
        args.reference_sidecar, experiment_name=args.experiment, expected_stems=stems, geometry_by_stem=geometry,
    )
    by_stem = {row["stem"]: row for row in records}; sidecar = {"records": records}
    from transformers import CLIPModel, CLIPProcessor
    from scripts.turbo_benchmark import _clip_score
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval(); processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    rows = []
    for step in OVERFIT_STEPS:
        image_for = lambda stem, current=step: output / "training_set" / stem / f"step_{current:06d}.png"
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for, detector=detector, confidence_threshold=.5, require_images=True)
        clip_rows = [{"stem": stem, "source": by_stem[stem]["source"], "cosine_similarity": _clip_score(clip, processor, device, _sample_by_stem(data, stem)["prompt"], image_for(stem))} for stem in stems]
        values = aggregate([row["cosine_similarity"] for row in clip_rows])
        rows.append({"checkpoint_step": step, "training_set_overfit": True, "evaluable_sample_count": len(stems), "pose": pose,
            "clip": {"mean_cosine_similarity": values["mean"], "sample_count": values["sample_count"], "per_sample": clip_rows}})
    metrics = {"experiment": args.experiment, "training_set_equals_evaluation_set": True, "evaluable_sample_count": len(stems),
               "training_resolution": provenance["training_resolution"], "evaluation_resolution": NATIVE_RESOLUTION_POLICY,
               "evaluation_provenance": provenance,
               "reference_sidecar": str(args.reference_sidecar.resolve()), "sidecar_records_sha256": metadata.get("records_sha256"),
               "danbooru_pck": "explicitly unavailable when authoritative targets are unavailable", "checkpoints": rows}
    _write(output / "training_set_overfit_metrics.json", metrics)
    _update_score_summary(output, metrics)


def report(args) -> None:
    checkpoints, output, stems, data = _inputs(args)
    provenance = _evaluation_provenance(
        checkpoints, data, stems, experiment_name=args.experiment,
        generated=_persisted_generation_metadata(output),
    )
    _verify_existing_generation(output, experiment_name=args.experiment, stems=stems, provenance=provenance)
    existing = _read_existing_summary(output); metrics = _read_existing_metrics(output)
    labels = ("Pose control", "Target training RGB", *(f"Step {step}" for step in OVERFIT_STEPS))
    rows = [(stem, [output / "training_set" / stem / "control.png", output / "training_set" / stem / "target.png", *(output / "training_set" / stem / f"step_{step:06d}.png" for step in OVERFIT_STEPS)]) for stem in stems]
    make_contact_sheet(rows[:4], output / "checkpoint_selection_grid.png", thumbnail_width=180, thumbnail_height=180, column_labels=labels)
    make_contact_sheet(rows, output / "full_training_set_contact_sheet.png", thumbnail_width=320, thumbnail_height=320, column_labels=labels)
    grids = {"checkpoint_selection": "checkpoint_selection_grid.png", "full_contact_sheet": "full_training_set_contact_sheet.png"}
    if metrics is not None:
        # The score artifact is authoritative when present; retain compatible
        # pre-existing report provenance that it does not supersede.
        summary = {**existing, **metrics}
    else:
        # Do not manufacture metric-shaped checkpoint rows or null values:
        # qualitative inspection is valid before every domain has a scorer.
        summary = {
            **existing,
            "experiment": args.experiment,
            "training_set_equals_evaluation_set": True,
            "sample_count": len(stems),
            "checkpoints": list(OVERFIT_STEPS),
            "training_resolution": provenance["training_resolution"],
            "evaluation_resolution": NATIVE_RESOLUTION_POLICY,
            "provenance": {
                "generation_metadata": "generation_results.json",
                "immutable_manifest_stems": list(stems),
                **provenance,
            },
            "quantitative_scoring": "not_yet_available",
        }
    summary["qualitative_grids"] = grids
    _write(output / "overfit_summary.json", summary)


def main() -> None:
    args = parser().parse_args()
    if args.stage in ("generate", "all"): generate(args)
    if args.stage in ("score", "score-only", "all"): score(args)
    if args.stage in ("report", "all"): report(args)


if __name__ == "__main__": main()
