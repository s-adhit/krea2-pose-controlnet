"""Read-only 8-step Turbo evaluation for the completed LR=5e-5 continuation.

This entrypoint is deliberately separate from ``turbo_benchmark.py``.  It
uses its unchanged sampler, PCK implementation, CLIP implementation, prompts,
controls, geometry, and seeds while accepting only the six exact
completion-marked checkpoints from ``pose-learning-900-lr5e5-to1500``.
It never constructs an optimizer or invokes any training operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.dataset_index import validate_posebridge_snapshot
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, save_image
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.turbo_evaluation import (
    LR5E5_CHECKPOINT_ROOT, LR5E5_HF_REPO_ID, LR5E5_HF_RUN_NAME,
    LR5E5_TURBO_CHECKPOINT_STEPS, LR5E5_TURBO_EVALUATION_ROOT,
    ORIGINAL_TURBO_EVALUATION_ROOT, assert_lr5e5_diagnostic_contract,
    assert_lr5e5_turbo_output_isolated, exact_lr5e5_turbo_checkpoints,
    raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_metadata,
    turbo_scoring_geometry,
)
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts.turbo_benchmark import (
    _clip_score, _dataset_and_spec as _original_dataset_and_spec,
    _missing_generation_checkpoints,
)


ROOT = Path("/lambda/nfs/adhit/krea2-pose")
OUTPUT = LR5E5_TURBO_EVALUATION_ROOT
ORIGINAL_OUTPUT = ORIGINAL_TURBO_EVALUATION_ROOT


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required immutable original Turbo result is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Malformed JSON object: {path}")
    return value


def _branch_spec(output: Path, args) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
    dataset, stems, spec = _original_dataset_and_spec(args)
    original_spec = _read_json(Path(args.original_output_dir) / "turbo_spec.json")
    assert_lr5e5_diagnostic_contract(spec, original_spec)
    destination = output / "turbo_spec.json"
    if destination.is_file() and _read_json(destination) != spec:
        raise ValueError(f"Existing LR=5e-5 Turbo spec conflicts with immutable diagnostic contract: {destination}")
    return dataset, stems, spec


def _write_spec_once(output: Path, spec: dict[str, Any]) -> None:
    path = output / "turbo_spec.json"
    if not path.is_file():
        _write(path, spec)


def _branch_checkpoints(args) -> list[tuple[int, Path]]:
    return exact_lr5e5_turbo_checkpoints(
        checkpoint_dir=args.checkpoint_dir, hf_repo_id=args.hf_repo_id,
        hf_recovery_dir=args.hf_recovery_dir, steps=LR5E5_TURBO_CHECKPOINT_STEPS,
    )


def _existing_branch_rows(output: Path) -> dict[int, dict[str, Any]]:
    path = output / "pck_clip_results.json"
    if not path.is_file():
        return {}
    rows = _read_json(path).get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError(f"Malformed LR=5e-5 Turbo score results: {path}")
    result = {row.get("checkpoint_step"): row for row in rows if isinstance(row, dict)}
    if set(result) - set(LR5E5_TURBO_CHECKPOINT_STEPS) or len(result) != len(rows):
        raise ValueError("LR=5e-5 Turbo scores contain a checkpoint outside the continuation branch")
    return result


def _merge_branch_generation(output: Path, generated: dict[str, list[int]]) -> dict[str, list[int]]:
    path = output / "generation_results.json"
    previous = _read_json(path).get("generated_steps", {}) if path.is_file() else {}
    if not isinstance(previous, dict):
        raise ValueError(f"Malformed LR=5e-5 generation results: {path}")
    merged: dict[str, list[int]] = {}
    for stem, steps in previous.items():
        if not isinstance(stem, str) or not isinstance(steps, list) or any(step not in LR5E5_TURBO_CHECKPOINT_STEPS for step in steps):
            raise ValueError("LR=5e-5 generation results contain a non-branch checkpoint")
        merged[stem] = list(steps)
    for stem, steps in generated.items():
        merged[stem] = sorted(set(merged.get(stem, []) + steps))
    return merged


def preflight(args) -> None:
    output = assert_lr5e5_turbo_output_isolated(args.output_dir)
    dataset, stems, spec = _branch_spec(output, args)
    checkpoints = _branch_checkpoints(args)
    _write_spec_once(output, spec)
    _write(output / "checkpoint_preflight.json", {
        "metadata": turbo_metadata(), "diagnostic_sample_count": len(dataset), "stems": list(stems),
        "local_checkpoint_root": str(LR5E5_CHECKPOINT_ROOT), "hf_repo_id": LR5E5_HF_REPO_ID,
        "hf_namespace": f"{LR5E5_HF_RUN_NAME}/full/",
        "checkpoints": [{"checkpoint_step": step, "remote_checkpoint": f"{LR5E5_HF_RUN_NAME}/full/step_{step:06d}.pt",
                         "remote_completion_marker": f"{LR5E5_HF_RUN_NAME}/full/step_{step:06d}.pt.complete.json",
                         "validated_path": str(path)} for step, path in checkpoints],
    })
    print(output / "checkpoint_preflight.json")


def generate(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run Turbo generation from the GH200 host shell with CUDA visible")
    output = assert_lr5e5_turbo_output_isolated(args.output_dir)
    dataset, stems, spec = _branch_spec(output, args); _write_spec_once(output, spec)
    checkpoints = _branch_checkpoints(args)
    shard_metadata = _read_json(Path(args.latent_root) / "shards.json")
    snapshot = validate_posebridge_snapshot(args.dataset_root or shard_metadata["dataset_root"])
    controls = {record.stem: record.control_path for record in snapshot.records_by_split["diagnostic_val"]}
    if set(controls) != set(stems):
        raise ValueError("Diagnostic controls differ from immutable diagnostic manifest")
    model = build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    vae = load_krea_vae("cuda")
    generated: dict[str, list[int]] = {}; compatibility: dict[str, dict[str, Any]] = {}
    for stem in stems:
        sample, directory = dict(_sample_by_stem(dataset, stem)), output / "fixed_pose" / stem
        directory.mkdir(parents=True, exist_ok=True)
        control_target = directory / "control.png"
        if not control_target.exists():
            control_target.write_bytes(Path(controls[stem]).read_bytes())
        metadata = {"stem": stem, "prompt": sample["prompt"], "control_path": str(controls[stem]),
                    "seed": spec["per_stem_seeds"][stem]["sampling"],
                    "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8], **turbo_metadata()}
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing LR=5e-5 Turbo metadata conflicts with immutable contract: {metadata_path}")
        _write(metadata_path, metadata)
        for step, checkpoint in _missing_generation_checkpoints(directory, checkpoints):
            state = load_training_state(checkpoint)
            if state["global_step"] != step:
                raise ValueError(f"LR=5e-5 checkpoint identity mismatch for step {step}")
            compatibility[str(step)] = raw_to_turbo_control_compatibility(model, state)
            load_trainable_state_dict(model, state["model"])
            pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                             torch.device("cuda"), metadata["seed"])
            save_image(pixels, directory / f"step_{step:06d}.png")
            generated.setdefault(stem, []).append(step)
    _write(output / "generation_results.json", {
        "metadata": turbo_metadata(), "stems": list(stems), "checkpoints": list(LR5E5_TURBO_CHECKPOINT_STEPS),
        "generated_steps": _merge_branch_generation(output, generated=generated),
        "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None),
        "raw_to_turbo_control_compatibility": compatibility,
    })
    print(output / "generation_results.json")


def score(args) -> None:
    output = assert_lr5e5_turbo_output_isolated(args.output_dir)
    dataset, stems, _ = _branch_spec(output, args)
    sidecar = _read_json(Path(args.reference_sidecar))
    geometry = {stem: turbo_scoring_geometry(_sample_by_stem(dataset, stem)) for stem in stems}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    unavailable = [{"stem": record["stem"], "pose_metric_status": "unavailable",
                    "pose_metric_reason": "authoritative_reference_pose_unavailable", "pck_005": None,
                    "pck_010": None, "pck_020": None}
                   for record in sidecar["records"] if record.get("status") != "available"]
    rows = _existing_branch_rows(output)
    for step in LR5E5_TURBO_CHECKPOINT_STEPS:
        image_for = lambda stem, current=step: output / "fixed_pose" / stem / f"step_{current:06d}.png"
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for,
                                       detector=detector, confidence_threshold=.5, require_images=True)
        clip_rows = [{"stem": stem, "source": next(row.get("source") for row in sidecar["records"] if row["stem"] == stem),
                      "cosine_similarity": _clip_score(clip, processor, device,
                          _read_json(output / "fixed_pose" / stem / "metadata.json")["prompt"], image_for(stem))}
                     for stem in stems]
        values = aggregate([row["cosine_similarity"] for row in clip_rows])
        rows[step] = {"checkpoint_step": step, "pose": pose, "pose_metric_unavailable_samples": unavailable,
                      "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"],
                               "std_cosine_similarity": values["std"], "sample_count": values["sample_count"],
                               "per_sample": clip_rows}}
    _write(output / "pck_clip_results.json", {"metadata": turbo_metadata(), "clip_model": args.clip_model_id,
           "confidence_threshold": .5, "checkpoints": [rows[step] for step in LR5E5_TURBO_CHECKPOINT_STEPS]})
    print(output / "pck_clip_results.json")


def _original_step_900(args, spec: dict[str, Any]) -> dict[str, Any]:
    original_output = Path(args.original_output_dir)
    assert_lr5e5_diagnostic_contract(spec, _read_json(original_output / "turbo_spec.json"))
    payload = _read_json(original_output / "pck_clip_results.json")
    if payload.get("metadata") != turbo_metadata() or not isinstance(payload.get("checkpoints"), list):
        raise ValueError("Original Turbo score result does not match the immutable Turbo contract")
    matches = [row for row in payload["checkpoints"] if row.get("checkpoint_step") == 900]
    if len(matches) != 1:
        raise ValueError("Original Turbo score result must contain exactly one reusable step-900 row")
    return matches[0]


def _summary_row(row: dict[str, Any], *, label: str, learning_rate: float) -> dict[str, Any]:
    pose, clip = row["pose"], row["clip"]
    return {"label": label, "checkpoint_step": row["checkpoint_step"], "learning_rate": learning_rate,
            **turbo_metadata(), "pooled_pck_005": pose["pck_005"], "pooled_pck_010": pose["pck_010"],
            "pooled_pck_020": pose["pck_020"], "single_person_pck": {key: pose["single_person"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "multi_person_pck": {key: pose["multi_person"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "coco_pck": {key: pose["per_source"]["COCO"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "human_art_pck": {key: pose["per_source"]["Human-Art"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "detection_coverage": pose["detection_coverage"], "joint_coverage": pose["joint_evaluation_coverage"],
            "matched_people": pose["matched_people"], "unmatched_reference_people": pose["unmatched_reference_people"],
            "unmatched_predicted_people": pose["unmatched_predicted_people"], "predicted_people": pose["predicted_people"],
            "evaluable_sample_count": pose["evaluable_sample_count"], "clip_mean_cosine_similarity": clip["mean_cosine_similarity"]}


def report(args) -> None:
    output = assert_lr5e5_turbo_output_isolated(args.output_dir)
    _, stems, spec = _branch_spec(output, args)
    scored = _read_json(output / "pck_clip_results.json")
    rows = scored.get("checkpoints")
    if not isinstance(rows, list) or tuple(row.get("checkpoint_step") for row in rows) != LR5E5_TURBO_CHECKPOINT_STEPS:
        raise ValueError("LR=5e-5 Turbo summary requires scores for exactly steps 1000, 1100, 1200, 1300, 1400, 1500")
    original = _original_step_900(args, spec)
    grid_rows = []
    for stem in stems:
        paths = [output / "fixed_pose" / stem / "control.png",
                 Path(args.original_output_dir) / "fixed_pose" / stem / "step_000900.png",
                 *(output / "fixed_pose" / stem / f"step_{step:06d}.png" for step in LR5E5_TURBO_CHECKPOINT_STEPS)]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Incomplete LR=5e-5 Turbo comparison row: {stem}")
        grid_rows.append((stem, paths))
    labels = ("control", "original 900 @ 1e-4", *(f"new {step} @ 5e-5" for step in LR5E5_TURBO_CHECKPOINT_STEPS))
    make_contact_sheet(grid_rows[:4], output / "turbo_lr5e5_checkpoint_selection_grid.png",
                       thumbnail_width=180, thumbnail_height=180, column_labels=labels)
    make_contact_sheet(grid_rows, output / "turbo_lr5e5_full_contact_sheet.png",
                       thumbnail_width=320, thumbnail_height=320, column_labels=labels)
    comparison = [_summary_row(original, label="original 900 @ 1e-4", learning_rate=1e-4),
                  *[_summary_row(row, label=f"new {row['checkpoint_step']} @ 5e-5", learning_rate=5e-5) for row in rows]]
    _write(output / "evaluation_summary.json", {
        "metadata": turbo_metadata(), "comparison": comparison, "original_baseline": original,
        "lr5e5_continuation_checkpoints": rows, "spec_sha256": hashlib.sha256((output / "turbo_spec.json").read_bytes()).hexdigest(),
        "qualitative_grids": {"checkpoint_selection": "turbo_lr5e5_checkpoint_selection_grid.png",
                                "full_contact_sheet": "turbo_lr5e5_full_contact_sheet.png"},
    })
    print(json.dumps(comparison, indent=2)); print(output / "evaluation_summary.json")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=str(OUTPUT))
    common.add_argument("--original-output-dir", default=str(ORIGINAL_OUTPUT))
    common.add_argument("--turbo-ckpt", default=os.environ.get("OSS_TURBO", str(ROOT / "models/krea-2-turbo/turbo.safetensors")))
    common.add_argument("--latent-root", default=str(ROOT / "posebridge_latents"))
    common.add_argument("--text-conditioning-root", default=str(ROOT / "text_conditioning"))
    common.add_argument("--checkpoint-dir", default=str(LR5E5_CHECKPOINT_ROOT))
    common.add_argument("--hf-repo-id", default=LR5E5_HF_REPO_ID)
    common.add_argument("--hf-recovery-dir", default=str(LR5E5_CHECKPOINT_ROOT / "hf-recovery-turbo"))
    common.add_argument("--diagnostic-manifest", default="data/manifests/diagnostic_val.jsonl")
    common.add_argument("--reference-sidecar", default="data/manifests/diagnostic_reference_pose.json")
    common.add_argument("--dataset-root")
    common.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    common.add_argument("--seed", type=int, default=420200)
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(required=True)
    for name, function in (("preflight", preflight), ("generate", generate), ("score", score), ("report", report)):
        item = sub.add_parser(name, parents=[common]); item.set_defaults(function=function)
    return parser


if __name__ == "__main__":
    args = parser().parse_args(); args.function(args)
