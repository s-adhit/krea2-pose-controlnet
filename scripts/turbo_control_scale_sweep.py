"""Read-only Krea-2 Turbo inference control-strength sweep for LR-only step 1500.

The five fixed runs differ solely in the clean pose latent multiplier immediately
before the established control-concatenation path.  This script never trains,
resumes, or mutates a checkpoint.
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
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict, trainable_state_dict
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.turbo_evaluation import (
    CONTROL_SCALE_TURBO_EVALUATION_ROOT, CONTROL_SCALE_VALUES,
    LR5E5_CHECKPOINT_ROOT, LR5E5_HF_REPO_ID, LR5E5_HF_RUN_NAME,
    LR5E5_TURBO_EVALUATION_ROOT, ORIGINAL_TURBO_EVALUATION_ROOT,
    assert_control_scale_turbo_output_isolated, assert_turbo_diagnostic_contract,
    exact_lr5e5_step1500_local_checkpoint, raw_to_turbo_control_compatibility,
    sample_turbo_pose_image, turbo_metadata, turbo_scoring_geometry,
)
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts.turbo_benchmark import _clip_score, _dataset_and_spec as _original_dataset_and_spec


ROOT = Path("/lambda/nfs/adhit/krea2-pose")
OUTPUT = CONTROL_SCALE_TURBO_EVALUATION_ROOT


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required immutable Turbo result is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Malformed JSON object: {path}")
    return value


def _scale_label(scale: float) -> str:
    if scale not in CONTROL_SCALE_VALUES:
        raise ValueError(f"Control scale must be one of {CONTROL_SCALE_VALUES}, got {scale}")
    return f"{scale:.2f}".replace(".", "p")


def _branch_spec(output: Path, args) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
    dataset, stems, spec = _original_dataset_and_spec(args)
    baseline_spec = _read_json(Path(args.lr5e5_output_dir) / "turbo_spec.json")
    assert_turbo_diagnostic_contract(spec, baseline_spec, branch_name="control-scale")
    destination = output / "turbo_spec.json"
    if destination.is_file() and _read_json(destination) != spec:
        raise ValueError(f"Existing control-scale Turbo spec conflicts with immutable diagnostic contract: {destination}")
    return dataset, stems, spec


def _source_checkpoint(args, output: Path) -> Path:
    return exact_lr5e5_step1500_local_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        hf_repo_id=args.hf_repo_id,
        marker_download_dir=output / "checkpoint-marker-validation",
    )


def _write_spec_once(output: Path, spec: dict[str, Any]) -> None:
    if not (output / "turbo_spec.json").is_file():
        _write(output / "turbo_spec.json", spec)


def _image_path(directory: Path, scale: float) -> Path:
    return directory / f"control_scale_{_scale_label(scale)}.png"


def _metadata(sample: dict[str, Any], control_path: str, seed: int, scale: float) -> dict[str, Any]:
    return {
        "stem": sample["stem"], "prompt": sample["prompt"], "control_path": control_path,
        "seed": seed, "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8],
        "control_scale": scale, "checkpoint_step": 1500,
        "checkpoint_path": str(LR5E5_CHECKPOINT_ROOT / "step_001500.pt"),
        "hf_namespace": f"{LR5E5_HF_RUN_NAME}/full/", **turbo_metadata(),
    }


def preflight(args) -> None:
    output = assert_control_scale_turbo_output_isolated(args.output_dir)
    dataset, stems, spec = _branch_spec(output, args)
    checkpoint = _source_checkpoint(args, output)
    _write_spec_once(output, spec)
    _write(output / "checkpoint_preflight.json", {
        "metadata": turbo_metadata(), "control_scales": list(CONTROL_SCALE_VALUES),
        "diagnostic_sample_count": len(dataset), "stems": list(stems),
        "local_checkpoint_root": str(LR5E5_CHECKPOINT_ROOT), "checkpoint": str(checkpoint),
        "checkpoint_step": 1500, "hf_repo_id": LR5E5_HF_REPO_ID,
        "hf_namespace": f"{LR5E5_HF_RUN_NAME}/full/",
    })
    print(output / "checkpoint_preflight.json")


def generate(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run Turbo generation from the GH200 host shell with CUDA visible")
    output = assert_control_scale_turbo_output_isolated(args.output_dir)
    dataset, stems, spec = _branch_spec(output, args)
    _write_spec_once(output, spec)
    checkpoint = _source_checkpoint(args, output)
    state = load_training_state(checkpoint)
    if state["global_step"] != 1500:
        raise ValueError("Control-scale source checkpoint must be exact embedded step 1500")
    shard_metadata = _read_json(Path(args.latent_root) / "shards.json")
    snapshot = validate_posebridge_snapshot(args.dataset_root or shard_metadata["dataset_root"])
    controls = {record.stem: record.control_path for record in snapshot.records_by_split["diagnostic_val"]}
    if set(controls) != set(stems):
        raise ValueError("Diagnostic controls differ from immutable diagnostic manifest")
    model = build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    compatibility = raw_to_turbo_control_compatibility(model, state)
    load_trainable_state_dict(model, state["model"])
    vae = load_krea_vae("cuda")
    before = trainable_state_dict(model)
    generated: dict[str, list[float]] = {}
    for stem in stems:
        sample, directory = dict(_sample_by_stem(dataset, stem)), output / "fixed_pose" / stem
        directory.mkdir(parents=True, exist_ok=True)
        control_target = directory / "control.png"
        if not control_target.exists():
            control_target.write_bytes(Path(controls[stem]).read_bytes())
        for scale in CONTROL_SCALE_VALUES:
            metadata = _metadata(sample, str(controls[stem]), spec["per_stem_seeds"][stem]["sampling"], scale)
            metadata_path = directory / f"control_scale_{_scale_label(scale)}.metadata.json"
            if metadata_path.exists() and _read_json(metadata_path) != metadata:
                raise ValueError(f"Existing control-scale metadata conflicts with immutable contract: {metadata_path}")
            _write(metadata_path, metadata)
            target = _image_path(directory, scale)
            if target.is_file():
                continue
            pixels = sample_turbo_pose_image(
                model, lambda latent: decode_normalized_latents(vae, latent), sample, torch.device("cuda"),
                metadata["seed"], control_scale=scale,
            )
            save_image(pixels, target)
            generated.setdefault(stem, []).append(scale)
    if any(not torch.equal(before[name], value) for name, value in trainable_state_dict(model).items()):
        raise RuntimeError("Read-only control-scale generation unexpectedly changed model weights")
    _write(output / "generation_results.json", {
        "metadata": turbo_metadata(), "checkpoint_step": 1500, "control_scales": list(CONTROL_SCALE_VALUES),
        "stems": list(stems), "generated_scales": generated,
        "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None),
        "raw_to_turbo_control_compatibility": compatibility,
    })
    print(output / "generation_results.json")


def score(args) -> None:
    output = assert_control_scale_turbo_output_isolated(args.output_dir)
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
    rows = []
    for scale in CONTROL_SCALE_VALUES:
        image_for = lambda stem, current=scale: _image_path(output / "fixed_pose" / stem, current)
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for,
                                       detector=detector, confidence_threshold=.5, require_images=True)
        clip_rows = [{"stem": stem,
                      "source": next(row.get("source") for row in sidecar["records"] if row["stem"] == stem),
                      "cosine_similarity": _clip_score(clip, processor, device,
                          _read_json(output / "fixed_pose" / stem / f"control_scale_{_scale_label(scale)}.metadata.json")["prompt"], image_for(stem))}
                     for stem in stems]
        values = aggregate([row["cosine_similarity"] for row in clip_rows])
        rows.append({"control_scale": scale, "pose": pose, "pose_metric_unavailable_samples": unavailable,
                     "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"],
                              "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_sample": clip_rows}})
    _write(output / "pck_clip_results.json", {"metadata": turbo_metadata(), "checkpoint_step": 1500,
           "clip_model": args.clip_model_id, "confidence_threshold": .5, "control_scales": list(CONTROL_SCALE_VALUES),
           "results": rows})
    print(output / "pck_clip_results.json")


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    pose, clip = row["pose"], row["clip"]
    return {"control_scale": row["control_scale"], **turbo_metadata(),
            "clip_mean": clip["mean_cosine_similarity"], "detection_coverage": pose["detection_coverage"],
            "joint_coverage": pose["joint_evaluation_coverage"], "pck_005": pose["pck_005"],
            "pck_010": pose["pck_010"], "pck_020": pose["pck_020"],
            "single_person_pck": {key: pose["single_person"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "multi_person_pck": {key: pose["multi_person"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "coco_pck": {key: pose["per_source"]["COCO"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "human_art_pck": {key: pose["per_source"]["Human-Art"][key] for key in ("pck_005", "pck_010", "pck_020")},
            "matched_people": pose["matched_people"], "unmatched_reference_people": pose["unmatched_reference_people"],
            "predicted_people": pose["predicted_people"], "unmatched_predicted_people": pose["unmatched_predicted_people"]}


def report(args) -> None:
    output = assert_control_scale_turbo_output_isolated(args.output_dir)
    _, stems, _ = _branch_spec(output, args)
    scored = _read_json(output / "pck_clip_results.json")
    rows = scored.get("results")
    if not isinstance(rows, list) or tuple(row.get("control_scale") for row in rows) != CONTROL_SCALE_VALUES:
        raise ValueError(f"Control-scale summary requires scores in exact scale order {CONTROL_SCALE_VALUES}")
    grid_rows = []
    for stem in stems:
        paths = [output / "fixed_pose" / stem / "control.png", *(_image_path(output / "fixed_pose" / stem, scale) for scale in CONTROL_SCALE_VALUES)]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Incomplete control-scale comparison row: {stem}")
        grid_rows.append((stem, paths))
    labels = ("control", *(f"scale {scale:.2f}" for scale in CONTROL_SCALE_VALUES))
    make_contact_sheet(grid_rows[:4], output / "turbo_control_scale_selection_grid.png", thumbnail_width=180,
                       thumbnail_height=180, column_labels=labels)
    make_contact_sheet(grid_rows, output / "turbo_control_scale_full_contact_sheet.png", thumbnail_width=320,
                       thumbnail_height=320, column_labels=labels)
    summary = [_summary_row(row) for row in rows]
    _write(output / "evaluation_summary.json", {
        "metadata": turbo_metadata(), "checkpoint_step": 1500, "control_scale_results": summary,
        "spec_sha256": hashlib.sha256((output / "turbo_spec.json").read_bytes()).hexdigest(),
        "qualitative_grids": {"selection": "turbo_control_scale_selection_grid.png", "full": "turbo_control_scale_full_contact_sheet.png"},
    })
    print(json.dumps(summary, indent=2))
    print(output / "evaluation_summary.json")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=str(OUTPUT))
    common.add_argument("--lr5e5-output-dir", default=str(LR5E5_TURBO_EVALUATION_ROOT))
    common.add_argument("--turbo-ckpt", default=os.environ.get("OSS_TURBO", str(ROOT / "models/krea-2-turbo/turbo.safetensors")))
    common.add_argument("--latent-root", default=str(ROOT / "posebridge_latents"))
    common.add_argument("--text-conditioning-root", default=str(ROOT / "text_conditioning"))
    common.add_argument("--checkpoint-dir", default=str(LR5E5_CHECKPOINT_ROOT))
    common.add_argument("--hf-repo-id", default=LR5E5_HF_REPO_ID)
    common.add_argument("--diagnostic-manifest", default="data/manifests/diagnostic_val.jsonl")
    common.add_argument("--reference-sidecar", default="data/manifests/diagnostic_reference_pose.json")
    common.add_argument("--dataset-root")
    common.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    common.add_argument("--seed", type=int, default=420200)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, function in (("preflight", preflight), ("generate", generate), ("score", score), ("report", report)):
        item = sub.add_parser(name, parents=[common])
        item.set_defaults(function=function)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    args.function(args)
