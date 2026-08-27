"""Evaluation-only 8-step CFG-0 Krea-2 Turbo Pose-ControlNet benchmark.

This command is intentionally separate from the canonical RAW evaluation tree.
It resolves only checkpoints 800 and 1500, generates the complete immutable
diagnostic set, then reuses the authoritative PCK and CLIP scoring helpers.
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
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.dataset_index import validate_posebridge_snapshot
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, make_evaluation_spec, save_image, write_spec
from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate, clip_feature_tensor, cosine_from_embeddings, prepare_clip_scoring_inputs
from pose_controlnet.turbo_evaluation import (TURBO_CHECKPOINT_STEPS, assert_exact_diagnostic_stems,
    assert_turbo_output_isolated, exact_turbo_checkpoints, raw_to_turbo_control_compatibility,
    sample_turbo_pose_image, turbo_metadata)
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from pose_controlnet.checkpointing import load_training_state


ROOT = Path("/lambda/nfs/adhit/krea2-pose")
OUTPUT = ROOT / "evaluation/turbo-8step-cfg0"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_stems(path: Path) -> tuple[str, ...]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(Path(record["file_name"]).stem for record in records)


def _dataset_and_spec(args):
    dataset = PreparedLatentShardDataset(args.latent_root, "diagnostic_val", text_conditioning_root=args.text_conditioning_root)
    stems = assert_exact_diagnostic_stems(_manifest_stems(Path(args.diagnostic_manifest)), (record[3] for record in dataset.records))
    spec = make_evaluation_spec(dataset, split="diagnostic_val", count=24, seed=args.seed, kind="turbo_fixed_pose", stems=list(stems))
    spec["turbo"] = turbo_metadata()
    return dataset, stems, spec


def preflight(args) -> None:
    output = assert_turbo_output_isolated(args.output_dir)
    dataset, stems, spec = _dataset_and_spec(args)
    checkpoints = exact_turbo_checkpoints(checkpoint_dir=args.checkpoint_dir, hf_repo_id=args.hf_repo_id,
                                          hf_recovery_dir=args.hf_recovery_dir)
    _write(output / "turbo_spec.json", spec)
    _write(output / "checkpoint_preflight.json", {"metadata": turbo_metadata(), "diagnostic_sample_count": len(dataset),
           "stems": list(stems), "checkpoints": [{"checkpoint_step": step, "path": str(path)} for step, path in checkpoints]})
    print(output / "checkpoint_preflight.json")


def generate(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run Turbo generation from the GH200 host shell with CUDA visible")
    output = assert_turbo_output_isolated(args.output_dir)
    dataset, stems, spec = _dataset_and_spec(args); write_spec(output / "turbo_spec.json", spec)
    checkpoints = exact_turbo_checkpoints(checkpoint_dir=args.checkpoint_dir, hf_repo_id=args.hf_repo_id,
                                          hf_recovery_dir=args.hf_recovery_dir)
    shard_metadata = json.loads((Path(args.latent_root) / "shards.json").read_text())
    snapshot = validate_posebridge_snapshot(args.dataset_root or shard_metadata["dataset_root"])
    controls = {record.stem: record.control_path for record in snapshot.records_by_split["diagnostic_val"]}
    if set(controls) != set(stems): raise ValueError("Diagnostic controls differ from immutable diagnostic manifest")
    model = build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    vae = load_krea_vae("cuda")
    generated: dict[str, list[int]] = {}; compatibility: dict[str, dict] = {}
    for stem in stems:
        sample, directory = dict(_sample_by_stem(dataset, stem)), output / "fixed_pose" / stem
        directory.mkdir(parents=True, exist_ok=True)
        control_target = directory / "control.png"
        if not control_target.exists():
            control_target.write_bytes(Path(controls[stem]).read_bytes())
        metadata = {"stem": stem, "prompt": sample["prompt"], "control_path": str(controls[stem]),
                    "seed": spec["per_stem_seeds"][stem]["sampling"], "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8],
                    **turbo_metadata()}
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f"Existing Turbo metadata conflicts with immutable contract: {metadata_path}")
        _write(metadata_path, metadata)
        for step, checkpoint in checkpoints:
            target = directory / f"step_{step:06d}.png"
            if target.exists(): continue
            state = load_training_state(checkpoint)
            if state["global_step"] != step: raise ValueError(f"Checkpoint identity mismatch for step {step}")
            compatibility[str(step)] = raw_to_turbo_control_compatibility(model, state)
            load_trainable_state_dict(model, state["model"])
            pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                             torch.device("cuda"), metadata["seed"])
            save_image(pixels, target); generated.setdefault(stem, []).append(step)
    _write(output / "generation_results.json", {"metadata": turbo_metadata(), "stems": list(stems),
           "checkpoints": list(TURBO_CHECKPOINT_STEPS), "generated_steps": generated,
           "turbo_base_checkpoint_report": getattr(model, "_krea_checkpoint_report", None),
           "raw_to_turbo_control_compatibility": compatibility})
    print(output / "generation_results.json")


def _clip_score(clip, processor, device: str, prompt: str, image_path: Path) -> float:
    # This is byte-for-byte equivalent in behavior to post1500_audit._clip_score.
    with Image.open(image_path) as image:
        inputs = prepare_clip_scoring_inputs(processor, prompt, image.convert("RGB"), clip.config.text_config.max_position_embeddings).to(device)
    with torch.inference_mode():
        image_features = clip_feature_tensor(clip.get_image_features(pixel_values=inputs.pixel_values))
        text_features = clip_feature_tensor(clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask))
    return float(cosine_from_embeddings(image_features.float().cpu().numpy(), text_features.float().cpu().numpy())[0])


def score(args) -> None:
    output = assert_turbo_output_isolated(args.output_dir)
    dataset, stems, _ = _dataset_and_spec(args)
    sidecar = json.loads(Path(args.reference_sidecar).read_text())
    geometry = {sample["stem"]: sample for sample in (_sample_by_stem(dataset, stem) for stem in stems)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    unavailable_statuses = [{"stem": record["stem"], "pose_metric_status": "unavailable",
                             "pose_metric_reason": "authoritative_reference_pose_unavailable",
                             "pck_005": None, "pck_010": None, "pck_020": None}
                            for record in sidecar["records"] if record.get("status") != "available"]
    rows = []
    for step in TURBO_CHECKPOINT_STEPS:
        image_for = lambda stem, current=step: output / "fixed_pose" / stem / f"step_{current:06d}.png"
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for, detector=detector,
                                       confidence_threshold=.5, require_images=True)
        clip_rows = [{"stem": stem, "source": next(row.get("source") for row in sidecar["records"] if row["stem"] == stem),
                      "cosine_similarity": _clip_score(clip, processor, device, json.loads((output / "fixed_pose" / stem / "metadata.json").read_text())["prompt"], image_for(stem))}
                     for stem in stems]
        values = aggregate([row["cosine_similarity"] for row in clip_rows])
        rows.append({"checkpoint_step": step, "pose": pose, "pose_metric_unavailable_samples": unavailable_statuses,
                     "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"], "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_sample": clip_rows}})
    _write(output / "pck_clip_results.json", {"metadata": turbo_metadata(), "clip_model": args.clip_model_id,
           "confidence_threshold": .5, "checkpoints": rows})
    print(output / "pck_clip_results.json")


def report(args) -> None:
    output = assert_turbo_output_isolated(args.output_dir)
    scored = json.loads((output / "pck_clip_results.json").read_text())
    rows = scored["checkpoints"]
    if tuple(row["checkpoint_step"] for row in rows) != TURBO_CHECKPOINT_STEPS: raise ValueError("Turbo summary requires exact 800 and 1500 scores")
    grid_rows = []
    for stem in json.loads((output / "turbo_spec.json").read_text())["stems"]:
        paths = [output / "fixed_pose" / stem / name for name in ("control.png", "step_000800.png", "step_001500.png")]
        if not all(path.is_file() for path in paths): raise FileNotFoundError(f"Incomplete Turbo comparison row: {stem}")
        grid_rows.append((stem, paths))
    make_contact_sheet(grid_rows, output / "turbo_comparison_grid.png", column_labels=("control", "checkpoint 800 Turbo", "checkpoint 1500 Turbo"))
    table = []
    for row in rows:
        pose, clip = row["pose"], row["clip"]
        table.append({"checkpoint": row["checkpoint_step"], **turbo_metadata(), "PCK.05": pose["pck_005"], "PCK.10": pose["pck_010"], "PCK.20": pose["pck_020"],
                      "single_PCK.20": pose["single_person"]["pck_020"], "multi_PCK.20": pose["multi_person"]["pck_020"],
                      "COCO_PCK.20": pose["per_source"]["COCO"]["pck_020"], "HumanArt_PCK.20": pose["per_source"]["Human-Art"]["pck_020"],
                      "detection_coverage": pose["generated_person_detection_coverage"], "joint_coverage": pose["joint_evaluation_coverage"], "CLIP": clip["mean_cosine_similarity"]})
    _write(output / "evaluation_summary.json", {"metadata": turbo_metadata(), "machine_readable_table": table, "checkpoints": rows,
           "spec_sha256": hashlib.sha256((output / "turbo_spec.json").read_bytes()).hexdigest()})
    print(json.dumps(table, indent=2)); print(output / "evaluation_summary.json")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=str(OUTPUT)); common.add_argument("--turbo-ckpt", default=os.environ.get("OSS_TURBO", str(ROOT / "models/krea-2-turbo/turbo.safetensors")), help="official OSS_TURBO checkpoint path")
    common.add_argument("--latent-root", default=str(ROOT / "posebridge_latents")); common.add_argument("--text-conditioning-root", default=str(ROOT / "text_conditioning"))
    common.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints/pose-learning-1500")); common.add_argument("--hf-repo-id", default="adhit-420/Krea-2-PoseControl-LoRA-checkpoints")
    common.add_argument("--hf-recovery-dir", default=str(ROOT / "checkpoints/pose-learning-1500/hf-recovery-turbo")); common.add_argument("--diagnostic-manifest", default="data/manifests/diagnostic_val.jsonl")
    common.add_argument("--reference-sidecar", default="data/manifests/diagnostic_reference_pose.json"); common.add_argument("--dataset-root"); common.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32"); common.add_argument("--seed", type=int, default=420200)
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(required=True)
    for name, function in (("preflight", preflight), ("generate", generate), ("score", score), ("report", report)):
        item = sub.add_parser(name, parents=[common]); item.set_defaults(function=function)
    return parser


if __name__ == "__main__":
    args = parser().parse_args(); args.function(args)
