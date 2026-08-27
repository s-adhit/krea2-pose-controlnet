"""Read-only post-1500 evaluation/audit entry point; never trains or steps an optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.evaluation import CHECKPOINT_STEPS, make_contact_sheet, ordered_checkpoints
from pose_controlnet.model import build_pose_model, load_trainable_state_dict
from pose_controlnet.post1500_evaluation import (POST_500_STEPS, fixed_timestep_loss_and_sensitivity,
    merge_checkpoint_results, score_authoritative_pck, source_balance_audit, telemetry_audit, timestep_distribution_audit)
from pose_controlnet.post500_evaluation import (KeypointRCNNEstimator, aggregate, clip_feature_tensor,
    cosine_from_embeddings, choose_best, plot_summary, prepare_clip_scoring_inputs)


ROOT = Path("/lambda/nfs/adhit/krea2-pose")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _requested_steps(raw: list[int] | None) -> tuple[int, ...]:
    steps = tuple(raw) if raw else CHECKPOINT_STEPS
    if (len(steps) != len(set(steps)) or any(step not in CHECKPOINT_STEPS for step in steps)
            or steps != tuple(step for step in CHECKPOINT_STEPS if step in steps)):
        raise ValueError("Steps must be an ordered subsequence of the exact canonical 0..1500 sequence")
    return steps


def _checkpoints(args, steps: tuple[int, ...]):
    return ordered_checkpoints(args.early_checkpoint_dir, steps=steps, later_checkpoint_dir=args.mid_checkpoint_dir,
                               archive_checkpoint_dir=args.final_checkpoint_dir, hf_repo_id=args.hf_repo_id,
                               hf_recovery_dir=args.hf_recovery_dir)


def _iter_shard_records(root: Path, split: str):
    for shard in sorted((root / split).glob(f"{split}-*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        for sample in payload["samples"]:
            yield sample


def _diagnostic_geometry(root: Path) -> dict[str, dict]:
    return {sample["stem"]: sample for sample in _iter_shard_records(root, "diagnostic_val")}


def cheap(args) -> None:
    latent_root = Path(args.latent_root)
    train_records = list(_iter_shard_records(latent_root, "train"))
    bucket_counts: dict[tuple[int, int], int] = {}
    for sample in train_records:
        bucket = tuple(sample["bucket"]); bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    cfg = TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=str(latent_root))
    output = Path(args.output_dir)
    _write(output / "timestep_distribution.json", timestep_distribution_audit(bucket_counts, cfg, samples_per_bucket=args.timestep_samples))
    sidecar = json.loads(Path(args.reference_sidecar).read_text())
    manifest = [json.loads(line) for line in Path(args.train_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets_by_stem = {sample["stem"]: sample["bucket"] for sample in train_records}
    manifest_stems = [Path(row["file_name"]).stem for row in manifest]
    if set(manifest_stems) != set(buckets_by_stem) or len(manifest_stems) != len(buckets_by_stem):
        raise ValueError("Immutable train manifest and prepared train shard membership differ")
    records_from_manifest = [{"stem": stem, "bucket": buckets_by_stem[stem]} for stem in manifest_stems]
    balance = source_balance_audit(records_from_manifest, [record["stem"] for record in sidecar["records"]])
    balance["immutable_manifest_path"] = str(Path(args.train_manifest))
    _write(output / "source_data_balance.json", balance)
    _write(output / "telemetry_audit.json", telemetry_audit(args.metrics_jsonl))
    print(output / "timestep_distribution.json")


def preflight(args) -> None:
    rows = []
    for step, path in _checkpoints(args, _requested_steps(args.steps)):
        rows.append({"checkpoint_step": step, "path": None if path is None else str(path), "source": "baseline" if path is None else ("hf-recovery" if "hf-recovery" in path.parts else "local")})
    _write(Path(args.output_dir) / "checkpoint_preflight.json", {"canonical_steps": list(_requested_steps(args.steps)), "checkpoints": rows})
    print(Path(args.output_dir) / "checkpoint_preflight.json")


def merge_flow(args) -> None:
    output = Path(args.output_dir)
    existing = json.loads((output / "fixed_flow_results.json").read_text())
    extension = json.loads((output / "fixed_flow_extension_results.json").read_text())
    _write(output / "fixed_flow_results.json", merge_checkpoint_results(existing, extension))
    print(output / "fixed_flow_results.json")


def _clip_score(clip, processor, device: str, prompt: str, image_path: Path) -> float:
    with Image.open(image_path) as image:
        inputs = prepare_clip_scoring_inputs(processor, prompt, image.convert("RGB"), clip.config.text_config.max_position_embeddings).to(device)
    with torch.inference_mode():
        image_features = clip_feature_tensor(clip.get_image_features(pixel_values=inputs.pixel_values))
        text_features = clip_feature_tensor(clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask))
    return float(cosine_from_embeddings(image_features.float().cpu().numpy(), text_features.float().cpu().numpy())[0])


def pck_clip(args) -> None:
    output, pose_root = Path(args.output_dir), Path(args.output_dir) / "fixed_pose"
    sidecar = json.loads(Path(args.reference_sidecar).read_text())
    geometry = _diagnostic_geometry(Path(args.latent_root)); steps = _requested_steps(args.steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, args.confidence_threshold)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    rows = []
    for step in steps:
        def image_for(stem: str, current=step) -> Path:
            return pose_root / stem / f"step_{current:06d}.png"
        pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for, detector=detector,
                                       confidence_threshold=args.confidence_threshold, require_images=not args.allow_missing_images)
        clip_rows = []
        for record in sidecar["records"]:
            image = image_for(record["stem"])
            metadata = image.parent / "metadata.json"
            if image.is_file() and metadata.is_file():
                clip_rows.append({"stem": record["stem"], "source": record.get("source"), "cosine_similarity": _clip_score(clip, processor, device, json.loads(metadata.read_text())["prompt"], image)})
            elif not args.allow_missing_images:
                raise FileNotFoundError(f"Missing generated image or metadata for CLIP: {image}")
        values = aggregate([row["cosine_similarity"] for row in clip_rows]) if clip_rows else {"sample_count": 0, "mean": None, "median": None, "std": None}
        by_source = {name: aggregate([row["cosine_similarity"] for row in clip_rows if row["source"] == source]) for name, source in (("Human-Art", "humanart"), ("COCO", "coco"), ("Danbooru", "danbooru")) if any(row["source"] == source for row in clip_rows)}
        rows.append({"checkpoint_step": step, "pose": pose, "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"], "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_source": by_source, "per_sample": clip_rows}})
    _write(output / "pck_clip_results.json", {"format_version": 1, "clip_model": args.clip_model_id, "confidence_threshold": args.confidence_threshold, "checkpoints": rows})
    print(output / "pck_clip_results.json")


def loss_control(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run loss/control audit from the GH200 host shell with CUDA visible")
    output = Path(args.output_dir); spec = json.loads((output / "fixed_flow_spec.json").read_text())
    stems = spec["stems"][:args.samples]
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    model = build_pose_model(args.raw_ckpt, 64, 64, "cuda").eval(); cfg = TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root)
    results = []
    for step, checkpoint in _checkpoints(args, (500, 800, 1100, 1500)):
        if checkpoint is None: raise ValueError("Loss/control audit requires trained checkpoints only")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False); load_trainable_state_dict(model, state["model"])
        results.append({"checkpoint_step": step, **fixed_timestep_loss_and_sensitivity(model, dataset, stems, cfg, torch.device("cuda"), seed=int(spec["seed"]))})
    _write(output / "loss_control_by_timestep.json", {"format_version": 1, "checkpoints": results})
    print(output / "loss_control_by_timestep.json")


def report(args) -> None:
    output = Path(args.output_dir)
    flow = json.loads((output / "fixed_flow_results.json").read_text())
    scored = json.loads((output / "pck_clip_results.json").read_text())
    flow_rows = {row["checkpoint_step"]: row for row in flow["checkpoints"]}
    score_rows = {row["checkpoint_step"]: row for row in scored["checkpoints"]}
    missing = [step for step in CHECKPOINT_STEPS if step not in flow_rows or step not in score_rows]
    if missing:
        raise ValueError(f"Cannot assemble final summary; missing canonical metrics for {missing}")
    rows = []
    for step in CHECKPOINT_STEPS:
        fixed, score = flow_rows[step], score_rows[step]
        rows.append({"checkpoint_step": step,
                     "fixed_flow": {"mean": fixed["mean_fixed_flow_loss"], "median": fixed["median_fixed_flow_loss"], "std": fixed["std_fixed_flow_loss"], "sample_count": fixed["sample_count"], "per_sample": fixed["per_sample"]},
                     "pose": score["pose"], "clip": score["clip"]})
    summary = {"format_version": 2, "metadata": {"fixed_flow_seed": 420100, "fixed_pose_seed": 420200,
               "fixed_flow_spec_sha256": hashlib.sha256((output / "fixed_flow_spec.json").read_bytes()).hexdigest(),
               "clip_model": scored["clip_model"], "pck_reference": "data/manifests/diagnostic_reference_pose.json",
               "pck_aggregation": "pooled eligible reference joints; Danbooru unavailable excluded"}, "checkpoints": rows}
    summary["best_checkpoints"] = choose_best(summary)
    _write(output / "evaluation_summary.json", summary)
    # Plot helpers consume the final summary.  Summary construction stays
    # intentionally separate from scoring so expensive detector/CLIP work is
    # never repeated when changing presentation only.
    print(*plot_summary(output / "evaluation_summary.json", output), sep="\n")
    _audit_plots(output)
    _comparison_grid(output)
    print("Step | Fixed-flow | CLIP | PCK@.05 | PCK@.10 | PCK@.20 | Detection | Joint coverage")
    for row in rows:
        pose, clip = row["pose"], row["clip"]
        print(f"{row['checkpoint_step']:>4} | {row['fixed_flow']['mean']:.6f} | {clip['mean_cosine_similarity']:.6f} | "
              f"{pose['pck_005'] if pose['pck_005'] is not None else 'N/A'} | {pose['pck_010'] if pose['pck_010'] is not None else 'N/A'} | "
              f"{pose['pck_020'] if pose['pck_020'] is not None else 'N/A'} | {pose['detection_coverage'] if pose['detection_coverage'] is not None else 'N/A'} | "
              f"{pose['joint_evaluation_coverage'] if pose['joint_evaluation_coverage'] is not None else 'N/A'}")
    print(output / "evaluation_summary.json")


def _audit_plots(output: Path) -> None:
    """Compact plots derived only from already-written audit JSON artifacts."""
    import matplotlib.pyplot as plt
    def draw(name: str, x, series, ylabel: str, xlabel: str = "optimizer step"):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label, values in series:
            if values: axis.plot(x[:len(values)], values, marker="o", label=label)
        axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.grid(True, alpha=.25); axis.legend(); figure.tight_layout(); figure.savefig(output / name, dpi=160); plt.close(figure)
    timestep_path = output / "timestep_distribution.json"
    if timestep_path.is_file():
        audit = json.loads(timestep_path.read_text()); rows = audit["per_bucket"]
        draw("timestep_distribution.png", list(range(len(rows))), [("mean shifted t", [row["distribution"]["mean"] for row in rows])], "mean shifted timestep", "bucket index")
    telemetry_path = output / "telemetry_audit.json"
    if telemetry_path.is_file():
        raw = json.loads(telemetry_path.read_text()).get("raw_series", {})
        mapping = {"train/loss": "train_loss_vs_step.png", "validation/flow_loss": "validation_loss_vs_step.png", "train/global_grad_norm": "gradient_norm_vs_step.png", "performance/samples_per_second": "throughput_vs_step.png", "cuda/peak_allocated_bytes": "memory_vs_step.png"}
        for key, name in mapping.items():
            points = raw.get(key, []); draw(name, [point["step"] for point in points], [(key, [point["value"] for point in points])], key)
    sensitivity_path = output / "loss_control_by_timestep.json"
    if sensitivity_path.is_file():
        rows = json.loads(sensitivity_path.read_text())["checkpoints"]
        ts = [entry["timestep"] for entry in rows[0]["timesteps"]] if rows else []
        draw("control_sensitivity_vs_timestep.png", ts, [(f"step {row['checkpoint_step']}", [entry["control_sensitivity_rms"]["mean"] for entry in row["timesteps"]]) for row in rows], "prediction delta RMS", "timestep")


def _comparison_grid(output: Path) -> None:
    """Keep the historical compact set; only use stems with all four images."""
    root, steps = output / "fixed_pose", (500, 800, 1100, 1500)
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else ():
        paths = [directory / "control.png", *(directory / f"step_{step:06d}.png" for step in steps)]
        if all(path.is_file() for path in paths): rows.append((directory.name, paths))
    if rows:
        make_contact_sheet(rows, output / "500_vs_800_vs_1100_vs_1500.png",
                           column_labels=("control", "step500", "step800", "step1100", "step1500"))


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # Extend the existing immutable 0..500 evaluation root in place.  The
    # fixed-pose generator reuses existing files and only writes 600..1500.
    common.add_argument("--output-dir", default=str(ROOT / "evaluation/pose-learning-500"))
    common.add_argument("--raw-ckpt", default=str(ROOT / "models/krea-2-raw/raw.safetensors"))
    common.add_argument("--latent-root", default=str(ROOT / "posebridge_latents"))
    common.add_argument("--text-conditioning-root", default=str(ROOT / "text_conditioning"))
    common.add_argument("--early-checkpoint-dir", default=str(ROOT / "checkpoints/pose-learning-100"))
    common.add_argument("--mid-checkpoint-dir", default=str(ROOT / "checkpoints/pose-learning-500"))
    common.add_argument("--final-checkpoint-dir", default=str(ROOT / "checkpoints/pose-learning-1500"))
    common.add_argument("--hf-repo-id", default="adhit-420/Krea-2-PoseControl-LoRA-checkpoints")
    common.add_argument("--hf-recovery-dir")
    common.add_argument("--steps", nargs="+", type=int)
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(required=True)
    item = sub.add_parser("preflight", parents=[common]); item.set_defaults(function=preflight)
    item = sub.add_parser("merge-flow", parents=[common]); item.set_defaults(function=merge_flow)
    item = sub.add_parser("cheap", parents=[common]); item.add_argument("--metrics-jsonl", default=str(ROOT / "checkpoints/pose-learning-1500/metrics.jsonl")); item.add_argument("--reference-sidecar", default="data/manifests/diagnostic_reference_pose.json"); item.add_argument("--train-manifest", default="data/manifests/train.jsonl"); item.add_argument("--timestep-samples", type=int, default=100_000); item.set_defaults(function=cheap)
    item = sub.add_parser("pck-clip", parents=[common]); item.add_argument("--reference-sidecar", default="data/manifests/diagnostic_reference_pose.json"); item.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32"); item.add_argument("--confidence-threshold", type=float, default=.5); item.add_argument("--allow-missing-images", action="store_true"); item.set_defaults(function=pck_clip)
    item = sub.add_parser("loss-control", parents=[common]); item.add_argument("--samples", type=int, default=8); item.set_defaults(function=loss_control)
    item = sub.add_parser("report", parents=[common]); item.set_defaults(function=report)
    return parser


if __name__ == "__main__":
    args = parser().parse_args(); args.function(args)
