"""Score existing deterministic fixed-pose images with CLIP; no training.

PCK is recorded as unavailable until authoritative source pose annotations plus
the rendering geometry needed to align them are supplied.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pose_controlnet.post500_evaluation import (aggregate, assert_checkpoint_order, choose_best, clip_feature_tensor, cosine_from_embeddings, prepare_clip_scoring_inputs, unavailable_pose_result)

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir", default="/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500"); p.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32"); p.add_argument("--confidence-threshold", type=float, default=.5); p.add_argument("--samples", type=int); args = p.parse_args()
    root = Path(args.output_dir); summary_path = root / "evaluation_summary.json"; summary_path.unlink(missing_ok=True)
    flow = json.loads((root / "fixed_flow_results.json").read_text()); assert_checkpoint_order([x["checkpoint_step"] for x in flow["checkpoints"]])
    pose_root = root / "fixed_pose"; stems = flow["spec"]["stems"] if False else json.loads((root / "fixed_pose_spec.json").read_text())["stems"]
    if args.samples is not None: stems = stems[:args.samples]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    rows = []
    for flow_row in flow["checkpoints"]:
        step = flow_row["checkpoint_step"]; similarities = []
        for stem in stems:
            directory = pose_root / stem; generated = directory / f"step_{step:06d}.png"; prompt = json.loads((directory / "metadata.json").read_text())["prompt"]
            inputs = prepare_clip_scoring_inputs(processor, prompt, Image.open(generated).convert("RGB"), clip.config.text_config.max_position_embeddings).to(device)
            with torch.inference_mode():
                image_features = clip_feature_tensor(clip.get_image_features(pixel_values=inputs.pixel_values))
                text_features = clip_feature_tensor(clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask))
                similarity = cosine_from_embeddings(image_features.float().cpu().numpy(), text_features.float().cpu().numpy())[0]
            similarities.append(float(similarity))
        pose = unavailable_pose_result()
        clip_values = aggregate(similarities); clip_result = {"mean_cosine_similarity": clip_values["mean"], "median_cosine_similarity": clip_values["median"], "std_cosine_similarity": clip_values["std"], "sample_count": clip_values["sample_count"], "per_sample": [{"stem": stem, "cosine_similarity": score} for stem, score in zip(stems, similarities)]}
        rows.append({"checkpoint_step": step, "fixed_flow": {"mean": flow_row["mean_fixed_flow_loss"], "median": flow_row["median_fixed_flow_loss"], "std": flow_row["std_fixed_flow_loss"], "sample_count": flow_row["sample_count"], "per_sample": flow_row["per_sample"]}, "pose": pose, "clip": clip_result})
    summary = {"format_version": 1, "metadata": {"fixed_flow_seed": 420100, "fixed_pose_seed": 420200, "fixed_flow_spec_sha256": __import__("hashlib").sha256((root / "fixed_flow_spec.json").read_bytes()).hexdigest(), "pose_metric_status": "unavailable", "pose_metric_reason": "authoritative_reference_pose_unavailable", "authoritative_pose_extension": "Provide per-stem source joints, person grouping, visibility/confidence, source dimensions, and renderer/crop geometry before enabling PCK.", "clip_model": args.clip_model_id, "clip_implementation": "transformers.CLIPModel"}, "checkpoints": rows}
    summary["best_checkpoints"] = choose_best(summary)
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(summary_path)
    print(summary_path)
if __name__ == "__main__": main()
