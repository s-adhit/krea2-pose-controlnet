"""Score existing deterministic fixed-pose images with PCK and CLIP; no training."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pose_controlnet.post500_evaluation import (CHECKPOINT_STEPS, KeypointRCNNEstimator, aggregate, assert_checkpoint_order, choose_best, cosine_from_embeddings, pck_for_people)

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir", default="/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500"); p.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32"); p.add_argument("--confidence-threshold", type=float, default=.5); p.add_argument("--samples", type=int); args = p.parse_args()
    root = Path(args.output_dir); flow = json.loads((root / "fixed_flow_results.json").read_text()); assert_checkpoint_order([x["checkpoint_step"] for x in flow["checkpoints"]])
    pose_root = root / "fixed_pose"; stems = flow["spec"]["stems"] if False else json.loads((root / "fixed_pose_spec.json").read_text())["stems"]
    if args.samples is not None: stems = stems[:args.samples]
    device = "cuda" if torch.cuda.is_available() else "cpu"; estimator = KeypointRCNNEstimator(device, args.confidence_threshold)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    reference = {stem: estimator(pose_root / stem / "control.png") for stem in stems}; rows = []
    for flow_row in flow["checkpoints"]:
        step = flow_row["checkpoint_step"]; pose_items = []; similarities = []
        for stem in stems:
            directory = pose_root / stem; generated = directory / f"step_{step:06d}.png"; prompt = json.loads((directory / "metadata.json").read_text())["prompt"]
            predicted = estimator(generated); metric = pck_for_people(reference[stem], predicted, args.confidence_threshold); metric["stem"] = stem; metric["reference_people"] = len(reference[stem]); metric["predicted_people"] = len(predicted); pose_items.append(metric)
            inputs = processor(text=[prompt], images=[Image.open(generated).convert("RGB")], return_tensors="pt", padding=True).to(device)
            with torch.inference_mode(): similarity = cosine_from_embeddings(clip.get_image_features(pixel_values=inputs.pixel_values).float().cpu().numpy(), clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).float().cpu().numpy())[0]
            similarities.append(float(similarity))
        joints = sum(x["evaluated_joint_count"] for x in pose_items)
        def weighted(key):
            return (sum((x[key] or 0) * x["evaluated_joint_count"] for x in pose_items) / joints) if joints else None
        pose = {"pck_005": weighted("pck_005"), "pck_010": weighted("pck_010"), "pck_020": weighted("pck_020"), "detection_coverage": float(np.mean([x["detection_coverage"] for x in pose_items])) if pose_items else 0.0, "evaluated_joint_count": joints, "excluded_sample_count": sum(bool(x["excluded"]) for x in pose_items), "per_image": pose_items}
        clip_values = aggregate(similarities); clip_result = {"mean_cosine_similarity": clip_values["mean"], "median_cosine_similarity": clip_values["median"], "std_cosine_similarity": clip_values["std"], "sample_count": clip_values["sample_count"], "per_sample": [{"stem": stem, "cosine_similarity": score} for stem, score in zip(stems, similarities)]}
        rows.append({"checkpoint_step": step, "fixed_flow": {"mean": flow_row["mean_fixed_flow_loss"], "median": flow_row["median_fixed_flow_loss"], "std": flow_row["std_fixed_flow_loss"], "sample_count": flow_row["sample_count"], "per_sample": flow_row["per_sample"]}, "pose": pose, "clip": clip_result})
    summary = {"format_version": 1, "metadata": {"fixed_flow_seed": 420100, "fixed_pose_seed": 420200, "fixed_flow_spec_sha256": __import__("hashlib").sha256((root / "fixed_flow_spec.json").read_bytes()).hexdigest(), "pose_estimator": estimator.identifier, "pck_normalization": "reference COCO-17 confident-keypoint bounding-box diagonal", "keypoint_confidence_threshold": args.confidence_threshold, "clip_model": args.clip_model_id, "clip_implementation": "transformers.CLIPModel"}, "checkpoints": rows}
    summary["best_checkpoints"] = choose_best(summary); (root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(root / "evaluation_summary.json")
if __name__ == "__main__": main()
