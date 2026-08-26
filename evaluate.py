"""Post-500 deterministic fixed-flow and fixed-pose comparison gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.evaluation import (DEFAULT_FIXED_FLOW_SEED, DEFAULT_FIXED_POSE_SEED, evaluate_fixed_flow,
    evaluate_fixed_pose, ordered_checkpoints, read_or_create_spec, write_spec)
from pose_controlnet.model import build_pose_model
from pose_controlnet.vae_preprocessing import load_krea_vae
from pose_controlnet.dataset_index import validate_posebridge_snapshot

def _cfg(args):
    return TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root, eval_steps=args.eval_steps, eval_guidance=args.eval_guidance,
                       metrics_jsonl_path=str(Path(args.output_dir) / "metrics.jsonl"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixed-flow", "fixed-pose"))
    parser.add_argument("--raw-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors")
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    parser.add_argument("--checkpoint-dir", default="/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100")
    parser.add_argument("--later-checkpoint-dir", default="/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500",
                        help="checkpoint root used for required steps above 100")
    parser.add_argument("--output-dir", default="/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500")
    parser.add_argument("--split", default=None); parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None); parser.add_argument("--eval-steps", type=int, default=8); parser.add_argument("--eval-guidance", type=float, default=3.5)
    parser.add_argument("--dataset-root", help="required for fixed-pose control PNG export; otherwise read from shards.json")
    parser.add_argument("--comparison-grid-thumbnail-width", type=int, default=320,
                        help="fixed-pose grid cell width in pixels (default: 320)")
    parser.add_argument("--comparison-grid-thumbnail-height", type=int, default=320,
                        help="fixed-pose grid cell height in pixels (default: 320)")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Run evaluation from the GH200 host shell with CUDA visible")
    split = args.split or ("val" if args.mode == "fixed-flow" else "diagnostic_val")
    count = args.samples or (32 if args.mode == "fixed-flow" else 8); seed = args.seed if args.seed is not None else (DEFAULT_FIXED_FLOW_SEED if args.mode == "fixed-flow" else DEFAULT_FIXED_POSE_SEED)
    output = Path(args.output_dir); dataset = PreparedLatentShardDataset(args.latent_root, split, text_conditioning_root=args.text_conditioning_root)
    kind = "fixed_flow" if args.mode == "fixed-flow" else "fixed_pose"; spec_path = output / f"{kind}_spec.json"
    spec = read_or_create_spec(spec_path, dataset, split=split, count=count, seed=seed, kind=kind); write_spec(spec_path, spec)
    cfg, device = _cfg(args), torch.device("cuda"); model = build_pose_model(args.raw_ckpt, 64, 64, "cuda").eval(); checkpoints = ordered_checkpoints(args.checkpoint_dir, later_checkpoint_dir=args.later_checkpoint_dir)
    if args.mode == "fixed-flow":
        result = evaluate_fixed_flow(model, dataset, spec, cfg, device, checkpoints); path = output / "fixed_flow_results.json"
    else:
        metadata = json.loads((Path(args.latent_root) / "shards.json").read_text()); dataset_root = Path(args.dataset_root or metadata["dataset_root"])
        snapshot = validate_posebridge_snapshot(dataset_root); records = {record.stem: record for record in snapshot.records_by_split[split]}
        controls = {stem: records[stem].control_path for stem in spec["stems"]}; vae = load_krea_vae(device)
        result = evaluate_fixed_pose(model, dataset, spec, cfg, device, checkpoints, vae, controls, output,
                                     thumbnail_width=args.comparison_grid_thumbnail_width,
                                     thumbnail_height=args.comparison_grid_thumbnail_height); path = output / "fixed_pose_results.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(path)
if __name__ == "__main__": main()
