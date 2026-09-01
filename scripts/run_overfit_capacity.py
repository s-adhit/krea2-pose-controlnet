"""One explicit runner for isolated Mixed-32 capacity experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys

from pose_controlnet.overfit_capacity import CapacityScientificConfig, validate_capacity_scientific_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", default="mixed32", choices=("mixed32",))
    parser.add_argument("--resolution", default="native", choices=("native", "current", "768"))
    parser.add_argument("--pose-loss", default="none", choices=("none", "normalized_coordinate_huber"))
    parser.add_argument("--lambda-pose", type=float, default=0.0)
    parser.add_argument("--forced-pose-exposure-probability", type=float, default=0.0)
    parser.add_argument("--pose-timestep-min", type=float); parser.add_argument("--pose-timestep-max", type=float)
    parser.add_argument("--pose-target-sidecar")
    parser.add_argument("--stage", default="train", choices=("train", "generate", "score", "report", "all"))
    parser.add_argument("--checkpoint-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints")
    parser.add_argument("--output-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    scientific = validate_capacity_scientific_config(CapacityScientificConfig(
        base_experiment=args.base_experiment, resolution=args.resolution, pose_loss=args.pose_loss,
        lambda_pose=args.lambda_pose, forced_pose_exposure_probability=args.forced_pose_exposure_probability,
        pose_timestep_min=args.pose_timestep_min, pose_timestep_max=args.pose_timestep_max,
    ))
    if scientific.pose_loss != "none" and not args.pose_target_sidecar:
        parser.error("--pose-target-sidecar is required when pose loss is enabled")
    train = [sys.executable, "scripts/train_overfit_capacity.py", "--base-experiment", args.base_experiment,
             "--resolution", scientific.resolution, "--pose-loss", scientific.pose_loss,
             "--lambda-pose", str(scientific.lambda_pose), "--forced-pose-exposure-probability", str(scientific.forced_pose_exposure_probability),
             "--checkpoint-root", args.checkpoint_root, "--evaluation-root", args.output_root]
    if scientific.pose_timestep_min is not None:
        train += ["--pose-timestep-min", str(scientific.pose_timestep_min), "--pose-timestep-max", str(scientific.pose_timestep_max)]
    if args.pose_target_sidecar: train += ["--pose-target-sidecar", args.pose_target_sidecar]
    if args.no_wandb: train.append("--no-wandb")
    evaluation = [sys.executable, "scripts/evaluate_overfit_capacity.py", "--experiment", scientific.experiment_name,
                  "--checkpoint-root", args.checkpoint_root, "--output-root", args.output_root]
    if args.stage in ("train", "all"): subprocess.run(train, check=True)
    if args.stage in ("generate", "score", "report"):
        subprocess.run([*evaluation, "--stage", "score-only" if args.stage == "score" else args.stage], check=True)
    elif args.stage == "all": subprocess.run([*evaluation, "--stage", "all"], check=True)


if __name__ == "__main__": main()
