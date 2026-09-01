"""Print dataset-equivalent wall-time projections from a measured GH200 step time."""
from __future__ import annotations

import argparse
import json

from pose_controlnet.throughput_benchmark import LOCKED_EFFECTIVE_BATCH, LOCKED_TRAINING_SAMPLES, projected_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds-per-optimizer-step", type=float, required=True)
    parser.add_argument("--effective-batch-size", type=int, default=LOCKED_EFFECTIVE_BATCH)
    parser.add_argument("--training-samples", type=int, default=LOCKED_TRAINING_SAMPLES)
    args = parser.parse_args()
    print(json.dumps({"seconds_per_optimizer_step": args.seconds_per_optimizer_step,
                      "effective_batch_size": args.effective_batch_size,
                      "training_samples": args.training_samples,
                      "projections": projected_runtime(seconds_per_optimizer_step=args.seconds_per_optimizer_step,
                                                       effective_batch_size=args.effective_batch_size,
                                                       training_samples=args.training_samples)}, indent=2))


if __name__ == "__main__":
    main()
