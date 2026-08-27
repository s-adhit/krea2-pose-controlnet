"""Read-only Monte Carlo comparison for the timestep-only continuation.

This script does not construct a model or optimizer and never writes a
checkpoint.  It samples the executable training sampler over the actual shard
bucket mix and prints JSON suitable for launch review.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import train
from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.diffusion import sample_flow_timestep


def _summary(values: np.ndarray, auxiliary_masks: np.ndarray, weights: np.ndarray) -> dict:
    total = float(weights.sum())
    def fraction(lower: float, upper: float, inclusive_upper: bool = False) -> float:
        selector = (values >= lower) & ((values <= upper) if inclusive_upper else (values < upper))
        return float(weights[selector].sum() / total)
    return {
        "sample_count": int(len(values)),
        "effective_bucket_weight": total,
        "mean": float(np.dot(values, weights) / total),
        "median": None,
        "0.0-0.2": fraction(0.0, .2), "0.2-0.4": fraction(.2, .4),
        "0.4-0.6": fraction(.4, .6), "0.6-0.8": fraction(.6, .8),
        "0.8-1.0": fraction(.8, 1.0, inclusive_upper=True),
        "actual_auxiliary_routing_fraction": float(np.dot(auxiliary_masks.astype(float), weights) / total),
    }


def audit(*, source_checkpoint: Path, latent_root: Path, samples_per_bucket: int, seed: int) -> dict:
    source_state = load_training_state(source_checkpoint)
    original_cfg = train.train_config_from_checkpoint_values(source_state["config"])
    proposed_cfg = train.timestep_branch_config_from_source_state(source_state)
    dataset = PreparedLatentShardDataset(str(latent_root), "train")
    bucket_counts = Counter(record[2] for record in dataset.records)
    originals, proposals, routes, buckets = [], [], [], []
    for index, (shape, weight) in enumerate(sorted(bucket_counts.items())):
        sequence_length = (shape[0] // 2) * (shape[1] // 2)
        original = sample_flow_timestep(samples_per_bucket, sequence_length, original_cfg, "cpu",
                                        torch.Generator().manual_seed(seed + index)).numpy()
        proposed, mask = sample_flow_timestep(samples_per_bucket, sequence_length, proposed_cfg, "cpu",
                                               torch.Generator().manual_seed(seed + 10_000 + index), return_aux_mask=True)
        originals.append(original); proposals.append(proposed.numpy()); routes.append(mask.numpy())
        buckets.append({"latent_shape": list(shape), "sequence_length": sequence_length, "weight": int(weight)})
    weights = np.concatenate([np.full(samples_per_bucket, item["weight"], dtype=np.float64) for item in buckets])
    def weighted(parts: list[np.ndarray], mask_parts: list[np.ndarray]) -> dict:
        values, mask = np.concatenate(parts), np.concatenate(mask_parts)
        # Deterministic replication-free bucket weighting.
        order = np.argsort(values)
        sorted_values, sorted_weights = values[order], weights[order]
        median = float(sorted_values[min(np.searchsorted(np.cumsum(sorted_weights), sorted_weights.sum() / 2, side="left"), len(values) - 1)])
        result = _summary(values, mask, weights)
        result["median"] = median
        return result
    return {"format_version": 1, "seed": seed, "samples_per_bucket": samples_per_bucket,
            "source_checkpoint": str(source_checkpoint), "bucket_counts": buckets,
            "original": weighted(originals, [np.zeros_like(route) for route in routes]),
            "proposed_80_20": weighted(proposals, routes),
            "auxiliary_pre_shift_support": [proposed_cfg.timestep_aux_min, proposed_cfg.timestep_aux_max],
            "auxiliary_probability": proposed_cfg.timestep_aux_prob}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=train.TIMESTEP_BRANCH_SOURCE_CHECKPOINT)
    parser.add_argument("--latent-root", type=Path, default=Path("/lambda/nfs/adhit/krea2-pose/posebridge_latents"))
    parser.add_argument("--samples-per-bucket", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=420_300)
    args = parser.parse_args()
    if args.samples_per_bucket < 1:
        parser.error("--samples-per-bucket must be positive")
    print(json.dumps(audit(source_checkpoint=args.source_checkpoint, latent_root=args.latent_root,
                           samples_per_bucket=args.samples_per_bucket, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
