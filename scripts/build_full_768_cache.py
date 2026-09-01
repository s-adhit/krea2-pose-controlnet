"""Build or safely resume the isolated full-train 768 latent cache and pose sidecar."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.full_768_cache import (  # noqa: E402
    Full768CacheError, build_full_768_pose_sidecar, prepare_full_768_cache,
    verify_full_768_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--train-manifest", type=Path, default=None,
                        help="Authoritative project full-train manifest (default: checked-in data/manifests/train.jsonl).")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pose-source", required=True, type=Path)
    parser.add_argument("--pose-output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-samples", type=int, default=256)
    args = parser.parse_args()
    try:
        cache = prepare_full_768_cache(dataset_root=args.dataset_root, output_root=args.output_root,
                                       device=args.device, shard_samples=args.shard_samples,
                                       **({"train_manifest": args.train_manifest} if args.train_manifest else {}))
        if args.pose_output.exists():
            # Existing output is never rewritten; verifier proves it is the requested artifact.
            pose = {"reused": True, **verify_full_768_cache(dataset_root=args.dataset_root, cache_root=args.output_root,
                                                               pose_sidecar=args.pose_output,
                                                               **({"train_manifest": args.train_manifest} if args.train_manifest else {}))}
        else:
            pose = build_full_768_pose_sidecar(cache_root=args.output_root, authoritative_source=args.pose_source,
                                               output_dir=args.pose_output)
        verified = verify_full_768_cache(dataset_root=args.dataset_root, cache_root=args.output_root,
                                         pose_sidecar=args.pose_output,
                                         **({"train_manifest": args.train_manifest} if args.train_manifest else {}))
    except (Full768CacheError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", "cache": cache, "pose": pose, "verified": verified}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
