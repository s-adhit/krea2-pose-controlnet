"""Fail-closed, no-GPU verification for the full 16,503-sample 768 benchmark inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.full_768_cache import Full768CacheError, verify_full_768_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--train-manifest", type=Path, default=None,
                        help="Authoritative project full-train manifest (default: checked-in data/manifests/train.jsonl).")
    parser.add_argument("--latent-root", required=True, type=Path)
    parser.add_argument("--pose-sidecar", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify_full_768_cache(dataset_root=args.dataset_root, cache_root=args.latent_root,
                                       pose_sidecar=args.pose_sidecar,
                                       **({"train_manifest": args.train_manifest} if args.train_manifest else {}))
    except (Full768CacheError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
