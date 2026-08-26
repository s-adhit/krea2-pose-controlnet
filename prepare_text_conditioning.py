"""Create atomic resumable exact Qwen conditioning archives for latent shards."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from pose_controlnet.text_conditioning import TextConditioningError, prepare_text_conditioning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Optional; defaults to immutable dataset_root recorded in latent shards")
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-samples", type=int, default=64)
    args = parser.parse_args()
    try:
        counts = prepare_text_conditioning(dataset_root=args.dataset_root, latent_root=args.latent_root,
            output_root=args.output_root, device=args.device, shard_samples=args.shard_samples)
    except TextConditioningError as exc: parser.error(str(exc))
    print(json.dumps({"status": "PASS", "counts": counts, "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__": main()
