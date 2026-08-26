"""Hard verification for exact persistent Qwen text-conditioning archives."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.text_conditioning import TextConditioningError, smoke_online_cached_equivalence, verify_text_conditioning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Optional; defaults to immutable dataset_root recorded in latent shards")
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--online-equivalence", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-split", type=int, default=2)
    args = parser.parse_args()
    try:
        result = {"counts": verify_text_conditioning(dataset_root=args.dataset_root, latent_root=args.latent_root, output_root=args.output_root)}
        if args.online_equivalence:
            if args.dataset_root is None: parser.error("--online-equivalence requires --dataset-root")
            result["equivalence"] = smoke_online_cached_equivalence(dataset_root=args.dataset_root, output_root=args.output_root, device=args.device, samples_per_split=args.samples_per_split)
    except TextConditioningError as exc: parser.error(str(exc))
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__": main()
