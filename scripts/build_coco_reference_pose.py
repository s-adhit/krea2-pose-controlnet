"""Build an immutable exact-manifest COCO capacity reference sidecar on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_controlnet.reference_pose import build_exact_coco_capacity_reference_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Exact capacity experiment whose immutable manifest is resolved.")
    parser.add_argument("--latent-root", type=Path, required=True, help="Explicit directory containing only direct verified train-*.pt PoseBridge shards.")
    parser.add_argument("--annotations", type=Path, nargs="+", required=True, help="Official COCO person_keypoints_{train,val}2017.json files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = build_exact_coco_capacity_reference_sidecar(
        experiment_name=args.experiment, latent_root=args.latent_root,
        annotation_paths=args.annotations, output=args.output,
    )
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": metadata["records_sha256"],
                      "records": metadata["output_record_count"], "people": metadata["people_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
