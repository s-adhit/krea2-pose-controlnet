"""Build an immutable COCO-only authoritative reference-pose sidecar on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pose_controlnet.reference_pose import build_coco_reference_records, write_reference_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-shard", type=Path, required=True, help="A verified shard whose persisted geometry is used verbatim.")
    parser.add_argument("--annotations", type=Path, nargs="+", required=True, help="Official COCO person_keypoints_{train,val}2017.json files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.latent_shard, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Not a verified v1 latent shard: {args.latent_shard}")
    records = build_coco_reference_records(payload["samples"], args.annotations)
    digest = write_reference_jsonl(records, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": digest, "records": len(records), "people": sum(len(record["people"]) for record in records)}, sort_keys=True))


if __name__ == "__main__":
    main()
