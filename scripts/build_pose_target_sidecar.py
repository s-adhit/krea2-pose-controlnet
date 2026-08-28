"""Build a new read-only unified pose-target sidecar from authoritative inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.pose_targets import (
    PoseTargetError,
    build_authoritative_sidecar_records,
    build_sidecar_records,
    write_sidecar,
)


def geometry_from_shards(root: Path) -> dict[str, dict]:
    geometry = {}
    for path in sorted(root.glob("*/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            stem = sample["stem"]
            if stem in geometry:
                raise PoseTargetError(f"Duplicate shard geometry stem: {stem}")
            geometry[stem] = {key: sample[key] for key in ("source_size", "resized_size", "crop_box", "bucket")}
    if not geometry:
        raise PoseTargetError(f"No latent-shard samples found under {root}")
    return geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--authoritative-jsonl", type=Path, help="Checked-in authoritative v1 COCO/Human-Art export.")
    source.add_argument("--source-spec", type=Path, help="Legacy v2 source/provenance contract.")
    parser.add_argument("--output", type=Path, required=True, help="New nonexistent sidecar directory.")
    args = parser.parse_args()
    try:
        geometry = geometry_from_shards(args.latent_root)
        if args.authoritative_jsonl:
            records, summary = build_authoritative_sidecar_records(
                geometry, authoritative_jsonl=args.authoritative_jsonl,
            )
            provenance = {"authoritative_jsonl": str(args.authoritative_jsonl.resolve())}
        else:
            spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
            records, summary = build_sidecar_records(geometry, spec.get("sources", {}))
            provenance = {"source_spec": str(args.source_spec.resolve()), "source_spec_sha256": __import__("hashlib").sha256(args.source_spec.read_bytes()).hexdigest()}
        metadata = write_sidecar(records, args.output, build_metadata={**provenance, **summary})
    except (OSError, KeyError, ValueError, PoseTargetError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", "output": str(args.output), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
