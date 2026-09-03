"""Build the immutable authoritative v3 pose sidecar for frozen final-val 48."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.pose_targets import PoseTargetError, build_authoritative_sidecar_records, write_sidecar
from pose_controlnet.turbo_evaluation import turbo_scoring_geometry


ROOT = Path("docs/evaluation/final-val-benchmark-selection")
SELECTION = ROOT / "final_val_benchmark_48.jsonl"
SPEC = ROOT / "final_val_benchmark_spec.json"
SPEC_SHA256 = "93a5254e57fa208263f6188573e0760ffedd954bf3b3b3425109ea0178957cd0"
SELECTION_SHA256 = "23d448d573a2ffd20adfd73fa88f34ebc08df280a051cb0931d9ecdcc1231ceb"
OUTPUT = ROOT / "final_val_benchmark_48_pose_targets_v3"
QUOTAS = {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_selection(path: Path, spec: Path) -> tuple[list[str], dict[str, Any]]:
    if _sha256(path) != SELECTION_SHA256:
        raise PoseTargetError(f"Frozen final-val selection SHA-256 mismatch: {path}")
    if _sha256(spec) != SPEC_SHA256:
        raise PoseTargetError(f"Frozen final-val spec SHA-256 mismatch: {spec}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseTargetError("Frozen final-val selection/spec is unreadable") from exc
    stems = [row.get("stem") for row in rows if isinstance(row, dict)]
    sources = [row.get("source") for row in rows if isinstance(row, dict)]
    if len(stems) != 48 or len(stems) != len(set(stems)) or any(not isinstance(stem, str) or not stem for stem in stems):
        raise PoseTargetError("Frozen final-val selection must contain exactly 48 unique stems")
    if list(zip(sources, stems)) != sorted(zip(sources, stems)) or Counter(sources) != QUOTAS:
        raise PoseTargetError("Frozen final-val selection order or source quotas mismatch")
    if payload.get("stems") != stems:
        raise PoseTargetError("Frozen final-val spec stem order does not match selection")
    provenance = payload.get("benchmark", {}).get("provenance", {}).get("final_val_benchmark_48", {})
    if provenance.get("sha256") != SELECTION_SHA256 or provenance.get("record_count") != 48:
        raise PoseTargetError("Frozen final-val spec selection provenance mismatch")
    return stems, payload


def _final_geometry(dataset: PreparedLatentShardDataset, stems: list[str]) -> dict[str, dict[str, list[int]]]:
    wanted = set(stems)
    indices: dict[str, int] = {}
    for index, record in enumerate(dataset.records):
        stem = record[3]
        if stem in wanted:
            if stem in indices:
                raise PoseTargetError(f"Duplicate final-val stem in latent shards: {stem}")
            indices[stem] = index
    missing = [stem for stem in stems if stem not in indices]
    if missing:
        raise PoseTargetError(f"Frozen final-val stems missing from latent shards: {missing[:3]}")
    return {stem: turbo_scoring_geometry(dataset[indices[stem]]) for stem in stems}


def build(*, latent_root: Path, text_conditioning_root: Path, selection: Path, spec: Path,
          authoritative_jsonl: Path, output: Path) -> dict[str, Any]:
    stems, _ = _read_selection(selection, spec)
    dataset = PreparedLatentShardDataset(latent_root, "val", text_conditioning_root=text_conditioning_root)
    geometry = _final_geometry(dataset, stems)
    records, summary = build_authoritative_sidecar_records(
        geometry, authoritative_jsonl=authoritative_jsonl, stem_order=stems,
    )
    if [record["stem"] for record in records] != stems or any(not record["pose_reward_available"] for record in records):
        raise PoseTargetError("Final-val authoritative pose targets are incomplete or out of frozen order")
    metadata = write_sidecar(records, output, preserve_order=True, build_metadata={
        **summary,
        "sidecar_kind": "final_val_benchmark_48_authoritative_pose_targets_v3",
        "frozen_stems": stems,
        "frozen_selection": {"path": str(selection.resolve()), "sha256": _sha256(selection), "record_count": len(stems)},
        "frozen_spec": {"path": str(spec.resolve()), "sha256": _sha256(spec)},
        "authoritative_source_pose_export": {"path": str(authoritative_jsonl.resolve()), "sha256": _sha256(authoritative_jsonl)},
    })
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, default=Path("/lambda/nfs/adhit/krea2-pose/posebridge_latents"))
    parser.add_argument("--text-conditioning-root", type=Path, default=Path("/lambda/nfs/adhit/krea2-pose/text_conditioning"))
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--authoritative-jsonl", type=Path, default=Path("data/pose_targets_authoritative_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        metadata = build(**vars(args))
    except (OSError, ValueError, PoseTargetError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", "output": str(args.output), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
