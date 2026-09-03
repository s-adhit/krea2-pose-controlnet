"""Bind the frozen final validation selection to immutable cached identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.evaluation import make_evaluation_spec
from pose_controlnet.turbo_runtime import turbo_metadata


FINAL_COUNT = 48
FINAL_VAL_SEED = 420_300
SOURCE_QUOTAS = {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8}
PROVENANCE_PATHS = {
    "final_val_benchmark_48": "docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl",
    "val_manifest": "data/manifests/val.jsonl",
    "diagnostic_val_manifest": "data/manifests/diagnostic_val.jsonl",
    "candidate_pool_96": "docs/evaluation/final-val-benchmark-selection/candidate_pool_96.jsonl",
}


class FinalValBenchmarkSpecError(ValueError):
    """Raised when the frozen selection cannot safely define the final spec."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalValBenchmarkSpecError(f"Malformed JSONL input: {path}") from exc


def _manifest_stems(path: Path) -> set[str]:
    try:
        stems = [Path(row["file_name"]).stem for row in _read_jsonl(path)]
    except (KeyError, TypeError) as exc:
        raise FinalValBenchmarkSpecError(f"Malformed manifest: {path}") from exc
    if len(stems) != len(set(stems)):
        raise FinalValBenchmarkSpecError(f"Manifest has duplicate stems: {path}")
    return set(stems)


def _provenance(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": PROVENANCE_PATHS[name], "sha256": _sha256(path), "record_count": len(_read_jsonl(path))}
        for name, path in paths.items()
    }


def build_final_val_benchmark_spec(
    dataset: PreparedLatentShardDataset,
    *,
    frozen_selection: Path,
    val_manifest: Path,
    diagnostic_manifest: Path,
    candidate_pool: Path,
    seed: int = FINAL_VAL_SEED,
) -> dict[str, Any]:
    """Create the final-val Turbo spec using the shared seed/identity builder."""
    frozen = _read_jsonl(frozen_selection)
    if len(frozen) != FINAL_COUNT:
        raise FinalValBenchmarkSpecError(f"Frozen selection must contain {FINAL_COUNT} records, got {len(frozen)}")
    try:
        stems = [row["stem"] for row in frozen]
        sources = [row["source"] for row in frozen]
        orientations = [row["orientation"] for row in frozen]
    except (KeyError, TypeError) as exc:
        raise FinalValBenchmarkSpecError("Frozen selection is missing required stem/source/orientation metadata") from exc
    if any(not isinstance(stem, str) or not stem for stem in stems) or len(stems) != len(set(stems)):
        raise FinalValBenchmarkSpecError("Frozen selection stems must be unique non-empty strings")
    if list(zip(sources, stems)) != sorted(zip(sources, stems)):
        raise FinalValBenchmarkSpecError("Frozen selection must retain the documented source-then-stem order")
    source_counts, orientation_counts = Counter(sources), Counter(orientations)
    if dict(sorted(source_counts.items())) != SOURCE_QUOTAS:
        raise FinalValBenchmarkSpecError(f"Frozen source quotas must be {SOURCE_QUOTAS}, got {dict(sorted(source_counts.items()))}")

    val_stems, diagnostic_stems = _manifest_stems(val_manifest), _manifest_stems(diagnostic_manifest)
    if set(stems) - val_stems:
        raise FinalValBenchmarkSpecError("Frozen selection contains a stem absent from val.jsonl")
    if set(stems) & diagnostic_stems:
        raise FinalValBenchmarkSpecError("Frozen selection overlaps diagnostic_val.jsonl")
    candidate_rows = _read_jsonl(candidate_pool)
    candidate_stems = {row.get("stem") for row in candidate_rows}
    if not set(stems) <= candidate_stems:
        raise FinalValBenchmarkSpecError("Frozen selection contains a stem absent from candidate_pool_96.jsonl")
    candidate_digest = _sha256(candidate_pool)
    if any(row.get("candidate_pool_sha256") != candidate_digest for row in frozen):
        raise FinalValBenchmarkSpecError("Frozen selection's candidate-pool digest does not match candidate_pool_96.jsonl")

    # make_evaluation_spec owns the identity hashes and deterministic per-stem seeds.
    spec = make_evaluation_spec(dataset, split="val", count=FINAL_COUNT, seed=seed,
                                kind="final_val_turbo_fixed_pose", stems=stems)
    spec["benchmark"] = {
        "name": "final_val_benchmark_48",
        "stem_order": "frozen source-then-stem ascending order",
        "source_counts": dict(sorted(source_counts.items())),
        "orientation_counts": dict(sorted(orientation_counts.items())),
        "provenance": _provenance({
            "final_val_benchmark_48": frozen_selection,
            "val_manifest": val_manifest,
            "diagnostic_val_manifest": diagnostic_manifest,
            "candidate_pool_96": candidate_pool,
        }),
    }
    spec["turbo"] = {**turbo_metadata(), "control_scale": 1.0}
    return spec


def write_immutable_spec(path: Path, spec: dict[str, Any]) -> Path:
    """Write once, or require a byte-for-byte equivalent JSON contract."""
    payload = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FinalValBenchmarkSpecError(f"Existing final benchmark spec conflicts with the frozen contract: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> None:
    root = Path("docs/evaluation/final-val-benchmark-selection")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    parser.add_argument("--frozen-selection", type=Path, default=root / "final_val_benchmark_48.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.jsonl"))
    parser.add_argument("--diagnostic-manifest", type=Path, default=Path("data/manifests/diagnostic_val.jsonl"))
    parser.add_argument("--candidate-pool", type=Path, default=root / "candidate_pool_96.jsonl")
    parser.add_argument("--output", type=Path, default=root / "final_val_benchmark_spec.json")
    args = parser.parse_args()
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    spec = build_final_val_benchmark_spec(dataset, frozen_selection=args.frozen_selection, val_manifest=args.val_manifest,
                                          diagnostic_manifest=args.diagnostic_manifest, candidate_pool=args.candidate_pool)
    write_immutable_spec(args.output, spec)
    print(f"wrote {len(spec['stems'])}-record final validation benchmark spec: {args.output}")


if __name__ == "__main__":
    main()
