"""Read-only provenance inventory; this never invents targets from control PNGs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.pose_targets import source_for_stem


def stems_from_jsonl(path: Path) -> list[str]:
    return [Path(json.loads(line)["file_name"]).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, help="Optional proposed build specification; only audited, never written.")
    args = parser.parse_args()
    root = args.dataset_root
    all_stems = stems_from_jsonl(root / "metadata.jsonl")
    used_stems = [stem for split in ("train", "val", "diagnostic_val") for stem in stems_from_jsonl(root / "manifests" / f"{split}.jsonl")]
    all_counts, used_counts = Counter(map(source_for_stem, all_stems)), Counter(map(source_for_stem, used_stems))
    sources = {name: {"target_provenance": "dwpose_pseudolabel" if name == "danbooru" else "original_annotation", "corpus_count": all_counts[name], "used_count": used_counts[name]} for name in sorted(all_counts)}
    if args.source_spec:
        spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
        for name, row in sources.items():
            supplied = spec.get("sources", {}).get(name)
            row["source_spec_present"] = supplied is not None
            row["authoritative_artifact_recoverable"] = bool(supplied and _paths_exist(supplied))
    report = {"schema_version": 1, "all_physical_pairs": len(all_stems), "used_manifest_samples": len(used_stems), "sources": sources, "status": "PASS" if all(row.get("authoritative_artifact_recoverable", False) for row in sources.values()) else "BLOCKED_MISSING_AUTHORITATIVE_ARTIFACTS"}
    print(json.dumps(report, indent=2, sort_keys=True))


def _paths_exist(spec: dict) -> bool:
    paths = spec.get("annotation_paths", []) or [spec.get("pseudolabel_path")]
    return bool(paths) and all(path and Path(path).is_file() for path in paths)


if __name__ == "__main__":
    main()
