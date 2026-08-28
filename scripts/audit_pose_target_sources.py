"""Read-only provenance inventory; this never invents targets from control PNGs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.pose_targets import (
    PoseTargetError, authoritative_source_oob_report, diagnostic_coverage,
    load_authoritative_export, source_for_stem,
)


def stems_from_jsonl(path: Path) -> list[str]:
    return [Path(json.loads(line)["file_name"]).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--authoritative-jsonl", type=Path, help="Checked-in authoritative v1 export to join by exact active stem.")
    parser.add_argument("--source-spec", type=Path, help="Optional proposed build specification; only audited, never written.")
    args = parser.parse_args()
    root = args.dataset_root
    all_stems = stems_from_jsonl(root / "metadata.jsonl")
    used_stems = [stem for split in ("train", "val", "diagnostic_val") for stem in stems_from_jsonl(root / "manifests" / f"{split}.jsonl")]
    all_counts, used_counts = Counter(map(source_for_stem, all_stems)), Counter(map(source_for_stem, used_stems))
    sources = {name: {"corpus_count": all_counts[name], "used_count": used_counts[name]} for name in sorted(all_counts)}
    total_used = sum(used_counts.values())
    blocking = []
    if args.authoritative_jsonl:
        try:
            export, export_metadata = load_authoritative_export(args.authoritative_jsonl)
        except (OSError, ValueError, PoseTargetError) as exc:
            parser.error(str(exc))
        active = set(used_stems)
        missing = sorted(stem for stem in active if source_for_stem(stem) != "danbooru" and stem not in export)
        unexpected_danbooru = sorted(stem for stem in active if source_for_stem(stem) == "danbooru" and stem in export)
        diagnostic_stems = [Path(json.loads(line)["file_name"]).stem for line in (root / "manifests/diagnostic_val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        diagnostic_records = {
            stem: {"stem": stem, "pose_reward_available": stem in export}
            for stem in diagnostic_stems if source_for_stem(stem) != "danbooru"
        }
        diagnostic = diagnostic_coverage(diagnostic_records)
        diagnostic["danbooru_stems"] = sorted(stem for stem in diagnostic_stems if source_for_stem(stem) == "danbooru")
        diagnostic["danbooru_available"] = [stem for stem in diagnostic["danbooru_stems"] if stem in export]
        source_oob = authoritative_source_oob_report(export, active_stems=active)
        for name, row in sources.items():
            row["pose_reward_available"] = name != "danbooru"
            row["target_provenance"] = "original_annotation" if name != "danbooru" else "unavailable"
            row["active_authoritative_records"] = sum(1 for stem in active if source_for_stem(stem) == name and stem in export)
        report = {
            "schema_version": 3, "all_physical_pairs": len(all_stems), "used_manifest_samples": len(used_stems),
            "coverage": _coverage(sources, used_counts, total_used, total_used - used_counts["danbooru"]),
            "sources": sources, "authoritative_export": export_metadata,
            "inactive_authoritative_records": len(set(export) - active),
            "unresolved_active_stems": missing, "unexpected_active_danbooru_records": unexpected_danbooru,
            "source_oob_contract": source_oob,
            "diagnostic_coverage": diagnostic,
            "blocking_available_sources": ["authoritative_export"] if missing or unexpected_danbooru or diagnostic["status"] != "PASS" or diagnostic["danbooru_available"] or source_oob["status"] != "PASS" else [],
            "status": "PASS" if not missing and not unexpected_danbooru and diagnostic["status"] == "PASS" and not diagnostic["danbooru_available"] and source_oob["status"] == "PASS" else "FAIL",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] != "PASS":
            raise SystemExit(1)
        return
    if args.source_spec:
        spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
        for name, row in sources.items():
            supplied = spec.get("sources", {}).get(name)
            row["source_spec_present"] = supplied is not None
            available = supplied.get("pose_reward_available") if isinstance(supplied, dict) else None
            row["pose_reward_available"] = available
            row["target_provenance"] = supplied.get("target_provenance") if isinstance(supplied, dict) else None
            if available is True:
                row["authoritative_artifact_recoverable"] = _paths_exist(supplied)
                if not row["authoritative_artifact_recoverable"]:
                    blocking.append(name)
            elif available is False:
                row["authoritative_artifact_recoverable"] = None
            else:
                blocking.append(name)
    else:
        blocking = sorted(sources)
    available_used = sum(used_counts[name] for name, row in sources.items() if row.get("pose_reward_available") is True)
    report = {
        "schema_version": 2, "all_physical_pairs": len(all_stems), "used_manifest_samples": len(used_stems),
        "coverage": _coverage(sources, used_counts, total_used, available_used), "sources": sources,
        "blocking_available_sources": sorted(blocking),
        "status": "PASS" if not blocking else "BLOCKED_MISSING_AUTHORITATIVE_ARTIFACTS",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _paths_exist(spec: dict) -> bool:
    paths = spec.get("annotation_paths", []) or [spec.get("adapter_path")]
    return bool(paths) and all(path and Path(path).is_file() for path in paths)


def _coverage(sources: dict, used_counts: Counter, total: int, available: int) -> dict:
    per_source = {}
    for name in sorted(sources):
        count = used_counts[name]
        has_target = sources[name].get("pose_reward_available") is True
        per_source[name] = {
            "total": count, "available": count if has_target else 0, "unavailable": 0 if has_target else count,
            "available_percent": 100.0 if has_target else 0.0, "unavailable_percent": 0.0 if has_target else 100.0,
        }
    return {
        "total": {"total": total, "available": available, "unavailable": total - available,
                  "available_percent": 0.0 if total == 0 else 100.0 * available / total,
                  "unavailable_percent": 0.0 if total == 0 else 100.0 * (total - available) / total},
        "sources": per_source,
    }


if __name__ == "__main__":
    main()
