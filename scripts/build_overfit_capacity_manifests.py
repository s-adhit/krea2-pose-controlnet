"""Deterministically materialize the six reviewed 32-sample capacity manifests.

The committed manifests are the experiment inputs.  This script is only their
auditable reproducer: it draws from immutable train metadata plus the checked-in
authoritative pose export, never source rasters or shard ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pose_controlnet.overfit_capacity import MIXED_COMPOSITION, OVERFIT_EXPERIMENTS, OVERFIT_MANIFEST_ROOT
from pose_controlnet.pose_targets import source_for_stem


def stable_key(domain: str, stem: str) -> str:
    return hashlib.sha256(f"overfit32-selection-v1:{domain}:{stem}".encode()).hexdigest()


def _choose(candidates: list[dict], domain: str) -> list[str]:
    # Fixed stratum quotas deliberately span one-person and multi-person
    # examples where authoritative targets exist.  Danbooru has no keypoint
    # targets, so its deterministic domain sample is intentionally unstratified.
    if domain == "danbooru":
        return [row["stem"] for row in sorted(candidates, key=lambda row: stable_key(domain, row["stem"]))[:32]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        people = len(row["people"])
        group = "one" if people == 1 else "two" if people == 2 else "three_four" if people in (3, 4) else "five_plus"
        groups[group].append(row)
    quotas = {"one": 16, "two": 8, "three_four": 4, "five_plus": 4}
    picked = []
    for group, quota in quotas.items():
        ranked = sorted(groups[group], key=lambda row: stable_key(domain + ":" + group, row["stem"]))
        if len(ranked) < quota:
            raise ValueError(f"{domain} has too few {group} candidates")
        picked.extend(row["stem"] for row in ranked[:quota])
    return picked


def build(*, train_manifest: Path, targets: Path) -> dict[str, list[dict]]:
    train = {Path(json.loads(line)["file_name"]).stem: json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for line in targets.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["stem"] in train:
            candidates[source_for_stem(row["stem"])].append(row)
    # Danbooru is explicitly unavailable for authoritative keypoint targets;
    # it still belongs in the capacity test, selected from immutable train rows.
    candidates["danbooru"] = [
        {"stem": stem, "people": []} for stem in train
        if source_for_stem(stem) == "danbooru"
    ]
    selected: dict[str, list[str]] = {}
    for name, domain in OVERFIT_EXPERIMENTS.items():
        if domain != "mixed": selected[name] = _choose(candidates[domain], domain)
    mixed = []
    positions = {6: (0, 5, 10, 15, 20, 25), 7: (0, 4, 9, 13, 18, 23, 28)}
    for domain, count in MIXED_COMPOSITION.items():
        name = next(key for key, value in OVERFIT_EXPERIMENTS.items() if value == domain)
        mixed.extend(selected[name][index] for index in positions[count])
    selected["overfit32-mixed-r64-mse"] = mixed
    return {name: [train[stem] for stem in stems] for name, stems in selected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    parser.add_argument("--targets", type=Path, default=Path("data/pose_targets_authoritative_v1.jsonl"))
    parser.add_argument("--out-root", type=Path, default=OVERFIT_MANIFEST_ROOT)
    parser.add_argument("--write", action="store_true", help="write committed experiment manifests")
    args = parser.parse_args(); manifests = build(train_manifest=args.train_manifest, targets=args.targets)
    for name, rows in manifests.items():
        if len(rows) != 32 or len({Path(row["file_name"]).stem for row in rows}) != 32: raise ValueError(f"{name} is not 32 unique rows")
        path = args.out_root / f"{name}.jsonl"
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"name": name, "stems": [Path(row["file_name"]).stem for row in rows]}))


if __name__ == "__main__": main()
