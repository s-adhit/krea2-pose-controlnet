#!/usr/bin/env python3
"""Read-only summary and consistency audit for the frozen release metrics."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/lambda/nfs/adhit/krea2-pose/evaluation")
FINAL = ROOT / "final-val-turbo"
BASELINE = ROOT / "turbo-baseline/turbo-baseline-final-val-v1/pck_clip_results.json"
REPO = Path("docs/evaluation")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(label: str, pose: dict, clip: dict, attempted: int) -> dict:
    return {
        "label": label,
        "attempted_generations": attempted,
        "pck_005": pose["pck_005"],
        "pck_010": pose["pck_010"],
        "pck_020": pose["pck_020"],
        "clip_mean": clip["mean_cosine_similarity"],
        "clip_sample_count": clip["sample_count"],
        "evaluable_samples": pose.get("evaluable_sample_count"),
        "rendered_reference_people": pose.get("rendered_reference_people"),
        "matched_people": pose.get("matched_people"),
        "predicted_people": pose.get("predicted_people"),
        "unmatched_reference_people": pose.get("unmatched_reference_people"),
        "unmatched_predicted_people": pose.get("unmatched_predicted_people"),
        "eligible_joint_count": pose.get("eligible_reference_joint_count"),
        "joint_coverage": pose.get("joint_evaluation_coverage"),
    }


def main() -> None:
    final_rows = []
    for label in ("parent-4000", "mix-025", "mix-050", "mix-075", "finish-control-a4300"):
        row = read(FINAL / label / "pck_clip_results.json")["checkpoints"][0]
        final_rows.append(metric_row(label, row["pose"], row["clip"], 48))

    baseline = read(BASELINE)
    baseline_rows = [
        metric_row(row["candidate"], row["pck"], row["clip"], row["generation_count"])
        for row in baseline["aggregate_by_candidate"]["rows"]
    ]

    native = read(REPO / "native-vs-dynamic768/results/mix-025-control1-turbo-v1/metrics_by_geometry.json")
    native_rows = [metric_row(row["geometry_mode"], row["pck"], row["clip"], row["generation_count"])
                   for row in native["rows"]]

    hard = read(REPO / "hard-pose-multiperson/results/hard-pose-multiperson-mix-025-v1/pck_clip_results.json")
    hard_rows = [metric_row(row["person_group"], row["pck"], row["clip"], row["generation_count"])
                 for row in hard["aggregate_by_person_group"]["rows"]]

    prompt = read(REPO / "prompt-injection-benchmark/results/mix-025/pck_clip_results.json")["checkpoints"][0]
    prompt_row = metric_row("mix-025", prompt["pose"], prompt["clip"], 48)

    # The committed rounded release summary must be a rendering of the raw
    # final-val payloads, not an independent source of numbers.
    rounded = read(REPO / "final-val-turbo/results_summary.json")["results"]
    for row in final_rows:
        expected = rounded[row["label"]]
        for raw_key, rounded_key in (("pck_005", "pck_005"), ("pck_010", "pck_010"),
                                     ("pck_020", "pck_020"), ("clip_mean", "clip_mean")):
            if round(row[raw_key], 4 if raw_key != "clip_mean" else 5) != expected[rounded_key]:
                raise SystemExit(f"rounded summary disagrees for {row['label']}:{raw_key}")

    print(json.dumps({
        "status": "PASS",
        "final_val_candidate_comparison": final_rows,
        "turbo_baseline": baseline_rows,
        "native_vs_dynamic_768": native_rows,
        "hard_pose_by_person_group": hard_rows,
        "prompt_injection_mix_025": prompt_row,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
