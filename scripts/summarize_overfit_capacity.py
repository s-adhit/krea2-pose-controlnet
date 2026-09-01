"""Aggregate completed TRAINING-SET OVERFIT summaries without evaluation work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_controlnet.overfit_capacity import NATIVE_RESOLUTION_POLICY, OVERFIT_EXPERIMENTS, canonical_resolution_policy


def _training_resolution(metadata: dict, payload: dict) -> str:
    value = metadata.get("training_resolution") or metadata.get("scientific_config", {}).get("resolution") or metadata.get("resolution_policy")
    observed = payload.get("training_resolution")
    resolution = canonical_resolution_policy(value)
    if observed != resolution:
        raise ValueError("Evaluation summary training-resolution provenance disagrees with checkpoint metadata")
    return resolution


def _comparison_label(training_resolution: str, metadata: dict) -> str:
    pose_loss = metadata.get("scientific_config", {}).get("pose_loss", "none")
    if training_resolution == "768" and pose_loss != "none":
        return "768+pose train / Native eval"
    return f"{training_resolution.capitalize()} train / Native eval"


def compact_checkpoint_comparison(summaries: dict[str, dict], baseline: str, candidate: str) -> dict:
    """Pair existing native-evaluation rows only; never choose a winner."""
    if baseline not in summaries or candidate not in summaries:
        raise ValueError("compact comparison requires two completed native-evaluation summaries")
    baseline_rows = {row.get("checkpoint_step"): row for row in summaries[baseline].get("quantitative_metrics", [])}
    candidate_rows = {row.get("checkpoint_step"): row for row in summaries[candidate].get("quantitative_metrics", [])}
    steps = (0, 50, 100, 200, 300, 400, 500)
    if set(baseline_rows) != set(steps) or set(candidate_rows) != set(steps):
        raise ValueError("compact comparison requires the exact authoritative 0/50/100/200/300/400/500 schedule")
    return {"baseline": baseline, "candidate": candidate, "evaluation_resolution": "native",
            "checkpoint_steps": list(steps), "winner_declared": False,
            "by_checkpoint": [{"checkpoint_step": step, "baseline": baseline_rows[step], "candidate": candidate_rows[step]}
                              for step in steps]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", required=True); parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), help="Emit a paired native-only checkpoint table without selecting a winner.")
    parser.add_argument("experiments", nargs="*")
    args = parser.parse_args(); names = args.experiments or (list(args.compare) if args.compare else list(OVERFIT_EXPERIMENTS)); root = Path(args.output_root)
    summaries, excluded = {}, {}
    for name in names:
        path = root / name / "overfit_summary.json"
        if not path.is_file(): raise FileNotFoundError(f"Completed training-set overfit summary is missing: {path}")
        payload = json.loads(path.read_text())
        checkpoint = Path(args.checkpoint_root) / name if args.checkpoint_root else None
        metadata = json.loads((checkpoint / "experiment_metadata.json").read_text()) if checkpoint and (checkpoint / "experiment_metadata.json").is_file() else {}
        if payload.get("evaluation_resolution") != NATIVE_RESOLUTION_POLICY:
            excluded[name] = "missing or non-native evaluation provenance"
            continue
        try:
            training_resolution = _training_resolution(metadata, payload)
        except (TypeError, ValueError) as exc:
            excluded[name] = str(exc)
            continue
        metrics_path = checkpoint / "metrics.jsonl" if checkpoint else None
        trajectory = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path and metrics_path.is_file() else []
        summaries[name] = {"config_provenance": metadata.get("scientific_config", {}), "training_resolution": training_resolution,
                           "evaluation_resolution": NATIVE_RESOLUTION_POLICY, "comparison_label": _comparison_label(training_resolution, metadata),
                           "trainable_parameter_count": metadata.get("parameter_audit", {}).get("trainable_parameter_count"),
                           "training_loss_trajectory": trajectory, "runtime": trajectory[-1].get("sec_per_step") if trajectory else None,
                           "qualitative_artifacts": payload.get("qualitative_grids", {}), "quantitative_metrics": payload.get("checkpoints"), "summary": payload}
    mixed = summaries.get("overfit32-mixed-r64-mse", {})
    mixed_rows = {row["stem"]: row for checkpoint in mixed.get("checkpoints", []) for row in checkpoint.get("clip", {}).get("per_sample", []) if checkpoint.get("checkpoint_step") == 500}
    same_sample = {}
    for name, payload in summaries.items():
        if name == "overfit32-mixed-r64-mse": continue
        rows = {row["stem"]: row for checkpoint in payload.get("checkpoints", []) for row in checkpoint.get("clip", {}).get("per_sample", []) if checkpoint.get("checkpoint_step") == 500}
        shared = sorted(set(rows) & set(mixed_rows)); same_sample[name] = {"checkpoint_step": 500, "shared_stems": shared, "homogeneous_clip": {stem: rows[stem] for stem in shared}, "mixed_clip": {stem: mixed_rows[stem] for stem in shared}}
    compact = compact_checkpoint_comparison(summaries, *args.compare) if args.compare else None
    destination = root / "capacity_comparison_summary.json"; destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"training_set_overfit_only": True, "evaluation_resolution": NATIVE_RESOLUTION_POLICY,
                                       "experiments": summaries, "excluded_experiments": excluded,
                                       "same_sample_step500": same_sample, "compact_checkpoint_comparison": compact}, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__": main()
