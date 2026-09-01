"""Emit a compact, axis-preserving table from production benchmark JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_controlnet.throughput_benchmark import validate_benchmark_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        result = json.loads(path.read_text(encoding="utf-8")); validate_benchmark_result(result)
        recipe = result["recipe"]
        rows.append({
            "label": result.get("label", path.stem), "objective": recipe["objective"],
            "microbatch": recipe["microbatch_size"], "accumulation": recipe["gradient_accumulation_steps"],
            "checkpoint_blocks": recipe["gradient_checkpointing_blocks"], "fused_adamw": recipe["fused_adamw"],
            "compile": recipe["compile"], "workers": recipe["data_loader_workers"],
            "step_s": round(float(result["optimizer_step_seconds_mean"]), 4),
            "effective_samples_s": round(float(result["effective_samples_per_second"]), 4),
            "forward_s": round(float(result["forward_seconds_mean"]), 4),
            "backward_s": round(float(result["backward_seconds_mean"]), 4),
            "optimizer_s": round(float(result["optimizer_seconds_mean"]), 4),
            "data_wait_s": round(float(result["data_wait_seconds_mean"]), 4),
            "peak_gib": round(float(result["cuda_peak_allocated_bytes"]) / 2**30, 2),
            "pose_active_fraction": round(float(result["pose_active_fraction"]), 4),
        })
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
