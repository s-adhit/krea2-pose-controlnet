"""Read-only post-1500 evaluation and audit primitives.

Nothing in this module constructs an optimizer, calls ``backward``, or changes
training/checkpoint/data state.  The command layer may write evaluation outputs
only after it has read immutable inputs and validated checkpoint identities.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position, sample_flow_timestep
from pose_controlnet.evaluation import CHECKPOINT_STEPS, _sample_by_stem
from pose_controlnet.post500_evaluation import PCK_THRESHOLDS, aggregate, pck_for_people
from pose_controlnet.reference_pose import reference_people_from_sidecar


POST_500_STEPS = CHECKPOINT_STEPS[11:]
TIMESTEP_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def assert_canonical_steps(steps: Iterable[int]) -> tuple[int, ...]:
    actual = tuple(steps)
    if actual != CHECKPOINT_STEPS:
        raise ValueError(f"canonical checkpoint order must be exactly {CHECKPOINT_STEPS}, got {actual}")
    return actual


def merge_checkpoint_results(existing: Mapping[str, Any], extension: Mapping[str, Any]) -> dict[str, Any]:
    """Merge fixed-flow results while refusing changed historical identities."""
    if existing.get("kind") != "fixed_flow" or extension.get("kind") != "fixed_flow":
        raise ValueError("Only fixed-flow result payloads may be merged")
    for key in ("spec",):
        if existing.get(key) != extension.get(key):
            raise ValueError(f"Fixed-flow extension changed immutable {key}")
    rows: dict[int, dict[str, Any]] = {}
    for payload in (existing, extension):
        for row in payload.get("checkpoints", []):
            step = row.get("checkpoint_step")
            if step not in CHECKPOINT_STEPS:
                raise ValueError(f"Unexpected checkpoint step {step}")
            if step in rows and rows[step] != row:
                raise ValueError(f"Conflicting fixed-flow results for step {step}")
            rows[step] = dict(row)
    missing = [step for step in CHECKPOINT_STEPS if step not in rows]
    if missing:
        raise ValueError(f"Fixed-flow series is incomplete; missing={missing}")
    return {**dict(existing), "checkpoints": [rows[step] for step in CHECKPOINT_STEPS]}


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "sample_count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
        "p01": float(np.quantile(values, .01)), "p05": float(np.quantile(values, .05)),
        "p10": float(np.quantile(values, .10)), "p25": float(np.quantile(values, .25)),
        "median": float(np.median(values)), "p75": float(np.quantile(values, .75)),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)),
        "fractions": {"0.0-0.2": float(((values >= 0) & (values < .2)).mean()),
                      "0.2-0.4": float(((values >= .2) & (values < .4)).mean()),
                      "0.4-0.6": float(((values >= .4) & (values < .6)).mean()),
                      "0.6-0.8": float(((values >= .6) & (values < .8)).mean()),
                      "0.8-1.0": float(((values >= .8) & (values <= 1)).mean())},
    }


def _weighted_distribution(values: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    """Exact bucket-weighted summary without materializing repeated samples."""
    if len(values) != len(weights) or not len(values) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid weighted timestep distribution")
    weights = weights.astype(float); total_weight = float(weights.sum())
    order = np.argsort(values); ordered_values, ordered_weights = values[order], weights[order]
    cumulative = np.cumsum(ordered_weights) / total_weight
    def quantile(q: float) -> float:
        return float(ordered_values[min(np.searchsorted(cumulative, q, side="left"), len(ordered_values) - 1)])
    mean = float(np.dot(values, weights) / total_weight)
    return {"sample_count": int(len(values)), "effective_bucket_weight": total_weight, "mean": mean,
            "std": float(np.sqrt(np.dot((values - mean) ** 2, weights) / total_weight)),
            "p01": quantile(.01), "p05": quantile(.05), "p10": quantile(.10), "p25": quantile(.25),
            "median": quantile(.5), "p75": quantile(.75), "p90": quantile(.9), "p95": quantile(.95), "p99": quantile(.99),
            "fractions": {"0.0-0.2": float(weights[(values >= 0) & (values < .2)].sum() / total_weight),
                          "0.2-0.4": float(weights[(values >= .2) & (values < .4)].sum() / total_weight),
                          "0.4-0.6": float(weights[(values >= .4) & (values < .6)].sum() / total_weight),
                          "0.6-0.8": float(weights[(values >= .6) & (values < .8)].sum() / total_weight),
                          "0.8-1.0": float(weights[(values >= .8) & (values <= 1)].sum() / total_weight)}}


def latent_tokens_for_bucket(bucket: tuple[int, int], *, patch: int = 2) -> int:
    width, height = map(int, bucket)
    if width <= 0 or height <= 0 or width % (8 * patch) or height % (8 * patch):
        raise ValueError(f"Invalid image bucket for VAE/patch tokenization: {bucket}")
    return (width // 8 // patch) * (height // 8 // patch)


def timestep_distribution_audit(bucket_counts: Mapping[tuple[int, int], int], cfg: Any, *, seed: int = 420_300,
                                samples_per_bucket: int = 100_000, patch: int = 2) -> dict[str, Any]:
    """Measure the actual logistic-normal-plus-shift training sampler on CPU."""
    if samples_per_bucket < 1 or not bucket_counts or any(count < 1 for count in bucket_counts.values()):
        raise ValueError("Timestep audit requires positive bucket weights and sample count")
    per_bucket, weighted = [], []
    for index, (bucket, weight) in enumerate(sorted(bucket_counts.items())):
        token_count = latent_tokens_for_bucket(bucket, patch=patch)
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        shifted = sample_flow_timestep(samples_per_bucket, token_count, cfg, "cpu", generator).numpy()
        mu = (cfg.mu_y2 - cfg.mu_y1) / (cfg.mu_x2 - cfg.mu_x1) * token_count + (cfg.mu_y1 - (cfg.mu_y2 - cfg.mu_y1) / (cfg.mu_x2 - cfg.mu_x1) * cfg.mu_x1)
        per_bucket.append({"bucket": list(bucket), "bucket_weight": int(weight), "latent_token_count": token_count,
                           "mu": float(mu), "distribution": _distribution(shifted)})
        # Replication represents the observed training mix without altering any
        # individual bucket's deterministic draw.
        weighted.append((shifted, int(weight)))
    all_values = np.concatenate([values for values, _ in weighted])
    all_weights = np.concatenate([np.full(len(values), weight, dtype=float) for values, weight in weighted])
    return {"format_version": 1, "seed": seed, "samples_per_bucket": samples_per_bucket,
            "base_distribution": "sigmoid(normal(0,1)); verified from sample_flow_timestep",
            "per_bucket": per_bucket, "overall_bucket_weighted": _weighted_distribution(all_values, all_weights)}


def source_family(stem: str) -> str:
    if stem.startswith("coco_"):
        return "COCO"
    if stem.startswith("danbooru_"):
        return "Danbooru"
    if "humanart_" in stem:
        return "Human-Art"
    return "unknown"


def source_balance_audit(records: Iterable[Mapping[str, Any]], diagnostic_stems: Iterable[str]) -> dict[str, Any]:
    records = list(records)
    sources = Counter(source_family(str(row["stem"])) for row in records)
    buckets = Counter("x".join(map(str, row["bucket"])) for row in records if row.get("bucket") is not None)
    diagnostic_sources = Counter(source_family(stem) for stem in diagnostic_stems)
    return {"training_sample_count": len(records), "source_family": dict(sorted(sources.items())),
            "aspect_bucket": dict(sorted(buckets.items())), "diagnostic_source_family": dict(sorted(diagnostic_sources.items())),
            "training_person_count_metadata": "unavailable in immutable manifests/shards; not fabricated"}


def _pool_pose(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sum(int(item["pck_eligible_joint_count"]) for item in metrics)
    rendered = sum(int(item.get("rendered_reference_people", item["reference_people"])) for item in metrics)
    matched = sum(int(item["matched_people"]) for item in metrics)
    result = {f"pck_{int(threshold * 100):03d}": (sum(int(item.get(f"pck_{int(threshold * 100):03d}_correct_count", 0)) for item in metrics) / eligible if eligible else None) for threshold in PCK_THRESHOLDS}
    result.update({"evaluable_sample_count": len(metrics), "rendered_reference_people": rendered,
                   "predicted_people": sum(int(item["predicted_people"]) for item in metrics), "matched_people": matched,
                   "unmatched_reference_people": sum(int(item["unmatched_reference_people"]) for item in metrics),
                   "unmatched_predicted_people": sum(int(item["unmatched_predicted_people"]) for item in metrics),
                   "source_visible_joint_count": sum(int(item.get("source_visible_joint_count", 0)) for item in metrics),
                   "rendered_joint_count": sum(int(item.get("rendered_joint_count", 0)) for item in metrics),
                   "eligible_reference_joint_count": eligible, "evaluated_joint_count": eligible,
                   "generated_person_detection_coverage": (matched / rendered if rendered else None),
                   "detection_coverage": (matched / rendered if rendered else None),
                   "joint_evaluation_coverage": (sum(int(item["joint_evaluation_covered_count"]) for item in metrics) / eligible if eligible else None)})
    return result


def score_authoritative_pck(*, sidecar: Mapping[str, Any], geometry_by_stem: Mapping[str, Mapping[str, Any]],
                            image_for: Callable[[str], Path], detector: Callable[[Path], list[dict]],
                            confidence_threshold: float = .5, require_images: bool = True) -> dict[str, Any]:
    """Run reference-only PCK and pool joint counts, never averaging images."""
    per_image, unavailable = [], []
    for record in sidecar.get("records", []):
        stem, source = str(record["stem"]), str(record.get("source", "unknown"))
        if record.get("status") != "available":
            unavailable.append({"stem": stem, "source": source, "reason": record.get("reason", "authoritative_reference_pose_unavailable")}); continue
        geometry = geometry_by_stem.get(stem)
        if geometry is None:
            raise ValueError(f"No persisted paired geometry for authoritative PCK stem {stem}")
        people = reference_people_from_sidecar(record, source_size=tuple(geometry["source_size"]), resized_size=tuple(geometry["resized_size"]), crop_box=tuple(geometry["crop_box"]))
        rendered = [person for person in people if person["reference_rendered"]]
        if not rendered:
            unavailable.append({"stem": stem, "source": source, "reason": "no_renderer_qualified_reference_people"}); continue
        image = image_for(stem)
        if not image.is_file():
            if require_images:
                raise FileNotFoundError(f"Missing required generated fixed-pose image: {image}")
            unavailable.append({"stem": stem, "source": source, "reason": "generated_image_missing"}); continue
        metric = pck_for_people([{"keypoints": person["keypoints"]} for person in rendered], detector(image), confidence_threshold)
        metric.update({"stem": stem, "source": source, "reference_available": True,
                       "rendered_reference_people": len(rendered),
                       "source_visible_joint_count": sum(state["source_visible"] for person in rendered for state in person["joint_states"]),
                       "rendered_joint_count": sum(state["rendered_in_control"] for person in rendered for state in person["joint_states"]),
                       "joint_evaluation_covered_count": int(metric["joint_evaluation_covered_count"]),
                       "person_group": "single-person" if len(rendered) == 1 else "multi-person"})
        per_image.append(metric)
    aggregate_result = _pool_pose(per_image)
    by_source = {label: _pool_pose([row for row in per_image if row["source"] == source]) for label, source in (("Human-Art", "humanart"), ("COCO", "coco"))}
    by_source["Danbooru unavailable"] = {"sample_count": sum(row["source"] == "danbooru" for row in unavailable), "pck_005": None, "pck_010": None, "pck_020": None}
    aggregate_result.update({"reference_available_sample_count": len(per_image), "unavailable_excluded_sample_count": len(unavailable),
                             "per_source": by_source,
                             "single_person": _pool_pose([row for row in per_image if row["person_group"] == "single-person"]),
                             "multi_person": _pool_pose([row for row in per_image if row["person_group"] == "multi-person"]),
                             "per_image": per_image, "unavailable": unavailable})
    return aggregate_result


def _stable_seed(seed: int, stem: str, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{stem}:{label}".encode()).digest()[:8], "big") % (2**63 - 1)


@torch.inference_mode()
def fixed_timestep_loss_and_sensitivity(model: Any, dataset: Any, stems: Iterable[str], cfg: Any, device: torch.device,
                                        *, timesteps: Iterable[float] = TIMESTEP_GRID, seed: int = 420_100) -> dict[str, Any]:
    """Forward-only real-control versus zero-control audit with fixed noise."""
    stems = tuple(stems)
    previous_training = model.training; model.eval(); rows = []
    try:
        for value in timesteps:
            if not 0 < value < 1:
                raise ValueError(f"Timestep must be in (0,1), got {value}")
            losses, sensitivities, normalized = [], [], []
            for stem in stems:
                sample = _sample_by_stem(dataset, stem); clean = sample["latent"][None].to(device=device, dtype=torch.float32)
                noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(_stable_seed(seed, stem, "loss-control-noise")), dtype=torch.float32).to(device)
                timestep = torch.full((1,), float(value), dtype=torch.float32, device=device); noisy, target = make_flow_pair(clean, noise, timestep)
                context, text_mask = sample["context"][None].to(device, torch.bfloat16), sample["mask"][None].to(device, torch.bool)
                image, pos, mask = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
                control, _, _ = patchify_and_position(sample["control"][None].to(device, torch.bfloat16), context.shape[1], model.config.patch, text_mask)
                target_tokens, _, _ = patchify_and_position(target, context.shape[1], model.config.patch, text_mask)
                real = forward_pose_control(model, image, control, context, timestep.to(torch.bfloat16), pos, mask, gradient_checkpointing_blocks=0)
                zero = forward_pose_control(model, image, torch.zeros_like(control), context, timestep.to(torch.bfloat16), pos, mask, gradient_checkpointing_blocks=0)
                delta = real.float() - zero.float(); rms = float(delta.square().mean().sqrt())
                losses.append(float(F.mse_loss(real.float(), target_tokens.float()))); sensitivities.append(rms)
                normalized.append(rms / max(float(real.float().square().mean().sqrt()), 1e-12))
            rows.append({"timestep": float(value), "flow_mse": aggregate(losses), "control_sensitivity_rms": aggregate(sensitivities), "control_sensitivity_normalized_rms": aggregate(normalized)})
    finally:
        model.train(previous_training)
    return {"seed": seed, "timesteps": rows, "sample_count": len(stems)}


def telemetry_audit(metrics_path: str | Path, *, start_step: int = 500, end_step: int = 1500) -> dict[str, Any]:
    """Parse durable JSONL telemetry without assuming every event has every field."""
    wanted = ("train/loss", "validation/flow_loss", "train/global_grad_norm", "train/learning_rate", "cuda/peak_allocated_bytes", "cuda/reserved_bytes", "cuda/allocated_bytes", "performance/samples_per_second", "performance/sec_per_step")
    series: dict[str, list[dict[str, float]]] = defaultdict(list); control: dict[str, list[dict[str, float]]] = defaultdict(list); lora: dict[str, list[dict[str, float]]] = defaultdict(list)
    with Path(metrics_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try: row = json.loads(line)
            except json.JSONDecodeError as error: raise ValueError(f"Invalid metrics JSONL at line {line_number}") from error
            step = row.get("global_step")
            if not isinstance(step, int) or not start_step <= step <= end_step: continue
            for key in wanted:
                if isinstance(row.get(key), (int, float)) and math.isfinite(row[key]): series[key].append({"step": step, "value": float(row[key])})
            for key, value in row.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    if key.startswith("diagnostics/control_input_grad_norm/"): control[key.rsplit("/", 1)[-1]].append({"step": step, "value": float(value)})
                    if key.startswith("diagnostics/lora_grad_norm/"): lora[key.rsplit("/", 1)[-1]].append({"step": step, "value": float(value)})
    def summarize(points: list[dict[str, float]]) -> dict[str, Any]:
        values = np.asarray([point["value"] for point in points], dtype=float)
        result = _distribution(values); result["first"] = float(values[0]); result["last"] = float(values[-1])
        result["linear_slope_per_step"] = float(np.polyfit([point["step"] for point in points], values, 1)[0]) if len(points) > 1 else None
        return result
    report = {"window": [start_step, end_step], "metrics": {key: summarize(value) for key, value in series.items() if value},
              "control_input_grad_norms": {key: summarize(value) for key, value in control.items() if value},
              "lora_grad_norms": {key: summarize(value) for key, value in lora.items() if value},
              "raw_series": dict(series)}
    grad = [point["value"] for point in series.get("train/global_grad_norm", [])]
    report["max_grad_norm_1_observability"] = {"logged_preclip_norm": bool(grad), "above_1_count": int(sum(value > 1.0 for value in grad)),
                                                "interpretation": "clip activation is evidenced only when logged preclip norm exceeds 1.0" if grad else "not logged"}
    return report
