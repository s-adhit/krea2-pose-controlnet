"""Metric, summary, plot, and allowlisted-export helpers for the post-500 gate.

This module is evaluation-only: it constructs neither an optimizer nor a training
data loader.  PCK requires authoritative source pose annotations and rendering
geometry; rendered control rasters are never treated as an annotation source.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

CHECKPOINT_STEPS = (0, 20, 40, 60, 80, 100, 200, 225, 350, 475, 500,
                    600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500)
PCK_THRESHOLDS = (0.05, 0.10, 0.20)
GITHUB_EXPORT_NAMES = frozenset({"comparison_grid.png", "500_vs_800_vs_1100_vs_1500.png", "evaluation_summary.json", "fixed_flow_vs_step.png", "clip_similarity_vs_step.png", "pck_vs_step.png", "detection_coverage_vs_step.png", "timestep_distribution.png", "control_sensitivity_vs_timestep.png", "validation_loss_vs_step.png", "throughput_vs_step.png", "train_loss_vs_step.png", "gradient_norm_vs_step.png", "memory_vs_step.png"})
POSE_METRIC_UNAVAILABLE_REASON = "authoritative_reference_pose_unavailable"


def assert_checkpoint_order(steps: Iterable[int]) -> tuple[int, ...]:
    actual = tuple(steps)
    if actual != CHECKPOINT_STEPS:
        raise ValueError(f"checkpoint order must be exactly {CHECKPOINT_STEPS}, got {actual}")
    return actual


def _valid_joints(person: dict, confidence_threshold: float) -> np.ndarray:
    points = np.asarray(person.get("keypoints", []), dtype=float)
    if points.shape != (17, 3):
        return np.zeros((17,), dtype=bool)
    return np.isfinite(points).all(axis=1) & (points[:, 2] >= confidence_threshold)


def _scale(person: dict, confidence_threshold: float) -> float | None:
    points = np.asarray(person.get("keypoints", []), dtype=float); valid = _valid_joints(person, confidence_threshold)
    if valid.sum() < 2: return None
    xy = points[valid, :2]; diagonal = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return diagonal if diagonal > 0 else None


def associate_people(reference: list[dict], predicted: list[dict], confidence_threshold: float) -> list[tuple[int, int]]:
    """Deterministic one-to-one minimum mean-joint-distance association."""
    from scipy.optimize import linear_sum_assignment
    if not reference or not predicted: return []
    cost = np.full((len(reference), len(predicted)), 1e9, dtype=float)
    for i, ref in enumerate(reference):
        r = np.asarray(ref.get("keypoints", []), dtype=float); rv = _valid_joints(ref, confidence_threshold)
        for j, pred in enumerate(predicted):
            p = np.asarray(pred.get("keypoints", []), dtype=float); pv = _valid_joints(pred, confidence_threshold); shared = rv & pv
            if shared.any(): cost[i, j] = float(np.linalg.norm(r[shared, :2] - p[shared, :2], axis=1).mean())
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e9]


def pck_for_people(reference: list[dict], predicted: list[dict], confidence_threshold: float) -> dict[str, Any]:
    """Score only renderer-encoded reference joints.

    Reference callers mark PCK eligibility in the confidence slot.  A missing
    detector joint is consequently a failed eligible joint, rather than being
    silently removed from the PCK denominator.
    """
    pairs = associate_people(reference, predicted, confidence_threshold); total = covered = 0
    thresholds = {f"pck_{int(t * 100):03d}": 0 for t in PCK_THRESHOLDS}; excluded = []
    usable_reference: set[int] = set()
    for ri, ref in enumerate(reference):
        if _scale(ref, confidence_threshold) is None:
            excluded.append({"reference_person": ri, "reason": "insufficient_reference_joints"})
        else:
            usable_reference.add(ri)
            total += int(_valid_joints(ref, confidence_threshold).sum())
    for ri, pi in pairs:
        ref, pred = reference[ri], predicted[pi]; scale = _scale(ref, confidence_threshold)
        if scale is None: continue
        r, p = np.asarray(ref["keypoints"], float), np.asarray(pred["keypoints"], float)
        eligible = _valid_joints(ref, confidence_threshold)
        predicted_valid = _valid_joints(pred, confidence_threshold)
        shared = eligible & predicted_valid
        covered += int(shared.sum())
        distances = np.linalg.norm(r[shared, :2] - p[shared, :2], axis=1)
        for threshold in PCK_THRESHOLDS:
            thresholds[f"pck_{int(threshold * 100):03d}"] += int((distances <= threshold * scale).sum())
    matched_reference = {i for i, _ in pairs}
    for index in usable_reference:
        if index not in matched_reference: excluded.append({"reference_person": index, "reason": "no_matched_prediction"})
    usable_pairs = [(i, j) for i, j in pairs if i in usable_reference]
    detection_coverage = len(usable_pairs) / len(usable_reference) if usable_reference else None
    return {key: (value / total if total else None) for key, value in thresholds.items()} | {
        **{f"{key}_correct_count": value for key, value in thresholds.items()},
        "reference_people": len(reference),
        "predicted_people": len(predicted),
        "matched_people": len(pairs),
        "unmatched_reference_people": len(reference) - len(matched_reference),
        "unmatched_predicted_people": len(predicted) - len({j for _, j in pairs}),
        "pck_eligible_joint_count": total,
        "evaluated_joint_count": total,
        "joint_evaluation_covered_count": covered,
        "joint_evaluation_coverage": (covered / total if total else None),
        "excluded": excluded,
        "generated_person_detection_coverage": detection_coverage,
        "detection_coverage": detection_coverage,
    }


def unavailable_pose_result(reason: str = POSE_METRIC_UNAVAILABLE_REASON) -> dict[str, Any]:
    """Structured result used until authoritative per-stem pose data is supplied."""
    return {
        "pose_metric_status": "unavailable",
        "pose_metric_reason": reason,
        "pck_005": None,
        "pck_010": None,
        "pck_020": None,
        "detection_coverage": None,
        "evaluated_joint_count": 0,
        "excluded_sample_count": 0,
        "per_image": [],
        "generated_image_detector": {"status": "not_run", "reason": reason},
    }


class KeypointRCNNEstimator:
    """COCO-17 estimator from the already-installed torchvision stack."""
    identifier = "torchvision/keypointrcnn_resnet50_fpn:COCO_V1"
    def __init__(self, device: str, confidence_threshold: float = 0.5) -> None:
        import torch
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
        self.torch, self.threshold = torch, confidence_threshold
        self.weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = keypointrcnn_resnet50_fpn(weights=self.weights).to(device).eval(); self.device = device
    def __call__(self, path: Path) -> list[dict]:
        from PIL import Image
        image = self.weights.transforms()(Image.open(path).convert("RGB")).to(self.device)
        with self.torch.inference_mode(): output = self.model([image])[0]
        people = []
        for box, points, score in zip(output["boxes"].cpu().numpy(), output["keypoints"].cpu().numpy(), output["scores"].cpu().numpy()):
            if float(score) >= self.threshold: people.append({"box": box.tolist(), "score": float(score), "keypoints": points.tolist()})
        return people


def cosine_from_embeddings(image: np.ndarray, text: np.ndarray) -> np.ndarray:
    image = np.asarray(image, float); text = np.asarray(text, float)
    return (image * text).sum(-1) / (np.linalg.norm(image, axis=-1) * np.linalg.norm(text, axis=-1))


def clip_feature_tensor(features: torch.Tensor | BaseModelOutputWithPooling) -> torch.Tensor:
    """Return a projected CLIP feature from supported Transformers API returns."""
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, BaseModelOutputWithPooling) and isinstance(features.pooler_output, torch.Tensor):
        return features.pooler_output
    raise TypeError(
        "CLIP get_*_features must return a torch.Tensor or "
        "BaseModelOutputWithPooling with a tensor pooler_output, got "
        f"{type(features).__name__}"
    )


def prepare_clip_scoring_inputs(processor: Any, caption: str, image: Any, context_length: int) -> Any:
    """Tokenize only the scoring copy of an immutable caption at CLIP's limit."""
    return processor(
        text=[caption], images=[image], return_tensors="pt", padding=True,
        truncation=True, max_length=context_length,
    )


def aggregate(values: list[float]) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=float)
    return {"sample_count": int(len(vector)), "mean": float(vector.mean()), "median": float(np.median(vector)), "std": float(vector.std())}


def _best_or_none(rows: list[dict[str, Any]], metric) -> int | None:
    valid = [row for row in rows if metric(row) is not None]
    return max(valid, key=metric)["checkpoint_step"] if valid else None


def choose_best(summary: dict[str, Any]) -> dict[str, int | None]:
    rows = summary["checkpoints"]
    return {"lowest_fixed_flow_mean": min(rows, key=lambda row: row["fixed_flow"]["mean"])["checkpoint_step"],
            "highest_pck_005": _best_or_none(rows, lambda row: row["pose"].get("pck_005")),
            "highest_pck_010": _best_or_none(rows, lambda row: row["pose"].get("pck_010")),
            "highest_pck_020": _best_or_none(rows, lambda row: row["pose"].get("pck_020")),
            "highest_detection_coverage": _best_or_none(rows, lambda row: row["pose"].get("detection_coverage")),
            "highest_clip_mean_cosine_similarity": max(rows, key=lambda row: row["clip"]["mean_cosine_similarity"])["checkpoint_step"],
            "highest_single_person_pck_020": _best_or_none(rows, lambda row: row["pose"].get("single_person", {}).get("pck_020")),
            "highest_multi_person_pck_020": _best_or_none(rows, lambda row: row["pose"].get("multi_person", {}).get("pck_020"))}


def plot_summary(summary_path: str | Path, output: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt
    summary = json.loads(Path(summary_path).read_text()); rows = summary["checkpoints"]; steps = [row["checkpoint_step"] for row in rows]; assert_checkpoint_order(steps)
    output = Path(output); output.mkdir(parents=True, exist_ok=True); made = []
    def draw(name, series, ylabel):
        figure, axis = plt.subplots(figsize=(8, 4.5));
        for label, values in series: axis.plot(steps, values, marker="o", label=label)
        axis.set_xticks(steps); axis.set_xlabel("optimizer step"); axis.set_ylabel(ylabel); axis.grid(True, alpha=.25); axis.legend(); figure.tight_layout(); path = output / name; figure.savefig(path, dpi=160); plt.close(figure); made.append(path)
    draw("fixed_flow_vs_step.png", [("mean fixed-flow MSE", [x["fixed_flow"]["mean"] for x in rows]), ("median", [x["fixed_flow"]["median"] for x in rows])], "fixed-flow MSE (lower is better)")
    draw("clip_similarity_vs_step.png", [("mean cosine similarity", [x["clip"]["mean_cosine_similarity"] for x in rows])], "CLIP cosine similarity (higher is better)")
    if all(row["pose"].get("pck_005") is not None for row in rows):
        draw("pck_vs_step.png", [("PCK@.05", [x["pose"]["pck_005"] for x in rows]), ("PCK@.10", [x["pose"]["pck_010"] for x in rows]), ("PCK@.20", [x["pose"]["pck_020"] for x in rows])], "pooled PCK (higher is better)")
        draw("detection_coverage_vs_step.png", [("person detection", [x["pose"]["detection_coverage"] for x in rows]), ("joint evaluation", [x["pose"].get("joint_evaluation_coverage") for x in rows])], "coverage (higher is better)")
    return made


def export_allowlisted(source: str | Path, destination: str | Path) -> list[Path]:
    source, destination = Path(source), Path(destination); destination.mkdir(parents=True, exist_ok=True); copied = []
    locations = {"comparison_grid.png": source / "fixed_pose" / "comparison_grid.png"} | {name: source / name for name in GITHUB_EXPORT_NAMES - {"comparison_grid.png"}}
    for name, path in locations.items():
        if path.exists(): shutil.copy2(path, destination / name); copied.append(destination / name)
    unexpected = [path for path in destination.rglob("*.png") if path.name not in GITHUB_EXPORT_NAMES]
    if unexpected: raise ValueError(f"refusing non-allowlisted generated images: {unexpected}")
    return copied
