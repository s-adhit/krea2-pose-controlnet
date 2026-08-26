"""Metric, summary, plot, and allowlisted-export helpers for the post-500 gate.

This module is evaluation-only: it constructs neither an optimizer nor a training
data loader.  PCK uses the COCO-17 Keypoint R-CNN estimator on both the rendered
control and generated image.  A control for which the estimator cannot produce a
confident person is explicitly excluded; it is never converted to zero joints.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CHECKPOINT_STEPS = (0, 20, 40, 60, 80, 100, 200, 300, 400, 500)
PCK_THRESHOLDS = (0.05, 0.10, 0.20)
GITHUB_EXPORT_NAMES = frozenset({"comparison_grid.png", "evaluation_summary.json", "fixed_flow_vs_step.png", "pck_vs_step.png", "clip_similarity_vs_step.png", "detection_coverage_vs_step.png", "evaluation_metrics.png"})


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
    pairs = associate_people(reference, predicted, confidence_threshold); total = matches = 0
    thresholds = {f"pck_{int(t * 100):03d}": 0 for t in PCK_THRESHOLDS}; excluded = []
    for ri, pi in pairs:
        ref, pred = reference[ri], predicted[pi]; scale = _scale(ref, confidence_threshold)
        if scale is None:
            excluded.append({"reference_person": ri, "reason": "insufficient_reference_joints"}); continue
        r, p = np.asarray(ref["keypoints"], float), np.asarray(pred["keypoints"], float)
        shared = _valid_joints(ref, confidence_threshold) & _valid_joints(pred, confidence_threshold)
        distances = np.linalg.norm(r[shared, :2] - p[shared, :2], axis=1); total += int(shared.sum())
        for threshold in PCK_THRESHOLDS: thresholds[f"pck_{int(threshold * 100):03d}"] += int((distances <= threshold * scale).sum())
    for index in range(len(reference)):
        if index not in {i for i, _ in pairs}: excluded.append({"reference_person": index, "reason": "no_matched_prediction"})
    return {key: (value / total if total else None) for key, value in thresholds.items()} | {"evaluated_joint_count": total, "matched_people": len(pairs), "excluded": excluded,
        "detection_coverage": (len(pairs) / len(reference) if reference else 0.0)}


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


def aggregate(values: list[float]) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=float)
    return {"sample_count": int(len(vector)), "mean": float(vector.mean()), "median": float(np.median(vector)), "std": float(vector.std())}


def choose_best(summary: dict[str, Any]) -> dict[str, int]:
    rows = summary["checkpoints"]
    return {"lowest_fixed_flow_mean": min(rows, key=lambda row: row["fixed_flow"]["mean"])["checkpoint_step"],
            "highest_pck_005": max(rows, key=lambda row: row["pose"]["pck_005"] if row["pose"]["pck_005"] is not None else -math.inf)["checkpoint_step"],
            "highest_pck_010": max(rows, key=lambda row: row["pose"]["pck_010"] if row["pose"]["pck_010"] is not None else -math.inf)["checkpoint_step"],
            "highest_pck_020": max(rows, key=lambda row: row["pose"]["pck_020"] if row["pose"]["pck_020"] is not None else -math.inf)["checkpoint_step"],
            "highest_detection_coverage": max(rows, key=lambda row: row["pose"]["detection_coverage"])["checkpoint_step"],
            "highest_clip_mean_cosine_similarity": max(rows, key=lambda row: row["clip"]["mean_cosine_similarity"])["checkpoint_step"]}


def plot_summary(summary_path: str | Path, output: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt
    summary = json.loads(Path(summary_path).read_text()); rows = summary["checkpoints"]; steps = [row["checkpoint_step"] for row in rows]; assert_checkpoint_order(steps)
    output = Path(output); output.mkdir(parents=True, exist_ok=True); made = []
    def draw(name, series, ylabel):
        figure, axis = plt.subplots(figsize=(8, 4.5));
        for label, values in series: axis.plot(steps, values, marker="o", label=label)
        axis.set_xticks(steps); axis.set_xlabel("optimizer step"); axis.set_ylabel(ylabel); axis.grid(True, alpha=.25); axis.legend(); figure.tight_layout(); path = output / name; figure.savefig(path, dpi=160); plt.close(figure); made.append(path)
    draw("fixed_flow_vs_step.png", [("mean fixed-flow MSE", [x["fixed_flow"]["mean"] for x in rows]), ("median", [x["fixed_flow"]["median"] for x in rows])], "fixed-flow MSE (lower is better)")
    draw("pck_vs_step.png", [("PCK@0.05", [x["pose"]["pck_005"] for x in rows]), ("PCK@0.10", [x["pose"]["pck_010"] for x in rows]), ("PCK@0.20", [x["pose"]["pck_020"] for x in rows])], "PCK (higher is better)")
    draw("clip_similarity_vs_step.png", [("mean cosine similarity", [x["clip"]["mean_cosine_similarity"] for x in rows])], "CLIP cosine similarity (higher is better)")
    draw("detection_coverage_vs_step.png", [("detection coverage", [x["pose"]["detection_coverage"] for x in rows])], "pose detection coverage (higher is better)")
    return made


def export_allowlisted(source: str | Path, destination: str | Path) -> list[Path]:
    source, destination = Path(source), Path(destination); destination.mkdir(parents=True, exist_ok=True); copied = []
    locations = {"comparison_grid.png": source / "fixed_pose" / "comparison_grid.png"} | {name: source / name for name in GITHUB_EXPORT_NAMES - {"comparison_grid.png"}}
    for name, path in locations.items():
        if path.exists(): shutil.copy2(path, destination / name); copied.append(destination / name)
    unexpected = [path for path in destination.rglob("*.png") if path.name not in GITHUB_EXPORT_NAMES]
    if unexpected: raise ValueError(f"refusing non-allowlisted generated images: {unexpected}")
    return copied
