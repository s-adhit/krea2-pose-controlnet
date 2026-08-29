"""Run the audit-only fixed-box Keypoint R-CNN critic on authoritative RGBs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
from pose_controlnet.keypoint_critic import (
    FixedBoxKeypointRCNNCritic,
    detached_pose_diagnostics,
    gaussian_heatmap_kl,
    gaussian_heatmap_target,
    masked_coordinate_huber,
    normalized_coordinate_huber,
    soft_coordinates,
)
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.pose_targets import load_sidecar


AUDIT_SOURCES = ("coco", "humanart_painting", "humanart_real_human", "humanart_sculpture")
GRADIENT_METRIC_NAMES = {
    "coordinate_huber_pixels": "rgb_grad_norm_coordinate_pixels",
    "coordinate_huber_normalized": "rgb_grad_norm_coordinate_normalized",
    "gaussian_heatmap_kl": "rgb_grad_norm_heatmap_kl",
}


def _rgb_tensor(image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(255.0)


def _person_tensors(record: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes, targets, valid = [], [], []
    for person in record["people"]:
        xywh = person.get("bbox_training_xywh")
        joints = person.get("joint_provenance")
        if not isinstance(xywh, list) or len(xywh) != 4 or not isinstance(joints, list) or len(joints) != 17:
            raise ValueError(f"{record['stem']}: incomplete authoritative person geometry")
        x, y, width, height = map(float, xywh)
        if width <= 0 or height <= 0:
            continue
        boxes.append((x, y, x + width, y + height))
        targets.append([joint["training_coordinate"] for joint in joints])
        valid.append([bool(joint["reward_joint_valid"]) for joint in joints])
    if not boxes:
        raise ValueError(f"{record['stem']}: no positive-area authoritative fixed boxes")
    return (
        torch.tensor(boxes, device=device, dtype=torch.float32),
        torch.tensor(targets, device=device, dtype=torch.float32),
        torch.tensor(valid, device=device, dtype=torch.bool),
    )


def _preprocessed_rgb(record: dict, index: DatasetIndex):
    stem = record["stem"]
    manifest = ManifestRecord(
        split="audit", stem=stem, file_name=f"{stem}.jpg", text="audit",
        rgb_path=index.rgb_by_stem[stem], control_path=index.control_by_stem[stem],
    )
    pair = preprocess_pair(manifest)
    actual = (list(pair.geometry.source_size), list(pair.geometry.resized_size), list(pair.geometry.crop_box), list(pair.geometry.bucket))
    expected = (record["source_size"], record["resized_size"], record["crop_box"], record["bucket"])
    if actual != expected:
        raise ValueError(f"{stem}: final paired preprocessing differs from authoritative sidecar geometry")
    return pair.rgb


def _usable(record: dict) -> bool:
    if record.get("pose_reward_available") is not True or not record.get("people"):
        return False
    return any(
        isinstance(person.get("bbox_training_xywh"), list)
        and person["bbox_training_xywh"][2] > 0 and person["bbox_training_xywh"][3] > 0
        and any(joint.get("reward_joint_valid") is True for joint in person.get("joint_provenance", []))
        for person in record["people"]
    )


def _weighted_add(total: dict[str, float], metrics: dict[str, float | int | None], weight: int) -> None:
    for key, value in metrics.items():
        if key != "joint_count" and value is not None:
            total[key] += float(value) * weight


def _candidate_loss(
    name: str, logits: torch.Tensor, boxes: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
    *, temperature: float, gaussian_sigma: float,
) -> torch.Tensor:
    """Return one valid-person/joint-mean candidate loss for the audit."""
    if name == "coordinate_huber_pixels":
        return masked_coordinate_huber(soft_coordinates(logits, boxes, temperature), targets, valid)
    if name == "coordinate_huber_normalized":
        return normalized_coordinate_huber(soft_coordinates(logits, boxes, temperature), targets, boxes, valid)
    if name == "gaussian_heatmap_kl":
        return gaussian_heatmap_kl(
            logits, targets, boxes, valid, sigma=gaussian_sigma, temperature=temperature,
        )
    raise ValueError(f"Unknown candidate loss: {name}")


def _loss_and_rgb_gradient(
    critic: FixedBoxKeypointRCNNCritic, rgb: torch.Tensor, boxes: torch.Tensor,
    targets: torch.Tensor, valid: torch.Tensor, name: str, *, temperature: float,
    gaussian_sigma: float,
) -> tuple[float, float]:
    """Run one isolated forward/autograd graph for one candidate loss.

    Every candidate rebuilds the critic graph from the same RGB values.  This
    prevents accidental graph reuse or accumulated RGB gradients from making a
    candidate's input-gradient norm depend on audit ordering.
    """
    rgb_leaf = rgb.detach().clone().requires_grad_(True)
    critic.zero_grad(set_to_none=True)
    heatmaps = critic(rgb_leaf, [boxes])
    loss = _candidate_loss(
        name, heatmaps.logits, heatmaps.boxes_training, targets, valid,
        temperature=temperature, gaussian_sigma=gaussian_sigma,
    )
    gradient, = torch.autograd.grad(loss, rgb_leaf)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{name}: non-finite loss or RGB gradient")
    gradient_norm = float(gradient.norm().item())
    if gradient_norm <= 0:
        raise RuntimeError(f"{name}: RGB gradient norm is zero")
    if any(parameter.grad is not None for parameter in critic.parameters()):
        raise RuntimeError("Frozen critic parameter unexpectedly received a gradient")
    return float(loss.item()), gradient_norm


def _gradient_statistics(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("gradient statistics require at least one value")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std()), "min": float(array.min()), "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True, help="Immutable authoritative v3 sidecar directory.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Read-only PoseBridge HF snapshot root.")
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--temperature-sweep", action="store_true",
                        help="Also report detached soft PCK/error at temperatures 0.5, 1.0, and 2.0.")
    args = parser.parse_args()
    if args.samples_per_source < 1 or args.temperature <= 0 or args.gaussian_sigma <= 0:
        parser.error("samples-per-source, temperature, and gaussian-sigma must be positive")
    device = torch.device(args.device)
    _, records = load_sidecar(args.sidecar)
    index = DatasetIndex.discover(args.dataset_root)
    selected = {
        source: [record for record in sorted(records, key=lambda item: item["stem"])
                 if record.get("source") == source and _usable(record)][:args.samples_per_source]
        for source in AUDIT_SOURCES
    }
    insufficient = {source: len(rows) for source, rows in selected.items() if len(rows) != args.samples_per_source}
    if insufficient:
        raise SystemExit(f"Insufficient usable deterministic records by source: {insufficient}")

    critic = FixedBoxKeypointRCNNCritic().to(device).eval()
    report = {"critic": critic.identifier, "temperature": args.temperature, "gaussian_sigma": args.gaussian_sigma,
              "samples_per_source": args.samples_per_source, "sources": {}}
    for source in AUDIT_SOURCES:
        weighted: dict[str, float] = defaultdict(float)
        joint_count = 0
        gradient_values: dict[str, list[float]] = defaultdict(list)
        samples = []
        for record in selected[source]:
            rgb = _rgb_tensor(_preprocessed_rgb(record, index)).to(device)
            boxes, targets, valid = _person_tensors(record, device)
            sample: dict[str, object] = {"stem": record["stem"]}
            for loss_name in ("coordinate_huber_pixels", "coordinate_huber_normalized", "gaussian_heatmap_kl"):
                loss_value, gradient_norm = _loss_and_rgb_gradient(
                    critic, rgb, boxes, targets, valid, loss_name,
                    temperature=args.temperature, gaussian_sigma=args.gaussian_sigma,
                )
                weighted[loss_name] += loss_value * int(valid.sum().item())
                gradient_values[loss_name].append(gradient_norm)
                sample[loss_name] = loss_value
                sample[GRADIENT_METRIC_NAMES[loss_name]] = gradient_norm
            with torch.inference_mode():
                logits = critic(rgb, [boxes]).logits
            with torch.no_grad():
                diagnostics = detached_pose_diagnostics(logits, boxes, targets, valid, temperature=args.temperature)
                # Construct this independently so target normalization is also audited.
                target_distribution = gaussian_heatmap_target(targets, boxes, tuple(logits.shape[-2:]), args.gaussian_sigma)
                if not torch.isfinite(target_distribution).all():
                    raise RuntimeError(f"{source}/{record['stem']}: non-finite Gaussian target")
            count = int(diagnostics["joint_count"])
            joint_count += count
            _weighted_add(weighted, diagnostics, count)
            sample["joint_count"] = count
            samples.append(sample)
        if joint_count == 0:
            raise RuntimeError(f"{source}: no valid joints were audited")
        sweep = None
        if args.temperature_sweep:
            sweep = {}
            for temperature in (0.5, 1.0, 2.0):
                temperature_weighted: dict[str, float] = defaultdict(float)
                for record in selected[source]:
                    rgb = _rgb_tensor(_preprocessed_rgb(record, index)).to(device)
                    boxes, targets, valid = _person_tensors(record, device)
                    with torch.inference_mode():
                        logits = critic(rgb, [boxes]).logits
                    metrics = detached_pose_diagnostics(
                        logits, boxes, targets, valid, temperature=temperature, include_argmax=False,
                    )
                    _weighted_add(temperature_weighted, metrics, int(metrics["joint_count"]))
                sweep[str(temperature)] = {
                    key: temperature_weighted[key] / joint_count
                    for key in ("soft_coordinate_error_normalized", "soft_pck_005", "soft_pck_010")
                }
        report["sources"][source] = {
            "sample_count": len(samples), "joint_count": joint_count,
            **{name: value / joint_count for name, value in sorted(weighted.items())},
            "rgb_gradient_statistics": {
                GRADIENT_METRIC_NAMES[name]: _gradient_statistics(values)
                for name, values in sorted(gradient_values.items())
            },
            "samples": samples,
        }
        if sweep is not None:
            report["sources"][source]["temperature_sweep"] = sweep
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
