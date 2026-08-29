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
    soft_coordinates,
)
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.pose_targets import load_sidecar


AUDIT_SOURCES = ("coco", "humanart_painting", "humanart_real_human", "humanart_sculpture")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True, help="Immutable authoritative v3 sidecar directory.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Read-only PoseBridge HF snapshot root.")
    parser.add_argument("--samples-per-source", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
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
        joint_count, grad_norm = 0, None
        samples = []
        for sample_index, record in enumerate(selected[source]):
            rgb = _rgb_tensor(_preprocessed_rgb(record, index)).to(device)
            boxes, targets, valid = _person_tensors(record, device)
            if sample_index == 0:
                rgb.requires_grad_(True)
                critic.zero_grad(set_to_none=True)
                heatmaps = critic(rgb, [boxes])
                predicted = soft_coordinates(heatmaps.logits, heatmaps.boxes_training, args.temperature)
                coordinate_loss = masked_coordinate_huber(predicted, targets, valid)
                coordinate_loss.backward()
                if rgb.grad is None or not torch.isfinite(rgb.grad).all() or rgb.grad.norm().item() <= 0:
                    raise RuntimeError(f"{source}/{record['stem']}: coordinate loss did not produce finite nonzero RGB gradient")
                if any(parameter.grad is not None for parameter in critic.parameters()):
                    raise RuntimeError("Frozen critic parameter unexpectedly received a gradient")
                grad_norm = float(rgb.grad.norm().item())
                logits = heatmaps.logits.detach()
            else:
                with torch.inference_mode():
                    logits = critic(rgb, [boxes]).logits
            with torch.no_grad():
                predicted = soft_coordinates(logits, boxes, args.temperature)
                coordinate_loss = masked_coordinate_huber(predicted, targets, valid)
                heatmap_loss = gaussian_heatmap_kl(
                    logits, targets, boxes, valid, sigma=args.gaussian_sigma, temperature=args.temperature,
                )
                diagnostics = detached_pose_diagnostics(logits, boxes, targets, valid, temperature=args.temperature)
                # Construct this independently so target normalization is also audited.
                target_distribution = gaussian_heatmap_target(targets, boxes, tuple(logits.shape[-2:]), args.gaussian_sigma)
                if not torch.isfinite(target_distribution).all():
                    raise RuntimeError(f"{source}/{record['stem']}: non-finite Gaussian target")
            count = int(diagnostics["joint_count"])
            joint_count += count
            _weighted_add(weighted, diagnostics, count)
            weighted["coordinate_huber"] += float(coordinate_loss.item()) * count
            weighted["gaussian_heatmap_kl"] += float(heatmap_loss.item()) * count
            samples.append({"stem": record["stem"], "joint_count": count})
        if joint_count == 0 or grad_norm is None:
            raise RuntimeError(f"{source}: no valid joints were audited")
        report["sources"][source] = {
            "sample_count": len(samples), "joint_count": joint_count,
            **{name: value / joint_count for name, value in sorted(weighted.items())},
            "rgb_input_gradient_norm_first_sample": grad_norm,
            "samples": samples,
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
