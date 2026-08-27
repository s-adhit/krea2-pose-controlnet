"""Renderer-aware source-control geometry gate and one-image PCK smoke.

This script is evaluation-only.  It neither constructs training state nor
imports optimizer/training modules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, pck_for_people
from pose_controlnet.reference_pose import reference_people_from_sidecar


def _geometry(geometry: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    return tuple(geometry["source_size"]), tuple(geometry["resized_size"]), tuple(geometry["crop_box"])


def _nearest_control_distance(control: Path, x: float, y: float) -> float:
    pixels = np.asarray(Image.open(control).convert("RGB"))
    drawn = np.argwhere(np.any(pixels != 0, axis=2))
    if not len(drawn):
        raise ValueError(f"{control}: no non-black control pixels")
    return float(np.hypot(drawn[:, 1] - x, drawn[:, 0] - y).min())


def validate_record(record: dict, control: Path, geometry: dict, tolerance: float = 1.5) -> dict:
    """Validate only analytically renderer-represented source joints."""
    source_size, resized_size, crop_box = _geometry(geometry)
    people = reference_people_from_sidecar(
        record, source_size=source_size, resized_size=resized_size, crop_box=crop_box,
    )
    checked, failures = [], []
    for person in people:
        if not person["reference_rendered"]:
            continue
        # Source control PNGs are the original renderer rasters.  Bucket-space
        # coordinates are retained for PCK, but geometry evidence here compares
        # the corresponding authoritative source-space joint.
        for state, point in zip(person["joint_states"], person["keypoints_source"]):
            if not state["rendered_in_control"]:
                continue
            distance = _nearest_control_distance(control, point[0], point[1])
            item = {"annotation_id": person["annotation_id"], "coco_index": state["coco_index"], "distance": distance}
            checked.append(item)
            if distance > tolerance:
                failures.append(item)
    return {
        "stem": record["stem"], "control_path": str(control), "checked_joint_count": len(checked),
        "max_distance": max((item["distance"] for item in checked), default=None),
        "failures": failures, "status": "PASS" if not failures else "FAIL",
    }


def _record_by_stem(sidecar: Path, stem: str) -> dict:
    records = json.loads(sidecar.read_text())["records"]
    return next(record for record in records if record["stem"] == stem)


def _geometry_by_stem(shard: Path) -> dict[str, dict]:
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    return {sample["stem"]: sample for sample in payload["samples"]}


def geometry_main(args: argparse.Namespace) -> None:
    geometry_by_stem = _geometry_by_stem(args.latent_shard)
    results = []
    for stem in args.stems:
        record = _record_by_stem(args.sidecar, stem)
        metadata_path = args.fixed_pose_root / stem / "metadata.json"
        if metadata_path.is_file():
            control = Path(json.loads(metadata_path.read_text())["control_path"])
        else:
            matches = list(args.control_root.glob(f"shard_* /{stem}.png".replace(" ", "")))
            if len(matches) != 1:
                raise ValueError(f"{stem}: expected one control raster under {args.control_root}, found {matches}")
            control = matches[0]
        results.append(validate_record(record, control, geometry_by_stem[stem], args.tolerance))
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL", "records": results}
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


def smoke_main(args: argparse.Namespace) -> None:
    record = _record_by_stem(args.sidecar, args.stem)
    metadata = json.loads((args.fixed_pose_root / args.stem / "metadata.json").read_text())
    source_size, resized_size, crop_box = _geometry(_geometry_by_stem(args.latent_shard)[args.stem])
    people = reference_people_from_sidecar(record, source_size=source_size, resized_size=resized_size, crop_box=crop_box)
    rendered = [person for person in people if person["reference_rendered"]]
    references = [{"keypoints": person["keypoints"]} for person in rendered]
    generated = args.fixed_pose_root / args.stem / f"step_{args.step:06d}.png"
    detector = KeypointRCNNEstimator(args.device, args.confidence_threshold)
    metric = pck_for_people(references, detector(generated), args.confidence_threshold)
    source_visible = sum(state["source_visible"] for person in rendered for state in person["joint_states"])
    rendered_joints = sum(state["rendered_in_control"] for person in rendered for state in person["joint_states"])
    result = metric | {
        "stem": args.stem, "step": args.step,
        "reference_people": len(people),
        "rendered_reference_people": len(rendered),
        "source_visible_joint_count": source_visible,
        "rendered_joint_count": rendered_joints,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sidecar", type=Path, default=Path("data/manifests/diagnostic_reference_pose.json"))
    common.add_argument("--fixed-pose-root", type=Path, default=Path("/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500/fixed_pose"))
    common.add_argument("--latent-shard", type=Path, default=Path("/lambda/nfs/adhit/krea2-pose/posebridge_latents/diagnostic_val/diagnostic_val-00000.pt"))
    common.add_argument("--control-root", type=Path, default=Path("/lambda/nfs/adhit/krea2-pose/posebridge_hf/conditioning_images"))
    geometry = subcommands.add_parser("geometry", parents=[common])
    geometry.add_argument("--tolerance", type=float, default=1.5)
    geometry.add_argument("--stems", nargs="+", default=["real_human_humanart_17000000000288", "coco_156320_crowd", "coco_299468_426600", "painting_humanart_10000000000838"])
    geometry.set_defaults(function=geometry_main)
    smoke = subcommands.add_parser("smoke", parents=[common])
    smoke.add_argument("--stem", default="real_human_humanart_17000000000288")
    smoke.add_argument("--step", type=int, default=500)
    smoke.add_argument("--confidence-threshold", type=float, default=0.5)
    smoke.add_argument("--device", default="cuda")
    smoke.set_defaults(function=smoke_main)
    args = parser.parse_args(); args.function(args)


if __name__ == "__main__":
    main()
