"""No-generation deployment smoke check for the standalone ComfyUI package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .krea2_pose_runtime.runtime import resolve_release_artifact, validate_release_artifact
from .nodes import extract_coco17_people
from .renderer import render_coco17_people


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--release-artifact", required=True, help="local file or hf://owner/repo/path")
    parser.add_argument("--turbo-path", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, help="optional Krea-2 Raw provenance file to validate")
    parser.add_argument("--output-condition", type=Path, default=Path("krea2_pose_preflight_condition.png"))
    parser.add_argument("--backend", choices=("dwpose", "keypoint_rcnn"), default="dwpose")
    parser.add_argument("--person-confidence", type=float, default=0.5)
    parser.add_argument("--keypoint-confidence", type=float, default=0.5)
    parser.add_argument("--max-people", type=int, default=0, help="0 retains every accepted person")
    parser.add_argument("--device", default=None, help="detector device, e.g. cuda or cpu")
    parser.add_argument("--dwpose-model-dir", type=Path,
                        help="optional ControlNet-Aux checkpoint root containing yzd-v/DWPose/")
    args = parser.parse_args(argv)
    for label, path in (("reference image", args.reference_image), ("Turbo checkpoint", args.turbo_path),
                        ("Krea-2 Raw checkpoint", args.raw_path)):
        if path is not None and not path.is_file():
            parser.error(f"{label} is missing: {path}")
    artifact = validate_release_artifact(resolve_release_artifact(args.release_artifact))
    with Image.open(args.reference_image) as source:
        source = source.convert("RGB")
        people = extract_coco17_people(
            source, backend=args.backend, person_confidence=args.person_confidence,
            keypoint_confidence=args.keypoint_confidence,
            max_people=None if args.max_people == 0 else args.max_people, device=args.device,
            dwpose_model_dir=args.dwpose_model_dir,
        )
        condition = render_coco17_people(source.size, people, keypoint_confidence_threshold=args.keypoint_confidence)
    args.output_condition.parent.mkdir(parents=True, exist_ok=True)
    condition.save(args.output_condition)
    print(json.dumps({"status": "PASS", "release_artifact": str(artifact), "turbo": str(args.turbo_path),
                      "raw": None if args.raw_path is None else str(args.raw_path),
                      "backend": args.backend, "condition": str(args.output_condition), "size": list(condition.size),
                      "detected_people": len(people),
                      "retained_body_joints_per_person": [sum(point[2] > 0 for point in person) for person in people]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
