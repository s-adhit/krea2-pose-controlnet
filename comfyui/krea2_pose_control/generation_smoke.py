"""Generate one frozen Krea-2 Pose Control release smoke image headlessly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .krea2_pose_runtime.runtime import generate, load_runtime
from .krea2_pose_runtime.turbo_runtime import TURBO_CFG, TURBO_MU, TURBO_STEPS


DEFAULT_PROMPT = (
    "An elegant wandering knight in tailored charcoal wool, aged steel accents and a deep green travel cloak, "
    "crossing a quiet stone moor beneath an overcast sky, soft directional daylight, refined realistic fantasy "
    "photography, natural fabric texture and muted cinematic color."
)
DEFAULT_SEED = 42
CONTROL_SCALE = 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-condition", type=Path, required=True)
    parser.add_argument("--release-artifact", required=True)
    parser.add_argument("--turbo-path", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)
    for label, path in (("pose condition", args.pose_condition), ("Turbo checkpoint", args.turbo_path),
                        ("Krea-2 Raw checkpoint", args.raw_path)):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    if not args.prompt.strip():
        parser.error("prompt must be non-empty")

    with Image.open(args.pose_condition) as condition_file:
        condition = condition_file.convert("RGB")
    runtime = load_runtime(turbo_path=args.turbo_path, release_artifact=args.release_artifact,
                           base_raw_path=args.raw_path)
    image = generate(runtime, prompt=args.prompt, pose_image=condition, seed=args.seed,
                     control_scale=CONTROL_SCALE, steps=TURBO_STEPS, mu=TURBO_MU)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output_image)
    print(json.dumps({
        "status": "PASS", "output_image": str(args.output_image), "condition": str(args.pose_condition),
        "seed": args.seed, "steps": TURBO_STEPS, "cfg": TURBO_CFG, "mu": TURBO_MU,
        "control_scale": CONTROL_SCALE, "geometry_mode": "native_aspect", "size": list(image.size),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
