#!/usr/bin/env python3
"""Launch the locked full 768 Krea-2 Pose Control-LoRA production recipe."""
from pose_controlnet.production_training import build_arg_parser, run


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
