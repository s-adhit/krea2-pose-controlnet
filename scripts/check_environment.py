from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the project environment without mutating installed packages."
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit nonzero unless CUDA and BF16 are available (use on the GH200)",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("Krea-2 Pose ControlNet Environment")
    print("=" * 60)

    print(f"Python:       {sys.version}")
    print(f"Executable:   {sys.executable}")
    print(f"Platform:     {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    project_root = Path(__file__).resolve().parents[1]

    print(f"Project root: {project_root}")
    print(f"Python prefix:{sys.prefix}")

    print()
    print("Managed project packages:")
    for package in (
        "Pillow", "diffusers", "einops", "huggingface-hub", "numpy", "pandas",
        "PyYAML", "safetensors", "tqdm", "transformers", "wandb", "pytest",
    ):
        print(f"  {package:18s} {package_version(package)}")

    print()
    print("Host-owned accelerator stack (must not be installed by uv):")
    try:
        import torch

        print(f"  torch:              {torch.__version__}")
        print(f"  torch location:     {Path(torch.__file__).resolve()}")
        print(f"  torch CUDA build:   {torch.version.cuda}")
        print(f"  CUDA available:     {torch.cuda.is_available()}")
        print(f"  cuDNN:              {torch.backends.cudnn.version()}")
        print(f"  BF16 CUDA support:  {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
        print(f"  torchvision:        {package_version('torchvision')}")
        print(f"  triton:             {package_version('triton')}")
    except ImportError:
        print("  torch:              NOT INSTALLED")
        if args.require_cuda:
            raise SystemExit("CUDA gate failed: torch is not importable")
        return

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA gate failed: CUDA is not available")
    if args.require_cuda and not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA gate failed: BF16 is not supported")


if __name__ == "__main__":
    main()
