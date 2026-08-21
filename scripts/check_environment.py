from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
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
    print("Installed core packages:")

    packages = [
        "PIL",
        "numpy",
        "pandas",
        "yaml",
        "tqdm",
    ]

    for package in packages:
        try:
            module = __import__(package)

            version = getattr(
                module,
                "__version__",
                "installed",
            )

            print(f"  {package:12s} {version}")

        except ImportError:
            print(f"  {package:12s} NOT INSTALLED")


if __name__ == "__main__":
    main()