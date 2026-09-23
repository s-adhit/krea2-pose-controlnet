"""Local discovery for ComfyUI-ControlNet-Aux's internal Python package.

ComfyUI loads every custom-node directory, but ControlNet-Aux keeps its Python
package below ``comfyui_controlnet_aux/src``.  That directory is not normally
on ``PYTHONPATH`` for a standalone module invocation, so discover precisely
that sibling rather than requiring users to add it themselves.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType


class ControlNetAuxDependencyError(RuntimeError):
    """Raised when the standalone package cannot use ControlNet-Aux."""


def discover_controlnet_aux_src(*, package_dir: Path | None = None,
                                search_paths: Iterable[str] | None = None) -> Path | None:
    """Return a sibling ControlNet-Aux ``src`` directory, if installed.

    Only explicit ComfyUI ``custom_nodes`` roots are considered.  This keeps
    discovery deterministic and avoids searching unrelated user directories.
    """
    package_dir = (package_dir or Path(__file__).resolve().parent).resolve()
    candidates = [package_dir.parent]
    for entry in search_paths if search_paths is not None else sys.path:
        if not entry:
            continue
        try:
            candidate = Path(entry).expanduser().resolve()
        except OSError:
            continue
        if candidate.name == "custom_nodes":
            candidates.append(candidate)
    seen: set[Path] = set()
    for custom_nodes in candidates:
        if custom_nodes.name != "custom_nodes" or custom_nodes in seen:
            continue
        seen.add(custom_nodes)
        source = custom_nodes / "comfyui_controlnet_aux" / "src"
        if source.is_dir():
            return source
    return None


def load_dwpose_module(*, package_dir: Path | None = None) -> ModuleType:
    """Import ControlNet-Aux DWPose, adding its discovered local ``src`` once.

    An already-importable package is always preferred.  A discovered path is
    process-local and added only for the duration of this Python process; it
    is required because ControlNet-Aux may import its sibling modules lazily.
    """
    module_name = "custom_controlnet_aux.dwpose"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {"custom_controlnet_aux", module_name}:
            raise ControlNetAuxDependencyError(
                "ComfyUI-ControlNet-Aux was found but one of its Python dependencies is missing. "
                "Install its documented DWPose/ONNX dependencies in ComfyUI's Python environment."
            ) from error

    source = discover_controlnet_aux_src(package_dir=package_dir)
    if source is None:
        raise ControlNetAuxDependencyError(
            "The dwpose backend requires ComfyUI-ControlNet-Aux. Install it as a sibling custom node at "
            "<ComfyUI>/custom_nodes/comfyui_controlnet_aux, then restart ComfyUI. "
            "Standalone preflight discovers its src directory automatically; do not add it to PYTHONPATH."
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
        importlib.invalidate_caches()
    try:
        return importlib.import_module(module_name)
    except (ImportError, AttributeError) as error:
        raise ControlNetAuxDependencyError(
            f"Unable to import DWPose from discovered ControlNet-Aux source: {source}. "
            "Verify that ComfyUI-ControlNet-Aux and its DWPose/ONNX dependencies are installed."
        ) from error
