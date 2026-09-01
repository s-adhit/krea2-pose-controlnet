"""Build an immutable source-space pose reference sidecar for an exact Mixed-32 manifest.

The builder consumes only the committed manifest and the reviewed
``pose_targets_v3`` export.  It never opens a generated image, a control raster,
or a pose detector.  Native evaluation geometry is intentionally deferred to
``--stage score-only`` and read from its persisted generation metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.reference_pose import (
    ReferencePoseError,
    build_exact_manifest_reference_records,
    write_exact_manifest_reference_jsonl,
)


DEFAULT_MANIFEST = Path("configs/overfit_capacity/manifests/overfit32-mixed-r64-mse.jsonl")
DEFAULT_AUTHORITATIVE_SOURCE = Path("/lambda/nfs/adhit/krea2-pose/pose_targets_v3")
DEFAULT_OUTPUT = Path("data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl")
DEFAULT_COMPATIBLE_EXPERIMENTS = (
    "overfit32-mixed-r64-mse",
    "overfit32-mixed-r64-mse-res768",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--authoritative-source", type=Path, default=DEFAULT_AUTHORITATIVE_SOURCE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--compatible-experiment", action="append", dest="compatible_experiments")
    return value


def build(*, manifest: Path, authoritative_source: Path, output: Path,
          compatible_experiments: tuple[str, ...] = DEFAULT_COMPATIBLE_EXPERIMENTS) -> dict:
    """Create one non-overwritable exact-manifest sidecar from reviewed targets."""
    source_metadata, source_records = load_sidecar(authoritative_source)
    records, metadata = build_exact_manifest_reference_records(
        manifest_path=manifest,
        authoritative_records=source_records,
        authoritative_metadata={**source_metadata, "source_path": str(authoritative_source.resolve())},
        compatible_experiments=compatible_experiments,
    )
    return write_exact_manifest_reference_jsonl(records, output, metadata=metadata)


def main() -> None:
    args = parser().parse_args()
    compatible = tuple(args.compatible_experiments or DEFAULT_COMPATIBLE_EXPERIMENTS)
    result = build(
        manifest=args.manifest,
        authoritative_source=args.authoritative_source,
        output=args.output,
        compatible_experiments=compatible,
    )
    print(result["records_sha256"])


if __name__ == "__main__":
    main()
