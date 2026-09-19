"""Materialize the manifest-defined selective hero-v2 retry batch.

This reuses the canonical runner's release validation and no-overwrite copy
policy, while retaining inference.py as the only model/sampling implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from generate_hero_v2_canonical import (
    EXPECTED_RELEASE_SHA256, HeroV2Error, ROOT, TURBO_CHECKPOINT,
    _copy_no_overwrite, _read_json, _write_json, sha256, validate_release,
)
from inference import (
    CANONICAL_CANDIDATE, NATIVE_GEOMETRY, PoseInferenceRequest, generate_pose,
    load_inference_runtime, metadata_path_for, resolve_pose_candidate,
)


MANIFEST = ROOT / "docs/showcase/final/hero-v2/retries/retry-a/retry_a.json"
EXPECTED_DEFAULTS = {
    "release_candidate": "mix-025", "mode": "turbo-pose-control",
    "geometry": "native_aspect_preserving", "steps": 8, "cfg": 0.0,
    "mu": 1.15, "mu_resolution_dependent": False, "style_lora": None,
}


def load_retries() -> tuple[list[dict[str, Any]], Path, Path]:
    payload = _read_json(MANIFEST)
    rows = payload.get("retries")
    if (payload.get("schema_version") != 1 or payload.get("status") != "approved_selective_retry_plan"
            or payload.get("defaults") != EXPECTED_DEFAULTS or not isinstance(rows, list) or len(rows) != 5):
        raise HeroV2Error("retry-a manifest identity or default contract drifted")
    output_root = Path(str(payload.get("output_root", "")))
    docs_root = ROOT / str(payload.get("presentation_root", ""))
    if output_root != Path("/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/retry-a"):
        raise HeroV2Error("retry-a output root drifted")
    if docs_root != ROOT / "docs/showcase/final/hero-v2/generations/retry-a":
        raise HeroV2Error("retry-a presentation root drifted")
    required = {"concept_name", "slug", "source_stem", "control_path", "dimensions", "people_count", "seed", "control_scale", "prompt"}
    seen: set[str] = set()
    retries: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise HeroV2Error("retry-a row is incomplete")
        slug, dimensions = row["slug"], row["dimensions"]
        control = ROOT / str(row["control_path"])
        if (not isinstance(slug, str) or slug in seen or not isinstance(row["seed"], int)
                or row["control_scale"] not in (1.0, 1.25) or not isinstance(row["prompt"], str)
                or not row["prompt"].strip() or not isinstance(dimensions, dict)
                or set(dimensions) != {"width", "height"} or not control.is_file()):
            raise HeroV2Error(f"Invalid retry-a row: {row.get('concept_name')}")
        width, height = dimensions["width"], dimensions["height"]
        if not isinstance(width, int) or not isinstance(height, int) or width % 16 or height % 16:
            raise HeroV2Error(f"Invalid retry-a dimensions: {row['concept_name']}")
        with Image.open(control) as image:
            if image.size != (width, height):
                raise HeroV2Error(f"Control geometry mismatch: {row['concept_name']}")
        seen.add(slug)
        retries.append(dict(row))
    return retries, output_root, docs_root


def _request(row: Mapping[str, Any], output: Path) -> PoseInferenceRequest:
    dimensions = row["dimensions"]
    return PoseInferenceRequest(
        turbo_checkpoint=TURBO_CHECKPOINT, pose_lora_checkpoint=None,
        release_artifact=Path("/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors"),
        candidate=CANONICAL_CANDIDATE, prompt=str(row["prompt"]),
        pose_image=ROOT / str(row["control_path"]), output=output, seed=int(row["seed"]),
        width=int(dimensions["width"]), height=int(dimensions["height"]), steps=8,
        cfg=0.0, mu=1.15, control_scale=float(row["control_scale"]), geometry_mode=NATIVE_GEOMETRY,
    )


def _validate_and_annotate(row: Mapping[str, Any], output: Path) -> dict[str, Any]:
    sidecar = metadata_path_for(output)
    if not output.is_file() or not sidecar.is_file():
        raise HeroV2Error(f"Missing retry output or provenance: {output}")
    with Image.open(output) as image:
        if image.size != (row["dimensions"]["width"], row["dimensions"]["height"]):
            raise HeroV2Error(f"Incorrect retry geometry: {output}")
    metadata = _read_json(sidecar)
    expected = {"prompt": row["prompt"], "seed": row["seed"], "control_scale": row["control_scale"], "steps": 8, "cfg": 0.0, "mu": 1.15, "geometry_mode": "explicit"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise HeroV2Error(f"Retry inference metadata drifted: {sidecar}")
    if metadata.get("candidate", {}).get("candidate_id") != CANONICAL_CANDIDATE:
        raise HeroV2Error(f"Unexpected release candidate: {sidecar}")
    if metadata.get("candidate", {}).get("release_artifact", {}).get("sha256") != EXPECTED_RELEASE_SHA256:
        raise HeroV2Error(f"Release artifact provenance mismatch: {sidecar}")
    metadata["hero_v2_retry_a"] = {
        "concept": row["concept_name"], "source_stem": row["source_stem"],
        "control_path": str((ROOT / str(row["control_path"])).resolve()),
        "width": row["dimensions"]["width"], "height": row["dimensions"]["height"],
        "candidate": "mix-025", "release_artifact_sha256": EXPECTED_RELEASE_SHA256,
        "steps": 8, "cfg": 0.0, "mu": 1.15, "mu_resolution_dependent": False,
        "control_scale": row["control_scale"], "style_lora": None,
    }
    _write_json(sidecar, metadata)
    return metadata


def generate() -> tuple[list[tuple[dict[str, Any], Path]], Path]:
    rows, output_root, docs_root = load_retries()
    validate_release()
    output_root.mkdir(parents=True, exist_ok=True)
    first = _request(rows[0], output_root / f"{rows[0]['slug']}.png")
    runtime = load_inference_runtime(first, candidate=resolve_pose_candidate(first))
    generated: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        output = output_root / f"{row['slug']}.png"
        if not output.exists() and not metadata_path_for(output).exists():
            generate_pose(_request(row, output), runtime=runtime)
        _validate_and_annotate(row, output)
        generated.append((row, output))
    return generated, docs_root


def package(generated: Sequence[tuple[dict[str, Any], Path]], docs_root: Path) -> Path:
    for row, output in generated:
        _copy_no_overwrite(output, docs_root / output.name)
        _copy_no_overwrite(metadata_path_for(output), docs_root / metadata_path_for(output).name)
        _copy_no_overwrite(ROOT / str(row["control_path"]), docs_root / "controls" / f"{row['slug']}.png")
    thumb, gutter, header = 300, 12, 58
    sheet = Image.new("RGB", (thumb * 2, header + thumb * len(generated)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, gutter), "Retry control", fill="black")
    draw.text((thumb + gutter, gutter), "Retry-a mix-025", fill="black")
    for index, (row, output) in enumerate(generated):
        y = header + index * thumb
        draw.text((gutter, y + gutter), str(row["concept_name"]), fill="black")
        for column, source in enumerate((ROOT / str(row["control_path"]), output)):
            with Image.open(source) as image:
                display = image.convert("RGB")
            display.thumbnail((thumb - 2 * gutter, thumb - 42))
            sheet.paste(display, (column * thumb + (thumb - display.width) // 2, y + 36 + (thumb - 42 - display.height) // 2))
    contact = docs_root / "retry-a_contact_sheet.png"
    sheet.save(contact)
    return contact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "generate"), nargs="?", default="generate")
    args = parser.parse_args(argv)
    rows, output_root, docs_root = load_retries()
    if args.command == "preflight":
        validate_release()
        print(f"PASS: {len(rows)} retry-a selections; outputs: {output_root}; presentation: {docs_root}")
        return
    generated, docs_root = generate()
    contact = package(generated, docs_root)
    print(f"PASS: retry-a generated at {output_root}; contact sheet: {contact}")


if __name__ == "__main__":  # pragma: no cover
    main()
