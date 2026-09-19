"""Materialize the seven frozen hero-v2 canonical pose-control candidates.

This is intentionally an orchestration layer: all model loading, control VAE
encoding, text conditioning, and Turbo sampling remain in ``inference.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from inference import (
    CANONICAL_CANDIDATE,
    NATIVE_GEOMETRY,
    PoseInferenceRequest,
    generate_pose,
    load_inference_runtime,
    metadata_path_for,
    resolve_pose_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "docs/showcase/final/hero-v2/final-selection/final_selection.json"
RELEASE_CONTRACT = ROOT / "docs/evaluation/release/final_release_v1.json"
RELEASE_ARTIFACT = Path("/lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors")
TURBO_CHECKPOINT = Path("/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/showcase/hero-v2/canonical-v1")
DOCS_ROOT = ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1"
EXPECTED_RELEASE_SHA256 = "6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1"
EXPECTED_RUNTIME = {"steps": 8, "cfg": 0.0, "mu": 1.15, "control_scale": 1.0}


class HeroV2Error(ValueError):
    """Raised when frozen hero-v2 inputs or outputs violate the contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HeroV2Error(f"Required JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise HeroV2Error(f"Malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HeroV2Error(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _slug(selection: Mapping[str, Any]) -> str:
    control = Path(str(selection["final_rendered_control_path"]))
    return control.parent.name


def load_selections() -> list[dict[str, Any]]:
    payload = _read_json(SELECTION)
    defaults = payload.get("defaults")
    rows = payload.get("selections")
    if (payload.get("schema_version") != 1 or payload.get("status") != "frozen_generation_plan"
            or defaults != {"release_candidate": "mix-025", "mode": "turbo-pose-control",
                            "geometry": "native_aspect_preserving", "control_scale": 1.0}
            or not isinstance(rows, list) or len(rows) != 7):
        raise HeroV2Error("Frozen final_selection.json identity or default contract drifted")
    seen: set[str] = set()
    selections: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise HeroV2Error("Frozen selection row is not an object")
        required = {"concept_name", "source_stem", "final_rendered_control_path", "dimensions", "prompt", "seed", "control_scale"}
        if not required.issubset(row) or row["control_scale"] != 1.0 or not isinstance(row["seed"], int):
            raise HeroV2Error("Frozen selection row is incomplete or does not use scale 1.0")
        dimensions = row["dimensions"]
        if not isinstance(dimensions, dict) or set(dimensions) != {"width", "height"}:
            raise HeroV2Error(f"{row.get('concept_name')}: invalid dimensions")
        width, height = dimensions["width"], dimensions["height"]
        control = ROOT / str(row["final_rendered_control_path"])
        if (not isinstance(width, int) or not isinstance(height, int) or width % 16 or height % 16
                or not isinstance(row["prompt"], str) or not row["prompt"].strip() or not control.is_file()):
            raise HeroV2Error(f"{row.get('concept_name')}: invalid frozen control, prompt, or dimensions")
        with Image.open(control) as image:
            if image.size != (width, height):
                raise HeroV2Error(f"{row['concept_name']}: control dimensions differ from frozen native geometry")
        slug = _slug(row)
        if slug in seen:
            raise HeroV2Error(f"Duplicate output identity: {slug}")
        seen.add(slug)
        selections.append(dict(row))
    return selections


def validate_release() -> None:
    contract = _read_json(RELEASE_CONTRACT)
    runtime = contract.get("runtime_defaults")
    if (contract.get("candidate", {}).get("id") != CANONICAL_CANDIDATE or not isinstance(runtime, dict)
            or runtime.get("model") != "Krea-2 Turbo" or runtime.get("steps") != 8
            or runtime.get("cfg") != 0.0 or runtime.get("mu") != 1.15
            or runtime.get("mu_resolution_dependent") is not False):
        raise HeroV2Error("Frozen release runtime contract drifted")
    if not RELEASE_ARTIFACT.is_file():
        raise HeroV2Error(f"Release artifact is missing: {RELEASE_ARTIFACT}")
    observed = sha256(RELEASE_ARTIFACT)
    if observed != EXPECTED_RELEASE_SHA256:
        raise HeroV2Error(f"Release artifact SHA-256 mismatch: {observed}")
    if not TURBO_CHECKPOINT.is_file():
        raise HeroV2Error(f"Turbo checkpoint is missing: {TURBO_CHECKPOINT}")


def _request(selection: Mapping[str, Any], output: Path) -> PoseInferenceRequest:
    dimensions = selection["dimensions"]
    return PoseInferenceRequest(
        turbo_checkpoint=TURBO_CHECKPOINT, pose_lora_checkpoint=None,
        release_artifact=RELEASE_ARTIFACT, candidate=CANONICAL_CANDIDATE,
        prompt=str(selection["prompt"]), pose_image=ROOT / str(selection["final_rendered_control_path"]),
        output=output, seed=int(selection["seed"]), width=int(dimensions["width"]), height=int(dimensions["height"]),
        steps=8, cfg=0.0, mu=1.15, control_scale=1.0, geometry_mode=NATIVE_GEOMETRY,
    )


def _augment_and_validate(selection: Mapping[str, Any], output: Path) -> dict[str, Any]:
    sidecar = metadata_path_for(output)
    if not output.is_file() or not sidecar.is_file():
        raise HeroV2Error(f"Missing canonical output or provenance: {output}")
    with Image.open(output) as image:
        if image.size != (selection["dimensions"]["width"], selection["dimensions"]["height"]):
            raise HeroV2Error(f"Incorrect output geometry: {output}")
    metadata = _read_json(sidecar)
    required = {"prompt": selection["prompt"], "seed": selection["seed"], "control_scale": 1.0,
                "steps": 8, "cfg": 0.0, "mu": 1.15, "geometry_mode": "explicit"}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise HeroV2Error(f"Canonical inference metadata drifted: {sidecar}")
    if metadata.get("candidate", {}).get("candidate_id") != CANONICAL_CANDIDATE:
        raise HeroV2Error(f"Unexpected release candidate in: {sidecar}")
    artifact = metadata.get("candidate", {}).get("release_artifact", {})
    if artifact.get("sha256") != EXPECTED_RELEASE_SHA256:
        raise HeroV2Error(f"Release artifact provenance mismatch: {sidecar}")
    metadata["hero_v2_canonical"] = {
        "concept": selection["concept_name"], "source_stem": selection["source_stem"],
        "control_path": str((ROOT / selection["final_rendered_control_path"]).resolve()),
        "width": selection["dimensions"]["width"], "height": selection["dimensions"]["height"],
        "candidate": "mix-025", "release_artifact_sha256": EXPECTED_RELEASE_SHA256,
        "steps": 8, "cfg": 0.0, "mu": 1.15, "mu_resolution_dependent": False,
        "control_scale": 1.0, "style_lora": None,
    }
    _write_json(sidecar, metadata)
    return metadata


def generate() -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    selections = load_selections()
    validate_release()  # Must happen before any model loading or generation.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    first = _request(selections[0], OUTPUT_ROOT / f"{_slug(selections[0])}.png")
    candidate = resolve_pose_candidate(first)
    runtime = load_inference_runtime(first, candidate=candidate)
    generated: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for selection in selections:
        output = OUTPUT_ROOT / f"{_slug(selection)}.png"
        if not output.exists() and not metadata_path_for(output).exists():
            generate_pose(_request(selection, output), runtime=runtime)
        metadata = _augment_and_validate(selection, output)
        generated.append((selection, output, metadata))
    return generated


def _copy_no_overwrite(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise HeroV2Error(f"Refusing to overwrite different presentation file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package(generated: Sequence[tuple[dict[str, Any], Path, dict[str, Any]]]) -> Path:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    for selection, output, _ in generated:
        _copy_no_overwrite(output, DOCS_ROOT / output.name)
        _copy_no_overwrite(metadata_path_for(output), DOCS_ROOT / metadata_path_for(output).name)
        control = ROOT / str(selection["final_rendered_control_path"])
        _copy_no_overwrite(control, DOCS_ROOT / "controls" / f"{_slug(selection)}.png")
    thumb_w, thumb_h, gutter, header = 300, 300, 12, 58
    sheet = Image.new("RGB", (thumb_w * 2, header + thumb_h * len(generated)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, gutter), "Frozen control", fill="black")
    draw.text((thumb_w + gutter, gutter), "Canonical mix-025", fill="black")
    for index, (selection, output, _) in enumerate(generated):
        y = header + index * thumb_h
        draw.text((gutter, y + gutter), str(selection["concept_name"]), fill="black")
        for column, source in enumerate((ROOT / str(selection["final_rendered_control_path"]), output)):
            with Image.open(source) as image:
                display = image.convert("RGB")
            display.thumbnail((thumb_w - 2 * gutter, thumb_h - 42))
            x = column * thumb_w + (thumb_w - display.width) // 2
            sheet.paste(display, (x, y + 36 + (thumb_h - 42 - display.height) // 2))
    contact = DOCS_ROOT / "canonical-v1_contact_sheet.png"
    sheet.save(contact)
    lines = ["# Hero-v2 canonical-v1 generation summary", "", "All seven frozen concepts were generated once at scale 1.0. No winner selection or duo retry was performed.", "",
             f"- Release artifact: `{RELEASE_ARTIFACT}`", f"- Release artifact SHA-256: `{EXPECTED_RELEASE_SHA256}`", "- Runtime: Krea-2 Turbo; 8 steps; CFG 0; mu 1.15; resolution-dependent mu disabled; no Style-LoRA.", "", "| Concept | Output | Control | Seed | Native size |", "| --- | --- | --- | ---: | ---: |"]
    for selection, output, _ in generated:
        size = selection["dimensions"]
        lines.append(f"| {selection['concept_name']} | `{output}` | `{selection['final_rendered_control_path']}` | {selection['seed']} | {size['width']} x {size['height']} |")
    (DOCS_ROOT / "GENERATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return contact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "generate"), nargs="?", default="generate")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        selections = load_selections(); validate_release()
        print(f"PASS: {len(selections)} frozen hero-v2 selections and release artifact SHA")
        return
    generated = generate()
    contact = package(generated)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "concepts": [row[0]["concept_name"] for row in generated], "contact_sheet": str(contact)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
