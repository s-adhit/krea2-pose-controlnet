"""Preflight and run the frozen Batch 1 native final-showcase manifest.

This is intentionally an orchestration layer: each output is generated only by
the canonical ``inference.py`` entrypoint, so its normal provenance sidecar is
preserved without duplicating model or sampling behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/showcase/final_showcase_batch1_v1.json"
EXPECTED_RELEASE_PATH = "docs/evaluation/release/final_release_v1.json"
EXPECTED_RELEASE_SHA256 = "9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b"
EXPECTED_OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/showcase/final/batch1-v1")
EXPECTED_ORDER = (
    "unique_fantasy_mage", "unique_dark_fantasy_jester", "unique_modern_sorcerer_mural",
    "unique_stained_glass_mage", "unique_comic_fashion", "unique_masked_swordsman",
    "matched_fantasy_mage_m1", "matched_stained_glass_mage_m1", "matched_dark_fantasy_jester_m2",
    "matched_modern_sorcerer_mural_m2", "matched_comic_fashion_m2", "matched_masked_swordsman_m2",
)
EXPECTED_TURBO = {"steps": 8, "cfg": 0.0, "mu": 1.15}
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
DEFAULT_TURBO_CHECKPOINT = "/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors"


class ShowcaseError(ValueError):
    """Raised when the immutable showcase contract cannot be honored."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ShowcaseError(f"Manifest is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ShowcaseError(f"Manifest is not valid JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ShowcaseError("Manifest must be a JSON object")
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, check_files: bool) -> list[Mapping[str, Any]]:
    if manifest.get("format_version") != 1 or manifest.get("kind") != "final_native_showcase_batch1_v1":
        raise ShowcaseError("Unexpected showcase manifest identity")
    if Path(manifest.get("output_root", "")) != EXPECTED_OUTPUT_ROOT:
        raise ShowcaseError("Showcase output root drifted")
    if manifest.get("release_contract") != {"path": EXPECTED_RELEASE_PATH, "sha256": EXPECTED_RELEASE_SHA256}:
        raise ShowcaseError("Top-level frozen release contract drifted")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ShowcaseError("Showcase must contain exactly 12 generation rows")
    if tuple(row.get("id") for row in rows if isinstance(row, dict)) != EXPECTED_ORDER:
        raise ShowcaseError("Showcase row ordering drifted")
    if [row.get("order") for row in rows] != list(range(1, 13)):
        raise ShowcaseError("Showcase order values must be exactly 1 through 12")
    concept_prompts: dict[str, str] = {}
    outputs: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ShowcaseError("Every showcase row must be an object")
        required = {"order", "id", "concept", "track", "control_stem", "control_path", "control_sha256", "width", "height", "prompt", "seed", "candidate", "release_contract_path", "release_contract_sha256", "control_scale", "turbo", "style_lora", "geometry", "output"}
        if set(row) - {"matched_pose_family"} != required:
            raise ShowcaseError(f"{row.get('id')}: manifest fields drifted")
        matched = row["track"] == "matched"
        if row["track"] not in {"unique", "matched"} or matched != ("matched_pose_family" in row):
            raise ShowcaseError(f"{row['id']}: invalid unique/matched assignment")
        if matched and row["matched_pose_family"] not in {"M1", "M2"}:
            raise ShowcaseError(f"{row['id']}: invalid matched pose family")
        if row["candidate"] != "mix-025" or row["control_scale"] != 1.0 or row["turbo"] != EXPECTED_TURBO:
            raise ShowcaseError(f"{row['id']}: release settings drifted")
        if row["style_lora"] is not None or row["geometry"] != NATIVE_GEOMETRY:
            raise ShowcaseError(f"{row['id']}: only native geometry with no Style-LoRA is permitted")
        if row["release_contract_path"] != EXPECTED_RELEASE_PATH or row["release_contract_sha256"] != EXPECTED_RELEASE_SHA256:
            raise ShowcaseError(f"{row['id']}: frozen release provenance drifted")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip() or not isinstance(row["seed"], int):
            raise ShowcaseError(f"{row['id']}: prompt or seed is invalid")
        prior = concept_prompts.setdefault(row["concept"], row["prompt"])
        if prior != row["prompt"]:
            raise ShowcaseError(f"{row['id']}: concept prompt does not match its paired row")
        width, height = row["width"], row["height"]
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ShowcaseError(f"{row['id']}: output dimensions are not canonical-inference valid")
        output = row["output"]
        if not isinstance(output, str) or not output.startswith("generations/") or output in outputs:
            raise ShowcaseError(f"{row['id']}: output path is invalid or duplicated")
        outputs.add(output)
        if check_files:
            control = Path(row["control_path"])
            if not control.is_file() or sha256(control) != row["control_sha256"]:
                raise ShowcaseError(f"{row['id']}: validated control is missing or SHA-256 mismatched")
            try:
                with Image.open(control) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                raise ShowcaseError(f"{row['id']}: control is not a readable image") from exc
    if set(concept_prompts) != {"fantasy_mage", "dark_fantasy_jester", "modern_sorcerer_mural", "stained_glass_mage", "comic_fashion", "masked_swordsman"}:
        raise ShowcaseError("Showcase concepts drifted")
    return rows


def preflight(manifest_path: Path) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    manifest = load_manifest(manifest_path)
    rows = validate_manifest(manifest, check_files=True)
    release = ROOT / EXPECTED_RELEASE_PATH
    if sha256(release) != EXPECTED_RELEASE_SHA256:
        raise ShowcaseError("Frozen final release contract SHA-256 mismatch")
    return manifest, rows


def generate(manifest_path: Path, turbo_checkpoint: Path) -> None:
    manifest, rows = preflight(manifest_path)
    if not turbo_checkpoint.is_file():
        raise ShowcaseError(f"Turbo checkpoint is missing: {turbo_checkpoint}")
    output_root = Path(manifest["output_root"])
    for row in rows:
        output = output_root / row["output"]
        sidecar = output.with_suffix(".json")
        if output.exists() or sidecar.exists():
            raise ShowcaseError(f"Refusing to overwrite existing output or provenance sidecar: {output}")
    for row in rows:
        output = output_root / row["output"]
        command = [sys.executable, str(ROOT / "inference.py"), "--turbo-ckpt", str(turbo_checkpoint), "--candidate", "mix-025", "--prompt", row["prompt"], "--pose-image", row["control_path"], "--output", str(output), "--seed", str(row["seed"]), "--width", str(row["width"]), "--height", str(row["height"]), "--control-scale", "1.0"]
        subprocess.run(command, cwd=ROOT, check=True)


def contact_sheet(manifest_path: Path) -> Path:
    manifest, rows = preflight(manifest_path)
    output_root = Path(manifest["output_root"])
    pairs: list[tuple[Mapping[str, Any], Path]] = []
    for row in rows:
        generated = output_root / row["output"]
        if not generated.is_file() or not generated.with_suffix(".json").is_file():
            raise ShowcaseError(f"Cannot make review sheet; generation or its provenance sidecar is missing: {generated}")
        pairs.append((row, generated))
    thumb_w, thumb_h, header, gutter = 280, 360, 26, 4
    sheet = Image.new("RGB", (2 * thumb_w, header + len(pairs) * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, gutter), "condition", fill="black")
    draw.text((thumb_w + gutter, gutter), "generation", fill="black")
    for index, (row, generated) in enumerate(pairs):
        y = header + index * thumb_h
        for column, path in enumerate((Path(row["control_path"]), generated)):
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((thumb_w - 2 * gutter, thumb_h - 2 * gutter))
            x = column * thumb_w + (thumb_w - image.width) // 2
            sheet.paste(image, (x, y + (thumb_h - image.height) // 2))
        draw.text((gutter, y + gutter), row["id"], fill="black", stroke_width=1, stroke_fill="white")
    path = output_root / "review_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "generate", "contact-sheet"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--turbo-ckpt", type=Path, default=Path(DEFAULT_TURBO_CHECKPOINT))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        _, rows = preflight(args.manifest)
        print(f"PASS: {len(rows)} frozen showcase rows")
    elif args.command == "generate":
        generate(args.manifest, args.turbo_ckpt)
    else:
        print(contact_sheet(args.manifest))


if __name__ == "__main__":
    main()
