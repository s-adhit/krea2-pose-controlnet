#!/usr/bin/env python3
"""Build the frozen hero-v1 winner contract and public/review collages.

The selection is intentionally the only editorial input in this script. Every
prompt, seed, control identity, runtime field, and release reference is read
from the frozen hero manifest, frozen batch manifests, and copied sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
HERO_MANIFEST = ROOT / "docs/showcase/final_hero_showcase_v1.json"
HERO_DIR = ROOT / "docs/showcase/final/hero-v1"
PROVENANCE = HERO_DIR / "hero_provenance.json"
WINNERS = HERO_DIR / "final_winners.json"
PUBLIC_COLLAGE = HERO_DIR / "final_showcase_collage.png"
LABELED_COLLAGE = HERO_DIR / "final_showcase_collage_labeled.png"

# The public collage is a dense, equal-pair composition.  A pair is always
# ``[condition | generation]`` with exactly equal panel dimensions.  Native
# aspect-ratio differences are represented with black containment padding;
# neither half is cropped, stretched, or treated as a thumbnail.
TOP_SIDE_SIZE = (700, 1000)
BOTTOM_SIDE_SIZE = (1054, 800)
GUTTER = 8
# The two slightly wider lower pairs complete the rectangular 3-over-2 grid
# without elevating either concept into a dominant hero panel.  Their area is
# about 20% above an upper pair, and their shorter height gives the wide jester
# and the tall painterly image equal visual treatment.
COLLAGE_SIZE = (
    3 * (2 * TOP_SIDE_SIZE[0] + GUTTER) + 2 * GUTTER,
    TOP_SIDE_SIZE[1] + BOTTOM_SIDE_SIZE[1] + GUTTER,
)
# concept, x, y, side width, side height. The swordswoman is first.
PAIR_ROWS = (
    (
        ("female_swordswoman_psychedelic", 0, 0, *TOP_SIDE_SIZE),
        ("fantasy_mage", 2 * TOP_SIDE_SIZE[0] + 2 * GUTTER, 0, *TOP_SIDE_SIZE),
        ("comic_fashion", 2 * (2 * TOP_SIDE_SIZE[0] + 2 * GUTTER), 0, *TOP_SIDE_SIZE),
    ),
    (
        ("starry_night_painterly", 0, TOP_SIDE_SIZE[1] + GUTTER, *BOTTOM_SIDE_SIZE),
        ("dark_fantasy_jester", 2 * BOTTOM_SIDE_SIZE[0] + 2 * GUTTER, TOP_SIDE_SIZE[1] + GUTTER, *BOTTOM_SIDE_SIZE),
    ),
)

# This is the user-approved public selection, expressed against frozen IDs.
SELECTIONS = {
    "fantasy_mage": ("hero_variant", "fantasy_mage_hero_b"),
    "dark_fantasy_jester": ("original_accepted", None),
    "comic_fashion": ("hero_variant", "comic_fashion_hero_b"),
    "female_swordswoman_psychedelic": ("original_accepted", None),
    "starry_night_painterly": ("hero_variant", "starry_night_painterly_hero_a"),
}
ALTERNATES = {
    "fantasy_mage": ("original", None),
    "comic_fashion": ("original", None),
    "female_swordswoman_psychedelic": ("hero_variant", "female_swordswoman_psychedelic_hero_a"),
    "starry_night_painterly": ("original", None),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def local_sidecar_path(concept: str, variant_id: str | None) -> Path:
    if variant_id is None:
        return HERO_DIR / "original_winners" / f"{concept}.json"
    return HERO_DIR / "hero_variants" / f"{variant_id}.json"


def local_image_path(concept: str, variant_id: str | None) -> Path:
    if variant_id is None:
        return HERO_DIR / "original_winners" / f"{concept}.png"
    return HERO_DIR / "hero_variants" / f"{variant_id}.png"


def load_frozen_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    hero = read_json(HERO_MANIFEST)
    provenance = read_json(PROVENANCE)
    if provenance.get("hero_manifest", {}).get("sha256") != sha256(HERO_MANIFEST):
        raise ValueError("hero provenance does not match the frozen hero manifest")
    source_manifests: dict[str, dict[str, Any]] = {}
    for reference in hero["source_manifests"]:
        path = ROOT / reference["path"]
        if sha256(path) != reference["sha256"]:
            raise ValueError(f"frozen source manifest hash mismatch: {path}")
        source_manifests[reference["id"]] = read_json(path)
    return hero, provenance, source_manifests


def source_row(manifest: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
    for row in manifest["rows"]:
        if row.get("id", row.get("row_id")) == row_id:
            return row
    raise ValueError(f"missing frozen source row: {row_id}")


def runtime_reference(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate": sidecar["candidate"],
        "runtime": {
            "mode": sidecar["mode"],
            "control_scale": sidecar["control_scale"],
            "geometry_mode": sidecar["geometry_mode"],
            "turbo": sidecar["turbo"],
            "style_lora": sidecar["style_lora"],
        },
        "release": sidecar["release"],
    }


def contract_record(
    concept: str,
    selection_kind: str,
    variant_id: str | None,
    hero_row: Mapping[str, Any],
    provenance_row: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_manifest_reference: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar_path = local_sidecar_path(concept, variant_id)
    image_path = local_image_path(concept, variant_id)
    condition_path = HERO_DIR / "conditions" / f"{concept}.png"
    for path in (sidecar_path, image_path, condition_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen showcase asset is missing: {path}")
    sidecar = read_json(sidecar_path)
    frozen_row = source_row(source_manifest, str(hero_row["source_row_id"]))
    if (
        sidecar["prompt"] != frozen_row["prompt"]
        or sidecar["pose_image_sha256"] != provenance_row["control_sha256"]
        or Path(sidecar["pose_image"]).stem != frozen_row["control_stem"]
    ):
        raise ValueError(f"sidecar does not agree with frozen source provenance: {concept}")
    selected_variant = (
        "original_accepted" if variant_id is None and selection_kind == "original_accepted"
        else "original" if variant_id is None
        else "hero_variant_" + variant_id.rsplit("_", 1)[-1]
    )
    return {
        "concept": concept,
        "selected_variant": selected_variant,
        "generation_id": variant_id if variant_id is not None else f"{concept}_original",
        "generation_path": repo_path(image_path),
        "condition_path": repo_path(condition_path),
        "prompt": sidecar["prompt"],
        "seed": sidecar["seed"],
        "control_stem": frozen_row["control_stem"],
        "control_sha256": sidecar["pose_image_sha256"],
        "generation_sidecar_path": repo_path(sidecar_path),
        "candidate_runtime_release_reference": runtime_reference(sidecar),
        "frozen_source_reference": {
            "hero_manifest_path": repo_path(HERO_MANIFEST),
            "hero_manifest_sha256": sha256(HERO_MANIFEST),
            "source_manifest_id": hero_row["source_manifest"],
            "source_manifest_path": source_manifest_reference["path"],
            "source_manifest_sha256": source_manifest_reference["sha256"],
            "source_row_id": hero_row["source_row_id"],
            "original_generation_sidecar": hero_row["original_generation_sidecar"],
        },
    }


def build_contract() -> dict[str, Any]:
    hero, provenance, source_manifests = load_frozen_sources()
    hero_rows = {row["id"]: row for row in hero["rows"]}
    provenance_rows = {row["id"]: row for row in provenance["concepts"]}
    source_references = {reference["id"]: reference for reference in hero["source_manifests"]}
    records = []
    alternates = []
    for concept, (kind, variant_id) in SELECTIONS.items():
        row = hero_rows[concept]
        records.append(contract_record(
            concept, kind, variant_id, row, provenance_rows[concept], source_manifests[row["source_manifest"]],
            source_references[row["source_manifest"]],
        ))
    for concept, (kind, variant_id) in ALTERNATES.items():
        row = hero_rows[concept]
        alternate = contract_record(
            concept, kind, variant_id, row, provenance_rows[concept], source_manifests[row["source_manifest"]],
            source_references[row["source_manifest"]],
        )
        alternate["alternate_variant"] = alternate.pop("selected_variant")
        alternates.append(alternate)
    return {
        "format_version": 1,
        "id": "final-hero-v1-winners",
        "winner_source": repo_path(HERO_MANIFEST),
        "winners": records,
        "metadata": {"alternates": alternates},
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def paste_contain(canvas: Image.Image, source: Path, box: tuple[int, int, int, int]) -> None:
    """Paste an uncropped source into a panel, padding rather than distorting.

    The showcase is a layout of attached pairs, not a reason to reframe the
    generated artwork.  A black panel backing makes any necessary letterbox or
    pillarbox part of the pair while retaining every source pixel.
    """
    image = Image.open(source).convert("RGB")
    panel = Image.new("RGB", (box[2], box[3]), "#000000")
    fitted = ImageOps.contain(image, panel.size, method=Image.Resampling.LANCZOS)
    offset = ((panel.width - fitted.width) // 2, (panel.height - fitted.height) // 2)
    panel.paste(fitted, offset)
    canvas.paste(panel, (box[0], box[1]))


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def pair_boxes(pair: tuple[str, int, int, int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return the equal displayed panels for one ``[condition | generation]`` pair."""
    _concept, x, y, side_width, side_height = pair
    return (
        (x, y, side_width, side_height),
        (x + side_width + GUTTER, y, side_width, side_height),
    )


def build_collage(contract: Mapping[str, Any], labeled: bool) -> Image.Image:
    """Build five native-aspect-preserving, equal-side condition/generation pairs."""
    canvas = Image.new("RGB", COLLAGE_SIZE, "#000000")
    draw = ImageDraw.Draw(canvas)
    entries = {entry["concept"]: entry for entry in contract["winners"]}
    labels = {
        "fantasy_mage": "Fantasy mage",
        "dark_fantasy_jester": "Dark-fantasy jester",
        "comic_fashion": "Comic fashion",
        "female_swordswoman_psychedelic": "Psychedelic swordswoman",
        "starry_night_painterly": "Starry-night painterly",
    }
    for row in PAIR_ROWS:
        for pair in row:
            concept, x, y, _side_width, _side_height = pair
            entry = entries[concept]
            condition_box, generation_box = pair_boxes(pair)
            paste_contain(canvas, ROOT / entry["condition_path"], condition_box)
            paste_contain(canvas, ROOT / entry["generation_path"], generation_box)
            if labeled:
                text = labels[concept]
                padding = 8
                text_box = draw.textbbox((0, 0), text, font=font(18))
                text_width = text_box[2] - text_box[0]
                draw.rectangle((x, y, x + text_width + 2 * padding, y + 30), fill="#17171a")
                draw.text((x + padding, y + 6), text, fill="#f4f0e8", font=font(18))
    return canvas


def save_image(path: Path, image: Image.Image) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_contract(contract: Mapping[str, Any]) -> None:
    if [entry["concept"] for entry in contract["winners"]] != list(SELECTIONS):
        raise ValueError("winner order or concepts drifted from the approved selection")
    for entry in [*contract["winners"], *contract["metadata"]["alternates"]]:
        for key in ("generation_path", "condition_path", "generation_sidecar_path"):
            if not (ROOT / entry[key]).is_file():
                raise FileNotFoundError(f"contract path does not exist: {entry[key]}")
        if not entry["prompt"] or not isinstance(entry["seed"], int):
            raise ValueError(f"incomplete winner provenance: {entry['concept']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate frozen inputs without writing outputs")
    args = parser.parse_args()
    contract = build_contract()
    verify_contract(contract)
    if args.check:
        return
    write_json(WINNERS, contract)
    save_image(PUBLIC_COLLAGE, build_collage(contract, labeled=False))
    save_image(LABELED_COLLAGE, build_collage(contract, labeled=True))


if __name__ == "__main__":
    main()
