"""Generate and package the frozen final native hero showcase.

This runner only orchestrates ``inference.py``.  It resolves prompts, prepared
controls, and native geometry from the accepted Batch 1/2 manifests and their
canonical sidecars; it never duplicates inference or sampling behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/showcase/final_hero_showcase_v1.json"
EXPECTED_OUTPUT_ROOT = Path("/lambda/nfs/adhit/krea2-pose/showcase/final/hero-v1")
EXPECTED_RELEASE = {
    "path": "docs/evaluation/release/final_release_v1.json",
    "sha256": "9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b",
}
EXPECTED_SOURCES = {
    "batch1-v1": {
        "path": "docs/showcase/final/batch1-v1/final_showcase_batch1_v1.json",
        "sha256": "629f3b87873b4a5a0bd0306ff6ffd1b9ac04e584976bb856226ab9311417c91b",
    },
    "batch2-v1": {
        "path": "docs/showcase/final/batch2-v1/final_showcase_batch2_v1.json",
        "sha256": "b152c97e29a9094260e860df1accdf66814db4898518313ba64300e1715250ae",
    },
}
EXPECTED_WINNERS = (
    ("fantasy_mage", "batch1-v1", "fantasy_mage_m1", "02_fantasy_mage_m1.json", 1847302951),
    ("dark_fantasy_jester", "batch1-v1", "dark_fantasy_jester_unique", "03_dark_fantasy_jester_unique.json", 3028147759),
    ("comic_fashion", "batch1-v1", "comic_fashion_unique", "09_comic_fashion_unique.json", 2519074836),
    ("female_swordswoman_psychedelic", "batch2-v1", "02_female_swordswoman_psychedelic_s2", "04_02_female_swordswoman_psychedelic_s2.json", 7194308222),
    ("starry_night_painterly", "batch2-v1", "05_starry_night_painterly_s1", "09_05_starry_night_painterly_s1.json", 7194308251),
)
EXPECTED_HERO_SEEDS = (7194308301, 7194308302, 7194308311, 7194308312, 7194308321,
                       7194308322, 7194308331, 7194308332, 7194308341, 7194308342)
EXPECTED_TURBO = {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0, "mu": 1.15,
                  "mu_resolution_dependent": False}
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
DEFAULT_TURBO_CHECKPOINT = "/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors"
DOCS_PACKAGE_ROOT = ROOT / "docs/showcase/final/hero-v1"


class HeroShowcaseError(ValueError):
    """Raised when the frozen hero-showcase contract cannot be honored."""


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
        raise HeroShowcaseError(f"Required JSON is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise HeroShowcaseError(f"Required JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise HeroShowcaseError(f"Required JSON object is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return _read_json(path)


def _source_row_key(source_id: str) -> str:
    return "id" if source_id == "batch1-v1" else "row_id"


def validate_manifest(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if manifest.get("format_version") != 1 or manifest.get("kind") != "final_native_hero_showcase_v1":
        raise HeroShowcaseError("Unexpected hero manifest identity")
    if manifest.get("id") != "final-hero-showcase-v1" or Path(manifest.get("output_root", "")) != EXPECTED_OUTPUT_ROOT:
        raise HeroShowcaseError("Hero manifest ID or output root drifted")
    if manifest.get("release_contract") != EXPECTED_RELEASE:
        raise HeroShowcaseError("Hero manifest release contract drifted")
    sources = manifest.get("source_manifests")
    if not isinstance(sources, list) or len(sources) != 2:
        raise HeroShowcaseError("Hero manifest requires exactly two source manifests")
    actual_sources = {item.get("id"): {key: item.get(key) for key in ("path", "sha256")}
                      for item in sources if isinstance(item, dict)}
    if actual_sources != EXPECTED_SOURCES:
        raise HeroShowcaseError("Canonical source manifest provenance drifted")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_WINNERS):
        raise HeroShowcaseError("Hero manifest requires exactly five accepted concepts")
    expected_by_id = {item[0]: item[1:] for item in EXPECTED_WINNERS}
    all_seeds: list[int] = []
    outputs: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "source_manifest", "source_row_id", "original_generation_sidecar", "original_seed", "hero_variants"}:
            raise HeroShowcaseError("Hero row fields drifted")
        expected = expected_by_id.get(row["id"])
        if expected is None:
            raise HeroShowcaseError("Hero concept is not an accepted winner")
        source_id, source_row_id, sidecar_name, original_seed = expected
        if (row["source_manifest"], row["source_row_id"], row["original_seed"]) != (source_id, source_row_id, original_seed):
            raise HeroShowcaseError(f"{row['id']}: accepted winner provenance drifted")
        expected_sidecar = f"docs/showcase/final/{'batch1-v1' if source_id == 'batch1-v1' else 'batch2-v1'}/generations/{sidecar_name}"
        if row["original_generation_sidecar"] != expected_sidecar:
            raise HeroShowcaseError(f"{row['id']}: original winning generation provenance drifted")
        variants = row["hero_variants"]
        if not isinstance(variants, list) or len(variants) != 2:
            raise HeroShowcaseError(f"{row['id']}: exactly two new hero variants are required")
        for suffix, variant in zip(("a", "b"), variants):
            if not isinstance(variant, dict) or set(variant) != {"id", "seed", "output"}:
                raise HeroShowcaseError(f"{row['id']}: hero variant fields drifted")
            expected_id = f"{row['id']}_hero_{suffix}"
            expected_output = f"generations/{expected_id}.png"
            if variant["id"] != expected_id or variant["output"] != expected_output or not isinstance(variant["seed"], int):
                raise HeroShowcaseError(f"{row['id']}: hero variant identity drifted")
            if variant["output"] in outputs:
                raise HeroShowcaseError("Hero output paths must be unique")
            outputs.add(variant["output"])
            all_seeds.append(variant["seed"])
    if tuple(row["id"] for row in rows) != tuple(item[0] for item in EXPECTED_WINNERS):
        raise HeroShowcaseError("Hero concept ordering drifted")
    if tuple(all_seeds) != EXPECTED_HERO_SEEDS or len(set(all_seeds)) != 10:
        raise HeroShowcaseError("Hero seeds drifted or are not unique")
    return rows


def _validate_release() -> None:
    release = ROOT / EXPECTED_RELEASE["path"]
    if sha256(release) != EXPECTED_RELEASE["sha256"]:
        raise HeroShowcaseError("Frozen final release contract SHA-256 mismatch")
    payload = _read_json(release)
    defaults = payload.get("runtime_defaults")
    if not isinstance(defaults, dict) or defaults.get("model") != "Krea-2 Turbo" or defaults.get("steps") != 8 or defaults.get("cfg") != 0.0 or defaults.get("mu") != 1.15 or defaults.get("mu_resolution_dependent") is not False:
        raise HeroShowcaseError("Frozen release runtime contract drifted")
    if defaults.get("control_scale") != {"default": 1.0, "optional_stronger_control_range": [1.25, 1.5]}:
        raise HeroShowcaseError("Frozen release control-scale contract drifted")


def _validate_source_row(source_id: str, row: Mapping[str, Any], source_manifest: Mapping[str, Any]) -> None:
    if source_id == "batch1-v1":
        if row.get("candidate") != "mix-025" or row.get("control_scale") != 1.0 or row.get("style_lora") is not None:
            raise HeroShowcaseError("Batch 1 accepted source row release settings drifted")
        runtime = row.get("runtime")
        if row.get("geometry") != "native_aspect_preserving" or runtime != EXPECTED_TURBO:
            raise HeroShowcaseError("Batch 1 accepted source row geometry/runtime drifted")
    else:
        if (source_manifest.get("candidate") != "mix-025" or source_manifest.get("control_scale") != 1.0
                or source_manifest.get("style_lora") is not None or source_manifest.get("geometry") != "native"
                or source_manifest.get("steps") != 8 or source_manifest.get("cfg") != 0.0
                or source_manifest.get("mu") != 1.15 or row.get("native_bucket") is None):
            raise HeroShowcaseError("Batch 2 accepted source row is missing native geometry")


def _validate_sidecar(sidecar: Mapping[str, Any], source: Mapping[str, Any], *, check_files: bool) -> None:
    bucket = list(source["native_bucket"])
    expected_control = str(Path(source["prepared_control_path"]))
    expected_seed = source["seed"]
    if (sidecar.get("mode") != "turbo-pose-control" or sidecar.get("prompt") != source.get("prompt")
            or sidecar.get("seed") != expected_seed or sidecar.get("pose_image") != expected_control
            or sidecar.get("output_dimensions") != bucket or sidecar.get("width") != bucket[0]
            or sidecar.get("height") != bucket[1] or sidecar.get("geometry_mode") != NATIVE_GEOMETRY):
        raise HeroShowcaseError("Accepted winner generation provenance disagrees with its frozen source row")
    if sidecar.get("steps") != 8 or sidecar.get("cfg") != 0.0 or sidecar.get("mu") != 1.15 or sidecar.get("control_scale") != 1.0:
        raise HeroShowcaseError("Accepted winner generation runtime drifted")
    turbo = sidecar.get("turbo")
    if not isinstance(turbo, dict) or any(turbo.get(key) != value for key, value in EXPECTED_TURBO.items()):
        raise HeroShowcaseError("Accepted winner Turbo provenance drifted")
    style = sidecar.get("style_lora")
    if not isinstance(style, dict) or any(style.get(key) is not None for key in ("name", "path", "sha256", "trigger_phrase")) or style.get("strength") != 0.0 or style.get("effective_prompt") != source["prompt"]:
        raise HeroShowcaseError("Accepted winner must have no Style-LoRA")
    candidate, release = sidecar.get("candidate"), sidecar.get("release")
    if (not isinstance(candidate, dict) or candidate.get("candidate_id") != "mix-025"
            or not isinstance(release, dict) or release.get("sha256") != EXPECTED_RELEASE["sha256"]):
        raise HeroShowcaseError("Accepted winner candidate/release provenance drifted")
    geometry = sidecar.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("bucket") != bucket:
        raise HeroShowcaseError("Accepted winner native geometry drifted")
    if check_files:
        control = Path(expected_control)
        original = Path(str(sidecar.get("output_path", "")))
        if not control.is_file() or sha256(control) != sidecar.get("pose_image_sha256"):
            raise HeroShowcaseError("Accepted winner control is missing or SHA-256 mismatched")
        _verify_image(control, bucket, "accepted winner control")
        if not original.is_file():
            raise HeroShowcaseError("Accepted winner image is missing")
        _verify_image(original, bucket, "accepted winner image")


def _verify_image(path: Path, size: Sequence[int], label: str) -> None:
    try:
        with Image.open(path) as image:
            actual = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        raise HeroShowcaseError(f"Unreadable {label}: {path}") from exc
    if tuple(actual) != tuple(size):
        raise HeroShowcaseError(f"{label} dimensions differ from the frozen native geometry: {path}")


def resolve_winners(manifest: Mapping[str, Any], *, check_files: bool) -> list[dict[str, Any]]:
    rows = validate_manifest(manifest)
    _validate_release()
    source_payloads: dict[str, dict[str, Any]] = {}
    source_seeds: set[int] = set()
    for source_id, source_spec in EXPECTED_SOURCES.items():
        source_path = ROOT / source_spec["path"]
        if sha256(source_path) != source_spec["sha256"]:
            raise HeroShowcaseError(f"Frozen {source_id} manifest SHA-256 mismatch")
        payload = _read_json(source_path)
        if payload.get("candidate") != "mix-025":
            raise HeroShowcaseError(f"Frozen {source_id} manifest candidate drifted")
        if source_id == "batch1-v1" and payload.get("release_contract") != EXPECTED_RELEASE:
            raise HeroShowcaseError("Frozen Batch 1 release contract provenance drifted")
        source_payloads[source_id] = payload
        for source_row in payload.get("rows", []):
            if isinstance(source_row, dict) and isinstance(source_row.get("seed"), int):
                source_seeds.add(source_row["seed"])
    resolved: list[dict[str, Any]] = []
    for hero in rows:
        source_id = hero["source_manifest"]
        key = _source_row_key(source_id)
        matching = [row for row in source_payloads[source_id].get("rows", [])
                    if isinstance(row, dict) and row.get(key) == hero["source_row_id"]]
        if len(matching) != 1:
            raise HeroShowcaseError(f"{hero['id']}: accepted source row does not resolve uniquely")
        source = matching[0]
        _validate_source_row(source_id, source, source_payloads[source_id])
        if source.get("concept") != hero["id"] or source.get("seed") != hero["original_seed"]:
            raise HeroShowcaseError(f"{hero['id']}: source concept or original seed drifted")
        sidecar_path = ROOT / hero["original_generation_sidecar"]
        sidecar = _read_json(sidecar_path)
        _validate_sidecar(sidecar, source, check_files=check_files)
        variants = []
        for variant in hero["hero_variants"]:
            if variant["seed"] in source_seeds:
                raise HeroShowcaseError(f"{variant['id']}: hero seed collides with a Batch 1/2 seed")
            variants.append({**variant, "output_path": str(EXPECTED_OUTPUT_ROOT / variant["output"])})
        resolved.append({
            "id": hero["id"], "prompt": source["prompt"], "control_stem": source["control_stem"],
            "control_path": source["prepared_control_path"], "control_sha256": sidecar["pose_image_sha256"],
            "native_bucket": source["native_bucket"], "original_seed": hero["original_seed"],
            "original_output_path": sidecar["output_path"], "original_generation_sidecar": str(sidecar_path),
            "hero_variants": variants,
        })
    return resolved


def preflight(manifest_path: Path = DEFAULT_MANIFEST, *, check_files: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(manifest_path)
    return manifest, resolve_winners(manifest, check_files=check_files)


def _expected_hero_sidecar(variant: Mapping[str, Any], winner: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt": winner["prompt"], "seed": variant["seed"], "pose_image": winner["control_path"],
        "pose_image_sha256": winner["control_sha256"], "output_path": variant["output_path"],
        "output_dimensions": winner["native_bucket"], "geometry_mode": NATIVE_GEOMETRY,
    }


def validate_hero_output(variant: Mapping[str, Any], winner: Mapping[str, Any]) -> Path:
    output = Path(variant["output_path"])
    sidecar_path = output.with_suffix(".json")
    if not output.is_file() or not sidecar_path.is_file():
        raise HeroShowcaseError(f"Hero output or provenance sidecar is missing: {output}")
    _verify_image(output, winner["native_bucket"], "hero generation")
    sidecar = _read_json(sidecar_path)
    expected = _expected_hero_sidecar(variant, winner)
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise HeroShowcaseError(f"Hero output provenance mismatch: {output}")
    if sidecar.get("steps") != 8 or sidecar.get("cfg") != 0.0 or sidecar.get("mu") != 1.15 or sidecar.get("control_scale") != 1.0:
        raise HeroShowcaseError(f"Hero output runtime mismatch: {output}")
    turbo = sidecar.get("turbo")
    geometry = sidecar.get("geometry")
    if (not isinstance(turbo, dict) or any(turbo.get(key) != value for key, value in EXPECTED_TURBO.items())
            or not isinstance(geometry, dict) or geometry.get("bucket") != winner["native_bucket"]):
        raise HeroShowcaseError(f"Hero output Turbo/native geometry provenance mismatch: {output}")
    if sidecar.get("candidate", {}).get("candidate_id") != "mix-025" or sidecar.get("release", {}).get("sha256") != EXPECTED_RELEASE["sha256"]:
        raise HeroShowcaseError(f"Hero output release provenance mismatch: {output}")
    style = sidecar.get("style_lora")
    if not isinstance(style, dict) or style.get("name") is not None or style.get("path") is not None or style.get("strength") != 0.0 or style.get("effective_prompt") != winner["prompt"]:
        raise HeroShowcaseError(f"Hero output unexpectedly uses Style-LoRA: {output}")
    return sidecar_path


def generate(manifest_path: Path, turbo_checkpoint: Path) -> None:
    manifest, winners = preflight(manifest_path, check_files=True)
    if not turbo_checkpoint.is_file():
        raise HeroShowcaseError(f"Turbo checkpoint is missing: {turbo_checkpoint}")
    for winner in winners:
        for variant in winner["hero_variants"]:
            output = Path(variant["output_path"])
            sidecar = output.with_suffix(".json")
            if output.exists() or sidecar.exists():
                validate_hero_output(variant, winner)
                continue
            command = [sys.executable, str(ROOT / "inference.py"), "--turbo-ckpt", str(turbo_checkpoint),
                       "--candidate", "mix-025", "--prompt", winner["prompt"], "--pose-image", winner["control_path"],
                       "--output", str(output), "--seed", str(variant["seed"]), "--control-scale", "1.0",
                       "--steps", "8", "--cfg", "0", "--mu", "1.15"]
            subprocess.run(command, cwd=ROOT, check=True)
            validate_hero_output(variant, winner)
    write_summary(manifest_path, manifest=manifest, winners=winners)


def _all_output_sidecars(winners: Sequence[Mapping[str, Any]]) -> list[Path]:
    return [validate_hero_output(variant, winner) for winner in winners for variant in winner["hero_variants"]]


def write_summary(manifest_path: Path, *, manifest: Mapping[str, Any] | None = None,
                  winners: Sequence[Mapping[str, Any]] | None = None) -> Path:
    if manifest is None or winners is None:
        manifest, winners = preflight(manifest_path, check_files=True)
    _all_output_sidecars(winners)
    payload = {
        "format_version": 1, "kind": "final_hero_showcase_provenance_v1",
        "hero_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "release_contract": EXPECTED_RELEASE,
        "source_manifests": EXPECTED_SOURCES,
        "runtime": {"candidate": "mix-025", "control_scale": 1.0, "turbo": EXPECTED_TURBO,
                    "style_lora": None, "geometry": NATIVE_GEOMETRY},
        "concepts": [{
            "id": winner["id"], "prompt": winner["prompt"], "control_stem": winner["control_stem"],
            "control_path": winner["control_path"], "control_sha256": winner["control_sha256"],
            "native_bucket": winner["native_bucket"], "original_seed": winner["original_seed"],
            "original_output_path": winner["original_output_path"],
            "original_generation_sidecar": winner["original_generation_sidecar"],
            "hero_variants": winner["hero_variants"],
        } for winner in winners],
    }
    path = Path(manifest["output_root"]) / "hero_provenance.json"
    _write_json(path, payload)
    return path


def review_sheet(manifest_path: Path) -> Path:
    manifest, winners = preflight(manifest_path, check_files=True)
    _all_output_sidecars(winners)
    thumb_w, thumb_h, header, gutter = 280, 360, 26, 4
    sheet = Image.new("RGB", (4 * thumb_w, header + len(winners) * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(("condition", "original accepted winner", "hero variant A", "hero variant B")):
        draw.text((column * thumb_w + gutter, gutter), label, fill="black")
    for row_index, winner in enumerate(winners):
        y = header + row_index * thumb_h
        paths = [Path(winner["control_path"]), Path(winner["original_output_path"])] + [Path(v["output_path"]) for v in winner["hero_variants"]]
        for column, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((thumb_w - 2 * gutter, thumb_h - 2 * gutter))
            x = column * thumb_w + (thumb_w - image.width) // 2
            sheet.paste(image, (x, y + (thumb_h - image.height) // 2))
        draw.text((gutter, y + gutter), winner["id"], fill="black", stroke_width=1, stroke_fill="white")
    path = Path(manifest["output_root"]) / "review_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    write_summary(manifest_path, manifest=manifest, winners=winners)
    return path


def _copy_if_matching(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise HeroShowcaseError(f"Cannot package missing file: {source}")
    if destination.exists():
        if not destination.is_file() or sha256(destination) != sha256(source):
            raise HeroShowcaseError(f"Refusing to overwrite conflicting docs package file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_to_docs(manifest_path: Path) -> Path:
    manifest, winners = preflight(manifest_path, check_files=True)
    review = review_sheet(manifest_path)
    summary = write_summary(manifest_path, manifest=manifest, winners=winners)
    _copy_if_matching(manifest_path, DOCS_PACKAGE_ROOT / manifest_path.name)
    _copy_if_matching(review, DOCS_PACKAGE_ROOT / review.name)
    _copy_if_matching(summary, DOCS_PACKAGE_ROOT / summary.name)
    for winner in winners:
        concept = winner["id"]
        _copy_if_matching(Path(winner["control_path"]), DOCS_PACKAGE_ROOT / "conditions" / f"{concept}.png")
        _copy_if_matching(Path(winner["original_output_path"]), DOCS_PACKAGE_ROOT / "original_winners" / f"{concept}.png")
        _copy_if_matching(Path(winner["original_generation_sidecar"]), DOCS_PACKAGE_ROOT / "original_winners" / f"{concept}.json")
        for variant in winner["hero_variants"]:
            output = Path(variant["output_path"])
            _copy_if_matching(output, DOCS_PACKAGE_ROOT / "hero_variants" / output.name)
            _copy_if_matching(output.with_suffix(".json"), DOCS_PACKAGE_ROOT / "hero_variants" / output.with_suffix(".json").name)
    return DOCS_PACKAGE_ROOT


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "generate", "review-sheet", "summary", "copy-to-docs"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--turbo-ckpt", type=Path, default=Path(DEFAULT_TURBO_CHECKPOINT))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        _, winners = preflight(args.manifest, check_files=True)
        print(f"PASS: {len(winners)} accepted hero concepts and 10 new deterministic variants")
    elif args.command == "generate":
        generate(args.manifest, args.turbo_ckpt)
    elif args.command == "review-sheet":
        print(review_sheet(args.manifest))
    elif args.command == "summary":
        print(write_summary(args.manifest))
    else:
        print(copy_to_docs(args.manifest))


if __name__ == "__main__":  # pragma: no cover
    main()
