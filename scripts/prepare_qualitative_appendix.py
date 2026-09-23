"""Prepare and, on a CUDA-visible GH200 shell, generate qualitative blog assets.

The script deliberately has two phases.  ``--prepare-only`` validates pinned
inputs, copies immutable frozen-evaluation selections, and writes manifests.
``--generate-appendix`` additionally runs exactly the 48 planned native-geometry
appendix samples using the canonical public inference entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from pose_controlnet.paired_preprocessing import REFERENCE_KREA_BUCKETS, choose_bucket


ROOT = Path(__file__).resolve().parents[1]
NFS = Path("/lambda/nfs/adhit/krea2-pose")
RELEASE = NFS / "release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors"
TURBO = NFS / "models/krea-2-turbo/turbo.safetensors"
RAW = NFS / "models/krea-2-raw/raw.safetensors"
RELEASE_SHA256 = "6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1"
APPENDIX = ROOT / "docs/blog-assets/appendix-v1"
QUAL = ROOT / "docs/blog-assets/qualitative"
NATIVE = NFS / "evaluation/native-vs-dynamic768/mix-025-control1-turbo-v1"
HARD = NFS / "evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1"

RUNTIME = {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0, "mu": 1.15,
           "control_scale": 1.0, "style_lora": None}
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"

# These are deliberately not person-count instructions.  The skeleton remains
# authoritative for count, framing, and body geometry in every family.
STYLE_PROMPTS = {
    "cinematic_fantasy_realism": (
        "Cinematic fantasy realism, elegant emerald wool garments with aged brass filigree "
        "and midnight-blue textiles, a ruined mountain observatory at storm-softened dawn, "
        "natural skin texture, tactile fabric and metal detail, restrained jewel tones, immersive filmic atmosphere."
    ),
    "painterly_storybook_fantasy": (
        "Painterly storybook fantasy, richly painted russet, cream, and moss-green garments, "
        "an enchanted orchard of lantern flowers and ancient stone markers, honeyed late-afternoon light, "
        "expressive brushwork, visible paper texture, warm lyrical color."
    ),
    "polished_modern_anime": (
        "Polished modern anime, sleek black-and-coral technical apparel with translucent accents, "
        "a rain-glossed neon botanical conservatory, cool cyan reflections and magenta rim light, "
        "crisp cel shading, refined linework, luminous contemporary color design."
    ),
    "editorial_high_fashion_photography": (
        "Editorial high-fashion photography, sculptural slate tailoring with silver jewelry and matte silk, "
        "a minimalist travertine gallery with sculpted light, soft directional studio illumination, "
        "premium medium-format detail, subtle film grain, quiet modern luxury."
    ),
}

# Eight singles: varied hero controls plus seated/crouched/arm-articulation
# frozen hard-pose controls.  Four multi/duo controls include two clean hero
# duos and two deliberately difficult frozen stress examples.
APPENDIX_CONTROLS = (
    ("single_01_female_warrior", "single", 1,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/realistic-female-warrior.png",
     "hero-v2 canonical-v1 / realistic female warrior"),
    ("single_02_wandering_knight", "single", 1,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/elegant-male-warrior-wandering-knight.png",
     "hero-v2 canonical-v1 / elegant male warrior"),
    ("single_03_floral_oracle", "single", 1,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/moonlit-priestess-dreamy-floral-oracle.png",
     "hero-v2 canonical-v1 / moonlit priestess"),
    ("single_04_stained_glass_saint", "single", 1,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/stained-glass-saint-celestial-figure.png",
     "hero-v2 canonical-v1 / stained-glass saint"),
    ("single_05_editorial", "single", 1,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/realistic-fashion-editorial-portrait.png",
     "hero-v2 canonical-v1 / realistic fashion editorial"),
    ("single_06_seated", "single", 1, HARD / "controls/coco_268556_2203816.png",
     "hard-pose-multiperson-mix-025-v1 / 01_seated"),
    ("single_07_crouched", "single", 1, HARD / "controls/real_human_humanart_15000000000477.png",
     "hard-pose-multiperson-mix-025-v1 / 02_crouched"),
    ("single_08_overhead_arms", "single", 1, HARD / "controls/coco_104715_474045.png",
     "hard-pose-multiperson-mix-025-v1 / 04_overhead_arms"),
    ("multi_01_gothic_duo", "duo", 2,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/gothic-masked-noble-with-attendant.png",
     "hero-v2 canonical-v1 / gothic masked noble with attendant"),
    ("multi_02_mythic_duo", "duo", 2,
     ROOT / "docs/showcase/final/hero-v2/generations/canonical-v1/controls/painterly-mythic-companions.png",
     "hero-v2 canonical-v1 / painterly mythic companions"),
    ("multi_03_overlapping_pair", "multi", 2, HARD / "controls/real_human_humanart_15000000001893.png",
     "hard-pose-multiperson-mix-025-v1 / 10_overlapping_two_person_pose"),
    ("multi_04_dense_group", "multi", 7, HARD / "controls/coco_374270_crowd.png",
     "hard-pose-multiperson-mix-025-v1 / 12_crowd_dense_group"),
)

NATIVE_SELECTIONS = (
    ("example_01", "real_human_humanart_15000000000521", "portrait / inversion stress"),
    ("example_02", "real_human_humanart_17000000002207", "multi-person stage-dance stress"),
)
HARD_SELECTIONS = (
    ("single_01", "06_strong_foreshortening"),
    ("single_02", "07_inversion"),
    ("multi_01", "09_two_person_interaction"),
    ("multi_02", "10_overlapping_two_person_pose"),
)

# Geometry verbs/terms that must not appear in fresh appendix prompts.  The
# evaluation prompts retained verbatim in frozen B/C manifests are historical
# source evidence, not new appendix prompting.
POSE_WORDS = re.compile(
    r"\b(stand(?:ing)?|walk(?:ing)?|sit(?:ting)?|seated|kneel(?:ing)?|crouch(?:ing)?|"
    r"lean(?:ing)?|arm|arms|leg|legs|hand|hands|foot|feet|limb|limbs|joint|joints|"
    r"pose|body\s+(?:position|geometry)|overhead|inversion|airborne)\b", re.I)


class AppendixError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AppendixError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required source is missing: {source}")
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise AppendixError(f"Refusing to overwrite non-identical asset: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def verify_png(path: Path, expected_size: tuple[int, int] | None = None) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        raise AppendixError(f"Invalid PNG: {path}") from exc
    if expected_size is not None and size != expected_size:
        raise AppendixError(f"Unexpected PNG size for {path}: {size} != {expected_size}")
    return size


def stable_seed(control_id: str, family: str, retry: int = 0) -> int:
    payload = f"appendix-v1::{control_id}::{family}::{retry}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def resolve_native_appendix_dimensions(source_size: tuple[int, int]) -> tuple[int, int]:
    """Resolve a standalone control through the frozen native bucket policy.

    Native evaluation persists this same nearest-aspect bucket with each cached
    latent.  Appendix controls are copied rasters rather than those cached
    records, so they must be resolved through the shared training/evaluation
    selector before they reach inference.  This is intentionally not the
    optional dynamic-768 policy.
    """
    try:
        return choose_bucket(source_size, REFERENCE_KREA_BUCKETS)
    except ValueError as exc:
        raise AppendixError(f"Invalid appendix control dimensions: {source_size!r}") from exc


def validate_inputs() -> None:
    for source in (RELEASE, TURBO, RAW, NATIVE / "native_dynamic_provenance.json",
                   NATIVE / "generation_results.json", NATIVE / "pck_clip_results.json",
                   NATIVE / "compact_summary.json", HARD / "hard_pose_provenance.json",
                   HARD / "generation_results.json", HARD / "pck_clip_results.json",
                   HARD / "compact_summary.json"):
        if not source.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {source}")
    if sha256(RELEASE) != RELEASE_SHA256:
        raise AppendixError("Release artifact SHA-256 differs from the frozen v1 release")
    if len(APPENDIX_CONTROLS) != 12 or sum(row[2] == 1 for row in APPENDIX_CONTROLS) != 8:
        raise AppendixError("Appendix selection must be exactly 8 single and 4 multi/duo controls")
    if sum(row[2] > 1 for row in APPENDIX_CONTROLS) != 4:
        raise AppendixError("Appendix selection must have exactly 4 multi/duo controls")
    for prompt in STYLE_PROMPTS.values():
        match = POSE_WORDS.search(prompt)
        if match:
            raise AppendixError(f"Appendix prompt violates geometry-neutral prompting rule: {match.group(0)!r}")
    for _, _, _, source, _ in APPENDIX_CONTROLS:
        verify_png(source)


def appendix_records() -> list[dict[str, Any]]:
    records = []
    seen_outputs: set[str] = set()
    for control_id, control_type, people, source, provenance in APPENDIX_CONTROLS:
        copied = APPENDIX / "controls" / f"{control_id}.png"
        source_size = verify_png(source)
        size = resolve_native_appendix_dimensions(source_size)
        copy_exact(source, copied)
        for family, prompt in STYLE_PROMPTS.items():
            output = APPENDIX / "generations" / f"{control_id}__{family}.png"
            relative = str(output.relative_to(ROOT))
            if relative in seen_outputs:
                raise AppendixError(f"Duplicate output path: {relative}")
            seen_outputs.add(relative)
            completed = output.is_file()
            stale_geometry = None
            if completed:
                actual_size = verify_png(output)
                if actual_size != size:
                    # These are appendix outputs produced before their copied
                    # raw control was resolved through the native bucket
                    # policy.  They are not valid completed samples, but are
                    # retained until the explicit CUDA generation replaces
                    # them.  Frozen evaluation/blog assets are not in this
                    # output tree and are never regenerated here.
                    completed = False
                    stale_geometry = f"stale appendix output geometry {actual_size}; expected {size}"
            records.append({
                "id": f"{control_id}__{family}", "control_id": control_id,
                "control_type": control_type, "control_source_path": str(source),
                "copied_control_path": str(copied.relative_to(ROOT)),
                "source_evaluation_showcase_provenance": provenance,
                "person_count": people, "style_family": family, "prompt": prompt,
                "source_width": source_size[0], "source_height": source_size[1],
                "seed": stable_seed(control_id, family), "width": size[0], "height": size[1],
                "geometry_mode": NATIVE_GEOMETRY, **RUNTIME,
                "release_artifact": {"path": str(RELEASE), "sha256": RELEASE_SHA256},
                "raw_provenance_path": str(RAW), "output_path": relative,
                "success": True if completed else None,
                "failure": stale_geometry, "retry_of": None,
                "status": "success" if completed else "pending_cuda",
            })
    if len(records) != 48:
        raise AppendixError("Appendix plan must be exactly 12 controls x 4 families")
    return records


def discard_stale_appendix_output(output: Path, row: dict[str, Any]) -> None:
    """Remove only a known wrong-geometry appendix output before replacement."""
    if not row["failure"] or not row["failure"].startswith("stale appendix output geometry"):
        raise AppendixError(f"Refusing to replace appendix output without stale-geometry proof: {output}")
    output.unlink()
    output.with_suffix(".json").unlink(missing_ok=True)


def build_sheet(rows: list[tuple[str, list[Path]]], destination: Path, labels: tuple[str, ...],
                title: str, thumb: tuple[int, int] = (240, 240)) -> None:
    """Create a labelled letterbox contact sheet without cropping images."""
    font = ImageFont.load_default()
    margin, header, row_label, label_height = 24, 42, 240, 28
    cols = len(labels)
    width = margin * 2 + row_label + cols * thumb[0] + (cols - 1) * margin
    height = header + margin + len(rows) * (thumb[1] + label_height + margin)
    sheet = Image.new("RGB", (width, height), "#101017")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 13), title, fill="#f3eeff", font=font)
    for col, label in enumerate(labels):
        x = margin + row_label + col * (thumb[0] + margin)
        draw.text((x, header), label, fill="#d9ccff", font=font)
    for row, (name, images) in enumerate(rows):
        y = header + margin + row * (thumb[1] + label_height + margin)
        draw.text((margin, y + 8), name, fill="#f3eeff", font=font)
        for col, path in enumerate(images):
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = margin + row_label + col * (thumb[0] + margin) + (thumb[0] - image.width) // 2
            iy = y + (thumb[1] - image.height) // 2
            sheet.paste(image, (x, iy))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def native_assets() -> int:
    output = QUAL / "native-vs-dynamic"
    results, provenance = read_json(NATIVE / "generation_results.json"), read_json(NATIVE / "native_dynamic_provenance.json")
    conditions = {row["stem"]: row for row in results["conditions"]}
    artifacts = results["generated_artifacts"]
    rows, manifest_rows = [], []
    for number, stem, selection_note in NATIVE_SELECTIONS:
        condition = conditions[stem]
        control = NATIVE / "controls" / f"{stem}.png"
        native = NATIVE / artifacts[stem][NATIVE_GEOMETRY]
        dynamic = NATIVE / artifacts[stem]["dynamic_768_bucket"]
        destinations = [output / f"{number}_control.png", output / f"{number}_native.png", output / f"{number}_dynamic.png"]
        for source, destination in zip((control, native, dynamic), destinations): copy_exact(source, destination); verify_png(destination)
        rows.append((f"{number}\n{selection_note}", destinations))
        manifest_rows.append({"id": number, "stem": stem, "selection_note": selection_note,
                              "control_source_path": str(control), "native_source_path": str(native),
                              "dynamic_source_path": str(dynamic), "control_path": str(destinations[0].relative_to(ROOT)),
                              "native_path": str(destinations[1].relative_to(ROOT)), "dynamic_path": str(destinations[2].relative_to(ROOT)),
                              "prompt": condition["prompt"], "seed": condition["sampling_seed"],
                              "native_bucket": condition["native_bucket"], "dynamic_768_bucket": condition["dynamic_768_bucket"],
                              "control_scale": 1.0, "runtime": RUNTIME,
                              "source_provenance": str(NATIVE / "native_dynamic_provenance.json")})
    build_sheet(rows, output / "comparison_sheet.png", ("control", "native", "dynamic-768"),
                "Native cached geometry vs dynamic-768")
    write_json(output / "manifest.json", {"kind": "targeted_native_vs_dynamic_blog_examples", "reused_frozen_evaluation_generations": True,
                                            "frozen_provenance": str(NATIVE / "native_dynamic_provenance.json"),
                                            "release_artifact": {"path": str(RELEASE), "sha256": RELEASE_SHA256}, "examples": manifest_rows})
    return 4


def hard_assets() -> int:
    output = QUAL / "hard-pose-single-multi"
    results, provenance = read_json(HARD / "generation_results.json"), read_json(HARD / "hard_pose_provenance.json")
    conditions = {row["condition_id"]: row for row in results["conditions"]}
    artifacts = results["generated_artifacts"]
    rows, manifest_rows = [], []
    for output_id, condition_id in HARD_SELECTIONS:
        condition = conditions[condition_id]
        control = HARD / "controls" / f"{condition['stem']}.png"
        generated = HARD / artifacts[condition_id]
        destinations = [output / f"{output_id}_control.png", output / f"{output_id}_output.png"]
        for source, destination in zip((control, generated), destinations): copy_exact(source, destination); verify_png(destination)
        rows.append((f"{output_id}\n{condition['pose_class']}", destinations))
        manifest_rows.append({"id": output_id, "condition_id": condition_id, "stem": condition["stem"],
                              "hard_pose_category": condition["hard_pose_category"], "person_group": condition["person_group"],
                              "person_count": condition["expected_reference_people"], "prompt": condition["prompt"], "seed": condition["seed"],
                              "control_source_path": str(control), "output_source_path": str(generated),
                              "control_path": str(destinations[0].relative_to(ROOT)), "output_path": str(destinations[1].relative_to(ROOT)),
                              "native_bucket": condition["native_bucket"], "runtime": RUNTIME,
                              "source_provenance": str(HARD / "hard_pose_provenance.json")})
    build_sheet(rows, output / "comparison_sheet.png", ("control", "generated"),
                "Hard pose: rows 1–2 single-person | rows 3–4 multi-person")
    write_json(output / "manifest.json", {"kind": "targeted_hard_pose_single_multi_blog_examples", "reused_frozen_evaluation_generations": True,
                                            "frozen_provenance": str(HARD / "hard_pose_provenance.json"),
                                            "release_artifact": {"path": str(RELEASE), "sha256": RELEASE_SHA256}, "examples": manifest_rows})
    return 4


def appendix_manifest(records: list[dict[str, Any]]) -> None:
    payload = {"kind": "appendix_v1_pose_transfer", "generation_count": 48,
               "selected_control_count": 12, "single_person_control_count": 8, "multi_duo_control_count": 4,
               "style_prompts": STYLE_PROMPTS, "runtime": RUNTIME, "geometry_mode": NATIVE_GEOMETRY,
               "release_artifact": {"path": str(RELEASE), "sha256": RELEASE_SHA256}, "raw_provenance_path": str(RAW),
               "records": records}
    write_json(APPENDIX / "manifest.json", payload)
    readme = """# Appendix v1

Twelve validated project controls × four geometry-neutral appearance families.
The skeleton is the sole body-geometry source; prompts only specify visual
treatment.  The frozen runtime is Krea-2 Turbo, 8 steps, CFG 0, mu 1.15,
control scale 1.0, no Style-LoRA, and native/aspect-preserving geometry.

`manifest.json` is the source of reproducibility facts.  A `pending_cuda`
record has been validated but requires execution from a CUDA-visible GH200
shell; it is not an implicit retry or a completed generation.
"""
    (APPENDIX / "README.md").write_text(readme, encoding="utf-8")


def generate_appendix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Appendix generation requires the CUDA-visible GH200 host shell")
    from inference import (CANONICAL_CANDIDATE, NATIVE_GEOMETRY as INFERENCE_NATIVE, PoseInferenceRequest,
                           generate_pose, load_inference_runtime, resolve_pose_candidate)
    first = next(row for row in records if row["status"] != "success")
    request = PoseInferenceRequest(turbo_checkpoint=TURBO, pose_lora_checkpoint=None, release_artifact=RELEASE,
                                  candidate=CANONICAL_CANDIDATE, prompt=first["prompt"],
                                  pose_image=ROOT / first["copied_control_path"], output=ROOT / first["output_path"],
                                  seed=first["seed"], steps=8, cfg=0.0, mu=1.15, control_scale=1.0,
                                  width=first["width"], height=first["height"],
                                  geometry_mode=INFERENCE_NATIVE)
    runtime = load_inference_runtime(request, candidate=resolve_pose_candidate(request))
    for row in records:
        output, control = ROOT / row["output_path"], ROOT / row["copied_control_path"]
        if output.is_file():
            if row["status"] == "success":
                verify_png(output, (row["width"], row["height"])); row.update(success=True, failure=None, status="success"); continue
            discard_stale_appendix_output(output, row)
        request = PoseInferenceRequest(turbo_checkpoint=TURBO, pose_lora_checkpoint=None, release_artifact=RELEASE,
                                      candidate=CANONICAL_CANDIDATE, prompt=row["prompt"], pose_image=control, output=output,
                                      seed=row["seed"], steps=8, cfg=0.0, mu=1.15, control_scale=1.0,
                                      width=row["width"], height=row["height"],
                                      geometry_mode=INFERENCE_NATIVE)
        try:
            generate_pose(request, runtime=runtime)
            verify_png(output, (row["width"], row["height"])); row.update(success=True, failure=None, status="success")
        except Exception as exc:
            # No silent retry: preserve the failed primary record and stop so a
            # human can classify whether this is a technical corruption eligible
            # for the single documented retry.
            row.update(success=False, failure=f"{type(exc).__name__}: {exc}", status="failed")
            appendix_manifest(records)
            raise
        appendix_manifest(records)
    return records


def appendix_sheet(records: list[dict[str, Any]]) -> None:
    if any(row["status"] != "success" for row in records): return
    grouped = []
    for control_id, _, _, _, _ in APPENDIX_CONTROLS:
        images = [ROOT / next(row["output_path"] for row in records if row["control_id"] == control_id and row["style_family"] == family)
                  for family in STYLE_PROMPTS]
        grouped.append((control_id, images))
    build_sheet(grouped, APPENDIX / "contact_sheet.png", tuple(STYLE_PROMPTS), "Appendix v1: control transfer across four appearance families", thumb=(190, 190))


def write_checksums() -> None:
    targets = [APPENDIX, QUAL / "native-vs-dynamic", QUAL / "hard-pose-single-multi"]
    files = sorted(path for target in targets for path in target.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in files]
    (ROOT / "docs/blog-assets/SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-appendix", action="store_true", help="run the 48 CUDA generations after preparation")
    args = parser.parse_args()
    validate_inputs()
    records = appendix_records()
    appendix_manifest(records)
    reused = native_assets() + hard_assets()
    if args.generate_appendix:
        records = generate_appendix(records)
    appendix_sheet(records)
    write_checksums()
    print(json.dumps({"appendix_success": sum(row["status"] == "success" for row in records),
                      "appendix_pending": sum(row["status"] == "pending_cuda" for row in records),
                      "reused_evaluation_outputs": reused, "appendix": str(APPENDIX), "qualitative": str(QUAL)}, indent=2))


if __name__ == "__main__":
    main()
