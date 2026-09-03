"""Isolated frozen English/Chinese/Telugu Turbo prompt smoke test.

This v2 experiment deliberately reuses the established scoring and generation
path without changing the completed English-versus-Chinese smoke experiment.
It produces exactly three images for one fixed pose and never uses source RGB.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from scripts import chinese_prompt_smoke as legacy
from scripts import final_val_turbo_benchmark as final_val


SMOKE_FILE = Path("docs/evaluation/prompting-guide/multilingual_prompt_smoke_v2.jsonl")
SMOKE_SHA256 = "bc178f1e6c0559b3bfc92c7d48edbdd2a825e9451eb9d230aed794edaf23d9e5"
STEM = "sculpture_humanart_14000000003803"
LANGUAGES = ("en", "zh", "te")
SMOKE_KIND = "prompting_guide_english_chinese_telugu_turbo_fixed_pose_v2"
GENERATION_COUNT = 3
PROMPTS = {
    "en": "A single adult woman wearing a simple cream outfit in a quiet botanical courtyard, soft overcast daylight, natural textures.",
    "zh": "一位成年女性，穿着简洁的奶油色服装，身处安静的植物庭院中，柔和的阴天天光，自然真实的材质质感。",
    "te": "ఒక వయోజన మహిళ సరళమైన క్రీమ్ రంగు దుస్తులు ధరించి, నిశ్శబ్దమైన బొటానికల్ ప్రాంగణంలో ఉంది, మృదువైన మేఘావృత దినకాంతి, సహజమైన వాస్తవిక పదార్థాల స్పర్శ.",
}


@contextmanager
def _v2_contract() -> Iterator[None]:
    """Temporarily select v2 in the shared, already-frozen smoke mechanics.

    The legacy module reads its small contract constants at execution time.
    This scoped substitution leaves its default EN/ZH contract and artifacts
    byte-for-byte untouched while retaining the exact established runtime,
    PCK, and CLIP behavior for the v2 rows.
    """
    names = ("SMOKE_FILE", "SMOKE_SHA256", "STEM", "LANGUAGES", "SMOKE_KIND", "GENERATION_COUNT")
    saved = {name: getattr(legacy, name) for name in names}
    values = {
        "SMOKE_FILE": SMOKE_FILE,
        "SMOKE_SHA256": SMOKE_SHA256,
        "STEM": STEM,
        "LANGUAGES": LANGUAGES,
        "SMOKE_KIND": SMOKE_KIND,
        "GENERATION_COUNT": GENERATION_COUNT,
    }
    try:
        for name, value in values.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)


def _sha256(path: Path) -> str:
    return legacy._sha256(path)


def load_smoke_rows(path: str | Path = SMOKE_FILE) -> list[dict[str, str]]:
    with _v2_contract():
        rows = legacy.load_smoke_rows(path)
    if tuple(row["language"] for row in rows) != LANGUAGES or any(row["prompt"] != PROMPTS[row["language"]] for row in rows):
        raise ValueError("Multilingual v2 rows do not match the exact frozen EN/ZH/TE prompts")
    return rows


def _generation_status(output: Path, rows: list[dict[str, str]], candidate: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    with _v2_contract():
        return legacy._generation_status(output, rows, candidate, contract)


def _score_records(payload: Mapping[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    with _v2_contract():
        return legacy._score_records(payload, rows)


def preflight(args: argparse.Namespace) -> None:
    with _v2_contract():
        legacy.preflight(args)


def generate(args: argparse.Namespace) -> None:
    with _v2_contract():
        legacy.generate(args)


def score(args: argparse.Namespace) -> None:
    # Existing CLIP prompt handling is intentionally used unchanged for all
    # three UTF-8 prompts; no language-specific metric behavior is introduced.
    with _v2_contract():
        legacy.score(args)


def report(args: argparse.Namespace) -> None:
    with _v2_contract():
        rows = legacy.load_smoke_rows(args.prompt_file)
        contract, _, _, candidate, _, training, output = legacy._inputs(args, rows)
        if legacy._generation_status(output, rows, candidate, contract) != "complete":
            raise FileNotFoundError("Multilingual v2 report requires the complete validated three-image generation set")
        records = legacy._validated_scores(output, contract, rows)
        by_language = {language: legacy._compact([record for record in records if record["language"] == language]) for language in LANGUAGES}
        if set(by_language) != set(LANGUAGES) or any(value["generation_count"] != 1 for value in by_language.values()):
            raise ValueError("Multilingual v2 scores do not contain exactly one EN/ZH/TE record")
        legacy._write(output / "metrics_by_language.json", {**contract, "group_by": "language", "rows": [{"language": language, **by_language[language]} for language in LANGUAGES]})
        paths = [output / "controls" / f"{STEM}.png"] + [legacy._directory(output, language) / final_val._image_name(candidate) for language in LANGUAGES]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("Multilingual v2 comparison requires the control and all three generations; no RGB fallback exists")
        legacy.make_contact_sheet([(STEM, paths)], output / "multilingual_comparison.png", thumbnail_width=320, thumbnail_height=320,
                                  column_labels=("pose control", "English", "Chinese", "Telugu"))
        legacy._write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "generation_count": GENERATION_COUNT,
                      "score_artifact": "pck_clip_results.json", "metrics_by_language": "metrics_by_language.json",
                      "comparison": "multilingual_comparison.png", "source_rgb_fallback_used": False,
                      "clip_behavior": "unchanged existing CLIP implementation applied to all three UTF-8 prompts"})
        print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    with _v2_contract():
        rows = legacy.load_smoke_rows(args.prompt_file)
        contract, _, _, candidate, _, _, output = legacy._inputs(args, rows)
        if legacy._generation_status(output, rows, candidate, contract) != "complete":
            raise FileNotFoundError("Multilingual v2 summary requires the complete validated three-image generation set")
        records = legacy._validated_scores(output, contract, rows)
        compact = {"candidate": candidate["label"], "stem": STEM, "sampling_seed": contract["sampling_seed"],
                   "generation_count": GENERATION_COUNT, "by_language": {language: legacy._compact([record for record in records if record["language"] == language]) for language in LANGUAGES}}
        legacy._write(output / "compact_summary.json", compact)
        print(json.dumps(compact, ensure_ascii=False, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "generate", "score", "report", "summary"))
    parser.add_argument("--candidate", choices=("mix-025",), default="mix-025")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC))
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    parser.add_argument("--dataset-root")
    parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
    parser.add_argument("--reference-sidecar", help="required only for score")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt-file", default=str(SMOKE_FILE))
    return parser


def main() -> None:
    args = parser().parse_args()
    {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__":
    main()
