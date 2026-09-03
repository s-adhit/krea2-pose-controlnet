"""Isolated frozen English-versus-Chinese Turbo prompt smoke test.

This tool is deliberately independent of the 64-row prompting-guide study.
It produces exactly one English and one Chinese image for one frozen pose;
the source RGB is never read, copied, or used as a fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.evaluation import _sample_by_stem, make_contact_sheet, save_image
from pose_controlnet.post1500_evaluation import _pool_pose, score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val
from scripts import prompting_guide_study as guide


SMOKE_FILE = Path("docs/evaluation/prompting-guide/chinese_prompt_smoke.jsonl")
SMOKE_SHA256 = "c782d6fecff1bc6393f9175a52cb9b66f11185dcf0a3a3c8cccf1ab3a095769e"
STEM = "sculpture_humanart_14000000003803"
LANGUAGES = ("en", "zh")
SMOKE_KIND = "prompting_guide_english_vs_chinese_turbo_fixed_pose"
GENERATION_COUNT = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required language-smoke artifact is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Language-smoke JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Language-smoke JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_smoke_rows(path: str | Path = SMOKE_FILE) -> list[dict[str, str]]:
    """Read exact UTF-8 rows and reject every contract or byte drift."""
    source = Path(path)
    if _sha256(source) != SMOKE_SHA256:
        raise ValueError(f"Frozen English-vs-Chinese prompt SHA-256 mismatch: {source}")
    try:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Required frozen language-smoke file is missing: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Frozen language-smoke file contains invalid JSON: {source}") from exc
    if (len(rows) != GENERATION_COUNT or any(not isinstance(row, dict) or set(row) != {"language", "stem", "prompt"} for row in rows)
            or tuple(row.get("language") for row in rows) != LANGUAGES
            or any(row.get("stem") != STEM for row in rows)
            or any(not isinstance(row.get("prompt"), str) or not row["prompt"] for row in rows)):
        raise ValueError("Language smoke must contain exactly ordered en and zh rows for the one frozen stem")
    return rows  # typing: exact schema checked above


def _candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    return guide._candidate_contract(candidate, checkpoint)


def _inputs(args: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[dict[str, Any], PreparedLatentShardDataset, Path, dict[str, Any], Path | None, dict[str, Any], Path]:
    guide._validate_locked_runtime_contract()
    spec, spec_digest = final_val.load_final_spec(args.final_spec)
    if STEM not in spec["stems"] or not isinstance(spec["per_stem_seeds"].get(STEM, {}).get("sampling"), int):
        raise ValueError("Frozen language-smoke stem lacks a frozen final-val sampling seed")
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, spec)
    shard_metadata = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shard_metadata.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Language smoke requires --dataset-root or latent shards.json.dataset_root")
    control = final_val.resolve_final_controls(dataset_root, [STEM]).get(STEM)
    if control is None:
        raise ValueError("Language-smoke control resolution did not return the frozen stem")
    candidate, checkpoint, training = final_val.resolve_candidate(args.candidate)
    if candidate.get("label") != "mix-025":
        raise ValueError("Language smoke is locked to candidate mix-025")
    endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
    final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    sample = _sample_by_stem(dataset, STEM)
    bucket = [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8]
    seed = int(spec["per_stem_seeds"][STEM]["sampling"])
    contract = {
        "kind": SMOKE_KIND, "candidate": "mix-025", "turbo": guide.TURBO,
        "geometry": guide.NATIVE_GEOMETRY,
        "prompt_file": {"path": str(Path(args.prompt_file)), "sha256": SMOKE_SHA256, "record_count": GENERATION_COUNT},
        "prompt_mapping": rows, "stem": STEM, "sampling_seed": seed,
        "control_sha256": _sha256(control), "bucket": bucket,
        "final_val_spec_sha256": spec_digest, **_candidate_contract(candidate, checkpoint),
    }
    output = Path(args.output_root)
    provenance = output / "language_smoke_provenance.json"
    if provenance.exists() and _read_json(provenance) != contract:
        raise ValueError(f"Existing output has conflicting immutable language-smoke provenance: {provenance}")
    if not provenance.exists():
        _write(provenance, contract)
    return contract, dataset, control, candidate, checkpoint, training, output


def _directory(output: Path, language: str) -> Path:
    return output / "generations" / STEM / language


def _copy_control(source: Path, target: Path) -> None:
    if target.exists() and _sha256(target) != _sha256(source):
        raise ValueError(f"Existing pose control conflicts with authoritative final-val control identity: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _metadata(row: Mapping[str, str], contract: Mapping[str, Any], control: Path) -> dict[str, Any]:
    return {"language": row["language"], "stem": STEM, "prompt": row["prompt"], "seed": contract["sampling_seed"],
            "control_path": str(control), "control_sha256": contract["control_sha256"], "bucket": contract["bucket"],
            "geometry": guide.NATIVE_GEOMETRY, "prompt_file_sha256": SMOKE_SHA256,
            "final_val_spec_sha256": contract["final_val_spec_sha256"], "candidate": "mix-025", **contract["turbo"],
            **{key: contract[key] for key in ("candidate_kind", "checkpoint_step", "checkpoint_interpolation", "checkpoint_sha256") if key in contract}}


def _generation_status(output: Path, rows: list[dict[str, str]], candidate: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    result_path = output / "generation_results.json"
    payload = _read_json(result_path) if result_path.exists() else None
    control = output / "controls" / f"{STEM}.png"
    observed, recorded = [], []
    for row in rows:
        directory = _directory(output, row["language"]); image = directory / final_val._image_name(candidate); metadata_path = directory / "metadata.json"
        if image.is_file():
            try:
                with Image.open(image) as opened: opened.verify()
                with Image.open(control) as opened: opened.verify()
                metadata = _read_json(metadata_path)
                # The stored authoritative source path may be host-specific,
                # but every other field is immutable and exact.
                expected = _metadata(row, contract, control)
                for key, value in expected.items():
                    if key != "control_path" and metadata.get(key) != value:
                        raise ValueError("generation metadata contract mismatch")
                if not isinstance(metadata.get("control_path"), str) or not metadata["control_path"]:
                    raise ValueError("generation metadata control path missing")
                if _sha256(control) != contract["control_sha256"]:
                    raise ValueError("generation control hash mismatch")
            except Exception as exc:
                raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if directory.exists() or control.exists():
                raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
        if payload is not None:
            recorded.append(payload.get("generated_artifacts", {}).get(row["language"]) == final_val._image_name(candidate))
    if not any(observed) and payload is None:
        return "missing"
    if (all(observed) and payload is not None and all(recorded) and payload.get("candidate") == "mix-025"
            and payload.get("prompt_file") == contract["prompt_file"] and payload.get("prompt_mapping") == rows
            and payload.get("generated_artifacts") == {language: final_val._image_name(candidate) for language in LANGUAGES}):
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


def preflight(args: argparse.Namespace) -> None:
    rows = load_smoke_rows(args.prompt_file)
    contract, dataset, control, _, _, training, output = _inputs(args, rows)
    _write(output / "checkpoint_preflight.json", {**contract, "generation_count": GENERATION_COUNT,
           "dataset_sample_count": len(dataset), "control_path_resolved": str(control), "training_metadata": training,
           "source_rgb_fallback_permitted": False, "text_conditioning": "online exact UTF-8 prompts; source captions are never sampled"})
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run language-smoke generation from the GH200 host shell with CUDA visible")
    rows = load_smoke_rows(args.prompt_file)
    contract, dataset, control, candidate, checkpoint, training, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) == "complete":
        print(json.dumps({"already_complete": "mix-025", "generation_count": GENERATION_COUNT})); return
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint)
    final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    vae, conditioner = load_krea_vae("cuda"), guide.PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    sample = dict(_sample_by_stem(dataset, STEM))
    if [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8] != contract["bucket"]:
        raise ValueError("Frozen native geometry changed for language-smoke stem")
    control_output = output / "controls" / f"{STEM}.png"; _copy_control(control, control_output)
    for row in rows:
        directory = _directory(output, row["language"]); metadata = _metadata(row, contract, control)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing generation metadata conflicts with frozen language-smoke contract: {metadata_path}")
        if not metadata_path.exists(): _write(metadata_path, metadata)
        context, mask = guide._conditioning(conditioner, row["prompt"], sample)
        generated_sample = dict(sample, prompt=row["prompt"], context=context, mask=mask)
        pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), generated_sample,
                                         torch.device("cuda"), contract["sampling_seed"], control_scale=1.0)
        save_image(pixels, directory / final_val._image_name(candidate))
    _write(output / "generation_results.json", {**contract, "generated_artifacts": {language: final_val._image_name(candidate) for language in LANGUAGES},
           "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training, "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _sidecar(path: str | Path) -> tuple[dict[str, Any], str]:
    spec, _ = final_val.load_final_spec(final_val.FINAL_SPEC)
    sidecar, digest = final_val._load_final_sidecar(path, list(spec["stems"]))
    records = {str(record["stem"]): record for record in sidecar["records"]}
    if STEM not in records:
        raise ValueError("Authoritative final-val pose sidecar is missing frozen language-smoke stem")
    return {"records": [records[STEM]]}, digest


def _score_records(payload: Mapping[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    if (not isinstance(records, list) or len(records) != GENERATION_COUNT
            or [record.get("language") for record in records] != list(LANGUAGES)
            or any(record.get("stem") != STEM or record.get("prompt") != row["prompt"] for record, row in zip(records, rows))):
        raise ValueError("Language-smoke score artifact is incomplete or does not match frozen prompt mapping")
    return records


def score(args: argparse.Namespace) -> None:
    rows = load_smoke_rows(args.prompt_file)
    contract, dataset, _, candidate, _, _, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) != "complete":
        raise FileNotFoundError("Language-smoke scoring requires the complete validated two-image generation set")
    if not args.reference_sidecar:
        raise ValueError("Language-smoke PCK requires the authoritative final-val --reference-sidecar")
    sidecar, sidecar_digest = _sidecar(args.reference_sidecar)
    existing_path = output / "pck_clip_results.json"
    if existing_path.exists():
        existing = _read_json(existing_path)
        if (any(existing.get(key) != value for key, value in contract.items()) or existing.get("reference_sidecar_sha256") != sidecar_digest
                or existing.get("clip_model") != args.clip_model_id or existing.get("confidence_threshold") != .5):
            raise ValueError("Existing language-smoke score artifact conflicts with frozen candidate/provenance contract")
    geometry = turbo_scoring_geometry(_sample_by_stem(dataset, STEM))
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    records = []
    for row in rows:
        image = _directory(output, row["language"]) / final_val._image_name(candidate)
        result = score_authoritative_pck(sidecar=sidecar, geometry_by_stem={STEM: geometry}, image_for=lambda _: image,
                                         detector=detector, confidence_threshold=.5, require_images=True)
        pck = result["per_image"][0] if len(result["per_image"]) == 1 else {"stem": STEM, "reference_available": False, "reason": result["unavailable"][0]["reason"]}
        records.append({"language": row["language"], "stem": STEM, "prompt": row["prompt"], "image": str(image.relative_to(output)),
                        "pck": pck, "clip_cosine_similarity": _clip_score(clip, processor, device, row["prompt"], image)})
    _write(existing_path, {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()), "reference_sidecar_sha256": sidecar_digest,
           "clip_model": args.clip_model_id, "confidence_threshold": .5, "per_generation": records})
    print(existing_path)


def _validated_scores(output: Path, contract: Mapping[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    scores = _read_json(output / "pck_clip_results.json")
    if any(scores.get(key) != value for key, value in contract.items()):
        raise ValueError("Language-smoke score artifact conflicts with frozen candidate/provenance contract")
    return _score_records(scores, rows)


def _compact(records: list[dict[str, Any]]) -> dict[str, Any]:
    pck = [record["pck"] for record in records if record["pck"].get("reference_available")]
    pose = _pool_pose(pck) if pck else {"evaluable_sample_count": 0, "pck_005": None, "pck_010": None, "pck_020": None}
    clips = aggregate([float(record["clip_cosine_similarity"]) for record in records])
    return {"generation_count": len(records), "pck": pose, "pck_unavailable_count": len(records) - len(pck),
            "clip": {"mean_cosine_similarity": clips["mean"], "median_cosine_similarity": clips["median"], "std_cosine_similarity": clips["std"], "sample_count": clips["sample_count"]}}


def report(args: argparse.Namespace) -> None:
    rows = load_smoke_rows(args.prompt_file)
    contract, _, _, candidate, _, training, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) != "complete":
        raise FileNotFoundError("Language-smoke report requires the complete validated two-image generation set")
    records = _validated_scores(output, contract, rows)
    by_language = {language: _compact([record for record in records if record["language"] == language]) for language in LANGUAGES}
    if set(by_language) != set(LANGUAGES) or any(value["generation_count"] != 1 for value in by_language.values()):
        raise ValueError("Language-smoke scores do not contain exactly one English and one Chinese record")
    _write(output / "metrics_by_language.json", {**contract, "group_by": "language", "rows": [{"language": language, **by_language[language]} for language in LANGUAGES]})
    paths = [output / "controls" / f"{STEM}.png"] + [_directory(output, language) / final_val._image_name(candidate) for language in LANGUAGES]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Language-smoke comparison requires the control and both generations; no RGB fallback exists")
    make_contact_sheet([(STEM, paths)], output / "english_vs_chinese_comparison.png", thumbnail_width=320, thumbnail_height=320,
                       column_labels=("pose control", "English", "Chinese"))
    _write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "generation_count": GENERATION_COUNT,
           "score_artifact": "pck_clip_results.json", "metrics_by_language": "metrics_by_language.json",
           "comparison": "english_vs_chinese_comparison.png", "source_rgb_fallback_used": False})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    rows = load_smoke_rows(args.prompt_file)
    contract, _, _, candidate, _, _, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) != "complete":
        raise FileNotFoundError("Language-smoke summary requires the complete validated two-image generation set")
    records = _validated_scores(output, contract, rows)
    print(json.dumps({"candidate": candidate["label"], "stem": STEM, "sampling_seed": contract["sampling_seed"], "generation_count": GENERATION_COUNT,
                      "by_language": {language: _compact([record for record in records if record["language"] == language]) for language in LANGUAGES}}, ensure_ascii=False, sort_keys=True))


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
