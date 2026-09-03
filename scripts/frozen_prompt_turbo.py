"""Frozen prompt-injection benchmark and same-pose hero generation.

This is intentionally separate from ``final_val_turbo_benchmark.py``.  It
reuses that evaluator's pinned candidate and Raw-to-Turbo compatibility rules,
but never writes into its source-caption output roots or changes its inputs.
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
from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
from pose_controlnet.text_conditioning import compact_valid_conditioning
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.turbo_evaluation import (
    raw_to_turbo_control_compatibility,
    sample_turbo_pose_image,
    turbo_metadata,
)
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val


PROMPT_INJECTION_FILE = Path("docs/evaluation/prompt-injection-benchmark/prompt_injection_48.jsonl")
PROMPT_INJECTION_SHA256 = "a7c6f3aa8aa1e18bc0767b9ad940b1c0d33fbabdfcdf568cffabb883b605bdf3"
HERO_FILE = Path("docs/evaluation/prompt-injection-benchmark/hero_same_pose_6.jsonl")
HERO_SHA256 = "1b28d8b9cc8754327727a317de03543aa71876ba0f878acd0ad8dc45897e9345"
HERO_STEM = "real_human_humanart_15000000000930"
HERO_SEED_ROOT = 420_600
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"


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
        raise FileNotFoundError(f"Required frozen benchmark artifact is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Frozen benchmark JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Frozen benchmark JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Frozen prompt-file SHA-256 mismatch: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required frozen prompt file is missing: {path}") from None
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Frozen prompt file has invalid JSON at line {number}: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Frozen prompt file row {number} must be an object: {path}")
        rows.append(row)
    return rows


def load_prompt_injection_rows(path: str | Path = PROMPT_INJECTION_FILE) -> list[dict[str, Any]]:
    """Load the exact 48 injected prompts in the frozen final-val stem order."""
    source = Path(path)
    rows = _read_jsonl(source, PROMPT_INJECTION_SHA256)
    spec, _ = final_val.load_final_spec(final_val.FINAL_SPEC)
    expected_stems = list(spec["stems"])
    expected_keys = {"stem", "prompt_id", "family", "subject_count", "prompt"}
    if len(rows) != final_val.FINAL_COUNT:
        raise ValueError("Prompt-injection file must contain exactly 48 rows")
    if any(set(row) != expected_keys for row in rows):
        raise ValueError("Prompt-injection rows must use the exact frozen schema")
    stems = [row.get("stem") for row in rows]
    if any(not isinstance(stem, str) or not stem for stem in stems) or len(stems) != len(set(stems)):
        raise ValueError("Prompt-injection file has missing or duplicate stems")
    if stems != expected_stems:
        raise ValueError("Prompt-injection stems must exactly match the frozen final-val order")
    for row in rows:
        if any(not isinstance(row[key], str) or not row[key].strip() for key in expected_keys):
            raise ValueError(f"Prompt-injection row has an empty required value: {row.get('stem')}")
    return rows


def load_hero_rows(path: str | Path = HERO_FILE) -> list[dict[str, Any]]:
    """Load six nonempty prompts that all bind to the one canonical hero pose."""
    source = Path(path)
    rows = _read_jsonl(source, HERO_SHA256)
    expected_keys = {"hero_id", "stem", "prompt"}
    if len(rows) != 6:
        raise ValueError("Same-pose hero file must contain exactly six rows")
    if any(set(row) != expected_keys for row in rows):
        raise ValueError("Same-pose hero rows must use the exact frozen schema")
    ids = [row.get("hero_id") for row in rows]
    if any(not isinstance(value, str) or not value.strip() for row in rows for value in (row["hero_id"], row["prompt"])):
        raise ValueError("Same-pose hero rows require nonempty IDs and prompts")
    if len(ids) != len(set(ids)):
        raise ValueError("Same-pose hero file has duplicate hero IDs")
    if any(row.get("stem") != HERO_STEM for row in rows):
        raise ValueError(f"Same-pose hero must use only canonical stem {HERO_STEM}")
    return rows


def _candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    if final_val._is_interpolation(candidate):
        return {"candidate_kind": candidate["kind"], "checkpoint_step": None,
                "checkpoint_interpolation": candidate["interpolation"]}
    if checkpoint is None:
        raise ValueError("Real candidate is missing its pinned checkpoint")
    return {"candidate_kind": "real_checkpoint", "checkpoint_step": candidate["step"],
            "checkpoint_sha256": _sha256(checkpoint)}


def _prompt_contract(kind: str, rows: list[dict[str, Any]], path: Path, digest: str,
                     candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    contract = {
        "kind": kind,
        "candidate": candidate["label"],
        "turbo": {**turbo_metadata(), "control_scale": 1.0},
        "geometry": NATIVE_GEOMETRY,
        "prompt_file": {"path": str(path), "sha256": digest, "record_count": len(rows)},
        # Retain every frozen row, including the exact stem -> prompt pairing.
        "prompt_mapping": rows,
        **_candidate_contract(candidate, checkpoint),
    }
    if kind == "prompt_injection_48_turbo_fixed_pose":
        spec, spec_digest = final_val.load_final_spec(final_val.FINAL_SPEC)
        contract["final_val_spec_sha256"] = spec_digest
        contract["sampling_seeds"] = {stem: int(spec["per_stem_seeds"][stem]["sampling"])
                                      for stem in spec["stems"]}
    elif kind == "hero_same_pose_6_turbo":
        contract["sampling_seeds"] = {row["hero_id"]: _hero_seed(row["hero_id"]) for row in rows}
    return contract


def _validate_or_write_output(output: Path, contract: Mapping[str, Any]) -> None:
    path = output / "frozen_prompt_provenance.json"
    if path.exists():
        if _read_json(path) != contract:
            raise ValueError(f"Existing output has conflicting immutable prompt provenance: {path}")
    else:
        _write(path, contract)


def _dataset(args: argparse.Namespace) -> tuple[dict[str, Any], str, PreparedLatentShardDataset, dict[str, Path]]:
    spec, spec_digest = final_val.load_final_spec(args.final_spec)
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    # This validates source-caption caches before we replace only the runtime
    # text conditioning.  It does not make source captions available to sampling.
    final_val.validate_cached_contract(dataset, spec)
    shards = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shards.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Frozen prompt evaluation requires --dataset-root or latent shards.json.dataset_root")
    return spec, spec_digest, dataset, final_val.resolve_final_controls(dataset_root, list(spec["stems"]))


def _conditioning(conditioner: PoseTextConditioner, prompt: str, sample: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, masks = conditioner([prompt])
    compact = compact_valid_conditioning(contexts, masks, 0)
    context, mask = compact["context"], compact["mask"]
    if context.dtype != torch.bfloat16:
        context = context.to(torch.bfloat16)
    if context.ndim != 3 or mask.ndim != 1 or context.shape[0] != mask.shape[0] or not mask.all().item() or not torch.isfinite(context).all().item():
        raise ValueError("Injected prompt conditioning is empty, non-finite, or malformed")
    if tuple(context.shape[1:]) != tuple(sample["context"].shape[1:]):
        raise ValueError("Injected prompt conditioning dimensions differ from frozen Qwen contract")
    return context.cpu().contiguous(), mask.cpu().to(torch.bool).contiguous()


def _image_name(candidate: Mapping[str, Any]) -> str:
    return final_val._image_name(candidate)


def _generation_status(output: Path, rows: list[dict[str, Any]], candidate: Mapping[str, Any],
                       contract: Mapping[str, Any], *, hero: bool) -> str:
    payload_path = output / "generation_results.json"
    payload = _read_json(payload_path) if payload_path.exists() else None
    observed, recorded = [], []
    for row in rows:
        identity = row["hero_id"] if hero else row["stem"]
        directory = output / "hero" / identity if hero else output / "fixed_pose" / identity
        image = directory / _image_name(candidate)
        if image.is_file():
            try:
                with Image.open(image) as opened:
                    opened.verify()
                control = directory / "control.png"
                with Image.open(control) as opened:
                    opened.verify()
                metadata = _read_json(directory / "metadata.json")
                candidate_fields = ("candidate_kind", "checkpoint_step", "checkpoint_interpolation", "checkpoint_sha256")
                expected_control_sha = contract["control_sha256"][HERO_STEM if hero else identity]
                if (metadata.get("candidate") != candidate["label"] or metadata.get("prompt") != row["prompt"]
                        or metadata.get("geometry") != NATIVE_GEOMETRY
                        or metadata.get("prompt_file_sha256") != contract["prompt_file"]["sha256"]
                        or metadata.get("seed") != contract["sampling_seeds"].get(identity)
                        or metadata.get("control_sha256") != expected_control_sha
                        or _sha256(control) != expected_control_sha
                        or any(metadata.get(key) != value for key, value in contract["turbo"].items())
                        or any(metadata.get(key) != contract.get(key) for key in candidate_fields if key in contract)):
                    raise ValueError("metadata contract mismatch")
                if hero and (metadata.get("hero_id") != identity or metadata.get("stem") != HERO_STEM):
                    raise ValueError("hero metadata contract mismatch")
                if not hero and (metadata.get("stem") != identity or metadata.get("prompt_id") != row["prompt_id"]):
                    raise ValueError("prompt-injection metadata contract mismatch")
            except Exception as exc:
                raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            observed.append(False)
        if payload is not None:
            recorded.append(payload.get("generated_artifacts", {}).get(identity) == [_image_name(candidate)])
    if not any(observed) and (payload is None or not any(recorded)):
        return "missing"
    if all(observed) and payload is not None and all(recorded) and payload.get("candidate") == candidate["label"] and payload.get("prompt_file") == contract["prompt_file"] and payload.get("prompt_mapping") == rows:
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


def _copy_control(source: Path, target: Path) -> None:
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError(f"Existing pose control conflicts with DatasetIndex resolution: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _load_candidate(candidate_id: str) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    candidate, checkpoint, training = final_val.resolve_candidate(candidate_id)
    if final_val._is_interpolation(candidate):
        endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
        final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    return candidate, checkpoint, training


def _load_model(args: argparse.Namespace, candidate: Mapping[str, Any], checkpoint: Path | None):
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint)
    final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    return model, compatibility


def _preflight_common(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path, digest: str,
                      kind: str) -> tuple[dict[str, Any], PreparedLatentShardDataset, dict[str, Path], dict[str, Any], Path | None, dict[str, Any], Path]:
    spec, spec_digest, dataset, controls = _dataset(args)
    candidate, checkpoint, training = _load_candidate(args.candidate)
    contract = _prompt_contract(kind, rows, path, digest, candidate, checkpoint)
    contract["control_sha256"] = {stem: _sha256(controls[stem]) for stem in
                                  ([row["stem"] for row in rows] if kind == "prompt_injection_48_turbo_fixed_pose" else [HERO_STEM])}
    if contract.get("final_val_spec_sha256", spec_digest) != spec_digest:
        raise ValueError("Prompt benchmark final-val spec provenance changed during preflight")
    _validate_or_write_output(Path(args.output_root), contract)
    return contract, dataset, controls, candidate, checkpoint, training, Path(args.output_root)


def prompt_preflight(args: argparse.Namespace) -> None:
    rows = load_prompt_injection_rows(args.prompt_file)
    contract, dataset, controls, candidate, checkpoint, training, output = _preflight_common(
        args, rows, Path(args.prompt_file), PROMPT_INJECTION_SHA256, "prompt_injection_48_turbo_fixed_pose")
    _write(output / "checkpoint_preflight.json", {**contract, "sample_count": len(dataset),
           "control_paths_resolved": len(controls), "training_metadata": training,
           "injected_text_conditioning": "online PoseTextConditioner; source-caption contexts are never sampled"})
    print(output / "checkpoint_preflight.json")


def prompt_generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run frozen prompt Turbo generation from the GH200 host shell with CUDA visible")
    rows = load_prompt_injection_rows(args.prompt_file)
    contract, dataset, controls, candidate, checkpoint, training, output = _preflight_common(
        args, rows, Path(args.prompt_file), PROMPT_INJECTION_SHA256, "prompt_injection_48_turbo_fixed_pose")
    if _generation_status(output, rows, candidate, contract, hero=False) == "complete":
        print(json.dumps({"already_complete": candidate["label"]})); return
    model, compatibility = _load_model(args, candidate, checkpoint)
    vae, conditioner = load_krea_vae("cuda"), PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    for row in rows:
        stem, directory = row["stem"], output / "fixed_pose" / row["stem"]
        sample = dict(_sample_by_stem(dataset, stem)); context, mask = _conditioning(conditioner, row["prompt"], sample)
        sample.update({"prompt": row["prompt"], "context": context, "mask": mask})
        _copy_control(controls[stem], directory / "control.png")
        metadata = {"stem": stem, "prompt": row["prompt"], "prompt_id": row["prompt_id"], "family": row["family"],
                    "subject_count": row["subject_count"], "seed": contract["sampling_seeds"][stem],
                    "control_path": str(controls[stem]), "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8],
                    "geometry": NATIVE_GEOMETRY, "prompt_file_sha256": PROMPT_INJECTION_SHA256,
                    "control_sha256": contract["control_sha256"][stem],
                    "final_val_spec_sha256": contract["final_val_spec_sha256"], "candidate": candidate["label"],
                    **contract["turbo"], **_candidate_contract(candidate, checkpoint)}
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing prompt-injection metadata conflicts with frozen contract: {metadata_path}")
        if not metadata_path.exists(): _write(metadata_path, metadata)
        pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                         torch.device("cuda"), metadata["seed"], control_scale=1.0)
        save_image(pixels, directory / _image_name(candidate))
    _write(output / "generation_results.json", {**contract, "stems": [row["stem"] for row in rows],
           "generated_artifacts": {row["stem"]: [_image_name(candidate)] for row in rows},
           "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training})
    print(output / "generation_results.json")


def prompt_score(args: argparse.Namespace) -> None:
    rows = load_prompt_injection_rows(args.prompt_file)
    contract, dataset, _, candidate, checkpoint, _, output = _preflight_common(
        args, rows, Path(args.prompt_file), PROMPT_INJECTION_SHA256, "prompt_injection_48_turbo_fixed_pose")
    if _generation_status(output, rows, candidate, contract, hero=False) != "complete":
        raise FileNotFoundError("Prompt-injection scoring requires the complete validated 48-image generation set")
    if not args.reference_sidecar:
        raise ValueError("Prompt-injection PCK scoring requires the canonical immutable --reference-sidecar")
    sidecar, sidecar_digest = final_val._load_final_sidecar(args.reference_sidecar, [row["stem"] for row in rows])
    geometry = {row["stem"]: final_val.turbo_scoring_geometry(_sample_by_stem(dataset, row["stem"])) for row in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    image_for = lambda row: output / "fixed_pose" / row["stem"] / _image_name(candidate)
    pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry,
                                   image_for=lambda stem: output / "fixed_pose" / stem / _image_name(candidate),
                                   detector=detector, confidence_threshold=.5, require_images=True)
    clip_rows = [{"stem": row["stem"], "prompt_id": row["prompt_id"], "cosine_similarity": _clip_score(clip, processor, device, row["prompt"], image_for(row))} for row in rows]
    values = aggregate([row["cosine_similarity"] for row in clip_rows])
    _write(output / "pck_clip_results.json", {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()),
           "reference_sidecar_sha256": sidecar_digest, "clip_model": args.clip_model_id, "confidence_threshold": .5,
           "checkpoints": [{"candidate": candidate["label"], "pose": pose, "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"], "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_sample": clip_rows}}]})
    print(output / "pck_clip_results.json")


def prompt_report(args: argparse.Namespace) -> None:
    rows = load_prompt_injection_rows(args.prompt_file)
    contract, _, _, candidate, _, training, output = _preflight_common(
        args, rows, Path(args.prompt_file), PROMPT_INJECTION_SHA256, "prompt_injection_48_turbo_fixed_pose")
    if _generation_status(output, rows, candidate, contract, hero=False) != "complete":
        raise FileNotFoundError("Prompt-injection report requires the complete validated 48-image generation set")
    scores = _read_json(output / "pck_clip_results.json")
    if any(scores.get(key) != value for key, value in contract.items()):
        raise ValueError("Prompt-injection score artifact conflicts with frozen provenance")
    grids = [(row["stem"], [output / "fixed_pose" / row["stem"] / "control.png", output / "fixed_pose" / row["stem"] / _image_name(candidate)]) for row in rows]
    if any(not path.is_file() for _, paths in grids for path in paths):
        raise FileNotFoundError("Prompt-injection report requires pose controls and generated outputs only")
    labels = ("pose control", f"generated output ({candidate['label']})")
    make_contact_sheet(grids[:4], output / "checkpoint_selection_grid.png", thumbnail_width=180, thumbnail_height=180, column_labels=labels)
    make_contact_sheet(grids, output / "full_contact_sheet.png", thumbnail_width=320, thumbnail_height=320, column_labels=labels)
    _write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "checkpoints": scores["checkpoints"],
           "qualitative_grids": {"checkpoint_selection": "checkpoint_selection_grid.png", "full_contact_sheet": "full_contact_sheet.png"},
           "clip_prompt_source": "frozen injected prompt text", "source_caption_benchmark_modified": False})
    print(output / "evaluation_summary.json")


def _hero_seed(hero_id: str) -> int:
    payload = f"{HERO_SEED_ROOT}:{hero_id}:sampling".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def hero_preflight(args: argparse.Namespace) -> None:
    rows = load_hero_rows(args.hero_file)
    contract, dataset, controls, candidate, checkpoint, training, output = _preflight_common(
        args, rows, Path(args.hero_file), HERO_SHA256, "hero_same_pose_6_turbo")
    if HERO_STEM not in controls or HERO_STEM not in {record[3] for record in dataset.records}:
        raise ValueError("Canonical hero stem is unavailable from the immutable validation dataset")
    _write(output / "checkpoint_preflight.json", {**contract, "sample_count": 1, "control_paths_resolved": 1,
           "hero_stem": HERO_STEM, "seed_derivation": f"SHA256('{HERO_SEED_ROOT}:<hero_id>:sampling')[:8] modulo 2^63-1",
           "per_prompt_seeds": {row["hero_id"]: _hero_seed(row["hero_id"]) for row in rows}, "training_metadata": training})
    print(output / "checkpoint_preflight.json")


def hero_generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run same-pose hero Turbo generation from the GH200 host shell with CUDA visible")
    rows = load_hero_rows(args.hero_file)
    contract, dataset, controls, candidate, checkpoint, training, output = _preflight_common(
        args, rows, Path(args.hero_file), HERO_SHA256, "hero_same_pose_6_turbo")
    if _generation_status(output, rows, candidate, contract, hero=True) == "complete":
        print(json.dumps({"already_complete": candidate["label"]})); return
    model, compatibility = _load_model(args, candidate, checkpoint)
    vae, conditioner = load_krea_vae("cuda"), PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    base_sample = dict(_sample_by_stem(dataset, HERO_STEM))
    for row in rows:
        directory = output / "hero" / row["hero_id"]
        context, mask = _conditioning(conditioner, row["prompt"], base_sample)
        sample = {**base_sample, "prompt": row["prompt"], "context": context, "mask": mask}
        _copy_control(controls[HERO_STEM], directory / "control.png")
        metadata = {"hero_id": row["hero_id"], "stem": HERO_STEM, "prompt": row["prompt"], "seed": contract["sampling_seeds"][row["hero_id"]],
                    "control_path": str(controls[HERO_STEM]), "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8],
                    "geometry": NATIVE_GEOMETRY, "prompt_file_sha256": HERO_SHA256, "candidate": candidate["label"],
                    "control_sha256": contract["control_sha256"][HERO_STEM],
                    **contract["turbo"], **_candidate_contract(candidate, checkpoint)}
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing hero metadata conflicts with frozen contract: {metadata_path}")
        if not metadata_path.exists(): _write(metadata_path, metadata)
        pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                         torch.device("cuda"), metadata["seed"], control_scale=1.0)
        save_image(pixels, directory / _image_name(candidate))
    _write(output / "generation_results.json", {**contract, "hero_stem": HERO_STEM,
           "generated_artifacts": {row["hero_id"]: [_image_name(candidate)] for row in rows},
           "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training})
    print(output / "generation_results.json")


def hero_report(args: argparse.Namespace) -> None:
    rows = load_hero_rows(args.hero_file)
    contract, _, _, candidate, _, training, output = _preflight_common(
        args, rows, Path(args.hero_file), HERO_SHA256, "hero_same_pose_6_turbo")
    if _generation_status(output, rows, candidate, contract, hero=True) != "complete":
        raise FileNotFoundError("Hero report requires all six generated interpretations")
    paths = [output / "hero" / rows[0]["hero_id"] / "control.png"] + [output / "hero" / row["hero_id"] / _image_name(candidate) for row in rows]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Hero contact sheet requires its pose control and all generated outputs; no RGB fallback exists")
    make_contact_sheet([(HERO_STEM, paths)], output / "hero_contact_sheet.png", thumbnail_width=320, thumbnail_height=320,
                       column_labels=("pose control",) + tuple(row["hero_id"] for row in rows))
    _write(output / "hero_summary.json", {**contract, "hero_stem": HERO_STEM, "training_metadata": training,
           "per_prompt": [{"hero_id": row["hero_id"], "prompt": row["prompt"], "seed": _hero_seed(row["hero_id"]),
                           "generated_image": str(Path("hero") / row["hero_id"] / _image_name(candidate))} for row in rows],
           "qualitative_grid": "hero_contact_sheet.png", "source_rgb_fallback_used": False})
    print(output / "hero_summary.json")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prompt-injection", "hero"))
    parser.add_argument("action", choices=("preflight", "generate", "score", "report"))
    parser.add_argument("--candidate", choices=tuple(final_val.CANDIDATES), default="mix-025")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--final-spec", default=str(final_val.FINAL_SPEC))
    parser.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    parser.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    parser.add_argument("--dataset-root")
    parser.add_argument("--turbo-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors")
    parser.add_argument("--reference-sidecar", help="required only for prompt-injection score")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt-file", default=str(PROMPT_INJECTION_FILE))
    parser.add_argument("--hero-file", default=str(HERO_FILE))
    return parser


def main() -> None:
    args = parser().parse_args()
    if args.mode == "hero" and args.action == "score":
        raise ValueError("Same-pose hero is generation-only and has no score action")
    actions = {
        ("prompt-injection", "preflight"): prompt_preflight,
        ("prompt-injection", "generate"): prompt_generate,
        ("prompt-injection", "score"): prompt_score,
        ("prompt-injection", "report"): prompt_report,
        ("hero", "preflight"): hero_preflight,
        ("hero", "generate"): hero_generate,
        ("hero", "report"): hero_report,
    }
    actions[(args.mode, args.action)](args)


if __name__ == "__main__":
    main()
