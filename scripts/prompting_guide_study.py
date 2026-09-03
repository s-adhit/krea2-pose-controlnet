"""Reproducible 8-pose x 8-prompt Krea-2 Turbo prompting-guide study.

This opt-in tool is deliberately isolated from the frozen source-caption and
prompt-injection evaluators.  It samples only the frozen pose controls and
online experimental text conditioning; source RGB is never an input or a
report fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
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
from pose_controlnet.text_conditioning import compact_valid_conditioning
from pose_controlnet.text_encoder import PoseTextConditioner
from pose_controlnet.turbo_evaluation import raw_to_turbo_control_compatibility, sample_turbo_pose_image, turbo_metadata, turbo_scoring_geometry
from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
from scripts import final_val_turbo_benchmark as final_val


STUDY_FILE = Path("docs/evaluation/prompting-guide/prompting_study.jsonl")
STUDY_SHA256 = "4fae6d39ac7354d451ca13556d3a2a89e303691ceb30727b8561b13b494450df"
STUDY_KIND = "prompting_guide_8_conditions_x_8_modes_turbo_fixed_pose"
NATIVE_GEOMETRY = "native_aspect_preserving_cached_latent_bucket"
MODES = (
    "P0_minimal", "P1_style", "P2_environment", "P3_neutral",
    "P4_supportive", "P5_conflicting", "P6_semantic_prior",
    "P7_framing_count_conflict",
)
CONDITION_COUNT = 8
GENERATION_COUNT = CONDITION_COUNT * len(MODES)
TURBO = {**turbo_metadata(), "control_scale": 1.0}


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
        raise FileNotFoundError(f"Required prompting-guide artifact is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompting-guide JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Prompting-guide JSON must be an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != STUDY_SHA256:
        raise ValueError(f"Frozen prompting-study SHA-256 mismatch: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required frozen prompting-study file is missing: {path}") from None
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Frozen prompting-study file has invalid JSON at line {number}: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Frozen prompting-study row {number} must be an object")
        rows.append(row)
    return rows


def load_study_rows(path: str | Path = STUDY_FILE) -> list[dict[str, Any]]:
    """Load the pinned 64-row mapping and reject any incomplete mode matrix."""
    rows = _read_jsonl(Path(path))
    if len(rows) != GENERATION_COUNT:
        raise ValueError(f"Prompting study must contain exactly {GENERATION_COUNT} rows")
    required = {"stem", "pose_class", "mode", "prompt"}
    if any(set(row) != required for row in rows):
        raise ValueError("Prompting-study rows must use the exact frozen schema")
    pairs: set[tuple[str, str]] = set()
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if any(not isinstance(row[key], str) or not row[key].strip() for key in required):
            raise ValueError("Prompting-study rows require nonempty stem, pose_class, mode, and prompt")
        if row["mode"] not in MODES:
            raise ValueError(f"Prompting study has unexpected mode name: {row['mode']}")
        pair = (row["stem"], row["mode"])
        if pair in pairs:
            raise ValueError(f"Prompting study has duplicate stem/mode pair: {pair}")
        pairs.add(pair); by_stem[row["stem"]].append(row)
    if len(by_stem) != CONDITION_COUNT:
        raise ValueError(f"Prompting study must contain exactly {CONDITION_COUNT} frozen pose conditions")
    if len({rows_for_stem[0]["pose_class"] for rows_for_stem in by_stem.values()}) != CONDITION_COUNT:
        raise ValueError("Each frozen pose condition must have one distinct pose class")
    for stem, stem_rows in by_stem.items():
        if len(stem_rows) != len(MODES) or tuple(row["mode"] for row in stem_rows) != MODES:
            raise ValueError(f"Prompting study has missing or misordered required modes for {stem}")
        if len({row["pose_class"] for row in stem_rows}) != 1:
            raise ValueError(f"Prompting study has conflicting pose classes for {stem}")
    return rows


def _candidate_contract(candidate: Mapping[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    if final_val._is_interpolation(candidate):
        return {"candidate_kind": candidate["kind"], "checkpoint_step": None,
                "checkpoint_interpolation": candidate["interpolation"]}
    if checkpoint is None:
        raise ValueError("Candidate is missing its pinned checkpoint")
    return {"candidate_kind": "real_checkpoint", "checkpoint_step": candidate["step"],
            "checkpoint_sha256": _sha256(checkpoint)}


def _validate_locked_runtime_contract() -> None:
    if (TURBO.get("steps"), TURBO.get("cfg"), TURBO.get("mu"), TURBO.get("control_scale"),
            TURBO.get("mu_resolution_dependent")) != (8, 0.0, 1.15, 1.0, False):
        raise ValueError("Prompting-guide study violates the locked Turbo 8-step CFG-0 mu=1.15 control-scale-1.0 contract")
    if NATIVE_GEOMETRY != "native_aspect_preserving_cached_latent_bucket":
        raise ValueError("Prompting-guide study violates the locked native/aspect-preserving geometry contract")


def _controls_for_stems(dataset_root: str | Path, stems: list[str]) -> dict[str, Path]:
    controls = final_val.resolve_final_controls(dataset_root, stems)
    if set(controls) != set(stems):
        raise ValueError("Prompting study control resolution did not return the exact frozen stem set")
    return controls


def _inputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], PreparedLatentShardDataset, dict[str, Path], dict[str, Any], Path | None, dict[str, Any], Path]:
    _validate_locked_runtime_contract()
    spec, spec_digest = final_val.load_final_spec(args.final_spec)
    dataset = PreparedLatentShardDataset(args.latent_root, "val", text_conditioning_root=args.text_conditioning_root)
    final_val.validate_cached_contract(dataset, spec)
    stems = list(dict.fromkeys(row["stem"] for row in rows))
    missing_seeds = [stem for stem in stems if not isinstance(spec["per_stem_seeds"].get(stem, {}).get("sampling"), int)]
    if missing_seeds:
        raise ValueError(f"Prompting study stems lack frozen final-val sampling seeds: {missing_seeds}")
    shard_metadata = _read_json(Path(args.latent_root) / "shards.json")
    dataset_root = args.dataset_root or shard_metadata.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError("Prompting study requires --dataset-root or latent shards.json.dataset_root")
    controls = _controls_for_stems(dataset_root, stems)
    candidate, checkpoint, training = final_val.resolve_candidate(args.candidate)
    if candidate["label"] != "mix-025":
        raise ValueError("Prompting-guide study is locked to candidate mix-025")
    endpoints = final_val._interpolation_endpoint_models(candidate["interpolation"])
    final_val.validate_interpolation_trainable_state(endpoints[0], endpoints[1])
    buckets = {stem: [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8]
               for stem in stems for sample in (_sample_by_stem(dataset, stem),)}
    contract = {
        "kind": STUDY_KIND,
        "candidate": candidate["label"],
        "turbo": TURBO,
        "geometry": NATIVE_GEOMETRY,
        "prompt_file": {"path": str(Path(args.prompt_file)), "sha256": STUDY_SHA256, "record_count": len(rows)},
        "prompt_mapping": rows,
        "conditions": [{"stem": stem, "pose_class": next(row["pose_class"] for row in rows if row["stem"] == stem)} for stem in stems],
        "final_val_spec_sha256": spec_digest,
        "sampling_seeds": {stem: int(spec["per_stem_seeds"][stem]["sampling"]) for stem in stems},
        "buckets": buckets,
        "control_sha256": {stem: _sha256(controls[stem]) for stem in stems},
        **_candidate_contract(candidate, checkpoint),
    }
    output = Path(args.output_root)
    provenance = output / "prompting_study_provenance.json"
    if provenance.exists() and _read_json(provenance) != contract:
        raise ValueError(f"Existing output has conflicting immutable prompting-study provenance: {provenance}")
    if not provenance.exists():
        _write(provenance, contract)
    return contract, dataset, controls, candidate, checkpoint, training, output


def _image_name(candidate: Mapping[str, Any]) -> str:
    return final_val._image_name(candidate)


def _directory(output: Path, row: Mapping[str, Any]) -> Path:
    return output / "generations" / str(row["stem"]) / str(row["mode"])


def _copy_control(source: Path, target: Path) -> None:
    if target.exists() and _sha256(target) != _sha256(source):
        raise ValueError(f"Existing pose control conflicts with authoritative final-val control identity: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _generation_metadata(row: Mapping[str, Any], contract: Mapping[str, Any], candidate: Mapping[str, Any], control: Path, bucket: list[int]) -> dict[str, Any]:
    stem = str(row["stem"])
    return {
        "stem": stem, "pose_class": row["pose_class"], "mode": row["mode"], "prompt": row["prompt"],
        "seed": contract["sampling_seeds"][stem], "control_path": str(control), "control_sha256": contract["control_sha256"][stem],
        "bucket": bucket, "geometry": NATIVE_GEOMETRY, "prompt_file_sha256": STUDY_SHA256,
        "final_val_spec_sha256": contract["final_val_spec_sha256"], "candidate": candidate["label"],
        **contract["turbo"], **{key: contract[key] for key in ("candidate_kind", "checkpoint_step", "checkpoint_interpolation", "checkpoint_sha256") if key in contract},
    }


def _generation_status(output: Path, rows: list[dict[str, Any]], candidate: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    payload_path = output / "generation_results.json"
    payload = _read_json(payload_path) if payload_path.exists() else None
    observed, recorded = [], []
    for row in rows:
        stem, mode = row["stem"], row["mode"]
        directory = _directory(output, row); image, metadata_path = directory / _image_name(candidate), directory / "metadata.json"
        control = output / "controls" / f"{stem}.png"
        if image.is_file():
            try:
                with Image.open(image) as opened: opened.verify()
                with Image.open(control) as opened: opened.verify()
                metadata = _read_json(metadata_path)
                if (metadata.get("control_sha256") != contract["control_sha256"][stem]
                        or _sha256(control) != contract["control_sha256"][stem]
                        or metadata.get("geometry") != NATIVE_GEOMETRY
                        or metadata.get("seed") != contract["sampling_seeds"][stem]
                        or metadata.get("prompt") != row["prompt"]
                        or metadata.get("mode") != mode
                        or metadata.get("pose_class") != row["pose_class"]
                        or metadata.get("candidate") != candidate["label"]
                        or metadata.get("prompt_file_sha256") != STUDY_SHA256
                        or metadata.get("bucket") != contract["buckets"][stem]
                        or any(metadata.get(key) != value for key, value in contract["turbo"].items())
                        or any(metadata.get(key) != contract.get(key) for key in ("candidate_kind", "checkpoint_step", "checkpoint_interpolation", "checkpoint_sha256") if key in contract)
                        or metadata.get("control_path") is None):
                    raise ValueError("generation metadata contract mismatch")
            except Exception as exc:
                raise ValueError(f"Generation artifact is corrupt or contract-inconsistent: {image}") from exc
            observed.append(True)
        else:
            if directory.exists() or control.exists():
                raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")
            observed.append(False)
        if payload is not None:
            recorded.append(payload.get("generated_artifacts", {}).get(stem, {}).get(mode) == _image_name(candidate))
    if not any(observed) and payload is None:
        return "missing"
    expected_artifacts = {stem: {mode: _image_name(candidate) for mode in MODES}
                          for stem in contract["sampling_seeds"]}
    if (all(observed) and payload is not None and all(recorded)
            and payload.get("candidate") == candidate["label"]
            and payload.get("prompt_file") == contract["prompt_file"]
            and payload.get("prompt_mapping") == rows
            and payload.get("generated_artifacts") == expected_artifacts):
        return "complete"
    raise ValueError("Existing generation output is incomplete or inconsistent; refusing to overwrite it")


def _conditioning(conditioner: PoseTextConditioner, prompt: str, sample: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, masks = conditioner([prompt])
    compact = compact_valid_conditioning(contexts, masks, 0)
    context, mask = compact["context"], compact["mask"]
    if context.dtype != torch.bfloat16: context = context.to(torch.bfloat16)
    if (context.ndim != 3 or mask.ndim != 1 or context.shape[0] != mask.shape[0]
            or not mask.all().item() or not torch.isfinite(context).all().item()
            or tuple(context.shape[1:]) != tuple(sample["context"].shape[1:])):
        raise ValueError("Experimental prompt conditioning is empty, non-finite, or incompatible with frozen Qwen dimensions")
    return context.cpu().contiguous(), mask.cpu().to(torch.bool).contiguous()


def preflight(args: argparse.Namespace) -> None:
    rows = load_study_rows(args.prompt_file)
    contract, dataset, controls, _, _, training, output = _inputs(args, rows)
    _write(output / "checkpoint_preflight.json", {**contract, "generation_count": GENERATION_COUNT,
           "condition_count": CONDITION_COUNT, "mode_count": len(MODES), "dataset_sample_count": len(dataset),
           "control_paths_resolved": len(controls), "training_metadata": training,
           "source_rgb_fallback_permitted": False,
           "text_conditioning": "online PoseTextConditioner with exact frozen experimental prompt; source captions are never sampled"})
    print(output / "checkpoint_preflight.json")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run prompting-guide generation from the GH200 host shell with CUDA visible")
    rows = load_study_rows(args.prompt_file)
    contract, dataset, controls, candidate, checkpoint, training, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) == "complete":
        print(json.dumps({"already_complete": candidate["label"], "generation_count": GENERATION_COUNT})); return
    model = final_val.build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    trainable = final_val.candidate_trainable_state(candidate, checkpoint)
    final_val.load_trainable_state_dict(model, trainable)
    compatibility = raw_to_turbo_control_compatibility(model, final_val.candidate_raw_to_turbo_state(candidate, checkpoint, trainable))
    vae, conditioner = load_krea_vae("cuda"), PoseTextConditioner(device="cuda", dtype=torch.bfloat16)
    samples = {stem: dict(_sample_by_stem(dataset, stem)) for stem in contract["sampling_seeds"]}
    for row in rows:
        stem, directory = row["stem"], _directory(output, row)
        control_output = output / "controls" / f"{stem}.png"; _copy_control(controls[stem], control_output)
        sample = dict(samples[stem]); context, mask = _conditioning(conditioner, row["prompt"], sample)
        sample.update({"prompt": row["prompt"], "context": context, "mask": mask})
        bucket = [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8]
        if bucket != contract["buckets"][stem]:
            raise ValueError(f"Frozen native geometry changed for prompting-study stem {stem}")
        metadata = _generation_metadata(row, contract, candidate, controls[stem], bucket)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and _read_json(metadata_path) != metadata:
            raise ValueError(f"Existing generation metadata conflicts with frozen prompting-study contract: {metadata_path}")
        if not metadata_path.exists(): _write(metadata_path, metadata)
        pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                         torch.device("cuda"), metadata["seed"], control_scale=1.0)
        save_image(pixels, directory / _image_name(candidate))
    _write(output / "generation_results.json", {**contract, "generated_artifacts": {
        stem: {mode: _image_name(candidate) for mode in MODES} for stem in contract["sampling_seeds"]},
        "raw_to_turbo_control_compatibility": compatibility, "training_metadata": training,
        "source_rgb_fallback_used": False})
    print(output / "generation_results.json")


def _study_sidecar(reference_sidecar: str | Path, stems: list[str]) -> tuple[dict[str, Any], str]:
    spec, _ = final_val.load_final_spec(final_val.FINAL_SPEC)
    sidecar, digest = final_val._load_final_sidecar(reference_sidecar, list(spec["stems"]))
    records = {str(record["stem"]): record for record in sidecar["records"]}
    if set(stems) - set(records):
        raise ValueError("Authoritative final-val pose sidecar is missing frozen prompting-study stems")
    return {"records": [records[stem] for stem in stems]}, digest


def _compact_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row["pck"] for row in rows if row["pck"].get("reference_available")]
    pose = _pool_pose(available) if available else {"evaluable_sample_count": 0, "pck_005": None, "pck_010": None, "pck_020": None}
    clips = aggregate([float(row["clip_cosine_similarity"]) for row in rows])
    return {"generation_count": len(rows), "pck": pose, "pck_unavailable_count": len(rows) - len(available),
            "clip": {"mean_cosine_similarity": clips["mean"], "median_cosine_similarity": clips["median"], "std_cosine_similarity": clips["std"], "sample_count": clips["sample_count"]}}


def _aggregate_tables(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[record["mode"]].append(record); by_class[record["pose_class"]].append(record)
    if tuple(by_mode) != MODES or any(len(by_mode[mode]) != CONDITION_COUNT for mode in MODES) or len(by_class) != CONDITION_COUNT:
        raise ValueError("Scored prompting-study records do not form the exact 8 x 8 matrix")
    return (
        {"group_by": "prompt_mode", "rows": [{"mode": mode, **_compact_group(by_mode[mode])} for mode in MODES]},
        {"group_by": "pose_class", "rows": [{"pose_class": pose_class, **_compact_group(by_class[pose_class])} for pose_class in sorted(by_class)]},
    )


def _score_records(payload: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = payload.get("per_generation")
    expected_pairs = [(row["stem"], row["mode"]) for row in rows]
    if (not isinstance(records, list) or len(records) != GENERATION_COUNT
            or [(row.get("stem"), row.get("mode")) for row in records] != expected_pairs):
        raise ValueError("Prompting-study score artifact is incomplete or does not match frozen prompt mapping")
    return records


def score(args: argparse.Namespace) -> None:
    rows = load_study_rows(args.prompt_file)
    contract, dataset, _, candidate, _, _, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) != "complete":
        raise FileNotFoundError("Prompting-study scoring requires the complete validated 64-image generation set")
    if not args.reference_sidecar:
        raise ValueError("Prompting-study PCK requires the authoritative final-val --reference-sidecar")
    stems = list(contract["sampling_seeds"]); sidecar, sidecar_digest = _study_sidecar(args.reference_sidecar, stems)
    existing_scores = output / "pck_clip_results.json"
    if existing_scores.exists():
        existing = _read_json(existing_scores)
        if (any(existing.get(key) != value for key, value in contract.items())
                or existing.get("reference_sidecar_sha256") != sidecar_digest
                or existing.get("clip_model") != args.clip_model_id
                or existing.get("confidence_threshold") != .5):
            raise ValueError("Existing prompting-study score artifact conflicts with frozen candidate/provenance contract")
    sidecar_by_stem = {record["stem"]: record for record in sidecar["records"]}
    geometry = {stem: turbo_scoring_geometry(_sample_by_stem(dataset, stem)) for stem in stems}
    device = "cuda" if torch.cuda.is_available() else "cpu"; detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id); clip = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    from scripts.turbo_benchmark import _clip_score
    per_generation = []
    for row in rows:
        image = _directory(output, row) / _image_name(candidate)
        pck_result = score_authoritative_pck(sidecar={"records": [sidecar_by_stem[row["stem"]]]},
                                             geometry_by_stem={row["stem"]: geometry[row["stem"]]},
                                             image_for=lambda _: image, detector=detector,
                                             confidence_threshold=.5, require_images=True)
        per_image = pck_result["per_image"][0] if len(pck_result["per_image"]) == 1 else {
            "stem": row["stem"], "reference_available": False, "reason": pck_result["unavailable"][0]["reason"]}
        per_generation.append({"stem": row["stem"], "pose_class": row["pose_class"], "mode": row["mode"], "prompt": row["prompt"],
                               "image": str(image.relative_to(output)), "pck": per_image,
                               "clip_cosine_similarity": _clip_score(clip, processor, device, row["prompt"], image)})
    by_mode, by_pose_class = _aggregate_tables(per_generation)
    _write(output / "pck_clip_results.json", {**contract, "reference_sidecar": str(Path(args.reference_sidecar).resolve()),
           "reference_sidecar_sha256": sidecar_digest, "clip_model": args.clip_model_id, "confidence_threshold": .5,
           "per_generation": per_generation, "aggregate_by_prompt_mode": by_mode, "aggregate_by_pose_class": by_pose_class})
    print(output / "pck_clip_results.json")


def _validated_scores(output: Path, contract: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = _read_json(output / "pck_clip_results.json")
    if any(scores.get(key) != value for key, value in contract.items()):
        raise ValueError("Prompting-study score artifact conflicts with frozen candidate/provenance contract")
    return scores, _score_records(scores, rows)


def report(args: argparse.Namespace) -> None:
    rows = load_study_rows(args.prompt_file)
    contract, _, _, candidate, _, training, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) != "complete":
        raise FileNotFoundError("Prompting-study report requires the complete validated 64-image generation set")
    scores, per_generation = _validated_scores(output, contract, rows)
    by_mode, by_pose_class = _aggregate_tables(per_generation)
    if scores.get("aggregate_by_prompt_mode") != by_mode or scores.get("aggregate_by_pose_class") != by_pose_class:
        raise ValueError("Prompting-study aggregate score tables do not match individual scored generations")
    _write(output / "metrics_by_prompt_mode.json", {**contract, **by_mode})
    _write(output / "metrics_by_pose_class.json", {**contract, **by_pose_class})
    for condition in contract["conditions"]:
        stem = condition["stem"]
        paths = [output / "controls" / f"{stem}.png"] + [_directory(output, next(row for row in rows if row["stem"] == stem and row["mode"] == mode)) / _image_name(candidate) for mode in MODES]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"Prompting-study comparison grid requires control and all eight generations for {stem}; no RGB fallback exists")
        make_contact_sheet([(stem, paths)], output / "comparison_grids" / f"{stem}.png", thumbnail_width=220, thumbnail_height=220,
                           column_labels=("pose control",) + MODES)
    aggregate_rows = []
    for condition in contract["conditions"]:
        stem = condition["stem"]
        aggregate_rows.append((stem, [output / "controls" / f"{stem}.png"] + [_directory(output, next(row for row in rows if row["stem"] == stem and row["mode"] == mode)) / _image_name(candidate) for mode in MODES]))
    make_contact_sheet(aggregate_rows, output / "prompting_study_contact_sheet.png", thumbnail_width=180, thumbnail_height=180,
                       column_labels=("pose control",) + MODES)
    _write(output / "evaluation_summary.json", {**contract, "training_metadata": training, "generation_count": GENERATION_COUNT,
           "score_artifact": "pck_clip_results.json", "compact_tables": {"by_prompt_mode": "metrics_by_prompt_mode.json", "by_pose_class": "metrics_by_pose_class.json"},
           "comparison_grids": {condition["stem"]: str(Path("comparison_grids") / f"{condition['stem']}.png") for condition in contract["conditions"]},
           "aggregate_contact_sheet": "prompting_study_contact_sheet.png", "source_rgb_fallback_used": False})
    print(output / "evaluation_summary.json")


def summary(args: argparse.Namespace) -> None:
    rows = load_study_rows(args.prompt_file)
    contract, _, _, candidate, _, _, output = _inputs(args, rows)
    if _generation_status(output, rows, candidate, contract) != "complete":
        raise FileNotFoundError("Prompting-study summary requires the complete validated 64-image generation set")
    _, records = _validated_scores(output, contract, rows)
    by_mode, by_pose_class = _aggregate_tables(records)
    print(json.dumps({"candidate": candidate["label"], "generation_count": GENERATION_COUNT,
                      "by_prompt_mode": by_mode["rows"], "by_pose_class": by_pose_class["rows"]}, sort_keys=True))


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
    parser.add_argument("--prompt-file", default=str(STUDY_FILE))
    return parser


def main() -> None:
    args = parser().parse_args()
    {"preflight": preflight, "generate": generate, "score": score, "report": report, "summary": summary}[args.action](args)


if __name__ == "__main__":
    main()
