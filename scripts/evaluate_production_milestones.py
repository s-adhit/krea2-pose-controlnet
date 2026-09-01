"""Dual-mode, local-only Turbo evaluation for the six production milestones.

The historical native diagnostic path is retained as an immutable benchmark.
Dynamic-768 is a separately rooted deployment-geometry check; it re-encodes
the resolved RGB/control pair with the production cache's shared bucket and
paired resize/crop implementation.  This entry point never trains or resumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.overfit_capacity import RESOLUTION_768_BUCKETS, deterministic_seed
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.production_milestone_evaluation import (
    EVALUATION_MODES,
    PRODUCTION_MILESTONE_STEPS,
    ProductionMilestoneEvaluationError,
    SUMMARY_COLUMNS,
    assert_mode_metadata,
    cross_checkpoint_summary,
    geometry_for_mode,
    mode_metadata,
    mode_output_root,
    normalize_modes,
)


def _turbo_metadata() -> dict[str, Any]:
    # Keep parser/preflight-style CPU checks independent of model-only imports.
    from pose_controlnet.turbo_evaluation import turbo_metadata
    return turbo_metadata()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing production milestone artifact: {path}") from None
    if not isinstance(value, dict):
        raise ProductionMilestoneEvaluationError(f"Malformed production milestone artifact: {path}")
    return value


def _manifest_stems(path: Path) -> tuple[str, ...]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return tuple(Path(record["file_name"]).stem for record in records)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProductionMilestoneEvaluationError(f"Malformed canonical diagnostic manifest: {path}") from exc


def _native_inputs(args: argparse.Namespace) -> tuple[Any, tuple[str, ...], dict[str, Any], dict[str, Any]]:
    from pose_controlnet.data import PreparedLatentShardDataset
    from pose_controlnet.dataset_index import validate_posebridge_snapshot
    from pose_controlnet.evaluation import _sample_by_stem, make_evaluation_spec
    from pose_controlnet.turbo_evaluation import assert_exact_diagnostic_stems, assert_turbo_diagnostic_contract
    dataset = PreparedLatentShardDataset(args.latent_root, "diagnostic_val", text_conditioning_root=args.text_conditioning_root)
    stems = assert_exact_diagnostic_stems(_manifest_stems(args.diagnostic_manifest), (record[3] for record in dataset.records), expected_count=24)
    spec = make_evaluation_spec(dataset, split="diagnostic_val", count=len(stems), seed=420200, kind="turbo_fixed_pose", stems=list(stems))
    spec["turbo"] = _turbo_metadata()
    canonical = _read(args.canonical_reference_spec)
    assert_turbo_diagnostic_contract(spec, canonical, branch_name="production milestones")
    snapshot = validate_posebridge_snapshot(args.dataset_root)
    physical = {record.stem: record for record in snapshot.records_by_split["diagnostic_val"]}
    if tuple(physical) != stems:
        # Membership is the invariant; snapshot's physical-record iteration is
        # allowed to differ from the immutable diagnostic manifest order.
        if set(physical) != set(stems):
            raise ProductionMilestoneEvaluationError("Resolved diagnostic pairs differ from the immutable diagnostic manifest")
    return dataset, stems, spec, physical


def _mode_root(args: argparse.Namespace, step: int, mode: str) -> Path:
    return mode_output_root(args.output_root, step, mode)


def _sample_geometry(mode: str, native_sample: Mapping[str, Any], physical_record: Any) -> dict[str, list[int]]:
    with Image.open(physical_record.rgb_path) as source:
        source_size = source.size
    return geometry_for_mode(mode=mode, native_sample=native_sample, source_size=source_size)


def _mode_sample(*, mode: str, native_sample: dict[str, Any], physical_record: Any, vae: Any | None) -> tuple[dict[str, Any], Any | None]:
    """Return a sample at native persisted geometry or dynamic cache geometry."""
    geometry = _sample_geometry(mode, native_sample, physical_record)
    if mode == "native":
        return native_sample, None
    if vae is None:
        raise ProductionMilestoneEvaluationError("Dynamic-768 generation requires the loaded VAE")
    pair = preprocess_pair(physical_record, buckets=RESOLUTION_768_BUCKETS)
    expected = {"source_size": list(pair.geometry.source_size), "resized_size": list(pair.geometry.resized_size),
                "crop_box": list(pair.geometry.crop_box), "bucket": list(pair.geometry.bucket)}
    if geometry != expected:
        raise ProductionMilestoneEvaluationError(f"Dynamic-768 geometry differs from shared paired preprocessing for {physical_record.stem}")
    generator = torch.Generator(device="cuda").manual_seed(deterministic_seed(physical_record.stem))
    from pose_controlnet.vae_preprocessing import encode_preprocessed_pair
    encoded = encode_preprocessed_pair(vae, pair, device="cuda", generator=generator)
    result = dict(native_sample)
    result.update({"latent": encoded.latent, "control": encoded.control, **expected})
    return result, pair


def _generation_complete(root: Path, stems: tuple[str, ...], step: int, mode: str) -> bool:
    index = root / "generation_results.json"
    if not index.is_file():
        if root.exists() and any(root.iterdir()):
            raise ProductionMilestoneEvaluationError(
                f"Partial production milestone output must be archived before retry: {root}"
            )
        return False
    payload = _read(index)
    if payload.get("checkpoint_step") != step or payload.get("mode") != mode or payload.get("stems") != list(stems):
        raise ProductionMilestoneEvaluationError(f"Existing generation index belongs to another checkpoint, mode, or diagnostic set: {index}")
    for stem in stems:
        directory = root / "fixed_pose" / stem
        image = directory / "generated.png"
        if not image.is_file() or not (directory / "metadata.json").is_file():
            raise ProductionMilestoneEvaluationError(f"Partial production milestone output must be archived before retry: {root}")
        assert_mode_metadata(_read(directory / "metadata.json"), mode=mode, stem=stem)
    return True


def _write_sample_artifacts(root: Path, *, mode: str, stem: str, sample: Mapping[str, Any], control: Image.Image | Path,
                            seed: int, geometry: Mapping[str, Any]) -> Path:
    directory = root / "fixed_pose" / stem
    directory.mkdir(parents=True, exist_ok=True)
    metadata = mode_metadata(mode=mode, stem=stem, prompt=str(sample["prompt"]), seed=seed, geometry=geometry) | {
        "control_scale": 1.0, **_turbo_metadata(),
    }
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file() and _read(metadata_path) != metadata:
        raise ProductionMilestoneEvaluationError(f"Existing sample metadata conflicts with immutable mode contract: {metadata_path}")
    if not metadata_path.is_file():
        _write(metadata_path, metadata)
    destination = directory / "control.png"
    if isinstance(control, Path):
        if destination.exists() and destination.read_bytes() != control.read_bytes():
            raise ProductionMilestoneEvaluationError(f"Existing native control conflicts with the locked historical control: {destination}")
        if not destination.exists():
            destination.write_bytes(control.read_bytes())
    else:
        if destination.exists():
            with Image.open(destination) as previous:
                if previous.size != control.size:
                    raise ProductionMilestoneEvaluationError(f"Existing dynamic control has the wrong bucket geometry: {destination}")
        else:
            control.save(destination)
    return directory


def generate(args: argparse.Namespace) -> None:
    from pose_controlnet.evaluation import _sample_by_stem
    from pose_controlnet.model import build_turbo_pose_model, load_trainable_state_dict
    from pose_controlnet.turbo_evaluation import exact_direct_local_turbo_checkpoints, raw_to_turbo_control_compatibility, sample_turbo_pose_image
    from pose_controlnet.vae_preprocessing import decode_normalized_latents, load_krea_vae
    if not torch.cuda.is_available():
        raise RuntimeError("Run production milestone Turbo generation only from the GH200 host shell with CUDA visible")
    modes = normalize_modes(args.modes)
    dataset, stems, spec, physical = _native_inputs(args)
    checkpoints = exact_direct_local_turbo_checkpoints(checkpoint_root=args.checkpoint_root, steps=tuple(args.steps))
    pending = [(step, mode) for step, _ in checkpoints for mode in modes if not _generation_complete(_mode_root(args, step, mode), stems, step, mode)]
    if not pending:
        print(json.dumps({"already_complete": [{"checkpoint_step": step, "mode": mode} for step, mode in pending]}))
        return
    model = build_turbo_pose_model(args.turbo_ckpt, 64, 64, "cuda").eval()
    vae = load_krea_vae("cuda")
    native_samples = {stem: dict(_sample_by_stem(dataset, stem)) for stem in stems}
    dynamic_samples: dict[str, tuple[dict[str, Any], Any]] = {}
    for step, checkpoint in checkpoints:
        active_modes = [mode for current, mode in pending if current == step]
        if not active_modes:
            continue
        state = load_training_state(checkpoint)
        if state["global_step"] != step:
            raise ProductionMilestoneEvaluationError(f"Checkpoint identity mismatch for step {step}")
        compatibility = raw_to_turbo_control_compatibility(model, state)
        load_trainable_state_dict(model, state["model"])
        for mode in active_modes:
            root = _mode_root(args, step, mode)
            for stem in stems:
                native = native_samples[stem]
                if mode == "dynamic-768":
                    if stem not in dynamic_samples:
                        sample, pair = _mode_sample(mode=mode, native_sample=native, physical_record=physical[stem], vae=vae)
                        dynamic_samples[stem] = (sample, pair)
                    sample, pair = dynamic_samples[stem]
                    control: Image.Image | Path = pair.control
                else:
                    sample, pair, control = native, None, Path(physical[stem].control_path)
                geometry = _sample_geometry(mode, native, physical[stem])
                directory = _write_sample_artifacts(root, mode=mode, stem=stem, sample=sample, control=control,
                                                    seed=int(spec["per_stem_seeds"][stem]["sampling"]), geometry=geometry)
                pixels = sample_turbo_pose_image(model, lambda latent: decode_normalized_latents(vae, latent), sample,
                                                 torch.device("cuda"), int(spec["per_stem_seeds"][stem]["sampling"]), control_scale=1.0)
                from pose_controlnet.evaluation import save_image
                save_image(pixels, directory / "generated.png")
            _write(root / "generation_results.json", {
                "checkpoint_step": step, "mode": mode, "stems": list(stems), "turbo": _turbo_metadata(),
                "control_scale": 1.0, "raw_to_turbo_control_compatibility": compatibility,
            })
    print(args.output_root)


def score(args: argparse.Namespace) -> None:
    from transformers import CLIPModel, CLIPProcessor
    from pose_controlnet.post1500_evaluation import score_authoritative_pck
    from pose_controlnet.post500_evaluation import KeypointRCNNEstimator, aggregate
    from pose_controlnet.evaluation import _sample_by_stem
    from scripts.turbo_benchmark import _clip_score
    modes = normalize_modes(args.modes)
    dataset, stems, _, physical = _native_inputs(args)
    sidecar = _read(args.reference_sidecar)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = KeypointRCNNEstimator(device, .5)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id, local_files_only=True)
    clip = CLIPModel.from_pretrained(args.clip_model_id, local_files_only=True).to(device).eval()
    for step in args.steps:
        for mode in modes:
            root = _mode_root(args, step, mode)
            if not _generation_complete(root, stems, step, mode):
                raise FileNotFoundError(f"Cannot score incomplete generation: {root}")
            result_path = root / "pck_clip_results.json"
            if result_path.is_file():
                existing = _read(result_path)
                if existing.get("checkpoint_step") == step and existing.get("mode") == mode:
                    continue
                raise ProductionMilestoneEvaluationError(f"Existing score belongs to another checkpoint/mode: {result_path}")
            geometry = {stem: _sample_geometry(mode, dict(_sample_by_stem(dataset, stem)), physical[stem]) for stem in stems}
            image_for = lambda stem, current=root: current / "fixed_pose" / stem / "generated.png"
            pose = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for,
                                           detector=detector, confidence_threshold=.5, require_images=True)
            clip_rows = [{"stem": stem, "source": next(row.get("source") for row in sidecar["records"] if row["stem"] == stem),
                          "cosine_similarity": _clip_score(clip, processor, device, _read(root / "fixed_pose" / stem / "metadata.json")["prompt"], image_for(stem))}
                         for stem in stems]
            values = aggregate([row["cosine_similarity"] for row in clip_rows])
            _write(result_path, {"checkpoint_step": step, "mode": mode, "turbo": _turbo_metadata(), "control_scale": 1.0,
                                 "confidence_threshold": .5, "pose": pose,
                                 "clip": {"mean_cosine_similarity": values["mean"], "median_cosine_similarity": values["median"],
                                          "std_cosine_similarity": values["std"], "sample_count": values["sample_count"], "per_sample": clip_rows}})
    print(args.output_root)


def report(args: argparse.Namespace) -> None:
    rows = []
    for step in args.steps:
        for mode in normalize_modes(args.modes):
            row = _read(_mode_root(args, step, mode) / "pck_clip_results.json")
            if row.get("checkpoint_step") != step or row.get("mode") != mode:
                raise ProductionMilestoneEvaluationError("Score artifacts may not cross checkpoint or mode boundaries")
            rows.append(row)
    summary = cross_checkpoint_summary(rows)
    root = Path(args.output_root)
    _write(root / "evaluation_summary.json", summary)
    with (root / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary["checkpoints"]:
            pose, clip = row["pose"], row["clip"]
            writer.writerow({"checkpoint_step": row["checkpoint_step"], "mode": row["mode"],
                             "pck_005": pose["pck_005"], "pck_010": pose["pck_010"], "pck_020": pose["pck_020"],
                             "clip_mean_cosine_similarity": clip["mean_cosine_similarity"]})
    print(root / "evaluation_summary.json")


def evaluate(args: argparse.Namespace) -> None:
    generate(args)
    score(args)
    report(args)


def contact_sheet_filename(mode: str) -> str:
    """Stable, mode-isolated filename for a full production contact sheet."""
    return f"{mode.replace('-', '')}_full_contact_sheet.png"


def _contact_sheet_stems(evaluation_root: Path, *, steps: tuple[int, ...], mode: str) -> tuple[str, ...]:
    expected: tuple[str, ...] | None = None
    for step in steps:
        index = evaluation_root / f"step_{step:06d}" / mode / "generation_results.json"
        payload = _read(index)
        stems = payload.get("stems")
        if payload.get("checkpoint_step") != step or payload.get("mode") != mode or not isinstance(stems, list):
            raise ProductionMilestoneEvaluationError(f"Generation index has the wrong checkpoint, mode, or stem list: {index}")
        normalized = tuple(str(stem) for stem in stems)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ProductionMilestoneEvaluationError(f"Generation index has an invalid stem list: {index}")
        if expected is None:
            expected = normalized
        elif normalized != expected:
            raise ProductionMilestoneEvaluationError(f"Generation stem order differs at step {step} for mode {mode}: {index}")
    assert expected is not None
    return expected


def contact_sheet(args: argparse.Namespace) -> None:
    """Render one read-only, full progression sheet for each requested mode."""
    from pose_controlnet.dataset_index import validate_posebridge_snapshot
    from pose_controlnet.evaluation import make_contact_sheet

    steps = tuple(args.steps)
    if not steps or len(set(steps)) != len(steps) or any(step <= 0 for step in steps):
        raise ProductionMilestoneEvaluationError("Contact-sheet steps must be unique positive optimizer steps")
    modes = normalize_modes(args.modes)
    snapshot = validate_posebridge_snapshot(args.dataset_root)
    physical = {record.stem: record for record in snapshot.records_by_split["diagnostic_val"]}
    for mode in modes:
        stems = _contact_sheet_stems(args.evaluation_root, steps=steps, mode=mode)
        missing_pairs = [stem for stem in stems if stem not in physical]
        if missing_pairs:
            raise ProductionMilestoneEvaluationError(
                f"Diagnostic source RGB/control pair is missing for contact-sheet stem(s): {missing_pairs[:4]}"
            )
        rows: list[tuple[str, list[Path]]] = []
        for stem in stems:
            record = physical[stem]
            images = [Path(record.control_path), Path(record.rgb_path)]
            for step in steps:
                generated = args.evaluation_root / f"step_{step:06d}" / mode / "fixed_pose" / stem / "generated.png"
                if not generated.is_file():
                    raise FileNotFoundError(f"Missing generated production milestone image: {generated}")
                images.append(generated)
            rows.append((stem, images))
        make_contact_sheet(rows, args.output_dir / contact_sheet_filename(mode), thumbnail_width=320,
                           thumbnail_height=320,
                           column_labels=("Pose control", "Target RGB", *(f"Step {step}" for step in steps)))
    print(args.output_dir)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="command_name")
    for name, function in (("generate", generate), ("score", score), ("report", report), ("evaluate", evaluate)):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--latent-root", type=str, required=True)
        command.add_argument("--text-conditioning-root", type=str, required=True)
        command.add_argument("--turbo-ckpt", type=Path, required=True)
        command.add_argument("--reference-sidecar", type=Path, required=True)
        command.add_argument("--diagnostic-manifest", type=Path, default=Path("data/manifests/diagnostic_val.jsonl"))
        command.add_argument("--canonical-reference-spec", type=Path, default=Path("docs/evaluation/turbo-8step-cfg0/turbo_spec.json"))
        command.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
        command.add_argument("--steps", type=int, nargs="+", default=list(PRODUCTION_MILESTONE_STEPS))
        command.add_argument("--modes", nargs="+", default=list(EVALUATION_MODES), choices=EVALUATION_MODES)
        command.set_defaults(function=function)
    contact = sub.add_parser("contact-sheet")
    contact.add_argument("--evaluation-root", type=Path, required=True)
    contact.add_argument("--dataset-root", type=Path, required=True)
    contact.add_argument("--steps", type=int, nargs="+", required=True)
    contact.add_argument("--modes", nargs="+", required=True, choices=EVALUATION_MODES)
    contact.add_argument("--output-dir", type=Path, required=True)
    contact.set_defaults(function=contact_sheet)
    return parser


def main() -> None:
    args = parser().parse_args()
    if args.command_name != "contact-sheet":
        steps = tuple(args.steps)
        if not steps or len(set(steps)) != len(steps) or any(step <= 0 for step in steps):
            raise ProductionMilestoneEvaluationError("Production milestone evaluation steps must be unique positive integers")
    args.function(args)


if __name__ == "__main__":
    main()
