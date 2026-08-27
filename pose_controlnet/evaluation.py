"""Deterministic, checkpoint-comparable flow and pose-control evaluation."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pose_controlnet.checkpointing import load_training_state, validated_hf_checkpoint_for_step
from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position, sample_eval_image, sample_flow_timestep
from pose_controlnet.model import load_trainable_state_dict
from pose_controlnet.vae_preprocessing import decode_normalized_latents


EVALUATION_FORMAT_VERSION = 1
DEFAULT_FIXED_FLOW_SEED = 420_100
DEFAULT_FIXED_POSE_SEED = 420_200
CHECKPOINT_STEPS = (0, 20, 40, 60, 80, 100, 200, 225, 350, 475, 500,
                    600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500)
POST_500_CHECKPOINT_STEPS = CHECKPOINT_STEPS[11:]
COMPARISON_GRID_COLUMNS = ("Control", "Step 0", "Step 20", "Step 40", "Step 60", "Step 80", "Step 100", "Step 200", "Step 225", "Step 350", "Step 475", "Step 500", "Step 600", "Step 700", "Step 800", "Step 900", "Step 1000", "Step 1100", "Step 1200", "Step 1300", "Step 1400", "Step 1500")
DEFAULT_COMPARISON_GRID_THUMBNAIL_WIDTH = 320
DEFAULT_COMPARISON_GRID_THUMBNAIL_HEIGHT = 320


def ordered_checkpoints(checkpoint_dir: str | Path, steps: Iterable[int] = CHECKPOINT_STEPS,
                        later_checkpoint_dir: str | Path | None = None, *, hf_repo_id: str = "",
                        hf_run_name: str = "pose-learning-1500",
                        hf_recovery_dir: str | Path | None = None,
                        archive_checkpoint_dir: str | Path | None = None) -> list[tuple[int, Path | None]]:
    """Resolve only the canonical archive identities, validating every state.

    The three local roots deliberately mirror the training archives: <=100,
    200..500, and 600..1500.  Only the final archive may recover a missing
    local file from HF, and recovery is exact-step/marker/checksum/schema
    validated by ``validated_hf_checkpoint_for_step``.
    """
    root = Path(checkpoint_dir)
    later_root = Path(later_checkpoint_dir) if later_checkpoint_dir is not None else root
    archive_root = Path(archive_checkpoint_dir) if archive_checkpoint_dir is not None else later_root
    result: list[tuple[int, Path | None]] = []
    for step in steps:
        if step == 0:
            result.append((0, None)); continue
        local_root = root if step <= 100 else (later_root if step <= 500 else archive_root)
        path = local_root / f"step_{step:06d}.pt"
        if not path.is_file() and step >= 600 and hf_repo_id:
            recovery_dir = Path(hf_recovery_dir) if hf_recovery_dir is not None else archive_root / "hf-recovery"
            recovered = validated_hf_checkpoint_for_step(repo_id=hf_repo_id, run_name=hf_run_name, step=step,
                                                         download_dir=recovery_dir)
            if recovered is None:
                raise FileNotFoundError(f"Required checkpoint step {step} is unavailable locally and has no validated completion-marked HF copy")
            path = recovered
        state = load_training_state(path)
        if state["global_step"] != step:
            raise ValueError(f"Checkpoint filename/embedded step mismatch: {path} has {state['global_step']}")
        result.append((step, path))
    return result


def _stable_seed(seed: int, stem: str, label: str) -> int:
    payload = f"{seed}:{stem}:{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def select_fixed_stems(dataset: PreparedLatentShardDataset, count: int, seed: int) -> list[str]:
    if count < 1 or count > len(dataset): raise ValueError(f"sample count must be in [1, {len(dataset)}]")
    ranked = sorted((_stable_seed(seed, record[3], "selection"), record[3]) for record in dataset.records)
    return [stem for _, stem in ranked[:count]]


def make_evaluation_spec(dataset: PreparedLatentShardDataset, *, split: str, count: int, seed: int,
                         kind: str, stems: list[str] | None = None) -> dict[str, Any]:
    selected = stems or select_fixed_stems(dataset, count, seed)
    available = {record[3] for record in dataset.records}
    if len(selected) != len(set(selected)) or any(stem not in available for stem in selected):
        raise ValueError("Evaluation stems must be unique and present in the immutable split")
    identities = {}
    for stem in selected:
        sample = _sample_by_stem(dataset, stem)
        identities[stem] = {"image_latent_sha256": _tensor_sha256(sample["latent"]),
                            "control_latent_sha256": _tensor_sha256(sample["control"]),
                            "context_sha256": _tensor_sha256(sample["context"]),
                            "mask_sha256": _tensor_sha256(sample["mask"])}
    return {"format_version": EVALUATION_FORMAT_VERSION, "kind": kind, "split": split,
            "seed": seed, "stems": selected,
            "per_stem_seeds": {stem: {"timestep": _stable_seed(seed, stem, "timestep"),
                                        "noise": _stable_seed(seed, stem, "noise"),
                                        "sampling": _stable_seed(seed, stem, "sampling")}
                               for stem in selected}, "sample_identities": identities}


def write_spec(path: str | Path, spec: dict[str, Any]) -> Path:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def read_or_create_spec(path: str | Path, dataset: PreparedLatentShardDataset, *, split: str,
                        count: int, seed: int, kind: str) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        spec = json.loads(destination.read_text(encoding="utf-8"))
        if spec.get("format_version") != EVALUATION_FORMAT_VERSION or spec.get("kind") != kind or spec.get("split") != split:
            raise ValueError(f"Existing evaluation spec is incompatible: {destination}")
        observed = make_evaluation_spec(dataset, split=split, count=len(spec.get("stems", [])), seed=int(spec.get("seed", -1)), kind=kind, stems=spec.get("stems"))
        if spec.get("sample_identities") != observed["sample_identities"] or spec.get("per_stem_seeds") != observed["per_stem_seeds"]:
            raise ValueError("Existing evaluation spec no longer matches the immutable latent/text inputs")
        return spec
    return make_evaluation_spec(dataset, split=split, count=count, seed=seed, kind=kind)


def _sample_by_stem(dataset: PreparedLatentShardDataset, stem: str) -> dict:
    for index, record in enumerate(dataset.records):
        if record[3] == stem: return dataset[index]
    raise KeyError(stem)


def fixed_flow_inputs(sample: dict, cfg: TrainConfig, model, *, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the fixed flow timestep/noise solely from sample identity and seed."""
    clean = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    seq_len = (clean.shape[-2] // model.config.patch) * (clean.shape[-1] // model.config.patch)
    timestep = sample_flow_timestep(1, seq_len, cfg, "cpu", torch.Generator().manual_seed(_stable_seed(seed, sample["stem"], "timestep"))).to(device)
    noise = torch.randn(clean.shape, dtype=torch.float32, generator=torch.Generator().manual_seed(_stable_seed(seed, sample["stem"], "noise"))).to(device)
    return timestep, noise


@torch.inference_mode()
def fixed_flow_loss(model, dataset: PreparedLatentShardDataset, spec: dict[str, Any], cfg: TrainConfig, device: torch.device) -> dict[str, Any]:
    was_training = model.training; model.eval(); per_sample = []
    try:
        for stem in spec["stems"]:
            sample = _sample_by_stem(dataset, stem); timestep, noise = fixed_flow_inputs(sample, cfg, model, seed=int(spec["seed"]), device=device)
            clean = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
            control = sample["control"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            noisy, target = make_flow_pair(clean, noise, timestep)
            context = sample["context"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            text_mask = sample["mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
            image, pos, mask = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
            control_tokens, _, _ = patchify_and_position(control, context.shape[1], model.config.patch, text_mask)
            target_tokens, _, _ = patchify_and_position(target, context.shape[1], model.config.patch, text_mask)
            prediction = forward_pose_control(model, image, control_tokens, context, timestep.to(torch.bfloat16), pos, mask, gradient_checkpointing_blocks=0)
            per_sample.append({"stem": stem, "loss": float(F.mse_loss(prediction.float(), target_tokens.float()).item())})
    finally:
        model.train(was_training)
    losses = torch.tensor([item["loss"] for item in per_sample], dtype=torch.float64)
    return {"sample_count": len(per_sample), "mean_fixed_flow_loss": float(losses.mean()),
            "median_fixed_flow_loss": float(losses.median()), "std_fixed_flow_loss": float(losses.std(unbiased=False)), "per_sample": per_sample}


def load_comparison_state(model, checkpoint: Path | None) -> int:
    """Load only a strict project trainable state; ``None`` is verified step-0 init."""
    if checkpoint is None:
        return 0
    state = load_training_state(checkpoint)
    load_trainable_state_dict(model, state["model"])
    return int(state["global_step"])


def evaluate_fixed_flow(model, dataset, spec, cfg, device, checkpoints) -> dict[str, Any]:
    results = []
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items() if name.startswith("first.") or ".A" in name or ".B" in name}
    for step, checkpoint in checkpoints:
        if checkpoint is None: load_trainable_state_dict(model, baseline)
        else: load_comparison_state(model, checkpoint)
        results.append({"checkpoint_step": step, **fixed_flow_loss(model, dataset, spec, cfg, device)})
    return {"format_version": EVALUATION_FORMAT_VERSION, "kind": "fixed_flow", "config": asdict(cfg), "spec": spec, "checkpoints": results}


def save_image(array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(array).save(path)


def make_contact_sheet(rows: list[tuple[str, list[Path]]], path: Path, *, thumbnail_width: int = DEFAULT_COMPARISON_GRID_THUMBNAIL_WIDTH,
                       thumbnail_height: int = DEFAULT_COMPARISON_GRID_THUMBNAIL_HEIGHT,
                       column_labels: tuple[str, ...] | None = None) -> None:
    """Render a compact, deterministic grid without altering any generated images."""
    if column_labels is None:
        column_labels = tuple(f"column {index}" for index in range(len(rows[0][1]) if rows else 0))
    if thumbnail_width < 1 or thumbnail_height < 1:
        raise ValueError("comparison-grid thumbnail dimensions must be positive")
    if not rows or len(column_labels) != len(rows[0][1]) or any(len(paths) != len(column_labels) for _, paths in rows):
        raise ValueError("comparison grid rows must have one image for every column label")
    header_height, gutter = 24, 4
    sheet = Image.new("RGB", (len(column_labels) * thumbnail_width, header_height + len(rows) * thumbnail_height), "white")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(column_labels):
        draw.text((column * thumbnail_width + gutter, gutter), label, fill="black")
    for row_index, (stem, paths) in enumerate(rows):
        y = header_height + row_index * thumbnail_height
        for column, item in enumerate(paths):
            with Image.open(item) as source:
                image = source.convert("RGB")
            image.thumbnail((thumbnail_width - 2 * gutter, thumbnail_height - 2 * gutter))
            x = column * thumbnail_width + (thumbnail_width - image.width) // 2
            sheet.paste(image, (x, y + (thumbnail_height - image.height) // 2))
        draw.text((gutter, y + gutter), stem, fill="black", stroke_width=1, stroke_fill="white")
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)


def evaluate_fixed_pose(model, dataset, spec, cfg, device, checkpoints, vae, control_paths: dict[str, Path], output: Path, *,
                        reuse_existing: bool = True,
                        thumbnail_width: int = DEFAULT_COMPARISON_GRID_THUMBNAIL_WIDTH,
                        thumbnail_height: int = DEFAULT_COMPARISON_GRID_THUMBNAIL_HEIGHT) -> dict[str, Any]:
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items() if name.startswith("first.") or ".A" in name or ".B" in name}
    rows = []; generated_steps: dict[str, list[int]] = {}; reused_steps: dict[str, list[int]] = {}
    was_training = model.training; model.eval()
    try:
        for stem in spec["stems"]:
            sample, directory = dict(_sample_by_stem(dataset, stem)), output / "fixed_pose" / stem
            sample["unconditional_context"] = dataset.text_conditioning.unconditional["context"]
            sample["unconditional_mask"] = dataset.text_conditioning.unconditional["mask"]
            directory.mkdir(parents=True, exist_ok=True); control_file = directory / "control.png"
            if not control_file.exists(): shutil.copy2(control_paths[stem], control_file)
            metadata = {"stem": stem, "prompt": sample["prompt"], "control_path": str(control_paths[stem]), "seed": spec["per_stem_seeds"][stem]["sampling"], "bucket": [sample["latent"].shape[-1] * 8, sample["latent"].shape[-2] * 8], "eval_steps": cfg.eval_steps, "eval_guidance": cfg.eval_guidance}
            metadata_path = directory / "metadata.json"
            if metadata_path.exists():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(existing.get(key) != value for key, value in metadata.items()):
                    raise ValueError(f"Existing fixed-pose metadata conflicts with immutable contract: {metadata_path}")
            else:
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            paths = [control_file]
            for step, checkpoint in checkpoints:
                image_path = directory / f"step_{step:06d}.png"
                if reuse_existing and image_path.exists():
                    paths.append(image_path); reused_steps.setdefault(stem, []).append(step); continue
                if checkpoint is None: load_trainable_state_dict(model, baseline)
                else: load_comparison_state(model, checkpoint)
                pixels = sample_eval_image(model, lambda latent: decode_normalized_latents(vae, latent), None, sample, cfg, device, int(spec["per_stem_seeds"][stem]["sampling"]))
                save_image(pixels, image_path); paths.append(image_path); generated_steps.setdefault(stem, []).append(step)
            rows.append((stem, paths))
    finally:
        model.train(was_training)
    # An incremental call cannot claim to make a longitudinal grid.  Build one
    # only when all selected images are present, which avoids overwriting the
    # historical 0..500 outputs simply to extend the series.
    labels = ("control", *(f"step{step}" for step, _ in checkpoints))
    comparison_grid = output / "fixed_pose" / "comparison_grid.png"
    if rows and not (reuse_existing and comparison_grid.exists()):
        make_contact_sheet(rows, output / "fixed_pose" / "comparison_grid.png", thumbnail_width=thumbnail_width,
                           thumbnail_height=thumbnail_height, column_labels=labels)
    return {"format_version": EVALUATION_FORMAT_VERSION, "kind": "fixed_pose", "spec": spec, "checkpoints": [step for step, _ in checkpoints], "output": str(output / "fixed_pose"), "generated_steps": generated_steps, "reused_steps": reused_steps}
