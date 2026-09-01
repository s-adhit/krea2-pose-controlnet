"""Read-only Mixed-32 768 coordinate-Huber gradient-balance calibration.

This operator audit reconstructs exact paired 768 inputs in memory and uses
``torch.autograd.grad`` on the production trainable parameter boundary.  It
never creates an optimizer, calls ``backward``, saves a checkpoint, writes a
latent cache, or changes model weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

import train
from pose_controlnet.capacity_pose import load_capacity_pose_records
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position
from pose_controlnet.keypoint_critic import FixedBoxKeypointRCNNCritic, differentiable_pose_loss
from pose_controlnet.keypoint_critic_audit import assert_frozen_no_parameter_grad, deterministic_noise_like
from pose_controlnet.model import audit_control_model, build_pose_model, trainable_params
from pose_controlnet.overfit_capacity import OVERFIT_SEED, RESOLUTION_768_BUCKETS, SelectedLatentShardDataset, validate_manifest
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.pose_reward_tools import gradient_interaction, lambda_calibration
from pose_controlnet.vae_preprocessing import decode_normalized_latents_autograd, encode_preprocessed_pair, load_krea_vae, qwen_decoded_to_unit_rgb
from scripts.audit_keypoint_critic import _person_tensors
from scripts.audit_keypoint_critic_timestep import unpatchify_latent_tokens


CALIBRATION_EXPERIMENT = "overfit32-mixed-r64-coord-calibration-res768"
DEFAULT_SIDECAR = Path("data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl")
DEFAULT_RAW_CKPT = "/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors"


class _InMemoryDataset:
    def __init__(self, samples: list[dict[str, Any]], text_conditioning: Any) -> None:
        self.samples, self.text_conditioning = samples, text_conditioning

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.samples[index])


def validate_candidate_lambdas(values: Iterable[float]) -> tuple[float, ...]:
    candidates = tuple(float(value) for value in values)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate lambdas must be a non-empty unique list")
    if any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("candidate lambdas must be finite and positive")
    return candidates


def validate_calibration_request(*, resolution: str, pose_loss: str, sidecar: Path,
                                 timestep_min: float, timestep_max: float,
                                 timesteps: Iterable[float], candidates: Iterable[float]) -> tuple[float, ...]:
    if resolution != "768" or pose_loss != "normalized_coordinate_huber":
        raise ValueError("calibration is defined only for Mixed-32 768 normalized_coordinate_huber")
    if sidecar.resolve() != DEFAULT_SIDECAR.resolve():
        raise ValueError(f"calibration must reuse the immutable authoritative Mixed sidecar: {DEFAULT_SIDECAR}")
    if not 0 < timestep_min <= timestep_max < 1:
        raise ValueError("pose timestep window must satisfy 0 < min <= max < 1")
    chosen = tuple(float(value) for value in timesteps)
    if not chosen or any(not timestep_min <= value <= timestep_max for value in chosen):
        raise ValueError("calibration timesteps must be non-empty values inside the selected pose timestep window")
    return validate_candidate_lambdas(candidates)


def candidate_weighted_ratios(interaction: dict[str, float | None], candidates: Iterable[float]) -> dict[str, float | None]:
    """Return ``||lambda grad L_pose|| / ||grad L_flow||`` per candidate."""
    ratio = interaction.get("ratio")
    return {f"{value:.9g}": None if ratio is None else float(value) * ratio for value in candidates}


def _encode_768_samples(*, selected: SelectedLatentShardDataset, dataset_root: Path, vae: Any,
                        device: torch.device) -> _InMemoryDataset:
    index = DatasetIndex.discover(dataset_root)
    samples: list[dict[str, Any]] = []
    for item_index in range(len(selected)):
        item, stem = selected[item_index], selected[item_index]["stem"]
        record = ManifestRecord(split="train", stem=stem, file_name=f"{stem}.jpg", text=item["prompt"],
                                rgb_path=index.rgb_by_stem[stem], control_path=index.control_by_stem[stem])
        pair = preprocess_pair(record, buckets=RESOLUTION_768_BUCKETS)
        generator = torch.Generator(device=device).manual_seed(
            int.from_bytes(f"mixed32-calibration:{stem}".encode(), "little") % (2**63 - 1)
        )
        with torch.no_grad():
            encoded = encode_preprocessed_pair(vae, pair, device=device, generator=generator)
        clean, control = encoded.latent.detach().float().cpu(), encoded.control.detach().float().cpu()
        if clean.shape != control.shape or not torch.isfinite(clean).all() or not torch.isfinite(control).all() or control.abs().max().item() == 0:
            raise ValueError(f"{stem}: in-memory 768 paired VAE encoding is invalid")
        samples.append({**item, "latent": clean, "control": control,
                        "source_size": list(pair.geometry.source_size), "resized_size": list(pair.geometry.resized_size),
                        "crop_box": list(pair.geometry.crop_box), "bucket": list(pair.geometry.bucket)})
    return _InMemoryDataset(samples, selected.text_conditioning)


def _gradients(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite calibration loss")
    gradients = list(torch.autograd.grad(loss, parameters, allow_unused=True))
    if any(gradient is not None and not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("non-finite calibration gradient")
    return gradients


def _forward_losses(model: torch.nn.Module, vae: Any, critic: Any, sample: dict[str, Any], pose_record: dict[str, Any],
                    *, timestep_value: float, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    clean = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    control = sample["control"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    timestep = torch.full((1,), timestep_value, device=device, dtype=torch.float32)
    noise = deterministic_noise_like(clean, seed=seed, stem=sample["stem"], label=f"mixed32-coordinate-{timestep_value:.6f}")
    noisy, target = make_flow_pair(clean, noise, timestep)
    context = sample["context"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    text_mask = sample["mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
    image_tokens, position, mask = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
    control_tokens, _, _ = patchify_and_position(control, context.shape[1], model.config.patch, text_mask)
    target_tokens, _, _ = patchify_and_position(target, context.shape[1], model.config.patch, text_mask)
    velocity = forward_pose_control(model, image_tokens, control_tokens, context, timestep.to(torch.bfloat16), position, mask,
                                    gradient_checkpointing_blocks=0)
    flow_loss = F.mse_loss(velocity.float(), target_tokens.float())
    x0_hat = unpatchify_latent_tokens(image_tokens - timestep.view(-1, 1, 1).to(image_tokens) * velocity,
                                      tuple(clean.shape[-2:]), model.config.patch)
    boxes, targets, valid = _person_tensors(pose_record, device)
    heatmaps = critic(qwen_decoded_to_unit_rgb(decode_normalized_latents_autograd(vae, x0_hat)).float(), [boxes])
    pose_loss = differentiable_pose_loss("normalized_coordinate_huber", heatmaps.logits, targets, heatmaps.boxes_training, valid)
    return flow_loss, pose_loss


def _panel(entries: list[dict[str, Any]], candidates: tuple[float, ...]) -> dict[str, Any]:
    if not entries:
        raise ValueError("gradient calibration has no eligible entries")
    flow = [entry["raw_flow_grad_norm"] for entry in entries]; pose = [entry["raw_pose_grad_norm"] for entry in entries]
    ratios = [entry["raw_pose_over_flow"] for entry in entries]
    mean_flow, mean_pose = sum(flow) / len(flow), sum(pose) / len(pose)
    interaction = {"flow_grad_norm": mean_flow, "pose_grad_norm": mean_pose,
                   "ratio": mean_pose / mean_flow if mean_flow > 0 else None, "dot": 0.0, "cosine": None}
    return {"sample_count": len(entries),
            "raw_flow_grad_norm": {"mean": mean_flow, "min": min(flow), "max": max(flow)},
            "raw_pose_grad_norm": {"mean": mean_pose, "min": min(pose), "max": max(pose)},
            "raw_pose_over_flow": {"mean": sum(ratios) / len(ratios), "min": min(ratios), "max": max(ratios)},
            "implied_lambda": lambda_calibration(interaction, targets=(.05, .10)),
            "candidate_weighted_pose_over_flow": candidate_weighted_ratios(interaction, candidates)}


def run_audit(*, sidecar: Path, dataset_root: Path, latent_root: Path, text_conditioning_root: Path,
              raw_ckpt: str, samples_per_source: int, timesteps: tuple[float, ...], seed: int,
              candidates: tuple[float, ...], timestep_min: float, timestep_max: float,
              device: torch.device) -> dict[str, Any]:
    stems = validate_manifest("overfit32-mixed-r64-mse")
    base = PreparedLatentShardDataset(str(latent_root), "train", text_conditioning_root=str(text_conditioning_root))
    selected = SelectedLatentShardDataset(base, stems)
    vae = load_krea_vae(device).eval()
    data = _encode_768_samples(selected=selected, dataset_root=dataset_root, vae=vae, device=device)
    sidecar_metadata, records = load_capacity_pose_records(sidecar=sidecar, experiment_name=CALIBRATION_EXPERIMENT, data=data, stems=stems)
    sources = ("coco", "humanart_painting", "humanart_real_human", "humanart_sculpture")
    chosen: list[str] = []
    for source in sources:
        rows = [stem for stem in stems if records[stem]["source"] == source and records[stem]["pose_reward_available"]]
        if len(rows) < samples_per_source:
            raise ValueError(f"{source}: insufficient eligible Mixed-32 pose records for calibration")
        chosen.extend(rows[:samples_per_source])
    model = build_pose_model(raw_ckpt, 64, 64, str(device)).eval(); model_report = audit_control_model(model, rank=64)
    parameters = trainable_params(model)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("calibration trainable parameter boundary is invalid")
    critic = FixedBoxKeypointRCNNCritic().to(device).eval(); assert_frozen_no_parameter_grad(vae, critic)
    entries: list[dict[str, Any]] = []
    for sample_index, stem in enumerate(chosen):
        item_index = stems.index(stem); batch = collate([data[item_index]])
        train.apply_cached_caption_dropout(batch, data.text_conditioning.unconditional, .10, seed, sample_index)
        sample = {**data[item_index], "context": batch["context"][0], "mask": batch["text_mask"][0]}
        for timestep in timesteps:
            model.zero_grad(set_to_none=True)
            flow_loss, _ = _forward_losses(model, vae, critic, sample, records[stem], timestep_value=timestep, seed=seed, device=device)
            flow_gradients = _gradients(flow_loss, parameters)
            model.zero_grad(set_to_none=True)
            _, pose_loss = _forward_losses(model, vae, critic, sample, records[stem], timestep_value=timestep, seed=seed, device=device)
            pose_gradients = _gradients(pose_loss, parameters)
            interaction = gradient_interaction(flow_gradients, pose_gradients)
            if interaction["flow_grad_norm"] <= 0 or interaction["pose_grad_norm"] <= 0:
                raise RuntimeError(f"{stem}: calibration requires finite nonzero raw flow and pose gradients")
            entries.append({"stem": stem, "source": records[stem]["source"], "timestep": timestep,
                            "flow_loss": float(flow_loss.item()), "pose_loss": float(pose_loss.item()),
                            "raw_flow_grad_norm": interaction["flow_grad_norm"], "raw_pose_grad_norm": interaction["pose_grad_norm"],
                            "raw_pose_over_flow": interaction["ratio"],
                            "implied_lambda": lambda_calibration(interaction, targets=(.05, .10)),
                            "candidate_weighted_pose_over_flow": candidate_weighted_ratios(interaction, candidates)})
            model.zero_grad(set_to_none=True); assert_frozen_no_parameter_grad(vae, critic)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("calibration left a model gradient")
    return {"audit_only": True, "optimizer_steps": 0, "model_weights_mutated": False,
            "experiment_contract": {"base": "Mixed-32", "training_resolution": 768, "pose_loss": "normalized_coordinate_huber",
                                    "evaluation_resolution": "native", "sidecar": str(sidecar), "stems": list(stems),
                                    "danbooru_numerical_pose_reward": "excluded"},
            "method": {"flow": "flow-matching MSE", "pose": "x0_hat -> autograd VAE decode -> frozen fixed-box Keypoint R-CNN -> soft expected normalized coordinates -> SmoothL1",
                       "caption_behavior": "existing deterministic cached 0.10 dropout", "timesteps": list(timesteps),
                       "pose_timestep_window": [timestep_min, timestep_max], "candidate_lambdas": list(candidates)},
            "sidecar_metadata": sidecar_metadata, "model": model_report,
            "selected_eligible_stems": chosen, "entries": entries, "aggregate": _panel(entries, candidates)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    value.add_argument("--dataset-root", type=Path, default="/lambda/nfs/adhit/krea2-pose/posebridge_hf")
    value.add_argument("--latent-root", type=Path, default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    value.add_argument("--text-conditioning-root", type=Path, default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    value.add_argument("--raw-ckpt", default=DEFAULT_RAW_CKPT); value.add_argument("--samples-per-source", type=int, default=2)
    value.add_argument("--pose-timestep-min", type=float, default=.10); value.add_argument("--pose-timestep-max", type=float, default=.20)
    value.add_argument("--timesteps", nargs="+", type=float, default=(.10, .15, .20))
    value.add_argument("--candidate-lambda", nargs="+", type=float, required=True)
    value.add_argument("--seed", type=int, default=OVERFIT_SEED); value.add_argument("--device", default="cuda")
    value.add_argument("--output-json", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.samples_per_source < 1: parser().error("--samples-per-source must be positive")
    candidates = validate_calibration_request(resolution="768", pose_loss="normalized_coordinate_huber", sidecar=args.sidecar,
        timestep_min=args.pose_timestep_min, timestep_max=args.pose_timestep_max, timesteps=args.timesteps, candidates=args.candidate_lambda)
    result = run_audit(sidecar=args.sidecar, dataset_root=args.dataset_root, latent_root=args.latent_root,
                       text_conditioning_root=args.text_conditioning_root, raw_ckpt=args.raw_ckpt,
                       samples_per_source=args.samples_per_source, timesteps=tuple(args.timesteps), seed=args.seed,
                       candidates=candidates, timestep_min=args.pose_timestep_min, timestep_max=args.pose_timestep_max,
                       device=torch.device(args.device))
    rendered = json.dumps(result, indent=2, sort_keys=True); print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
