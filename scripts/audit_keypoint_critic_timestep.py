"""Gate C audit: step-1500 x0_hat pose quality and output-gradient exposure.

This is a read-only evaluation of the exact project model/VAE/control path. It
does not alter ``train.py``, add a pose loss, or call an optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.dataset_index import DatasetIndex
from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position
from pose_controlnet.keypoint_critic import (
    FixedBoxKeypointRCNNCritic,
    detached_pose_diagnostics,
    gaussian_heatmap_kl,
    normalized_coordinate_huber,
    soft_coordinates,
)
from pose_controlnet.keypoint_critic_audit import (
    AUDIT_SOURCES,
    assert_authoritative_geometry_unchanged,
    assert_frozen_no_parameter_grad,
    authoritative_geometry_snapshot,
    deterministic_noise_like,
    distribution_statistics,
    metric_deltas,
    parse_timesteps,
    reconstruct_clean_latent,
    weighted_metric_mean,
)
from pose_controlnet.model import build_pose_model, load_trainable_state_dict
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.vae_preprocessing import (
    decode_normalized_latents,
    decode_normalized_latents_autograd,
    load_krea_vae,
    qwen_decoded_to_unit_rgb,
)
from scripts.audit_keypoint_critic import _person_tensors, _preprocessed_rgb, _rgb_tensor, _usable


DEFAULT_CHECKPOINT = Path("/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt")
DEFAULT_CHECKPOINT_SHA256 = "6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0"
DEFAULT_RAW_CKPT = "/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpatchify_latent_tokens(tokens: torch.Tensor, latent_hw: tuple[int, int], patch: int) -> torch.Tensor:
    """Invert the project image-token layout without changing token order."""
    height, width = latent_hw
    if height % patch or width % patch:
        raise ValueError("latent spatial dimensions must be divisible by model patch size")
    expected_tokens = (height // patch) * (width // patch)
    if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
        raise ValueError("token count does not match requested latent spatial dimensions")
    channels = tokens.shape[2] // (patch * patch)
    if channels * patch * patch != tokens.shape[2]:
        raise ValueError("token feature width is not an integral patch layout")
    return rearrange(tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                     h=height // patch, w=width // patch, c=channels, ph=patch, pw=patch)


def _critic_metrics(critic: FixedBoxKeypointRCNNCritic, rgb: torch.Tensor, boxes: torch.Tensor,
                    targets: torch.Tensor, valid: torch.Tensor, *, temperature: float,
                    gaussian_sigma: float) -> dict[str, float | int | None]:
    heatmaps = critic(rgb, [boxes])
    logits = heatmaps.logits
    coordinates = soft_coordinates(logits, heatmaps.boxes_training, temperature)
    result: dict[str, float | int | None] = {
        "gaussian_heatmap_kl": float(gaussian_heatmap_kl(
            logits, targets, heatmaps.boxes_training, valid, sigma=gaussian_sigma, temperature=temperature,
        ).item()),
        "normalized_coordinate_huber": float(normalized_coordinate_huber(
            coordinates, targets, heatmaps.boxes_training, valid,
        ).item()),
    }
    result.update(detached_pose_diagnostics(
        logits, heatmaps.boxes_training, targets, valid, temperature=temperature, include_argmax=False,
    ))
    return result


def _clear_and_assert_frozen(*modules: torch.nn.Module) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)
    assert_frozen_no_parameter_grad(*modules)


def _model_x0_hat(model: torch.nn.Module, sample: dict[str, Any], timestep_value: float,
                  *, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact cached-conditioning/control forward path and recover x0_hat.

    The model receives BF16 patchified ``x_t`` just as production does.  The
    reconstruction uses that exact token tensor, so it is the same numerical
    input state on which ``v_hat`` was predicted.
    """
    clean = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    control = sample["control"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    noise = deterministic_noise_like(clean, seed=seed, stem=sample["stem"], label="gate-c-noise")
    timestep = torch.full((1,), timestep_value, dtype=torch.float32, device=device)
    noisy, _ = make_flow_pair(clean, noise, timestep)
    context = sample["context"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    text_mask = sample["mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
    image_tokens, pos, mask = patchify_and_position(
        noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask,
    )
    control_tokens, _, _ = patchify_and_position(control, context.shape[1], model.config.patch, text_mask)
    velocity = forward_pose_control(
        model, image_tokens, control_tokens, context, timestep.to(torch.bfloat16), pos, mask,
        gradient_checkpointing_blocks=0,
    )
    x0_hat_tokens = reconstruct_clean_latent(image_tokens, velocity, timestep)
    x0_hat = unpatchify_latent_tokens(x0_hat_tokens, tuple(clean.shape[-2:]), model.config.patch)
    return velocity, x0_hat


def _gradient_audit(model: torch.nn.Module, vae: Any, critic: FixedBoxKeypointRCNNCritic, sample: dict[str, Any],
                    boxes: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, timestep: float, *,
                    device: torch.device, seed: int, temperature: float, gaussian_sigma: float,
                    loss_name: str) -> dict[str, float]:
    """Measure one independent dL/dv_hat and dL/dx0_hat graph; never accumulates param grads."""
    model.zero_grad(set_to_none=True)
    _clear_and_assert_frozen(vae, critic)
    velocity, x0_hat = _model_x0_hat(model, sample, timestep, device=device, seed=seed)
    decoded = decode_normalized_latents_autograd(vae, x0_hat)
    heatmaps = critic(qwen_decoded_to_unit_rgb(decoded).float(), [boxes])
    if loss_name == "gaussian_heatmap_kl":
        loss = gaussian_heatmap_kl(heatmaps.logits, targets, heatmaps.boxes_training, valid,
                                   sigma=gaussian_sigma, temperature=temperature)
    elif loss_name == "normalized_coordinate_huber":
        loss = normalized_coordinate_huber(
            soft_coordinates(heatmaps.logits, heatmaps.boxes_training, temperature),
            targets, heatmaps.boxes_training, valid,
        )
    else:
        raise ValueError(f"unknown output-gradient audit loss: {loss_name}")
    velocity_grad, x0_hat_grad = torch.autograd.grad(loss, (velocity, x0_hat))
    if not torch.isfinite(loss) or not torch.isfinite(velocity_grad).all() or not torch.isfinite(x0_hat_grad).all():
        raise RuntimeError(f"{loss_name}: non-finite loss or output gradient")
    result = {"v_hat_grad_norm": float(velocity_grad.float().norm().item()),
              "x0_hat_grad_norm": float(x0_hat_grad.float().norm().item())}
    if result["v_hat_grad_norm"] <= 0 or result["x0_hat_grad_norm"] <= 0:
        raise RuntimeError(f"{loss_name}: output gradient norm is zero")
    _clear_and_assert_frozen(vae, critic)
    # autograd.grad does not populate model parameter .grad; this guards the
    # audit promise without treating trainable LoRA parameters as frozen.
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Gate C unexpectedly accumulated a model parameter gradient")
    return result


def _select(records: list[dict], available_stems: set[str], samples_per_source: int) -> dict[str, list[dict]]:
    selected = {
        source: [record for record in sorted(records, key=lambda item: item["stem"])
                 if record.get("source") == source and record["stem"] in available_stems and _usable(record)][:samples_per_source]
        for source in AUDIT_SOURCES
    }
    insufficient = {source: len(rows) for source, rows in selected.items() if len(rows) != samples_per_source}
    if insufficient:
        raise ValueError(f"Insufficient usable deterministic latent/text records by source: {insufficient}")
    return selected


def _sample_by_stem(dataset: PreparedLatentShardDataset, stem: str) -> dict[str, Any]:
    for index, record in enumerate(dataset.records):
        if record[3] == stem:
            return dataset[index]
    raise KeyError(stem)


def _validate_geometry(record: Mapping[str, Any], sample: Mapping[str, Any]) -> None:
    expected = {"source_size": record["source_size"], "resized_size": record["resized_size"],
                "crop_box": record["crop_box"], "bucket": record["bucket"]}
    actual = {key: sample.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"{record['stem']}: prepared latent geometry does not equal immutable sidecar geometry")


def _load_step_1500_model(*, raw_ckpt: str, checkpoint: Path, expected_sha256: str, device: str) -> tuple[torch.nn.Module, dict[str, Any], str]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required exact step-1500 checkpoint is unavailable: {checkpoint}")
    observed_sha256 = _sha256(checkpoint)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {observed_sha256}")
    state = load_training_state(checkpoint)
    if state.get("global_step") != 1500:
        raise ValueError(f"Gate C requires embedded step 1500, got {state.get('global_step')}")
    model = build_pose_model(raw_ckpt, 64, 64, device).eval()
    load_trainable_state_dict(model, state["model"])
    return model, state, observed_sha256


def run_audit(*, sidecar: Path, dataset_root: Path, latent_root: Path, text_conditioning_root: Path,
              split: str, raw_ckpt: str, checkpoint: Path, expected_checkpoint_sha256: str,
              samples_per_source: int, timesteps: tuple[float, ...], seed: int, temperature: float,
              gaussian_sigma: float, device: torch.device) -> dict[str, object]:
    _, records = load_sidecar(sidecar)
    index = DatasetIndex.discover(dataset_root)
    dataset = PreparedLatentShardDataset(str(latent_root), split, text_conditioning_root=str(text_conditioning_root))
    selected = _select(records, {record[3] for record in dataset.records}, samples_per_source)
    model, state, checkpoint_sha256 = _load_step_1500_model(
        raw_ckpt=raw_ckpt, checkpoint=checkpoint, expected_sha256=expected_checkpoint_sha256, device=str(device),
    )
    vae = load_krea_vae(device)
    critic = FixedBoxKeypointRCNNCritic().to(device).eval()
    _clear_and_assert_frozen(vae, critic)
    report: dict[str, object] = {
        "gate": "C", "audit_only": True, "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256,
        "global_step": int(state["global_step"])}, "model_base": {"raw_checkpoint": raw_ckpt, "rank": 64, "alpha": 64},
        "flow_convention": {"x_t": "t * noise + (1 - t) * x0", "target": "noise - x0",
                            "x0_hat": "x_t - t * v_hat", "gradient": "d x0_hat / d v_hat = -t"},
        "timesteps": list(timesteps), "samples_per_source": samples_per_source, "split": split,
        "seed_policy": {"base_seed": seed, "noise": "sha256(base_seed:stem:gate-c-noise)"},
        "temperature": temperature, "gaussian_sigma": gaussian_sigma, "critic": critic.identifier,
        "vae": {"repository": "Qwen/Qwen-Image", "subfolder": "vae", "class": type(vae).__name__,
                "dtype": str(next(vae.parameters()).dtype)}, "sources": {},
        "frozen_contract_checks": {"vae_parameters_frozen": True, "critic_parameters_frozen": True,
                                   "vae_parameter_grads_absent": True, "critic_parameter_grads_absent": True},
    }
    for source, rows in selected.items():
        samples: list[dict[str, Any]] = []
        for record in rows:
            sample = _sample_by_stem(dataset, record["stem"])
            _validate_geometry(record, sample)
            boxes, targets, valid = _person_tensors(record, device)
            geometry_snapshot = authoritative_geometry_snapshot(boxes, targets, valid)
            rgb = _rgb_tensor(_preprocessed_rgb(record, index)).to(device)
            with torch.inference_mode():
                original = _critic_metrics(critic, rgb.unsqueeze(0), boxes, targets, valid,
                                           temperature=temperature, gaussian_sigma=gaussian_sigma)
                vae_rgb = qwen_decoded_to_unit_rgb(decode_normalized_latents(
                    vae, sample["latent"].unsqueeze(0).to(device, torch.bfloat16),
                )).float()
                roundtrip = _critic_metrics(critic, vae_rgb, boxes, targets, valid,
                                             temperature=temperature, gaussian_sigma=gaussian_sigma)
            if tuple(vae_rgb.shape[-2:]) != tuple(rgb.shape[-2:]):
                raise RuntimeError(f"{source}/{record['stem']}: VAE round-trip geometry differs from source RGB")
            timestep_rows: list[dict[str, Any]] = []
            for timestep in timesteps:
                with torch.inference_mode():
                    _, x0_hat = _model_x0_hat(model, sample, timestep, device=device, seed=seed)
                    predicted_rgb = qwen_decoded_to_unit_rgb(decode_normalized_latents(vae, x0_hat)).float()
                    metrics = _critic_metrics(critic, predicted_rgb, boxes, targets, valid,
                                               temperature=temperature, gaussian_sigma=gaussian_sigma)
                gradients = {
                    loss_name: _gradient_audit(model, vae, critic, sample, boxes, targets, valid, timestep,
                                                device=device, seed=seed, temperature=temperature,
                                                gaussian_sigma=gaussian_sigma, loss_name=loss_name)
                    for loss_name in ("gaussian_heatmap_kl", "normalized_coordinate_huber")
                }
                timestep_rows.append({"timestep": timestep, "metrics": metrics,
                                      "delta_vs_vae_roundtrip": metric_deltas(metrics, roundtrip),
                                      "gradients": gradients})
            assert_authoritative_geometry_unchanged(geometry_snapshot, boxes, targets, valid)
            samples.append({"stem": record["stem"], "joint_count": int(valid.sum().item()), "original_rgb": original,
                            "vae_roundtrip": roundtrip, "delta_roundtrip_minus_original": metric_deltas(roundtrip, original),
                            "timesteps": timestep_rows})
        original_aggregate = weighted_metric_mean([item["original_rgb"] for item in samples])
        vae_aggregate = weighted_metric_mean([item["vae_roundtrip"] for item in samples])
        timestep_aggregate = []
        for timestep in timesteps:
            rows_at_t = [next(row for row in item["timesteps"] if row["timestep"] == timestep) for item in samples]
            metrics = weighted_metric_mean([row["metrics"] for row in rows_at_t])
            gradients: dict[str, dict[str, dict[str, float]]] = {}
            for loss_name in ("gaussian_heatmap_kl", "normalized_coordinate_huber"):
                gradients[loss_name] = {
                    "v_hat_grad_norm": distribution_statistics(row["gradients"][loss_name]["v_hat_grad_norm"] for row in rows_at_t),
                    "x0_hat_grad_norm": distribution_statistics(row["gradients"][loss_name]["x0_hat_grad_norm"] for row in rows_at_t),
                }
            timestep_aggregate.append({"timestep": timestep, "metrics": metrics,
                                        "delta_vs_vae_roundtrip": metric_deltas(metrics, vae_aggregate),
                                        "gradient_statistics": gradients})
        report["sources"][source] = {"sample_count": len(samples), "original_rgb": original_aggregate,
                                      "vae_roundtrip": vae_aggregate,
                                      "delta_roundtrip_minus_original": metric_deltas(vae_aggregate, original_aggregate),
                                      "timesteps": timestep_aggregate, "samples": samples}
    _clear_and_assert_frozen(vae, critic)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--text-conditioning-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-ckpt", default=DEFAULT_RAW_CKPT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--samples-per-source", type=int, default=4)
    parser.add_argument("--timesteps", nargs="+", type=float, default=(0.02, 0.05, 0.10, 0.20, 0.30, 0.40))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.samples_per_source < 1 or args.temperature <= 0 or args.gaussian_sigma <= 0:
        parser.error("samples-per-source, temperature, and gaussian-sigma must be positive")
    try:
        timesteps = parse_timesteps(args.timesteps)
    except ValueError as error:
        parser.error(str(error))
    report = run_audit(sidecar=args.sidecar, dataset_root=args.dataset_root, latent_root=args.latent_root,
                       text_conditioning_root=args.text_conditioning_root, split=args.split, raw_ckpt=args.raw_ckpt,
                       checkpoint=args.checkpoint, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                       samples_per_source=args.samples_per_source, timesteps=timesteps, seed=args.seed,
                       temperature=args.temperature, gaussian_sigma=args.gaussian_sigma,
                       device=torch.device(args.device))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
