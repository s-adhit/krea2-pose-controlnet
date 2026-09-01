"""Gate B audit: exact frozen Krea VAE round trip plus critic-to-latent gradients.

This is deliberately read-only.  It never imports ``train.py``, adds a loss to
training, or updates a parameter.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
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
    distribution_statistics,
    metric_deltas,
    stable_seed,
    weighted_metric_mean,
)
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.vae_preprocessing import (
    decode_normalized_latents,
    decode_normalized_latents_autograd,
    encode_preprocessed_pair,
    load_krea_vae,
    qwen_decoded_to_unit_rgb,
)
from scripts.audit_keypoint_critic import _person_tensors, _preprocessed_rgb, _rgb_tensor, _usable


def _critic_metrics(critic: FixedBoxKeypointRCNNCritic, rgb: torch.Tensor, boxes: torch.Tensor,
                    targets: torch.Tensor, valid: torch.Tensor, *, temperature: float,
                    gaussian_sigma: float) -> dict[str, float | int | None]:
    """Compute the two retained candidates and detached diagnostics once."""
    heatmaps = critic(rgb, [boxes])
    logits = heatmaps.logits
    coordinates = soft_coordinates(logits, heatmaps.boxes_training, temperature)
    metrics: dict[str, float | int | None] = {
        "gaussian_heatmap_kl": float(gaussian_heatmap_kl(
            logits, targets, heatmaps.boxes_training, valid, sigma=gaussian_sigma, temperature=temperature,
        ).item()),
        "normalized_coordinate_huber": float(normalized_coordinate_huber(
            coordinates, targets, heatmaps.boxes_training, valid,
        ).item()),
    }
    metrics.update(detached_pose_diagnostics(
        logits, heatmaps.boxes_training, targets, valid, temperature=temperature, include_argmax=False,
    ))
    return metrics


def _require_frozen_clean(*modules: torch.nn.Module) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)
    assert_frozen_no_parameter_grad(*modules)


def _latent_gradient(vae: Any, critic: FixedBoxKeypointRCNNCritic, z0: torch.Tensor, boxes: torch.Tensor,
                     targets: torch.Tensor, valid: torch.Tensor, loss_name: str, *, temperature: float,
                     gaussian_sigma: float) -> float:
    """Use an isolated graph for one VAE/critic loss and return ||dL/dz0||."""
    _require_frozen_clean(vae, critic)
    latent = z0.detach().clone().unsqueeze(0).requires_grad_(True)
    decoded = decode_normalized_latents_autograd(vae, latent)
    rgb = qwen_decoded_to_unit_rgb(decoded).float()
    heatmaps = critic(rgb, [boxes])
    if loss_name == "gaussian_heatmap_kl":
        loss = gaussian_heatmap_kl(
            heatmaps.logits, targets, heatmaps.boxes_training, valid,
            sigma=gaussian_sigma, temperature=temperature,
        )
    elif loss_name == "normalized_coordinate_huber":
        loss = normalized_coordinate_huber(
            soft_coordinates(heatmaps.logits, heatmaps.boxes_training, temperature),
            targets, heatmaps.boxes_training, valid,
        )
    else:
        raise ValueError(f"unknown latent audit loss: {loss_name}")
    gradient, = torch.autograd.grad(loss, latent)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{loss_name}: non-finite latent loss or gradient")
    norm = float(gradient.float().norm().item())
    if norm <= 0:
        raise RuntimeError(f"{loss_name}: latent gradient norm is zero")
    _require_frozen_clean(vae, critic)
    return norm


def _select(records: list[dict], samples_per_source: int) -> dict[str, list[dict]]:
    selected = {
        source: [record for record in sorted(records, key=lambda item: item["stem"])
                 if record.get("source") == source and _usable(record)][:samples_per_source]
        for source in AUDIT_SOURCES
    }
    insufficient = {source: len(rows) for source, rows in selected.items() if len(rows) != samples_per_source}
    if insufficient:
        raise ValueError(f"Insufficient usable deterministic records by source: {insufficient}")
    return selected


def _vae_identity(vae: Any) -> dict[str, object]:
    config = vae.config
    return {"repository": "Qwen/Qwen-Image", "subfolder": "vae", "class": type(vae).__name__,
            "dtype": str(next(vae.parameters()).dtype), "z_dim": int(config.z_dim),
            "latents_mean": list(config.latents_mean), "latents_std": list(config.latents_std)}


def _aggregate_stage(samples: list[Mapping[str, Any]], key: str) -> dict[str, float | int | None]:
    return weighted_metric_mean([sample[key] for sample in samples])


def run_audit(*, sidecar: Path, dataset_root: Path, samples_per_source: int, seed: int,
              temperature: float, gaussian_sigma: float, device: torch.device) -> dict[str, object]:
    """Run Gate B. Exposed for focused tests; real weights are only loaded by CLI."""
    _, records = load_sidecar(sidecar)
    index = DatasetIndex.discover(dataset_root)
    selected = _select(records, samples_per_source)
    vae = load_krea_vae(device)
    critic = FixedBoxKeypointRCNNCritic().to(device).eval()
    _require_frozen_clean(vae, critic)
    report: dict[str, object] = {
        "gate": "B", "audit_only": True, "seed_policy": {
            "base_seed": seed, "vae_posterior": "sha256(base_seed:stem:vae-posterior)",
        }, "samples_per_source": samples_per_source, "temperature": temperature,
        "gaussian_sigma": gaussian_sigma, "vae": _vae_identity(vae), "critic": critic.identifier,
        "frozen_contract_checks": {"vae_parameters_frozen": True, "critic_parameters_frozen": True,
                                   "vae_parameter_grads_absent": True, "critic_parameter_grads_absent": True},
        "sources": {},
    }
    for source, rows in selected.items():
        samples: list[dict[str, object]] = []
        gradients: dict[str, list[float]] = {"gaussian_heatmap_kl": [], "normalized_coordinate_huber": []}
        for record in rows:
            rgb = _rgb_tensor(_preprocessed_rgb(record, index)).to(device)
            boxes, targets, valid = _person_tensors(record, device)
            geometry_snapshot = authoritative_geometry_snapshot(boxes, targets, valid)
            with torch.inference_mode():
                original = _critic_metrics(critic, rgb.unsqueeze(0), boxes, targets, valid,
                                           temperature=temperature, gaussian_sigma=gaussian_sigma)
                # The shard encoder samples posterior latents.  This explicit, stem-derived
                # generator makes the audit round trip reproducible without mutating shards.
                posterior_generator = torch.Generator(device=device).manual_seed(stable_seed(seed, record["stem"], "vae-posterior"))
                manifest = ManifestRecord(
                    split="audit", stem=record["stem"], file_name=f"{record['stem']}.jpg", text="audit",
                    rgb_path=index.rgb_by_stem[record["stem"]], control_path=index.control_by_stem[record["stem"]],
                )
                encoded = encode_preprocessed_pair(
                    vae, preprocess_pair(manifest), device=device, generator=posterior_generator,
                )
                reconstructed = qwen_decoded_to_unit_rgb(decode_normalized_latents(vae, encoded.latent.unsqueeze(0))).float()
                roundtrip = _critic_metrics(critic, reconstructed, boxes, targets, valid,
                                             temperature=temperature, gaussian_sigma=gaussian_sigma)
            if tuple(reconstructed.shape[-2:]) != tuple(rgb.shape[-2:]):
                raise RuntimeError(f"{source}/{record['stem']}: VAE output geometry differs from authoritative RGB")
            gradient_report = {}
            for loss_name in gradients:
                value = _latent_gradient(vae, critic, encoded.latent, boxes, targets, valid, loss_name,
                                         temperature=temperature, gaussian_sigma=gaussian_sigma)
                gradients[loss_name].append(value)
                gradient_report[loss_name] = value
            assert_authoritative_geometry_unchanged(geometry_snapshot, boxes, targets, valid)
            reconstruction_l1 = float((reconstructed[0] - rgb).abs().mean().item())
            reconstruction_mse = float((reconstructed[0] - rgb).square().mean().item())
            samples.append({"stem": record["stem"], "joint_count": int(valid.sum().item()),
                            "original_rgb": original, "vae_roundtrip": roundtrip,
                            "delta_roundtrip_minus_original": metric_deltas(roundtrip, original),
                            "detached_rgb_reconstruction_l1": reconstruction_l1,
                            "detached_rgb_reconstruction_mse": reconstruction_mse,
                            "latent_gradient_norms": gradient_report})
        original_aggregate = _aggregate_stage(samples, "original_rgb")
        roundtrip_aggregate = _aggregate_stage(samples, "vae_roundtrip")
        report["sources"][source] = {
            "sample_count": len(samples), "original_rgb": original_aggregate,
            "vae_roundtrip": roundtrip_aggregate,
            "delta_roundtrip_minus_original": metric_deltas(roundtrip_aggregate, original_aggregate),
            "latent_gradient_statistics": {name: distribution_statistics(values) for name, values in gradients.items()},
            "samples": samples,
        }
    _require_frozen_clean(vae, critic)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.samples_per_source < 1 or args.temperature <= 0 or args.gaussian_sigma <= 0:
        parser.error("samples-per-source, temperature, and gaussian-sigma must be positive")
    report = run_audit(sidecar=args.sidecar, dataset_root=args.dataset_root,
                       samples_per_source=args.samples_per_source, seed=args.seed,
                       temperature=args.temperature, gaussian_sigma=args.gaussian_sigma,
                       device=torch.device(args.device))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
