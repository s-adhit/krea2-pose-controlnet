"""Audit-only VAE-decode autograd into the frozen RTMPose SimCC critic.

This is intentionally independent of ``train.py`` and never constructs an
optimizer or a training loss.  Its only backward path is:

    normalized RGB latent -> frozen Qwen/Krea VAE decode -> fixed crop
    -> frozen RTMPose raw SimCC head -> official_simcc_kl

The sidecar supplies fixed people and targets.  No detector, NMS, argmax,
PCK, DWPose, or rendered skeleton is part of that path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
from pose_controlnet.paired_preprocessing import PreprocessedPair, preprocess_pair
from pose_controlnet.pose_critic import (
    crop_to_critic,
    load_official_rtmpose,
    pose_loss,
    sidecar_person_target,
    simcc_statistics,
)
from pose_controlnet.pose_targets import load_sidecar
from pose_controlnet.vae_preprocessing import (
    decode_normalized_latents_autograd,
    encode_preprocessed_pair,
    pil_to_qwen_vae_tensor,
    qwen_decoded_to_unit_rgb,
    load_krea_vae,
)


SOURCES = ("coco", "humanart_painting", "humanart_real_human", "humanart_sculpture")
DEFAULT_SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_body_joint_count(row: Mapping[str, Any]) -> int:
    people = row.get("people")
    if not isinstance(people, list):
        return 0
    return sum(int(valid.sum().item()) for _, _, valid in map(sidecar_person_target, people))


def select_reward_available_rows(
    rows: Iterable[Mapping[str, Any]], *, source: str, count: int
) -> list[Mapping[str, Any]]:
    """Select stable, valid-body-joint sidecar rows without physical-path heuristics."""
    if count < 1:
        raise ValueError("per-source must be at least one")
    candidates = sorted(
        (
            row
            for row in rows
            if row.get("source") == source
            and row.get("pose_reward_available") is True
            and _valid_body_joint_count(row) > 0
        ),
        key=lambda row: str(row["stem"]),
    )
    if len(candidates) < count:
        raise RuntimeError(
            f"{source}: need {count} reward-available rows with valid body joints; "
            f"found {len(candidates)}"
        )
    return candidates[:count]


def _audit_record(index: DatasetIndex, row: Mapping[str, Any]) -> PreprocessedPair:
    stem = str(row["stem"])
    rgb_path, control_path = index.resolve(f"{stem}.jpg")
    record = ManifestRecord(
        split="pose_critic_vae_audit",
        stem=stem,
        file_name=f"{stem}.jpg",
        text="pose critic VAE audit",
        rgb_path=rgb_path,
        control_path=control_path,
    )
    pair = preprocess_pair(record)
    expected = {
        "source_size": list(pair.geometry.source_size),
        "resized_size": list(pair.geometry.resized_size),
        "crop_box": list(pair.geometry.crop_box),
        "bucket": list(pair.geometry.bucket),
    }
    mismatches = {
        name: {"sidecar": row.get(name), "preprocessed": value}
        for name, value in expected.items()
        if row.get(name) != value
    }
    if mismatches:
        raise AssertionError(f"{stem}: sidecar geometry disagrees with paired preprocessing: {mismatches}")
    return pair


def _assert_frozen_grad_free(module: torch.nn.Module, label: str) -> None:
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise AssertionError(f"{label} has trainable parameters: {trainable[:3]}")
    grads = [name for name, parameter in module.named_parameters() if parameter.grad is not None]
    if grads:
        raise AssertionError(f"{label} parameter received a gradient: {grads[:3]}")


def _cuda_peak(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_cuda_memory_allocated_bytes": None, "peak_cuda_memory_reserved_bytes": None}
    torch.cuda.synchronize(device)
    return {
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _pck(values: list[float], threshold: float) -> float | None:
    return float(np.mean(np.asarray(values) <= threshold)) if values else None


def _source_summary(samples: list[dict[str, Any]], peak: dict[str, int | None]) -> dict[str, Any]:
    soft_errors = [error for sample in samples for error in sample["soft_expectation_error_over_diag"]]
    return {
        "sample_count": len(samples),
        "official_simcc_kl": _mean([sample["official_simcc_kl"] for sample in samples]),
        "latent_gradient_norm": _mean([sample["latent_gradient_norm"] for sample in samples]),
        "decoded_image_gradient_norm": _mean([sample["decoded_image_gradient_norm"] for sample in samples]),
        "valid_joint_count": sum(sample["valid_joint_count"] for sample in samples),
        "reconstruction_rgb_l1": _mean([sample["reconstruction_rgb_l1"] for sample in samples]),
        "reconstruction_rgb_mse": _mean([sample["reconstruction_rgb_mse"] for sample in samples]),
        "soft_expectation_pck_005": _pck(soft_errors, 0.05),
        "soft_expectation_pck_010": _pck(soft_errors, 0.10),
        **peak,
    }


def _latent_seed(seed: int, source: str, stem: str) -> int:
    payload = f"{seed}:{source}:{stem}:vae-posterior".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def audit_one(
    *,
    vae: torch.nn.Module,
    critic: torch.nn.Module,
    pair: PreprocessedPair,
    row: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run one finite/nonzero latent-gradient assertion and detached metrics."""
    source, stem = str(row["source"]), str(row["stem"])
    generator = torch.Generator(device=device).manual_seed(_latent_seed(seed, source, stem))
    encoded = encode_preprocessed_pair(vae, pair, device=device, generator=generator)
    latent = encoded.latent.unsqueeze(0).detach().clone().requires_grad_(True)
    decoded = decode_normalized_latents_autograd(vae, latent)
    decoded_rgb = qwen_decoded_to_unit_rgb(decoded)
    if not decoded_rgb.requires_grad or decoded_rgb.grad_fn is None:
        raise AssertionError(f"{stem}: decoded RGB is detached from the latent graph")
    decoded_rgb.retain_grad()

    input_rgb = qwen_decoded_to_unit_rgb(
        pil_to_qwen_vae_tensor(pair.rgb).to(device=device, dtype=decoded.dtype).squeeze(2)
    )
    if decoded_rgb.shape != input_rgb.shape:
        raise AssertionError(
            f"{stem}: VAE reconstruction shape {tuple(decoded_rgb.shape)} != final frame {tuple(input_rgb.shape)}"
        )
    reconstruction_l1 = float(F.l1_loss(decoded_rgb.float(), input_rgb.float()).detach().cpu())
    reconstruction_mse = float(F.mse_loss(decoded_rgb.float(), input_rgb.float()).detach().cpu())

    weighted_losses: list[torch.Tensor] = []
    valid_joint_count = 0
    soft_errors: list[float] = []
    # The following forward path ends at official_simcc_kl.  PCK/error values
    # detach immediately and are audit reporting only.
    for person in row["people"]:
        crop, target, valid = sidecar_person_target(person)
        valid = valid.to(device=device)
        if not bool(valid.any()):
            continue
        target = target.to(device=device)
        logits_x, logits_y = critic(crop_to_critic(decoded_rgb, crop, critic.spec))
        person_loss = pose_loss(
            logits_x, logits_y, target[None], valid[None], kind="official_simcc_kl", spec=critic.spec
        )
        if not torch.isfinite(person_loss):
            raise FloatingPointError(f"{stem}: official_simcc_kl is NaN or Inf")
        count = int(valid.sum().item())
        weighted_losses.append(person_loss * count)
        valid_joint_count += count
        with torch.no_grad():
            predicted = simcc_statistics(logits_x, logits_y, critic.spec)["coords"][0]
            error = (predicted - target).square().sum(dim=-1).sqrt()
            box = person["bbox_training_xywh"]
            diagonal = float((box[2] ** 2 + box[3] ** 2) ** 0.5)
            soft_errors.extend((error[valid] / diagonal).detach().float().cpu().tolist())
    if not weighted_losses or valid_joint_count == 0:
        raise AssertionError(f"{stem}: no valid body joints reached official_simcc_kl")

    loss = torch.stack(weighted_losses).sum() / valid_joint_count
    loss.backward()
    if latent.grad is None:
        raise AssertionError(f"{stem}: latent.grad is absent")
    if not torch.isfinite(latent.grad).all():
        raise FloatingPointError(f"{stem}: latent.grad is NaN or Inf")
    latent_gradient_norm = float(latent.grad.float().norm().detach().cpu())
    if not latent_gradient_norm > 0.0:
        raise FloatingPointError(f"{stem}: latent gradient norm is zero")
    if decoded_rgb.grad is None or not torch.isfinite(decoded_rgb.grad).all():
        raise FloatingPointError(f"{stem}: decoded RGB gradient is absent, NaN, or Inf")
    decoded_gradient_norm = float(decoded_rgb.grad.float().norm().detach().cpu())
    _assert_frozen_grad_free(vae, "VAE")
    _assert_frozen_grad_free(critic, "RTMPose critic")

    return {
        "stem": stem,
        "bucket": list(pair.geometry.bucket),
        "vae_posterior_seed": _latent_seed(seed, source, stem),
        "official_simcc_kl": float(loss.detach().float().cpu()),
        "latent_gradient_norm": latent_gradient_norm,
        "decoded_image_gradient_norm": decoded_gradient_norm,
        "valid_joint_count": valid_joint_count,
        "reconstruction_rgb_l1": reconstruction_l1,
        "reconstruction_rgb_mse": reconstruction_mse,
        "soft_expectation_error_over_diag": soft_errors,
        "soft_expectation_pck_005": _pck(soft_errors, 0.05),
        "soft_expectation_pck_010": _pck(soft_errors, 0.10),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Krea VAE decode autograd into RTMPose.")
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Staged official RTMPose config")
    parser.add_argument("--weights", type=Path, required=True, help="Staged official RTMPose weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error("--per-source must be at least one")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but unavailable: {device}")
    if not args.config.is_file() or not args.weights.is_file():
        parser.error("--config and --weights must be existing staged official files")

    metadata, rows = load_sidecar(args.sidecar)
    selected = {source: select_reward_available_rows(rows, source=source, count=args.per_source) for source in SOURCES}
    index = DatasetIndex.discover(args.dataset_root)
    vae = load_krea_vae(device)
    critic = load_official_rtmpose(args.config, args.weights, str(device))
    _assert_frozen_grad_free(vae, "VAE")
    _assert_frozen_grad_free(critic, "RTMPose critic")

    report: dict[str, Any] = {
        "status": "PASS",
        "audit": "frozen_vae_decode_autograd_to_frozen_rtmpose_official_simcc_kl",
        "seed": args.seed,
        "per_source": args.per_source,
        "sidecar": str(args.sidecar),
        "sidecar_records_sha256": metadata["records_sha256"],
        "rtmpose_config": str(args.config),
        "rtmpose_weights": str(args.weights),
        "rtmpose_weights_sha256": _sha256(args.weights),
        "sources": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for row in selected[source]:
            samples.append(audit_one(
                vae=vae,
                critic=critic,
                pair=_audit_record(index, row),
                row=row,
                device=device,
                seed=args.seed,
            ))
        report["sources"][source] = _source_summary(samples, _cuda_peak(device)) | {"samples": samples}

    destination = args.output_dir / "vae_decode_pose_critic_audit.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
