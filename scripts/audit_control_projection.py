"""Read-only control-versus-image magnitude audit for LR-only checkpoint step 1500.

It measures real cached pose latents and real flow-noised image latents at the
actual learned ControlInputLayer.  It does not generate images or alter model
or checkpoint state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pose_controlnet.checkpointing import load_training_state
from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.diffusion import make_flow_pair, patchify_and_position
from pose_controlnet.evaluation import _sample_by_stem
from pose_controlnet.model import build_pose_model, load_trainable_state_dict, trainable_state_dict
from pose_controlnet.turbo_evaluation import (
    CONTROL_SCALE_TURBO_EVALUATION_ROOT,
    LR5E5_CHECKPOINT_ROOT, LR5E5_HF_REPO_ID, LR5E5_HF_RUN_NAME,
    LR5E5_TURBO_EVALUATION_ROOT, ORIGINAL_TURBO_EVALUATION_ROOT,
    TIMESTEP_TURBO_EVALUATION_ROOT,
    assert_exact_diagnostic_stems, exact_lr5e5_step1500_local_checkpoint,
)
from scripts.turbo_benchmark import _dataset_and_spec as _original_dataset_and_spec


ROOT = Path("/lambda/nfs/adhit/krea2-pose")
OUTPUT = ROOT / "evaluation/control-projection-step1500"
FIXED_TIMESTEPS = (0.1, 0.3, 0.5, 0.7, 0.9)
AUDIT_SEED = 420_500


def assert_projection_audit_output_isolated(output_dir: str | Path) -> Path:
    """Reject every existing Turbo evaluation namespace and its descendants."""
    output = Path(output_dir).resolve()
    protected = (ORIGINAL_TURBO_EVALUATION_ROOT, LR5E5_TURBO_EVALUATION_ROOT,
                 TIMESTEP_TURBO_EVALUATION_ROOT, CONTROL_SCALE_TURBO_EVALUATION_ROOT)
    for root in protected:
        root = root.resolve()
        if output == root or root in output.parents:
            raise ValueError(f"Projection audit output must not collide with an existing Turbo evaluation tree: {root}")
    return output


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stable_seed(seed: int, stem: str, timestep: float) -> int:
    payload = f"{seed}:{stem}:control-projection:{timestep:.1f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _tensor_measurement(tensor: torch.Tensor) -> dict[str, float | int]:
    value = tensor.float()
    sum_squares = float(value.square().sum().item())
    count = value.numel()
    return {"rms": (sum_squares / count) ** 0.5, "l2": sum_squares ** 0.5,
            "sum_squares": sum_squares, "element_count": count}


def _public_measurement(measurement: dict[str, float | int]) -> dict[str, float | int]:
    return {key: measurement[key] for key in ("rms", "l2", "element_count")}


def _reduce_measurements(measurements: Iterable[dict[str, float | int]]) -> dict[str, float | int]:
    values = list(measurements)
    if not values:
        raise ValueError("Cannot aggregate an empty projection audit group")
    sum_squares = sum(float(value["sum_squares"]) for value in values)
    count = sum(int(value["element_count"]) for value in values)
    return {"rms": (sum_squares / count) ** 0.5, "l2": sum_squares ** 0.5, "element_count": count}


@torch.inference_mode()
def projection_contributions(model: Any, image_tokens: torch.Tensor,
                             control_tokens: torch.Tensor) -> dict[str, Any]:
    """Invoke the actual ControlInputLayer with the executable feature order.

    ``forward_pose_control`` concatenates ``[noisy_img, pose_ctrl]`` before
    ``model.first``.  Patchification/BF16 conversion happen in the caller; no
    positional encoding or normalization is applied before this projection.
    """
    if image_tokens.shape != control_tokens.shape:
        raise ValueError("Image and control patch tokens must have identical shapes")
    if image_tokens.ndim != 3:
        raise ValueError("Projected image/control tokens must be [batch, tokens, features]")
    zeros_image, zeros_control = torch.zeros_like(image_tokens), torch.zeros_like(control_tokens)
    image_only_input = torch.cat([image_tokens, zeros_control], dim=-1)
    control_only_input = torch.cat([zeros_image, control_tokens], dim=-1)
    both_input = torch.cat([image_tokens, control_tokens], dim=-1)
    expected_features = getattr(model.first, "weight").shape[1]
    if any(value.shape[-1] != expected_features for value in (image_only_input, control_only_input, both_input)):
        raise ValueError("ControlInputLayer feature width does not match concatenated image/control token width")
    return {
        "image_input": image_tokens, "control_input": control_tokens,
        "image_only_input": image_only_input, "control_only_input": control_only_input, "both_input": both_input,
        "image_only_output": model.first(image_only_input),
        "control_only_output": model.first(control_only_input),
        "both_output": model.first(both_input),
    }


def _record_measurements(paths: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    return {
        "image_input": _tensor_measurement(paths["image_input"]),
        "control_input": _tensor_measurement(paths["control_input"]),
        "image_only_projection_output": _tensor_measurement(paths["image_only_output"]),
        "control_only_projection_output": _tensor_measurement(paths["control_only_output"]),
        "both_projection_output": _tensor_measurement(paths["both_output"]),
    }


def _summary(measurements: list[dict[str, dict[str, float | int]]]) -> dict[str, Any]:
    merged = {key: _reduce_measurements([value[key] for value in measurements]) for key in measurements[0]}
    image_rms = float(merged["image_only_projection_output"]["rms"])
    control_rms = float(merged["control_only_projection_output"]["rms"])
    summary = {
        "image_input_rms": merged["image_input"]["rms"], "control_input_rms": merged["control_input"]["rms"],
        "image_only_projection_output_rms": image_rms,
        "control_only_projection_output_rms": control_rms,
        "both_projection_output_rms": merged["both_projection_output"]["rms"],
        "control_to_image_projection_ratio": control_rms / image_rms if image_rms else None,
        "l2": {key: value["l2"] for key, value in merged.items()},
        "rms": {key: value["rms"] for key, value in merged.items()},
        "element_count": {key: value["element_count"] for key, value in merged.items()},
    }
    if "noisy_image_latent" in merged:
        summary["noisy_image_latent_rms"] = merged["noisy_image_latent"]["rms"]
        summary["control_latent_rms"] = merged["control_latent"]["rms"]
    return summary


def _source_checkpoint(args, output: Path) -> Path:
    return exact_lr5e5_step1500_local_checkpoint(
        checkpoint_dir=args.checkpoint_dir, hf_repo_id=args.hf_repo_id,
        marker_download_dir=output / "checkpoint-marker-validation",
    )


def _dataset(args) -> tuple[PreparedLatentShardDataset, tuple[str, ...], dict[str, Any]]:
    dataset, stems, spec = _original_dataset_and_spec(args)
    assert_exact_diagnostic_stems(stems, (record[3] for record in dataset.records))
    return dataset, stems, spec


def _raw_checkpoint(args, state: dict[str, Any]) -> str:
    config = state.get("config")
    recorded = config.get("raw_ckpt") if isinstance(config, dict) else None
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("Source training state lacks recorded Krea-2 Raw checkpoint provenance")
    if args.raw_ckpt is not None and Path(args.raw_ckpt).resolve() != Path(recorded).resolve():
        raise ValueError("Projection audit --raw-ckpt must exactly match the source checkpoint's recorded Raw provenance")
    return recorded


def preflight(args) -> None:
    output = assert_projection_audit_output_isolated(args.output_dir)
    checkpoint = _source_checkpoint(args, output)
    state = load_training_state(checkpoint)
    if state["global_step"] != 1500:
        raise ValueError("Projection audit source checkpoint must be exact embedded step 1500")
    dataset, stems, _ = _dataset(args)
    buckets = sorted({tuple(_sample_by_stem(dataset, stem)["latent"].shape[-2:]) for stem in stems})
    if len(buckets) < 2:
        raise ValueError("Projection audit requires multiple real diagnostic bucket shapes")
    _write(output / "checkpoint_preflight.json", {
        "checkpoint": str(checkpoint), "checkpoint_step": 1500, "local_checkpoint_root": str(LR5E5_CHECKPOINT_ROOT),
        "hf_repo_id": LR5E5_HF_REPO_ID, "hf_namespace": f"{LR5E5_HF_RUN_NAME}/full/",
        "recorded_raw_checkpoint": _raw_checkpoint(args, state), "diagnostic_sample_count": len(stems),
        "latent_bucket_shapes": [list(bucket) for bucket in buckets], "fixed_timesteps": list(FIXED_TIMESTEPS),
        "deterministic_noise_seed": AUDIT_SEED,
    })
    print(output / "checkpoint_preflight.json")


def audit(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run the projection audit from the GH200 host shell with CUDA visible")
    output = assert_projection_audit_output_isolated(args.output_dir)
    checkpoint = _source_checkpoint(args, output)
    state = load_training_state(checkpoint)
    if state["global_step"] != 1500:
        raise ValueError("Projection audit source checkpoint must be exact embedded step 1500")
    dataset, stems, spec = _dataset(args)
    raw_checkpoint = _raw_checkpoint(args, state)
    model = build_pose_model(raw_checkpoint, 64, 64, "cuda").eval()
    load_trainable_state_dict(model, state["model"])
    before = trainable_state_dict(model)
    all_measurements: list[dict[str, dict[str, float | int]]] = []
    records: list[dict[str, Any]] = []
    for timestep_value in FIXED_TIMESTEPS:
        for stem in stems:
            sample = _sample_by_stem(dataset, stem)
            clean = sample["latent"][None].to(device="cuda", dtype=torch.float32)
            control_latent = sample["control"][None].to(device="cuda", dtype=torch.bfloat16)
            noise = torch.randn(clean.shape, dtype=torch.float32,
                                generator=torch.Generator().manual_seed(_stable_seed(AUDIT_SEED, stem, timestep_value))).to("cuda")
            timestep = torch.full((1,), timestep_value, dtype=torch.float32, device="cuda")
            noisy, _ = make_flow_pair(clean, noise, timestep)
            context = sample["context"][None].to(device="cuda", dtype=torch.bfloat16)
            text_mask = sample["mask"][None].to(device="cuda", dtype=torch.bool)
            image_tokens, _, _ = patchify_and_position(noisy.to(torch.bfloat16), context.shape[1], model.config.patch, text_mask)
            control_tokens, _, _ = patchify_and_position(control_latent, context.shape[1], model.config.patch, text_mask)
            paths = projection_contributions(model, image_tokens, control_tokens)
            measurements = _record_measurements(paths)
            measurements["noisy_image_latent"] = _tensor_measurement(noisy)
            measurements["control_latent"] = _tensor_measurement(control_latent)
            all_measurements.append(measurements)
            image_rms, control_rms = measurements["image_only_projection_output"]["rms"], measurements["control_only_projection_output"]["rms"]
            records.append({"stem": stem, "timestep": timestep_value, "bucket": [clean.shape[-1] * 8, clean.shape[-2] * 8],
                            "noise_seed": _stable_seed(AUDIT_SEED, stem, timestep_value),
                            "measurements": {key: _public_measurement(value) for key, value in measurements.items()},
                            "control_to_image_projection_ratio": float(control_rms) / float(image_rms) if image_rms else None})
    if any(not torch.equal(before[name], value) for name, value in trainable_state_dict(model).items()):
        raise RuntimeError("Read-only projection audit unexpectedly changed model weights")
    full_measurements = all_measurements
    by_timestep: dict[float, list[dict[str, dict[str, float | int]]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, dict[str, float | int]]]] = defaultdict(list)
    for record, measurement in zip(records, full_measurements):
        by_timestep[record["timestep"]].append(measurement)
        by_bucket["x".join(map(str, record["bucket"]))].append(measurement)
    _write(output / "control_projection_audit.json", {
        "format_version": 1, "checkpoint": {"path": str(checkpoint), "step": 1500, "hf_repo_id": LR5E5_HF_REPO_ID,
                                                "hf_namespace": f"{LR5E5_HF_RUN_NAME}/full/", "raw_checkpoint": raw_checkpoint},
        "diagnostic_spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "sample_count": len(stems), "record_count": len(records), "fixed_timesteps": list(FIXED_TIMESTEPS),
        "deterministic_noise_seed": AUDIT_SEED,
        "feature_ordering": {"concatenation": "[noisy_image_patch_tokens, clean_control_patch_tokens]",
                             "projection": "model.first (ControlInputLayer)",
                             "before_projection": "BF16 cast then patchify; no positional encoding or normalization"},
        "measurement_definitions": {"rms": "sqrt(sum(x^2) / element_count), computed in float32",
                                    "l2": "sqrt(sum(x^2)), computed in float32",
                                    "noisy_image": "x_t = timestep * deterministic_noise + (1 - timestep) * clean_image_latent",
                                    "projection_paths": {"image_only": "[image, zeros_like(control)]", "control_only": "[zeros_like(image), control]", "both": "[image, control]"}},
        "aggregate": _summary(full_measurements),
        "per_timestep": {str(value): _summary(by_timestep[value]) for value in FIXED_TIMESTEPS},
        "per_bucket": {bucket: _summary(measurements) for bucket, measurements in sorted(by_bucket.items())},
        "per_record": records,
    })
    print(output / "control_projection_audit.json")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=str(OUTPUT))
    common.add_argument("--latent-root", default=str(ROOT / "posebridge_latents"))
    common.add_argument("--text-conditioning-root", default=str(ROOT / "text_conditioning"))
    common.add_argument("--checkpoint-dir", default=str(LR5E5_CHECKPOINT_ROOT))
    common.add_argument("--hf-repo-id", default=LR5E5_HF_REPO_ID)
    common.add_argument("--diagnostic-manifest", default="data/manifests/diagnostic_val.jsonl")
    common.add_argument("--seed", type=int, default=420200)
    common.add_argument("--raw-ckpt")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, function in (("preflight", preflight), ("audit", audit)):
        item = sub.add_parser(name, parents=[common])
        item.set_defaults(function=function)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    args.function(args)
