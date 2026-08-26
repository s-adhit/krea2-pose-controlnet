"""One-step real Krea-2 Raw pose-control path diagnostic for Gate E only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pose_controlnet.data import load_prepared_sample  # noqa: E402
from pose_controlnet.diffusion import (  # noqa: E402
    forward_pose_control,
    make_flow_pair,
    patchify_and_position,
    sample_flow_timestep,
)
from pose_controlnet.model import (  # noqa: E402
    POSE_CONFIG,
    audit_control_model,
    build_pose_model,
    trainable_params,
)


class _FlowConfig:
    mu_x1 = 256.0
    mu_y1 = 0.5
    mu_x2 = 6400.0
    mu_y2 = 1.15


def _rms(tensor: torch.Tensor) -> float:
    return tensor.float().square().mean().sqrt().item()


def _norm(tensor: torch.Tensor) -> float:
    return torch.linalg.vector_norm(tensor.float()).item()


def _gradient_norm(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise AssertionError(f"Missing gradient for {name}")
    value = _norm(parameter.grad)
    if not math.isfinite(value):
        raise AssertionError(f"Non-finite gradient for {name}: {value}")
    return value


def _assert_frozen_gradients_absent(model: torch.nn.Module) -> None:
    offenders = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if offenders:
        raise AssertionError(f"Frozen backbone gradients are present: {offenders[:8]}")


def _representative_lora(model: torch.nn.Module):
    module = model.blocks[0].attn.wq
    return "blocks.0.attn.wq", module.A, module.B


def _output_difference(real: torch.Tensor, zero: torch.Tensor) -> dict[str, float]:
    difference = (real.float() - zero.float()).abs()
    result = {
        "max_abs": difference.max().item(),
        "rms": _rms(difference),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise AssertionError(f"Non-finite real-control/zero-control difference: {result}")
    return result


def run(args: argparse.Namespace) -> dict:
    if args.device != "cuda":
        raise ValueError("The real 13B Gate-E diagnostic must run on the GH200 with --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; run this command from the normal GH200 host shell")
    if args.rank != 64:
        raise ValueError(f"Gate E requires rank 64, got {args.rank}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("[gate-e] loading one verified persistent latent sample", flush=True)
    sample = load_prepared_sample(
        args.latent_root, split=args.split,
        shard_number=args.shard_number, sample_number=args.sample_number,
    )
    clean = sample["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    control_clean = sample["control"].unsqueeze(0).to(device=device, dtype=torch.float32)
    data_report = {
        "shard_path": sample["shard_path"],
        "sample_number": sample["sample_number"],
        "stem": sample["stem"],
        "bucket": sample["bucket"],
        "caption_nonempty": bool(sample["prompt"].strip()),
        "image_latent_shape": list(clean.shape),
        "image_latent_rms": _rms(clean),
        "image_latent_std": clean.std(unbiased=False).item(),
        "control_latent_shape": list(control_clean.shape),
        "control_latent_rms": _rms(control_clean),
        "control_latent_std": control_clean.std(unbiased=False).item(),
    }

    print("[gate-e] strict-loading Krea-2 Raw and applying rank-64 surgery", flush=True)
    model = build_pose_model(args.raw_ckpt, rank=args.rank, alpha=args.alpha, device=args.device)
    model.train()
    checkpoint_report = model._krea_checkpoint_report
    architecture_report = audit_control_model(model, rank=args.rank)
    if checkpoint_report["missing_keys"] or checkpoint_report["unexpected_keys"]:
        raise AssertionError("Strict checkpoint load reported incompatible keys")

    input_width = POSE_CONFIG.channels * POSE_CONFIG.patch**2
    with safe_open(args.raw_ckpt, framework="pt", device="cpu") as handle:
        pretrained_first = handle.get_tensor("first.weight").float()
    image_half = model.first.weight[:, :input_width]
    control_half = model.first.weight[:, input_width:]
    if not torch.equal(image_half.detach().cpu(), pretrained_first):
        raise AssertionError("ControlInputLayer image half does not exactly match raw first.weight")
    if torch.count_nonzero(control_half.detach()).item() != 0:
        raise AssertionError("ControlInputLayer control half is not zero-initialized")

    patch = POSE_CONFIG.patch
    image_tokens_count = (clean.shape[-2] // patch) * (clean.shape[-1] // patch)
    timestep = sample_flow_timestep(1, image_tokens_count, _FlowConfig, device, generator)
    noise = torch.randn(clean.shape, device=device, dtype=torch.float32, generator=generator)
    noisy, flow_target = make_flow_pair(clean, noise, timestep)

    context = torch.randn(
        1, args.text_tokens, POSE_CONFIG.txtlayers, POSE_CONFIG.txtdim,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    text_mask = torch.ones(1, args.text_tokens, device=device, dtype=torch.bool)
    image_tokens, pos, mask = patchify_and_position(
        noisy.to(torch.bfloat16), args.text_tokens, patch, text_mask
    )
    control_tokens, _, _ = patchify_and_position(
        control_clean.to(torch.bfloat16), args.text_tokens, patch, text_mask
    )
    zero_control_tokens = torch.zeros_like(control_tokens)
    target_tokens, _, _ = patchify_and_position(
        flow_target, args.text_tokens, patch, text_mask
    )
    concatenated = torch.cat((image_tokens, control_tokens), dim=-1)
    if image_tokens.shape[1] != control_tokens.shape[1] or concatenated.shape[1] != image_tokens.shape[1]:
        raise AssertionError("Channel concatenation changed spatial token count")
    if concatenated.shape[-1] != input_width * 2:
        raise AssertionError(f"Wrong concatenated width: {concatenated.shape[-1]}")
    with torch.no_grad():
        hidden_shape = list(model.first(concatenated).shape)
    if hidden_shape != [1, image_tokens_count, POSE_CONFIG.features]:
        raise AssertionError(f"Unexpected ControlInputLayer output shape: {hidden_shape}")

    control_report = {
        "image_tokens_shape": list(image_tokens.shape),
        "control_tokens_shape": list(control_tokens.shape),
        "concatenated_shape": list(concatenated.shape),
        "hidden_shape": hidden_shape,
        "token_count_unchanged": True,
        "image_half_exact_match": True,
        "image_half_weight_norm": _norm(image_half.detach()),
        "control_half_weight_norm_before_backward": _norm(control_half.detach()),
    }

    parameters = trainable_params(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0
    )
    print("[gate-e] first real flow-matching forward/backward", flush=True)
    prediction_before = forward_pose_control(
        model, image_tokens, control_tokens, context, timestep, pos, mask, grad_ckpt=True
    )
    if prediction_before.shape != target_tokens.shape:
        raise AssertionError(
            f"Prediction/target mismatch: {tuple(prediction_before.shape)} vs {tuple(target_tokens.shape)}"
        )
    loss_before = F.mse_loss(prediction_before.float(), target_tokens.float())
    if not math.isfinite(loss_before.item()):
        raise AssertionError(f"Non-finite flow loss: {loss_before.item()}")
    loss_before.backward()

    control_gradient = _gradient_norm(model.first.weight, "first.weight")
    control_half_gradient = _norm(model.first.weight.grad[:, input_width:])
    if not math.isfinite(control_half_gradient) or control_half_gradient <= 0:
        raise AssertionError(f"Control-half gradient gate failed: {control_half_gradient}")
    lora_name, lora_a, lora_b = _representative_lora(model)
    lora_a_first = _gradient_norm(lora_a, f"{lora_name}.A")
    lora_b_first = _gradient_norm(lora_b, f"{lora_name}.B")
    if lora_a_first != 0.0:
        raise AssertionError(f"Expected zero first-step LoRA A gradient, got {lora_a_first}")
    if lora_b_first <= 0:
        raise AssertionError(f"Representative LoRA B gradient is zero: {lora_b_first}")
    _assert_frozen_gradients_absent(model)

    with torch.no_grad():
        prediction_zero_before = forward_pose_control(
            model, image_tokens, zero_control_tokens, context, timestep, pos, mask,
            grad_ckpt=False,
        )
    initial_difference = _output_difference(prediction_before.detach(), prediction_zero_before)
    if initial_difference["max_abs"] != 0.0:
        raise AssertionError(f"Zero-initialized control changed step-0 output: {initial_difference}")

    print("[gate-e] taking exactly one AdamW step", flush=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    control_half_after_step = _norm(model.first.weight[:, input_width:].detach())
    if control_half_after_step <= 0 or not math.isfinite(control_half_after_step):
        raise AssertionError(f"Control half did not update: {control_half_after_step}")

    prediction_real_after = forward_pose_control(
        model, image_tokens, control_tokens, context, timestep, pos, mask, grad_ckpt=True
    )
    with torch.no_grad():
        prediction_zero_after = forward_pose_control(
            model, image_tokens, zero_control_tokens, context, timestep, pos, mask,
            grad_ckpt=False,
        )
    post_step_difference = _output_difference(prediction_real_after.detach(), prediction_zero_after)
    if post_step_difference["max_abs"] <= 0 or post_step_difference["rms"] <= 0:
        raise AssertionError(f"Real/zero control outputs did not diverge: {post_step_difference}")

    print("[gate-e] post-step backward proving LoRA A and B gradients", flush=True)
    loss_after = F.mse_loss(prediction_real_after.float(), target_tokens.float())
    if not math.isfinite(loss_after.item()):
        raise AssertionError(f"Non-finite post-step flow loss: {loss_after.item()}")
    loss_after.backward()
    lora_a_after = _gradient_norm(lora_a, f"{lora_name}.A")
    lora_b_after = _gradient_norm(lora_b, f"{lora_name}.B")
    if lora_a_after <= 0 or lora_b_after <= 0:
        raise AssertionError(
            f"Representative post-step LoRA gradients must be nonzero: A={lora_a_after}, B={lora_b_after}"
        )
    _assert_frozen_gradients_absent(model)

    result = {
        "status": "PASS",
        "seed": args.seed,
        "device": torch.cuda.get_device_name(),
        "checkpoint": checkpoint_report,
        "architecture": architecture_report,
        "data": data_report,
        "flow": {
            "timestep": timestep.item(),
            "loss_before_step": loss_before.item(),
            "loss_after_step": loss_after.item(),
            "control_input_full_gradient_norm_first_backward": control_gradient,
            "control_half_gradient_norm_first_backward": control_half_gradient,
            "representative_lora": lora_name,
            "lora_a_gradient_norm_first_backward_expected_zero": lora_a_first,
            "lora_b_gradient_norm_first_backward": lora_b_first,
            "lora_a_gradient_norm_post_step_backward": lora_a_after,
            "lora_b_gradient_norm_post_step_backward": lora_b_after,
            "control_half_weight_norm_after_one_step": control_half_after_step,
            "initial_real_vs_zero_control": initial_difference,
            "post_step_real_vs_zero_control": post_step_difference,
            "frozen_backbone_gradients_absent": True,
            "optimizer_steps": 1,
        },
        "control_input": control_report,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ckpt", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "diagnostic_val"))
    parser.add_argument("--shard-number", type=int, default=0)
    parser.add_argument("--sample-number", type=int, default=0)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--text-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
