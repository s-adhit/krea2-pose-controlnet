"""Model surgery for Krea-2 depth control: expanded input layer + LoRA injection.

Imports SingleStreamDiT from the krea-2 repo (must be on sys.path).
"""

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from mmdit import SingleMMDiTConfig, SingleStreamDiT

K2_RAW_CONFIG = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

LORA_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
                "mlp.gate", "mlp.up", "mlp.down")


def inspect_raw_checkpoint(ckpt_path: str) -> dict:
    """Verify the real raw checkpoint key/shape contract without loading 26 GB."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Krea-2 Raw checkpoint not found: {path}")

    with torch.device("meta"):
        expected_model = SingleStreamDiT(K2_RAW_CONFIG)
    expected = {name: tuple(tensor.shape) for name, tensor in expected_model.state_dict().items()}
    del expected_model

    with safe_open(path, framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
        checkpoint_shapes = {
            name: tuple(handle.get_slice(name).get_shape()) for name in checkpoint_keys
        }

    expected_keys = set(expected)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    mismatched = sorted(
        name for name in expected_keys & checkpoint_keys
        if expected[name] != checkpoint_shapes[name]
    )
    if missing or unexpected or mismatched:
        summary = (
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatches={len(mismatched)}"
        )
        raise RuntimeError(f"Krea-2 Raw checkpoint architecture mismatch: {summary}")

    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "tensor_count": len(checkpoint_keys),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
        "config": asdict(K2_RAW_CONFIG),
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        y = self.base(x)
        lora = (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)
        return y + lora * self.scale


class ControlInputLayer(nn.Module):
    """Replaces model.first: takes (B, L, 2*C) = [noisy patches, control patches].

    Pretrained weights on the first half, zeros on the control half. Fully trained,
    fp32 master weights.
    """

    def __init__(self, pretrained: nn.Linear):
        super().__init__()
        in_f, out_f = pretrained.in_features, pretrained.out_features
        self.weight = nn.Parameter(torch.zeros(out_f, in_f * 2, dtype=torch.float32))
        self.bias = nn.Parameter(pretrained.bias.detach().float().clone())
        with torch.no_grad():
            self.weight[:, :in_f] = pretrained.weight.detach().float()

    def forward(self, x):
        return nn.functional.linear(
            x, self.weight.to(x.dtype), self.bias.to(x.dtype)
        )


def _get_module(root: nn.Module, path: str) -> nn.Module:
    mod = root
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def _set_module(root: nn.Module, path: str, new: nn.Module):
    parts = path.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], new)


def build_control_model(
    ckpt_path: str,
    rank: int = 64,
    alpha: float | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    targets: tuple[str, ...] = LORA_TARGETS,
) -> SingleStreamDiT:
    alpha = alpha if alpha is not None else rank
    if rank != 64:
        raise ValueError(f"Pose Control-LoRA rank must be exactly 64, got {rank}")
    if tuple(targets) != LORA_TARGETS:
        raise ValueError("Pose Control-LoRA targets must exactly match LORA_TARGETS")
    checkpoint_report = inspect_raw_checkpoint(ckpt_path)
    with torch.device("meta"):
        model = SingleStreamDiT(K2_RAW_CONFIG)
    incompatible = model.load_state_dict(load_file(ckpt_path), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict Krea-2 Raw load unexpectedly returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    model.first = ControlInputLayer(model.first).to(device)

    for i in range(K2_RAW_CONFIG.layers):
        for t in targets:
            path = f"blocks.{i}.{t}"
            base = _get_module(model, path)
            _set_module(model, path, LoRALinear(base, rank, alpha).to(device))
            _get_module(model, path).base.requires_grad_(False)

    model._krea_checkpoint_report = checkpoint_report

    return model


def audit_control_model(model: nn.Module, rank: int = 64) -> dict:
    """Fail loudly unless surgery and trainability match the Gate-E contract."""
    if rank != 64:
        raise AssertionError(f"LoRA rank must be 64, got {rank}")
    cfg = model.config
    expected = K2_RAW_CONFIG
    for field in asdict(expected):
        if getattr(cfg, field) != getattr(expected, field):
            raise AssertionError(f"Unexpected model config {field}={getattr(cfg, field)}")
    if len(model.blocks) != 28:
        raise AssertionError(f"Expected 28 transformer blocks, got {len(model.blocks)}")
    if model.first.weight.shape != (expected.features, expected.channels * expected.patch**2 * 2):
        raise AssertionError(f"Unexpected ControlInputLayer shape: {tuple(model.first.weight.shape)}")
    text_blocks = [*model.txtfusion.layerwise_blocks, *model.txtfusion.refiner_blocks]
    if len(model.txtfusion.layerwise_blocks) != 2 or len(model.txtfusion.refiner_blocks) != 2:
        raise AssertionError("Expected two layerwise and two refiner text-fusion blocks")
    for block_number, block in enumerate(text_blocks):
        if block.attn.heads != 20 or block.attn.kvheads != 20 or block.attn.headdim != 128:
            raise AssertionError(f"Unexpected text attention structure in block {block_number}")

    lora_names = []
    for block_number, block in enumerate(model.blocks):
        if block.attn.heads != 48 or block.attn.kvheads != 12 or block.attn.headdim != 128:
            raise AssertionError(f"Unexpected attention structure in block {block_number}")
        for target in LORA_TARGETS:
            path = f"blocks.{block_number}.{target}"
            module = _get_module(model, path)
            if not isinstance(module, LoRALinear):
                raise AssertionError(f"Missing LoRA target: {path}")
            if module.A.shape[0] != rank or module.B.shape[1] != rank:
                raise AssertionError(f"Wrong LoRA rank at {path}")
            lora_names.append(path)
    if len(lora_names) != 28 * len(LORA_TARGETS):
        raise AssertionError(f"Expected 224 LoRA target modules, got {len(lora_names)}")

    intended = {"first.weight", "first.bias"}
    intended.update(f"{path}.{suffix}" for path in lora_names for suffix in ("A", "B"))
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != intended:
        raise AssertionError(
            f"Unexpected trainability: missing={sorted(intended - actual)}, "
            f"unexpected={sorted(actual - intended)}"
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {
        "hidden_features": expected.features,
        "blocks": len(model.blocks),
        "attention_heads": 48,
        "attention_kv_heads": 12,
        "attention_head_dim": 128,
        "mlp_hidden_features": model.blocks[0].mlp.gate.base.out_features,
        "text_input_layers": expected.txtlayers,
        "text_layerwise_blocks": len(model.txtfusion.layerwise_blocks),
        "text_refiner_blocks": len(model.txtfusion.refiner_blocks),
        "text_attention_heads": 20,
        "text_attention_kv_heads": 20,
        "lora_rank": rank,
        "lora_target_names": list(LORA_TARGETS),
        "lora_target_modules": len(lora_names),
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".A" in k or ".B" in k or k.startswith("first.")
    }


def load_trainable_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]):
    expected = set(trainable_state_dict(model))
    actual = set(sd)
    if actual != expected:
        raise ValueError(
            "Trainable checkpoint state does not exactly match the control/LoRA contract: "
            f"missing={sorted(expected - actual)[:5]}, unexpected={sorted(actual - expected)[:5]}"
        )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    frozen = set(model.state_dict()) - expected
    if set(missing) != frozen or unexpected:
        raise ValueError(
            "Strict trainable-state load failed: "
            f"missing={sorted(set(missing) - frozen)[:5]}, unexpected={unexpected}"
        )
