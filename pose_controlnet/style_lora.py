"""Strict, reversible Krea-2 Style-LoRA composition for evaluation only.

This module deliberately keeps external Style-LoRAs distinct from the trained
Pose Control-LoRA state.  It never changes a model parameter: a temporary
forward hook adds each validated low-rank delta and is removed after the
generation that requested it.
"""
from __future__ import annotations

import contextlib
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
from safetensors import safe_open

from pose_controlnet.model import POSE_CONFIG


STYLE_LORA_SPECS = {
    "darkbrush": {
        "path": "/lambda/nfs/adhit/krea2-pose/style_loras/darkbrush/darkbrush.safetensors",
        "sha256": "f476ad1c0679bc6b14c815187e78a6ece43248f6d232faeccbfed0c4f37f36de",
        "namespace": "official_transformer",
    },
    "rainywindow": {
        "path": "/lambda/nfs/adhit/krea2-pose/style_loras/rainywindow/rainywindow.safetensors",
        "sha256": "7063a6f15ec6112ad3c06d79097b2a30a3ea7d9072821cb36021010d55989fe5",
        "namespace": "official_transformer",
    },
    "retroanime": {
        "path": "/lambda/nfs/adhit/krea2-pose/style_loras/retroanime/retroanime.safetensors",
        "sha256": "ca42107783d9e517c5d62cb9a9db9ab2ba4887d90e9dad97a9d1a7fe6ff14c56",
        "namespace": "official_transformer",
    },
    "realism": {
        "path": "/lambda/nfs/adhit/krea2-pose/style_loras/realism/krea2_realism_lora.safetensors",
        "sha256": "6c38a7934c54a56e0f67753660a4500a094d6dce28a0ee4a0d1dc9f4975d32d2",
        "namespace": "base_model_model",
    },
}
STYLE_RANK = 32
_PAIR = re.compile(r"^(?P<target>.+)\.lora_(?P<part>[AB])\.weight$")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_runtime_targets() -> set[str]:
    block_targets = {f"blocks.{index}.{kind}.{name}" for index in range(28)
                     for kind, names in (("attn", ("wq", "wk", "wv", "gate", "wo")),
                                         ("mlp", ("gate", "up", "down"))) for name in names}
    text_blocks = {f"txtfusion.{family}.{index}.{kind}.{name}"
                   for family in ("layerwise_blocks", "refiner_blocks") for index in range(2)
                   for kind, names in (("attn", ("wq", "wk", "wv", "gate", "wo")),
                                       ("mlp", ("gate", "up", "down"))) for name in names}
    return block_targets | text_blocks | {
        "first", "last.linear", "tmlp.0", "tmlp.2", "tproj.1",
        "txtfusion.projector", "txtmlp.1", "txtmlp.3",
    }


RUNTIME_TARGETS = frozenset(_required_runtime_targets())
if len(RUNTIME_TARGETS) != 264:  # import-time guard against accidental mapper drift
    raise RuntimeError(f"Expected exactly 264 Style-LoRA runtime targets, got {len(RUNTIME_TARGETS)}")


def _official_target(source: str) -> str:
    """Map the public Krea transformer namespace to this MMDiT runtime."""
    direct = {
        "transformer.img_in": "first",
        "transformer.final_layer.linear": "last.linear",
        "transformer.time_embed.linear_1": "tmlp.0",
        "transformer.time_embed.linear_2": "tmlp.2",
        "transformer.time_mod_proj": "tproj.1",
        "transformer.txt_in.linear_1": "txtmlp.1",
        "transformer.txt_in.linear_2": "txtmlp.3",
        "transformer.text_fusion.projector": "txtfusion.projector",
    }
    if source in direct:
        return direct[source]
    if source.startswith("transformer.transformer_blocks."):
        mapped = "blocks." + source.removeprefix("transformer.transformer_blocks.")
    elif source.startswith("transformer.text_fusion."):
        mapped = "txtfusion." + source.removeprefix("transformer.text_fusion.")
    else:
        raise ValueError(f"Unsupported official Krea Style-LoRA target: {source}")
    return (mapped.replace(".attn.to_q", ".attn.wq").replace(".attn.to_k", ".attn.wk")
            .replace(".attn.to_v", ".attn.wv").replace(".attn.to_gate", ".attn.gate")
            .replace(".attn.to_out.0", ".attn.wo").replace(".ff.", ".mlp."))


def _realism_target(source: str) -> str:
    prefix = "base_model.model."
    if not source.startswith(prefix):
        raise ValueError(f"Unsupported realism Style-LoRA target: {source}")
    return source.removeprefix(prefix)


def map_style_target(source: str, namespace: str) -> str:
    if namespace == "official_transformer":
        return _official_target(source)
    if namespace == "base_model_model":
        return _realism_target(source)
    raise ValueError(f"Unsupported Style-LoRA namespace: {namespace}")


def _runtime_linear_shapes() -> dict[str, tuple[int, int]]:
    # This is a meta-device structural audit of the exact shared Raw/Turbo
    # MMDiT config.  It does not load model weights or allocate accelerator RAM.
    from base_model.mmdit import SingleStreamDiT

    with torch.device("meta"):
        model = SingleStreamDiT(POSE_CONFIG)
    shapes = {name.removesuffix(".weight"): tuple(parameter.shape)
              for name, parameter in model.named_parameters() if name.endswith(".weight") and parameter.ndim == 2}
    missing = RUNTIME_TARGETS - set(shapes)
    if missing:
        raise RuntimeError(f"Krea-2 runtime is missing Style-LoRA targets: {sorted(missing)[:5]}")
    return {name: shapes[name] for name in RUNTIME_TARGETS}


@dataclass(frozen=True)
class StyleLoRAAudit:
    style_id: str
    path: str
    sha256: str
    namespace: str
    supported: bool
    tensor_count: int
    target_count: int
    rank: int | None
    dtype: str | None
    metadata: Mapping[str, str] | None
    scaling_rule: Mapping[str, Any]
    mapping: Mapping[str, str]
    errors: tuple[str, ...]

    def json(self) -> dict[str, Any]:
        return asdict(self)


def audit_style_lora(style_id: str, *, expected_path: str | Path | None = None,
                     expected_sha256: str | None = None) -> StyleLoRAAudit:
    """Validate a complete external adapter against the current Turbo runtime.

    Every missing, ambiguous, unpaired, non-FP32, or shape-incompatible target
    is reported as unsupported.  Callers must only apply a supported audit.
    """
    if style_id not in STYLE_LORA_SPECS:
        raise ValueError(f"Unknown Style-LoRA: {style_id}")
    spec = STYLE_LORA_SPECS[style_id]
    path = Path(expected_path or spec["path"])
    expected_digest = expected_sha256 or spec["sha256"]
    errors: list[str] = []
    observed_digest = ""
    try:
        observed_digest = sha256(path)
        if observed_digest != expected_digest:
            errors.append("frozen_sha256_mismatch")
    except (FileNotFoundError, OSError) as exc:
        errors.append(f"unreadable:{type(exc).__name__}")
    pairs: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    metadata: Mapping[str, str] | None = None
    if not errors:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for key in handle.keys():
                    matched = _PAIR.fullmatch(key)
                    if matched is None:
                        errors.append(f"unexpected_tensor_key:{key}")
                        continue
                    pairs.setdefault(matched["target"], {})[matched["part"]] = (
                        tuple(handle.get_slice(key).get_shape()), str(handle.get_slice(key).get_dtype())
                    )
        except Exception as exc:  # safetensors reports several format-specific exception types
            errors.append(f"unreadable_safetensors:{type(exc).__name__}")
    mapping: dict[str, str] = {}
    shapes = _runtime_linear_shapes()
    for source, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            errors.append(f"unpaired_A_B:{source}")
            continue
        try:
            runtime = map_style_target(source, str(spec["namespace"]))
        except ValueError as exc:
            errors.append(str(exc)); continue
        if runtime in mapping.values():
            errors.append(f"ambiguous_runtime_target:{runtime}")
            continue
        mapping[source] = runtime
        a_shape, a_dtype = pair["A"]; b_shape, b_dtype = pair["B"]
        if a_dtype != "F32" or b_dtype != "F32":
            errors.append(f"non_fp32:{source}:{a_dtype}/{b_dtype}")
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != STYLE_RANK or b_shape[1] != STYLE_RANK:
            errors.append(f"rank_mismatch:{source}:{a_shape}/{b_shape}")
            continue
        expected_out, expected_in = shapes.get(runtime, (None, None))
        if runtime == "first":
            # ControlInputLayer is expanded to 128 inputs; style img_in is
            # defined on the original 64-channel image half only.
            expected_in = expected_in
        if a_shape[1] != expected_in or b_shape[0] != expected_out:
            errors.append(f"shape_mismatch:{source}:{a_shape}/{b_shape}!={(expected_out, expected_in)}")
    mapped_targets = set(mapping.values())
    if mapped_targets != RUNTIME_TARGETS:
        errors.append(f"incomplete_target_resolution:missing={len(RUNTIME_TARGETS - mapped_targets)}:unexpected={len(mapped_targets - RUNTIME_TARGETS)}")
    if len(pairs) != 264:
        errors.append(f"tensor_pair_count:{len(pairs)}")
    declared_alpha = metadata.get("lora_alpha") if metadata else None
    if declared_alpha is not None:
        try:
            alpha = float(declared_alpha)
        except ValueError:
            errors.append(f"invalid_lora_alpha:{declared_alpha}")
            alpha = None
        scaling_rule: dict[str, Any] = {"file_declared_alpha": alpha, "rank": STYLE_RANK,
                                         "effective_multiplier": (alpha / STYLE_RANK if alpha is not None else None),
                                         "formula": "style_strength * (lora_alpha / rank) * (B @ A)"}
    else:
        # The three official files contain no safetensors metadata at all.
        # Preserve their observable contract: do not introduce an alpha/rank
        # convention that the file does not declare.
        scaling_rule = {"file_declared_alpha": None, "effective_multiplier": 1.0,
                        "formula": "style_strength * (B @ A)",
                        "reason": "no_file_declared_alpha; no inferred alpha scaling"}
    return StyleLoRAAudit(style_id=style_id, path=str(path), sha256=observed_digest, namespace=str(spec["namespace"]),
                          supported=not errors, tensor_count=len(pairs) * 2, target_count=len(mapping), rank=STYLE_RANK,
                          dtype="F32" if not any(error.startswith("non_fp32") for error in errors) else None,
                          metadata=metadata, scaling_rule=scaling_rule, mapping=mapping, errors=tuple(errors))


@dataclass
class _Delta:
    runtime_target: str
    a: torch.Tensor
    b: torch.Tensor


class StyleLoRAAdapter:
    """Validated Style-LoRA tensors kept separate from the pose checkpoint."""

    def __init__(self, audit: StyleLoRAAudit, deltas: list[_Delta]):
        if not audit.supported:
            raise ValueError(f"Cannot load unsupported Style-LoRA {audit.style_id}: {audit.errors}")
        self.audit, self.deltas = audit, deltas

    @classmethod
    def load(cls, audit: StyleLoRAAudit, *, device: str | torch.device) -> "StyleLoRAAdapter":
        deltas: list[_Delta] = []
        # safetensors' CPU reader is portable across host/service CUDA setups;
        # move only the validated low-rank tensors to the active model device.
        with safe_open(audit.path, framework="pt", device="cpu") as handle:
            for source, runtime in sorted(audit.mapping.items()):
                a = handle.get_tensor(source + ".lora_A.weight").to(device=device)
                b = handle.get_tensor(source + ".lora_B.weight").to(device=device)
                if a.dtype != torch.float32 or b.dtype != torch.float32 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError(f"Style-LoRA tensor is non-FP32 or non-finite: {source}")
                deltas.append(_Delta(runtime, a.contiguous(), b.contiguous()))
        if len(deltas) != 264:
            raise ValueError(f"Style-LoRA load lost targets: {len(deltas)}")
        return cls(audit, deltas)


def _runtime_module(model: nn.Module, target: str) -> nn.Module:
    try:
        module = model.get_submodule(target)
    except AttributeError as exc:
        raise ValueError(f"Style-LoRA runtime target is missing: {target}") from exc
    if target == "first":
        # Pose control expands this projection from the original 64 image
        # inputs to 128 `[image | control]` inputs.  It intentionally is not
        # nn.Linear, but remains a linear projection with a `.weight` tensor.
        if not isinstance(getattr(module, "weight", None), torch.Tensor) or module.weight.ndim != 2:
            raise ValueError("Style-LoRA expanded image-input target is malformed")
        return module
    linear = getattr(module, "base", module)
    if not isinstance(linear, nn.Linear):
        raise ValueError(f"Style-LoRA runtime target is not linear: {target} ({type(module).__name__})")
    return module


@contextlib.contextmanager
def applied_style_lora(model: nn.Module, adapter: StyleLoRAAdapter, strength: float) -> Iterator[None]:
    """Temporarily add one style adapter without mutating shared model weights."""
    if not isinstance(strength, (int, float)) or not torch.isfinite(torch.tensor(float(strength))):
        raise ValueError(f"Style strength must be finite, got {strength!r}")
    if strength == 0:
        yield
        return
    multiplier = float(strength) * float(adapter.audit.scaling_rule["effective_multiplier"])
    handles = []
    try:
        for delta in adapter.deltas:
            module = _runtime_module(model, delta.runtime_target)
            if delta.runtime_target == "first":
                if (module.weight.shape[0] != delta.b.shape[0]
                        or module.weight.shape[1] != delta.a.shape[1] * 2):
                    raise ValueError("Style-LoRA expanded image-input shape changed")
            else:
                linear = getattr(module, "base", module)
                if delta.a.shape != (STYLE_RANK, linear.in_features) or delta.b.shape != (linear.out_features, STYLE_RANK):
                    raise ValueError(f"Style-LoRA runtime shape changed at {delta.runtime_target}")
            if delta.a.shape[0] != STYLE_RANK or delta.b.shape[1] != STYLE_RANK:
                raise ValueError(f"Style-LoRA runtime shape changed at {delta.runtime_target}")

            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor,
                     *, a: torch.Tensor = delta.a, b: torch.Tensor = delta.b, target: str = delta.runtime_target) -> torch.Tensor:
                x = inputs[0]
                if target == "first":
                    x = x[..., :a.shape[1]]
                addition = (x @ a.T.to(dtype=x.dtype)) @ b.T.to(dtype=x.dtype)
                return output + addition * multiplier

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
