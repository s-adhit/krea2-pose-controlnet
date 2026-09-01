"""Small dependency-free helpers shared by the read-only critic audits."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


AUDIT_SOURCES = ("coco", "humanart_painting", "humanart_real_human", "humanart_sculpture")


def parse_timesteps(values: Sequence[float]) -> tuple[float, ...]:
    """Validate an explicitly ordered audit timestep sweep."""
    parsed = tuple(float(value) for value in values)
    if not parsed or any(not 0.0 < value < 1.0 for value in parsed):
        raise ValueError("timesteps must be a non-empty sequence strictly inside (0, 1)")
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError("timesteps must be strictly increasing without duplicates")
    return parsed


def stable_seed(seed: int, stem: str, label: str) -> int:
    """Derive a cross-process deterministic torch seed from audit identity."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    payload = f"{seed}:{stem}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def deterministic_noise_like(clean: torch.Tensor, *, seed: int, stem: str, label: str = "noise") -> torch.Tensor:
    """Generate CPU-seeded Gaussian noise, then move it to the clean latent device."""
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(seed, stem, label))
    noise = torch.randn(clean.shape, dtype=clean.dtype, device="cpu", generator=generator)
    return noise.to(device=clean.device)


def reconstruct_clean_latent(noisy: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor | float) -> torch.Tensor:
    """Apply the repository's flow identity ``x0_hat = x_t - t * v_hat``."""
    if noisy.shape != velocity.shape:
        raise ValueError("noisy latent/tokens and velocity must have identical shapes")
    if isinstance(timestep, torch.Tensor):
        if timestep.ndim == 0:
            scale = timestep
        elif timestep.ndim == 1 and timestep.shape[0] == noisy.shape[0]:
            scale = timestep.view(-1, *([1] * (noisy.ndim - 1)))
        else:
            raise ValueError("tensor timestep must be scalar or have one value per batch entry")
        scale = scale.to(device=noisy.device, dtype=noisy.dtype)
    else:
        scale = float(timestep)
    return noisy - scale * velocity


def distribution_statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("statistics require one or more finite values")
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "std": float(array.std()), "min": float(array.min()), "max": float(array.max())}


def weighted_metric_mean(rows: Iterable[Mapping[str, float | int | None]]) -> dict[str, float | int | None]:
    """Pool per-image valid-joint means without changing their observation weight."""
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for row in rows:
        weight = int(row.get("joint_count", 0) or 0)
        if weight < 0:
            raise ValueError("joint_count must be non-negative")
        count += weight
        for key, value in row.items():
            if key != "joint_count" and isinstance(value, (int, float)):
                totals[key] += float(value) * weight
    if count == 0:
        raise ValueError("cannot aggregate metrics with zero valid joints")
    return {"joint_count": count, **{key: value / count for key, value in sorted(totals.items())}}


def metric_deltas(current: Mapping[str, float | int | None], baseline: Mapping[str, float | int | None]) -> dict[str, float]:
    """Return signed current-minus-baseline deltas for shared numeric metrics."""
    return {
        key: float(value) - float(baseline[key])
        for key, value in current.items()
        if key != "joint_count" and isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def assert_frozen_no_parameter_grad(*modules: torch.nn.Module) -> None:
    """Fail if a frozen audit boundary has trainable or accumulated parameter grads."""
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise RuntimeError("audit boundary parameter is unexpectedly trainable")
            if parameter.grad is not None:
                raise RuntimeError("frozen audit boundary parameter unexpectedly received a gradient")


def authoritative_geometry_snapshot(boxes: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture immutable fixed-box/COCO17 inputs for post-audit identity checks."""
    return boxes.detach().clone(), targets.detach().clone(), valid.detach().clone()


def assert_authoritative_geometry_unchanged(snapshot: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                            boxes: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
    """Ensure VAE/model audits never edit their Phase-1 boxes, targets, or mask."""
    if not (torch.equal(snapshot[0], boxes) and torch.equal(snapshot[1], targets) and torch.equal(snapshot[2], valid)):
        raise RuntimeError("authoritative fixed boxes, COCO17 targets, or validity mask were mutated")
