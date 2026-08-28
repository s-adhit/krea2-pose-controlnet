"""Audit-only differentiable RTMPose SimCC utilities.

This module is intentionally not imported by ``train.py``.  It keeps people
fixed to sidecar-v3 boxes and exposes raw body-17 SimCC logits; it never calls
an MMPose inferencer, detector, NMS, or renderer. The one raw SimCC maximum
decoder below is detached and strictly for audit metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

RTMPOSE_M_WHOLEBODY_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"
RTMPOSE_M_WHOLEBODY_WEIGHTS_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
COCO17_BODY = tuple(range(17))


@dataclass(frozen=True)
class CriticSpec:
    input_size: tuple[int, int] = (192, 256)  # official config: (W, H)
    split_ratio: float = 2.0
    sigma: tuple[float, float] = (4.9, 5.66)  # SimCC-bin units
    beta: float = 10.0  # official KLDiscretLoss prediction logit scale
    label_beta: float = 10.0  # official KLDiscretLoss label logit scale
    bbox_padding: float = 1.25
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: tuple[float, float, float] = (58.395, 57.12, 57.375)

    @property
    def vector_lengths(self) -> tuple[int, int]:
        return (round(self.input_size[0] * self.split_ratio), round(self.input_size[1] * self.split_ratio))


@dataclass(frozen=True)
class FixedCrop:
    """MMPose validation affine expressed as center/scale in final-image pixels."""
    center: tuple[float, float]
    scale: tuple[float, float]


def fixed_crop_from_xywh(box: Iterable[float], spec: CriticSpec = CriticSpec()) -> FixedCrop:
    """Exact non-UDP GetBBoxCenterScale + TopdownAffine geometry.

    The official validation pipeline applies padding=1.25 then expands the
    scale to input aspect W/H.  It does not clip the box: grid_sample supplies
    zero padding outside the decoded image, matching cv2.warpAffine's default.
    """
    x, y, w, h = (float(v) for v in box)
    if not w > 0 or not h > 0:
        raise ValueError(f"bbox_xywh must have positive size, got {box}")
    center = (x + w / 2.0, y + h / 2.0)
    w, h = w * spec.bbox_padding, h * spec.bbox_padding
    aspect = spec.input_size[0] / spec.input_size[1]
    scale = (w, w / aspect) if w > h * aspect else (h * aspect, h)
    return FixedCrop(center, scale)


def crop_to_critic(image: torch.Tensor, crop: FixedCrop, spec: CriticSpec = CriticSpec()) -> torch.Tensor:
    """Differentiably warp Bx3xHxW RGB image into one RTMPose crop."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected Bx3xHxW RGB, got {tuple(image.shape)}")
    b, _, h, w = image.shape; out_w, out_h = spec.input_size
    yy, xx = torch.meshgrid(torch.arange(out_h, device=image.device, dtype=image.dtype), torch.arange(out_w, device=image.device, dtype=image.dtype), indexing="ij")
    sx = crop.center[0] + (xx - out_w / 2.0) * crop.scale[0] / out_w
    sy = crop.center[1] + (yy - out_h / 2.0) * crop.scale[1] / out_h
    grid = torch.stack(((2.0 * sx + 1.0) / w - 1.0, (2.0 * sy + 1.0) / h - 1.0), -1).expand(b, -1, -1, -1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def image_to_crop_coords(points_xy: torch.Tensor, crop: FixedCrop, spec: CriticSpec = CriticSpec()) -> torch.Tensor:
    """Map final-training-frame coordinates to RTMPose input coordinates."""
    center = points_xy.new_tensor(crop.center); scale = points_xy.new_tensor(crop.scale)
    return (points_xy - center) / scale * points_xy.new_tensor(spec.input_size) + points_xy.new_tensor(spec.input_size) / 2.0


def sidecar_person_target(person: Mapping[str, Any], spec: CriticSpec = CriticSpec()) -> tuple[FixedCrop, torch.Tensor, torch.Tensor]:
    """Return fixed crop, COCO17 crop coords and fail-closed reward-valid mask."""
    box = person.get("bbox_training_xywh")
    if box is None:
        raise ValueError("Sidecar person lacks bbox_training_xywh")
    crop = fixed_crop_from_xywh(box, spec)
    points = torch.tensor([[p[0], p[1]] for p in person["keypoints_training"]], dtype=torch.float32)
    if points.shape != (17, 2): raise ValueError("Only physical COCO-17 targets are supported")
    valid = torch.tensor(person["reward_visible_mask"], dtype=torch.bool)
    provenance = person.get("joint_provenance", [])
    if len(provenance) == 17:
        valid &= torch.tensor([bool(p.get("reward_joint_valid", False)) for p in provenance])
    coords = image_to_crop_coords(points, crop, spec)
    split = coords * spec.split_ratio; lx, ly = spec.vector_lengths
    valid &= (split[:, 0] >= 0) & (split[:, 0] < lx) & (split[:, 1] >= 0) & (split[:, 1] < ly)
    return crop, coords, valid


def simcc_gaussian_targets(coords: torch.Tensor, valid: torch.Tensor, spec: CriticSpec = CriticSpec()) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official ``normalize=False`` raw Gaussian SimCC label vectors.

    ``valid`` is kept in the signature to make target construction and loss
    call sites explicit; invalid joints are masked by :func:`pose_loss`, not
    altered here. These vectors are deliberately not normalized before the
    KLDiscretLoss-style label softmax.
    """
    if coords.shape[-2:] != (17, 2): raise ValueError("coords must be ...x17x2")
    lx, ly = spec.vector_lengths; bins_x = torch.arange(lx, device=coords.device, dtype=coords.dtype); bins_y = torch.arange(ly, device=coords.device, dtype=coords.dtype)
    mu = torch.round(coords * spec.split_ratio)
    tx = torch.exp(-((bins_x - mu[..., 0, None]) ** 2) / (2 * spec.sigma[0] ** 2))
    ty = torch.exp(-((bins_y - mu[..., 1, None]) ** 2) / (2 * spec.sigma[1] ** 2))
    return tx, ty


def simcc_statistics(logits_x: torch.Tensor, logits_y: torch.Tensor, spec: CriticSpec = CriticSpec()) -> dict[str, torch.Tensor]:
    """Differentiable beta-softmax expectation statistics for audit use."""
    px = (logits_x.float() * spec.beta).softmax(-1)
    py = (logits_y.float() * spec.beta).softmax(-1)
    bx = torch.arange(px.shape[-1], device=px.device, dtype=px.dtype); by = torch.arange(py.shape[-1], device=py.device, dtype=py.dtype)
    coords = torch.stack(((px * bx).sum(-1) / spec.split_ratio, (py * by).sum(-1) / spec.split_ratio), -1)
    entropy = -(px * px.clamp_min(1e-12).log()).sum(-1) - (py * py.clamp_min(1e-12).log()).sum(-1)
    x_peak, y_peak = px.max(-1).values, py.max(-1).values
    return {
        "coords": coords,
        "entropy": entropy,
        "x_beta_softmax_peak_probability": x_peak,
        "y_beta_softmax_peak_probability": y_peak,
        "beta_softmax_confidence": x_peak * y_peak,
    }


def simcc_argmax_decode(logits_x: torch.Tensor, logits_y: torch.Tensor, spec: CriticSpec = CriticSpec()) -> torch.Tensor:
    """Official-style raw SimCC maximum decode, detached and audit-only.

    This mirrors ``SimCCLabel.decode`` with ``use_dark=False``: maximize raw
    x/y vectors independently, then divide SimCC-bin positions by split ratio.
    It must never be used in a loss or other gradient-bearing path.
    """
    return torch.stack((
        logits_x.detach().argmax(dim=-1).to(torch.float32) / spec.split_ratio,
        logits_y.detach().argmax(dim=-1).to(torch.float32) / spec.split_ratio,
    ), dim=-1)


def pose_loss(logits_x: torch.Tensor, logits_y: torch.Tensor, coords: torch.Tensor, valid: torch.Tensor, *, kind: str, spec: CriticSpec = CriticSpec()) -> torch.Tensor:
    """Differentiable audit candidates; no argmax coordinate is ever used."""
    if not bool(valid.any()): return (logits_x.sum() + logits_y.sum()) * 0.0
    stat = simcc_statistics(logits_x, logits_y, spec)
    if kind == "expectation_huber":
        per = F.huber_loss(stat["coords"], coords, reduction="none", delta=4.0).mean(-1)
    elif kind == "official_simcc_kl":
        tx, ty = simcc_gaussian_targets(coords, valid, spec)
        log_px = F.log_softmax(logits_x.float() * spec.beta, dim=-1)
        log_py = F.log_softmax(logits_y.float() * spec.beta, dim=-1)
        target_px = F.softmax(tx.float() * spec.label_beta, dim=-1)
        target_py = F.softmax(ty.float() * spec.label_beta, dim=-1)
        per = (
            F.kl_div(log_px, target_px, reduction="none").mean(dim=-1)
            + F.kl_div(log_py, target_py, reduction="none").mean(dim=-1)
        )
    else: raise ValueError(f"Unknown audit loss {kind!r}")
    return per.masked_select(valid).mean()


class FrozenPoseCritic(nn.Module):
    """A frozen raw-logit wrapper. The input graph is deliberately preserved."""
    def __init__(self, model: nn.Module, spec: CriticSpec = CriticSpec()) -> None:
        super().__init__(); self.model, self.spec = model.eval().requires_grad_(False), spec

    def forward(self, rgb_crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = rgb_crops.new_tensor(self.spec.mean)[None, :, None, None]; std = rgb_crops.new_tensor(self.spec.std)[None, :, None, None]
        normalized = (rgb_crops * 255.0 - mean) / std
        feats = self.model.extract_feat(normalized)
        x, y = self.model.head.forward(feats)
        return x[:, :17], y[:, :17]


def load_official_rtmpose(config: str | Path, weights: str | Path, device: str) -> FrozenPoseCritic:
    """Build top-down MMPose directly; intentionally bypasses inference APIs."""
    try:
        import mmdet  # noqa: F401  # config's CSPNeXt registry owner
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import load_checkpoint
        from mmpose.registry import MODELS
    except ImportError as exc:
        raise RuntimeError("Official audit stack requires mmpose==1.3.2, mmdet==3.3.0, mmengine==0.10.7 and mmcv-lite==2.1.0") from exc
    cfg = Config.fromfile(str(config)); cfg.model.backbone.init_cfg = None
    init_default_scope("mmpose"); model = MODELS.build(cfg.model); load_checkpoint(model, str(weights), map_location="cpu", strict=True)
    return FrozenPoseCritic(model.to(device))
