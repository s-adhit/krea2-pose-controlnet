"""Audit-only fixed-box differentiable torchvision Keypoint R-CNN critic.

This module deliberately exposes only the frozen COCO keypoint branch.  It is
not connected to training and it does not run the detector, RPN, box head,
box regression, scoring, NMS, or torchvision's keypoint argmax decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


COCO17_JOINT_COUNT = 17
POSE_LOSS_NAMES = ("gaussian_heatmap_kl", "normalized_coordinate_huber")


@dataclass(frozen=True)
class FixedBoxHeatmaps:
    """Raw heatmaps and the two coordinate systems used to create them.

    ``boxes_training`` are the supplied, authoritative boxes in the input RGB
    frame. ``boxes_model`` are torchvision-transform-resized copies used by
    ROIAlign.  Rows are ordered image-major, then in the supplied box order.
    """

    logits: Tensor
    boxes_training: Tensor
    boxes_model: Tensor
    box_image_indices: Tensor
    training_image_sizes: tuple[tuple[int, int], ...]
    model_image_sizes: tuple[tuple[int, int], ...]


class FixedBoxKeypointRCNNCritic(nn.Module):
    """Frozen COCO_V1 Keypoint R-CNN keypoint branch with supplied boxes.

    Inputs are RGB tensors in ``[0, 1]`` and authoritative ``xyxy`` boxes in
    the same, final training RGB coordinate frame.  The call path is exactly
    ``transform -> backbone -> keypoint_roi_pool -> keypoint_head ->
    keypoint_predictor``.  No detector-derived values enter the result.
    """

    identifier = "torchvision/keypointrcnn_resnet50_fpn:COCO_V1"

    def __init__(self, model: nn.Module | None = None, *, progress: bool = True) -> None:
        super().__init__()
        if model is None:
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights,
                keypointrcnn_resnet50_fpn,
            )

            self.weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
            model = keypointrcnn_resnet50_fpn(weights=self.weights, progress=progress)
        else:
            self.weights = None
        self.model = model.requires_grad_(False).eval()
        self._assert_keypoint_branch()
        self.train(False)

    def train(self, mode: bool = True) -> "FixedBoxKeypointRCNNCritic":
        """Keep the frozen critic in evaluation mode even if a parent trains."""
        super().train(False)
        self.model.eval()
        return self

    def _assert_keypoint_branch(self) -> None:
        for path in ("transform", "backbone", "roi_heads"):
            if not hasattr(self.model, path):
                raise TypeError(f"Fixed-box critic model lacks {path}")
        roi_heads = self.model.roi_heads
        for path in ("keypoint_roi_pool", "keypoint_head", "keypoint_predictor"):
            if getattr(roi_heads, path, None) is None:
                raise TypeError(f"Fixed-box critic model lacks roi_heads.{path}")

    def forward(self, rgb: Tensor, boxes: Sequence[Tensor] | Tensor) -> FixedBoxHeatmaps:
        """Return raw ``[total_people, 17, 56, 56]`` COCO heatmap logits.

        The model's ``GeneralizedRCNNTransform`` normalizes RGB by the model
        mean/std, resizes each image to its eval min/max policy, and pads the
        batch.  Passing boxes as transform targets makes its canonical resize
        operation apply to the fixed data boxes; the transformed boxes then go
        straight to the keypoint ROI branch.
        """
        images, original_boxes = _validate_rgb_and_boxes(rgb, boxes)
        targets = [{"boxes": item} for item in original_boxes]
        image_list, transformed_targets = self.model.transform(images, targets)
        if transformed_targets is None:  # defensive; targets were supplied
            raise RuntimeError("torchvision transform unexpectedly dropped fixed boxes")
        model_boxes = [target["boxes"] for target in transformed_targets]
        features = self.model.backbone(image_list.tensors)
        if isinstance(features, Tensor):
            features = {"0": features}
        roi_heads = self.model.roi_heads
        keypoint_features = roi_heads.keypoint_roi_pool(features, model_boxes, image_list.image_sizes)
        keypoint_features = roi_heads.keypoint_head(keypoint_features)
        logits = roi_heads.keypoint_predictor(keypoint_features)
        if logits.ndim != 4 or logits.shape[1] < COCO17_JOINT_COUNT:
            raise RuntimeError(f"Expected [people, >=17, H, W] keypoint logits, got {tuple(logits.shape)}")
        image_indices = torch.cat([
            torch.full((item.shape[0],), index, dtype=torch.int64, device=logits.device)
            for index, item in enumerate(original_boxes)
        ])
        return FixedBoxHeatmaps(
            logits=logits[:, :COCO17_JOINT_COUNT],
            boxes_training=torch.cat(original_boxes, dim=0),
            boxes_model=torch.cat(model_boxes, dim=0),
            box_image_indices=image_indices,
            training_image_sizes=tuple((int(image.shape[-2]), int(image.shape[-1])) for image in images),
            model_image_sizes=tuple((int(height), int(width)) for height, width in image_list.image_sizes),
        )


def spatial_softmax(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Return a per-joint spatial distribution without decoding a keypoint."""
    _validate_logits(logits)
    if not isinstance(temperature, (float, int)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softmax((logits / float(temperature)).flatten(-2), dim=-1).reshape_as(logits)


def heatmap_expectation(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Return expected ``[x_heatmap, y_heatmap]`` coordinates per heatmap."""
    probabilities = spatial_softmax(logits, temperature)
    height, width = logits.shape[-2:]
    x_grid = torch.arange(width, dtype=logits.dtype, device=logits.device)
    y_grid = torch.arange(height, dtype=logits.dtype, device=logits.device)
    x = (probabilities.sum(dim=-2) * x_grid).sum(dim=-1)
    y = (probabilities.sum(dim=-1) * y_grid).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def map_heatmap_coordinates_to_boxes(heatmap_coordinates: Tensor, boxes_xyxy: Tensor, heatmap_size: tuple[int, int]) -> Tensor:
    """Map heatmap-index coordinates to continuous image coordinates in ROI boxes.

    A heatmap index refers to a cell center, hence the explicit ``+0.5``.
    This is the differentiable analogue of torchvision's Heckbert-style
    keypoint decoder, without interpolation, argmax, or Python scalar reads.
    """
    if heatmap_coordinates.ndim != 3 or heatmap_coordinates.shape[-1] != 2:
        raise ValueError("heatmap_coordinates must have shape [people, joints, 2]")
    _validate_boxes(boxes_xyxy, expected=len(heatmap_coordinates))
    height, width = heatmap_size
    if height < 1 or width < 1:
        raise ValueError("heatmap_size must be positive")
    boxes = boxes_xyxy.to(dtype=heatmap_coordinates.dtype, device=heatmap_coordinates.device)
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    x = x0[:, None] + (heatmap_coordinates[..., 0] + 0.5) * ((x1 - x0)[:, None] / width)
    y = y0[:, None] + (heatmap_coordinates[..., 1] + 0.5) * ((y1 - y0)[:, None] / height)
    return torch.stack((x, y), dim=-1)


def soft_coordinates(logits: Tensor, boxes_xyxy: Tensor, temperature: float = 1.0) -> Tensor:
    """Differentiably map spatial-softmax expectations into training RGB space."""
    return map_heatmap_coordinates_to_boxes(
        heatmap_expectation(logits, temperature), boxes_xyxy, tuple(logits.shape[-2:]),
    )


def masked_coordinate_huber(predicted: Tensor, target: Tensor, reward_joint_valid: Tensor, delta: float = 1.0) -> Tensor:
    """Mean coordinate Huber loss over only valid COCO17 person/joint entries."""
    _validate_coordinate_inputs(predicted, target, reward_joint_valid)
    if not isinstance(delta, (float, int)) or delta <= 0:
        raise ValueError("delta must be positive")
    valid = reward_joint_valid.to(dtype=predicted.dtype)
    per_joint = F.huber_loss(predicted, target.to(predicted), reduction="none", delta=float(delta)).mean(dim=-1)
    denominator = valid.sum()
    return (per_joint * valid).sum() / denominator.clamp_min(1) if bool(denominator.detach().item()) else predicted.sum() * 0


coordinate_huber = masked_coordinate_huber


def normalize_coordinates_to_boxes(coordinates: Tensor, boxes_xyxy: Tensor, *, eps: float | None = None) -> Tensor:
    """Express COCO17 coordinates in their authoritative fixed ROI frame.

    The input coordinates remain untouched.  This is solely the shared
    ``(x - x0) / width, (y - y0) / height`` transform used by the normalized
    coordinate audit loss.  Width and height are clamped defensively so the
    function remains numerically safe at the public API boundary.
    """
    if coordinates.ndim != 3 or coordinates.shape[1:] != (COCO17_JOINT_COUNT, 2):
        raise ValueError("coordinates must have shape [people, 17, 2]")
    if not coordinates.is_floating_point() or not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite floating-point values")
    _validate_boxes(boxes_xyxy, expected=coordinates.shape[0])
    if eps is None:
        eps = torch.finfo(coordinates.dtype).eps
    if not isinstance(eps, (float, int)) or eps <= 0:
        raise ValueError("eps must be positive")
    boxes = boxes_xyxy.to(dtype=coordinates.dtype, device=coordinates.device)
    origin = boxes[:, None, :2]
    scale = (boxes[:, 2:] - boxes[:, :2]).clamp_min(float(eps))[:, None]
    return (coordinates - origin) / scale


def normalized_coordinate_huber(
    predicted: Tensor, target: Tensor, boxes_xyxy: Tensor, reward_joint_valid: Tensor,
    delta: float = 1.0, *, eps: float | None = None,
) -> Tensor:
    """Mean Huber loss over valid joints after person-box normalization.

    Both the soft prediction and the authoritative target are normalized in
    exactly the same fixed ROI coordinate system.  The final reduction is the
    same valid-person/joint mean as :func:`masked_coordinate_huber`.
    """
    _validate_coordinate_inputs(predicted, target, reward_joint_valid)
    _validate_boxes(boxes_xyxy, expected=predicted.shape[0])
    return masked_coordinate_huber(
        normalize_coordinates_to_boxes(predicted, boxes_xyxy, eps=eps),
        normalize_coordinates_to_boxes(target.to(predicted), boxes_xyxy, eps=eps),
        reward_joint_valid,
        delta,
    )


def gaussian_heatmap_target(
    target_coordinates: Tensor, boxes_xyxy: Tensor, heatmap_size: tuple[int, int], sigma: float = 1.5,
) -> Tensor:
    """Create normalized Gaussian targets in the raw keypoint-heatmap frame."""
    if target_coordinates.ndim != 3 or target_coordinates.shape[-1] != 2:
        raise ValueError("target_coordinates must have shape [people, joints, 2]")
    if not torch.isfinite(target_coordinates).all():
        raise ValueError("target_coordinates must be finite")
    _validate_boxes(boxes_xyxy, expected=target_coordinates.shape[0])
    height, width = heatmap_size
    if height < 1 or width < 1 or not isinstance(sigma, (float, int)) or sigma <= 0:
        raise ValueError("heatmap_size and sigma must be positive")
    boxes = boxes_xyxy.to(dtype=target_coordinates.dtype, device=target_coordinates.device)
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    target_x = ((target_coordinates[..., 0] - x0[:, None]) / (x1 - x0)[:, None]) * width - 0.5
    target_y = ((target_coordinates[..., 1] - y0[:, None]) / (y1 - y0)[:, None]) * height - 0.5
    x_grid = torch.arange(width, dtype=target_coordinates.dtype, device=target_coordinates.device)
    y_grid = torch.arange(height, dtype=target_coordinates.dtype, device=target_coordinates.device)
    squared_distance = (x_grid[None, None, None, :] - target_x[..., None, None]).square()
    squared_distance = squared_distance + (y_grid[None, None, :, None] - target_y[..., None, None]).square()
    target = torch.exp(-squared_distance / (2 * float(sigma) ** 2))
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(target.dtype).tiny)


def masked_gaussian_heatmap_kl(
    logits: Tensor, target_coordinates: Tensor, boxes_xyxy: Tensor, reward_joint_valid: Tensor,
    *, sigma: float = 1.5, temperature: float = 1.0,
) -> Tensor:
    """Mean target-KL-predicted heatmap divergence over valid COCO17 joints."""
    _validate_logits(logits)
    _validate_coordinate_inputs(
        torch.empty((*logits.shape[:2], 2), dtype=logits.dtype, device=logits.device),
        target_coordinates, reward_joint_valid,
    )
    target = gaussian_heatmap_target(target_coordinates.to(logits), boxes_xyxy, tuple(logits.shape[-2:]), sigma)
    log_probabilities = F.log_softmax((logits / float(temperature)).flatten(-2), dim=-1).reshape_as(logits)
    per_joint = (target * (target.clamp_min(torch.finfo(target.dtype).tiny).log() - log_probabilities)).sum(dim=(-2, -1))
    valid = reward_joint_valid.to(dtype=logits.dtype)
    denominator = valid.sum()
    return (per_joint * valid).sum() / denominator.clamp_min(1) if bool(denominator.detach().item()) else logits.sum() * 0


gaussian_heatmap_kl = masked_gaussian_heatmap_kl


def differentiable_pose_loss(
    pose_loss: str, logits: Tensor, target_coordinates: Tensor, boxes_xyxy: Tensor,
    reward_joint_valid: Tensor, *, temperature: float = 1.0, gaussian_sigma: float = 1.5,
    coordinate_huber_delta: float = 1.0,
) -> Tensor:
    """Select a differentiable fixed-box pose loss from raw critic heatmap logits.

    Both choices use only the supplied authoritative ROI geometry and
    ``reward_joint_valid`` mask.  The coordinate variant deliberately uses the
    differentiable spatial expectation; no decoder or detector operation is in
    this backward path.
    """
    if pose_loss == "gaussian_heatmap_kl":
        return gaussian_heatmap_kl(
            logits, target_coordinates, boxes_xyxy, reward_joint_valid,
            sigma=gaussian_sigma, temperature=temperature,
        )
    if pose_loss == "normalized_coordinate_huber":
        return normalized_coordinate_huber(
            soft_coordinates(logits, boxes_xyxy, temperature), target_coordinates,
            boxes_xyxy, reward_joint_valid, delta=coordinate_huber_delta,
        )
    raise ValueError(f"Unsupported pose loss {pose_loss!r}; expected one of {POSE_LOSS_NAMES}")


@torch.no_grad()
def detached_pose_diagnostics(
    logits: Tensor, boxes_xyxy: Tensor, target_coordinates: Tensor, reward_joint_valid: Tensor,
    *, temperature: float = 1.0, include_argmax: bool = True,
) -> dict[str, float | int | None]:
    """Detached audit-only soft/argmax diagnostics; never use this for a loss."""
    _validate_logits(logits)
    _validate_coordinate_inputs(
        torch.empty((*logits.shape[:2], 2), dtype=logits.dtype, device=logits.device),
        target_coordinates, reward_joint_valid,
    )
    probability = spatial_softmax(logits, temperature)
    soft = soft_coordinates(logits, boxes_xyxy, temperature)
    valid = reward_joint_valid.bool()
    count = int(valid.sum().item())
    if count == 0:
        return {"joint_count": 0, "soft_coordinate_error_normalized": None, "soft_pck_005": None,
                "soft_pck_010": None, "argmax_pck_005": None, "argmax_pck_010": None,
                "heatmap_entropy": None, "heatmap_peak_probability": None}
    distance = torch.linalg.vector_norm(soft - target_coordinates.to(soft), dim=-1)
    diagonal = torch.linalg.vector_norm(boxes_xyxy[:, 2:] - boxes_xyxy[:, :2], dim=-1).clamp_min(torch.finfo(distance.dtype).eps)
    normalized = distance / diagonal[:, None]
    result: dict[str, float | int | None] = {
        "joint_count": count,
        "soft_coordinate_error_normalized": float(normalized[valid].mean().item()),
        "soft_pck_005": float((normalized[valid] <= 0.05).float().mean().item()),
        "soft_pck_010": float((normalized[valid] <= 0.10).float().mean().item()),
        "heatmap_entropy": float((-(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=(-2, -1)))[valid].mean().item()),
        "heatmap_peak_probability": float(probability.amax(dim=(-2, -1))[valid].mean().item()),
        "argmax_pck_005": None,
        "argmax_pck_010": None,
    }
    if include_argmax:
        height, width = logits.shape[-2:]
        maxima = logits.flatten(-2).argmax(dim=-1)
        heatmap_xy = torch.stack((maxima.remainder(width), torch.div(maxima, width, rounding_mode="floor")), dim=-1).to(logits.dtype)
        argmax_coordinates = map_heatmap_coordinates_to_boxes(heatmap_xy, boxes_xyxy, (height, width))
        argmax_normalized = torch.linalg.vector_norm(argmax_coordinates - target_coordinates.to(argmax_coordinates), dim=-1) / diagonal[:, None]
        result["argmax_pck_005"] = float((argmax_normalized[valid] <= 0.05).float().mean().item())
        result["argmax_pck_010"] = float((argmax_normalized[valid] <= 0.10).float().mean().item())
    return result


def _validate_rgb_and_boxes(rgb: Tensor, boxes: Sequence[Tensor] | Tensor) -> tuple[list[Tensor], list[Tensor]]:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [3, H, W] or [batch, 3, H, W]")
    if not rgb.is_floating_point() or not torch.isfinite(rgb).all():
        raise ValueError("rgb must be finite floating-point RGB in [0, 1]")
    if isinstance(boxes, Tensor):
        if boxes.ndim != 3 or boxes.shape[0] != rgb.shape[0]:
            raise ValueError("tensor boxes must have shape [batch, people, 4]")
        boxes_list = [boxes[index] for index in range(boxes.shape[0])]
    else:
        boxes_list = list(boxes)
    if len(boxes_list) != rgb.shape[0]:
        raise ValueError("one fixed-box tensor is required per RGB image")
    images = [rgb[index] for index in range(rgb.shape[0])]
    normalized_boxes = []
    for image, item in zip(images, boxes_list):
        if not isinstance(item, Tensor):
            raise TypeError("fixed boxes must be tensors")
        item = item.detach().to(device=image.device, dtype=image.dtype)
        _validate_boxes(item)
        if item.numel() == 0:
            raise ValueError("every audit RGB must provide at least one fixed person box")
        height, width = image.shape[-2:]
        if (item[:, 0] < 0).any() or (item[:, 1] < 0).any() or (item[:, 2] > width).any() or (item[:, 3] > height).any():
            raise ValueError("fixed boxes must lie in the RGB training frame")
        normalized_boxes.append(item)
    return images, normalized_boxes


def _validate_boxes(boxes_xyxy: Tensor, expected: int | None = None) -> None:
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4 or (expected is not None and boxes_xyxy.shape[0] != expected):
        raise ValueError("boxes_xyxy must have shape [people, 4]")
    if not boxes_xyxy.is_floating_point() or not torch.isfinite(boxes_xyxy).all():
        raise ValueError("boxes_xyxy must be finite floating-point values")
    if (boxes_xyxy[:, 2] <= boxes_xyxy[:, 0]).any() or (boxes_xyxy[:, 3] <= boxes_xyxy[:, 1]).any():
        raise ValueError("boxes_xyxy must have positive width and height")


def _validate_logits(logits: Tensor) -> None:
    if logits.ndim != 4 or logits.shape[1] != COCO17_JOINT_COUNT or logits.shape[-2] < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have shape [people, 17, heatmap_height, heatmap_width]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")


def _validate_coordinate_inputs(predicted: Tensor, target: Tensor, valid: Tensor) -> None:
    if predicted.ndim != 3 or predicted.shape[1:] != (COCO17_JOINT_COUNT, 2):
        raise ValueError("predicted coordinates must have shape [people, 17, 2]")
    if target.shape != predicted.shape or not target.is_floating_point() or not torch.isfinite(target).all():
        raise ValueError("target coordinates must be finite with shape [people, 17, 2]")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("reward_joint_valid must have shape [people, 17]")
