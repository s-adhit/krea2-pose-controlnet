"""Historical compatibility re-exports for :mod:`pose_controlnet.pose_critic`.

New production code must import ``pose_controlnet.pose_critic`` directly.
"""
from pose_controlnet.pose_critic import (
    COCO17_JOINT_COUNT,
    POSE_LOSS_NAMES,
    FixedBoxHeatmaps,
    FixedBoxKeypointRCNNCritic,
    coordinate_huber,
    detached_pose_diagnostics,
    differentiable_pose_loss,
    gaussian_heatmap_kl,
    gaussian_heatmap_target,
    heatmap_expectation,
    map_heatmap_coordinates_to_boxes,
    masked_coordinate_huber,
    masked_gaussian_heatmap_kl,
    normalize_coordinates_to_boxes,
    normalized_coordinate_distances,
    normalized_coordinate_huber,
    soft_coordinates,
    spatial_softmax,
)

__all__ = [
    "COCO17_JOINT_COUNT", "POSE_LOSS_NAMES", "FixedBoxHeatmaps", "FixedBoxKeypointRCNNCritic",
    "coordinate_huber", "detached_pose_diagnostics", "differentiable_pose_loss", "gaussian_heatmap_kl",
    "gaussian_heatmap_target", "heatmap_expectation", "map_heatmap_coordinates_to_boxes",
    "masked_coordinate_huber", "masked_gaussian_heatmap_kl", "normalize_coordinates_to_boxes",
    "normalized_coordinate_distances", "normalized_coordinate_huber", "soft_coordinates", "spatial_softmax",
]
