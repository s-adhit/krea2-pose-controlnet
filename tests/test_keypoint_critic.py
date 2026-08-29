"""CPU-only contracts for the audit-only fixed-box Keypoint R-CNN critic."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from pose_controlnet.keypoint_critic import (
    COCO17_JOINT_COUNT,
    FixedBoxKeypointRCNNCritic,
    detached_pose_diagnostics,
    gaussian_heatmap_kl,
    gaussian_heatmap_target,
    map_heatmap_coordinates_to_boxes,
    masked_coordinate_huber,
    soft_coordinates,
)


def _boxes() -> torch.Tensor:
    return torch.tensor([[10.0, 20.0, 110.0, 220.0]])


def _coordinates(x: float = 50.0, y: float = 90.0) -> torch.Tensor:
    return torch.tensor([[[x, y]] * COCO17_JOINT_COUNT])


class _ToyROIHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.keypoint_roi_pool = nn.Identity()
        self.keypoint_head = nn.Identity()
        self.keypoint_predictor = nn.Identity()


class _ToyKeypointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = nn.Identity()
        self.backbone = nn.Identity()
        self.roi_heads = _ToyROIHeads()
        self.parameter = nn.Parameter(torch.ones(()))


class KeypointCriticTests(unittest.TestCase):
    def test_spatial_expectation_known_peak(self) -> None:
        logits = torch.full((1, COCO17_JOINT_COUNT, 4, 5), -40.0)
        logits[:, :, 2, 3] = 40.0
        coordinates = soft_coordinates(logits, _boxes(), temperature=1.0)
        expected_x = 10.0 + (3.5 / 5.0) * 100.0
        expected_y = 20.0 + (2.5 / 4.0) * 200.0
        self.assertTrue(torch.allclose(coordinates[..., 0], torch.full((1, 17), expected_x), atol=1e-4))
        self.assertTrue(torch.allclose(coordinates[..., 1], torch.full((1, 17), expected_y), atol=1e-4))

    def test_heatmap_coordinate_mapping_to_training_image(self) -> None:
        indices = torch.tensor([[[0.0, 0.0], [3.0, 1.0]] * 8 + [[2.0, 2.0]]])
        result = map_heatmap_coordinates_to_boxes(indices, _boxes(), (4, 5))
        self.assertAlmostEqual(result[0, 0, 0].item(), 20.0)
        self.assertAlmostEqual(result[0, 0, 1].item(), 45.0)
        self.assertAlmostEqual(result[0, 1, 0].item(), 80.0)
        self.assertAlmostEqual(result[0, 1, 1].item(), 95.0)

    def test_validity_masking_and_invalid_zero_contribution(self) -> None:
        predicted = _coordinates(20.0, 30.0).requires_grad_()
        target = _coordinates(20.0, 30.0)
        target[:, 1] = torch.tensor([10000.0, -10000.0])
        valid = torch.zeros((1, COCO17_JOINT_COUNT), dtype=torch.bool)
        valid[:, 0] = True
        loss = masked_coordinate_huber(predicted, target, valid)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(predicted.grad[:, 1], torch.zeros_like(predicted.grad[:, 1])))

    def test_all_invalid_coordinate_loss_is_differentiable_zero(self) -> None:
        predicted = _coordinates().requires_grad_()
        loss = masked_coordinate_huber(predicted, _coordinates(99.0, 99.0), torch.zeros((1, 17), dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(predicted.grad, torch.zeros_like(predicted)))

    def test_gaussian_target_is_normalized(self) -> None:
        target = gaussian_heatmap_target(_coordinates(), _boxes(), (7, 9), sigma=1.5)
        self.assertEqual(tuple(target.shape), (1, 17, 7, 9))
        self.assertTrue(torch.allclose(target.sum(dim=(-2, -1)), torch.ones((1, 17)), atol=1e-6))

    def test_heatmap_kl_is_finite(self) -> None:
        logits = torch.randn((1, 17, 6, 8))
        loss = gaussian_heatmap_kl(logits, _coordinates(), _boxes(), torch.ones((1, 17), dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))

    def test_invalid_joints_do_not_change_heatmap_kl(self) -> None:
        logits = torch.randn((1, 17, 6, 8))
        target = _coordinates()
        valid = torch.zeros((1, 17), dtype=torch.bool)
        valid[:, 0] = True
        altered = target.clone()
        altered[:, 1:] = 9999.0
        baseline = gaussian_heatmap_kl(logits, target, _boxes(), valid)
        self.assertTrue(torch.allclose(baseline, gaussian_heatmap_kl(logits, altered, _boxes(), valid)))

    def test_gradients_propagate_through_synthetic_heatmaps(self) -> None:
        logits = torch.randn((1, 17, 6, 8), requires_grad=True)
        prediction = soft_coordinates(logits, _boxes())
        coordinate_loss = masked_coordinate_huber(prediction, _coordinates(), torch.ones((1, 17), dtype=torch.bool))
        distribution_loss = gaussian_heatmap_kl(logits, _coordinates(), _boxes(), torch.ones((1, 17), dtype=torch.bool))
        (coordinate_loss + distribution_loss).backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.norm().item(), 0.0)

    def test_argmax_and_pck_diagnostics_are_detached(self) -> None:
        logits = torch.randn((1, 17, 6, 8), requires_grad=True)
        metrics = detached_pose_diagnostics(logits, _boxes(), _coordinates(), torch.ones((1, 17), dtype=torch.bool))
        self.assertIn("argmax_pck_005", metrics)
        self.assertIn("soft_pck_010", metrics)
        self.assertTrue(all(not isinstance(value, torch.Tensor) for value in metrics.values()))
        self.assertIsNone(logits.grad)

    def test_constructor_freezes_mocked_heavy_model_boundary(self) -> None:
        model = _ToyKeypointModel()
        critic = FixedBoxKeypointRCNNCritic(model=model)
        self.assertFalse(critic.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in critic.parameters()))
        critic.train(True)
        self.assertFalse(critic.training)
        self.assertFalse(critic.model.training)


if __name__ == "__main__":
    unittest.main()
