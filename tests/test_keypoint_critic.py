"""CPU-only contracts for the audit-only fixed-box Keypoint R-CNN critic."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from pose_controlnet.keypoint_critic import (
    COCO17_JOINT_COUNT,
    FixedBoxHeatmaps,
    FixedBoxKeypointRCNNCritic,
    detached_pose_diagnostics,
    differentiable_pose_loss,
    gaussian_heatmap_kl,
    gaussian_heatmap_target,
    map_heatmap_coordinates_to_boxes,
    masked_coordinate_huber,
    normalize_coordinates_to_boxes,
    normalized_coordinate_huber,
    soft_coordinates,
)
from scripts.audit_keypoint_critic import _loss_and_rgb_gradient


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


class _GradientToyCritic(nn.Module):
    """Tiny frozen-boundary critic that makes each candidate depend on RGB."""
    def forward(self, rgb: torch.Tensor, boxes: list[torch.Tensor]) -> FixedBoxHeatmaps:
        logits = rgb.mean(dim=1, keepdim=True).expand(-1, COCO17_JOINT_COUNT, -1, -1)
        fixed_boxes = boxes[0]
        return FixedBoxHeatmaps(
            logits=logits,
            boxes_training=fixed_boxes,
            boxes_model=fixed_boxes,
            box_image_indices=torch.zeros((len(fixed_boxes),), dtype=torch.int64),
            training_image_sizes=((int(rgb.shape[-2]), int(rgb.shape[-1])),),
            model_image_sizes=((int(rgb.shape[-2]), int(rgb.shape[-1])),),
        )


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

    def test_normalized_roi_coordinate_mapping(self) -> None:
        coordinates = torch.tensor([[[60.0, 120.0]] * COCO17_JOINT_COUNT])
        normalized = normalize_coordinates_to_boxes(coordinates, _boxes())
        self.assertTrue(torch.allclose(normalized, torch.tensor([[[0.5, 0.5]] * COCO17_JOINT_COUNT])))

    def test_normalized_coordinate_huber_is_box_scale_invariant(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 100.0, 200.0], [10.0, 20.0, 210.0, 420.0]])
        target = torch.tensor([[[50.0, 100.0]] * 17, [[110.0, 220.0]] * 17])
        prediction = torch.tensor([[[60.0, 120.0]] * 17, [[130.0, 260.0]] * 17])
        valid = torch.ones((2, 17), dtype=torch.bool)
        combined = normalized_coordinate_huber(prediction, target, boxes, valid)
        first = normalized_coordinate_huber(prediction[:1], target[:1], boxes[:1], valid[:1])
        second = normalized_coordinate_huber(prediction[1:], target[1:], boxes[1:], valid[1:])
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(combined, first))

    def test_normalized_coordinate_huber_has_finite_nonzero_gradients(self) -> None:
        prediction = _coordinates(65.0, 125.0).requires_grad_()
        loss = normalized_coordinate_huber(prediction, _coordinates(), _boxes(), torch.ones((1, 17), dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.norm().item(), 0.0)

    def test_coordinate_losses_average_only_valid_person_joint_observations(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0]])
        target = torch.zeros((2, 17, 2))
        prediction = target.clone()
        prediction[0, 0, 0] = 4.0
        prediction[1, 0, 0] = 2.0
        valid = torch.zeros((2, 17), dtype=torch.bool)
        valid[:, 0] = True
        # Huber(4) = 3.5 and Huber(2) = 1.5 per x coordinate; the y terms
        # are zero and each joint reduces across x/y, so the valid-joint mean is 1.25.
        self.assertAlmostEqual(masked_coordinate_huber(prediction, target, valid).item(), 1.25)
        normalized = normalized_coordinate_huber(prediction, target, boxes, valid)
        # The normalized errors are .4 and .1, both in the quadratic branch.
        self.assertAlmostEqual(normalized.item(), (0.5 * .4 ** 2 / 2 + 0.5 * .1 ** 2 / 2) / 2, places=7)

    def test_gaussian_target_is_normalized(self) -> None:
        target = gaussian_heatmap_target(_coordinates(), _boxes(), (7, 9), sigma=1.5)
        self.assertEqual(tuple(target.shape), (1, 17, 7, 9))
        self.assertTrue(torch.allclose(target.sum(dim=(-2, -1)), torch.ones((1, 17)), atol=1e-6))

    def test_heatmap_kl_is_finite(self) -> None:
        logits = torch.randn((1, 17, 6, 8))
        loss = gaussian_heatmap_kl(logits, _coordinates(), _boxes(), torch.ones((1, 17), dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))

    def test_selectable_losses_preserve_gaussian_kl_and_coordinate_huber_is_differentiable(self) -> None:
        logits = torch.randn((1, 17, 6, 8), requires_grad=True)
        target, boxes = _coordinates(), _boxes()
        valid = torch.ones((1, 17), dtype=torch.bool)
        baseline = gaussian_heatmap_kl(logits, target, boxes, valid, sigma=1.5, temperature=1.0)
        selected_kl = differentiable_pose_loss("gaussian_heatmap_kl", logits, target, boxes, valid)
        self.assertTrue(torch.allclose(selected_kl, baseline))
        coordinate = differentiable_pose_loss("normalized_coordinate_huber", logits, target, boxes, valid)
        coordinate.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.norm().item(), 0.0)

    def test_coordinate_selector_ignores_invalid_oob_joints(self) -> None:
        logits = torch.randn((1, 17, 6, 8), requires_grad=True)
        target = _coordinates()
        valid = torch.zeros((1, 17), dtype=torch.bool)
        valid[:, 0] = True
        changed = target.clone()
        changed[:, 1:] = 1e6
        baseline = differentiable_pose_loss("normalized_coordinate_huber", logits, target, _boxes(), valid)
        self.assertTrue(torch.allclose(
            baseline, differentiable_pose_loss("normalized_coordinate_huber", logits, changed, _boxes(), valid),
        ))

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

    def test_three_candidate_losses_use_independent_rgb_gradient_graphs(self) -> None:
        critic = _GradientToyCritic()
        rgb = torch.tensor([[[[0.1, 0.5], [0.9, 0.2]], [[0.3, 0.7], [0.4, 0.8]], [[0.9, 0.2], [0.6, 0.1]]]])
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        targets = torch.tensor([[[0.1, 0.2]] * COCO17_JOINT_COUNT])
        valid = torch.ones((1, COCO17_JOINT_COUNT), dtype=torch.bool)
        results = [
            _loss_and_rgb_gradient(critic, rgb, boxes, targets, valid, name, temperature=1.0, gaussian_sigma=1.5)
            for name in ("coordinate_huber_pixels", "coordinate_huber_normalized", "gaussian_heatmap_kl")
        ]
        for loss, gradient_norm in results:
            self.assertGreater(loss, 0.0)
            self.assertGreater(gradient_norm, 0.0)
            self.assertTrue(torch.isfinite(torch.tensor((loss, gradient_norm))).all())
        self.assertFalse(rgb.requires_grad)
        self.assertIsNone(rgb.grad)

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
