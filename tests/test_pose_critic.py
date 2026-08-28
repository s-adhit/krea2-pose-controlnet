import unittest

import torch
import torch.nn as nn

from pose_controlnet.pose_critic import (CriticSpec, FrozenPoseCritic, crop_to_critic,
    fixed_crop_from_xywh, image_to_crop_coords, pose_loss, sidecar_person_target,
    simcc_gaussian_targets, simcc_statistics)


class _ToyHead(nn.Module):
    def forward(self, feats):
        x = feats.mean((2, 3))[:, :1].unsqueeze(-1).expand(-1, 17, 384)
        y = feats.mean((2, 3))[:, :1].unsqueeze(-1).expand(-1, 17, 512)
        return x, y


class _ToyPose(nn.Module):
    def __init__(self): super().__init__(); self.scale = nn.Parameter(torch.ones(())); self.head = _ToyHead()
    def extract_feat(self, x): return x * self.scale


def _person():
    return {"bbox_training_xywh": [10, 20, 60, 100], "keypoints_training": [[40, 70, 2]] * 17,
            "reward_visible_mask": [True] * 17,
            "joint_provenance": [{"reward_joint_valid": True}] * 17}


class PoseCriticTests(unittest.TestCase):
    def test_fixed_box_and_coordinate_mapping(self):
        crop = fixed_crop_from_xywh([10, 20, 60, 100])
        self.assertEqual(crop.center, (40.0, 70.0)); self.assertEqual(crop.scale, (93.75, 125.0))
        got = image_to_crop_coords(torch.tensor([[40., 70.]]), crop)
        self.assertTrue(torch.allclose(got, torch.tensor([[96., 128.]])))

    def test_sidecar_oob_and_multi_person_semantics(self):
        person = _person(); person["joint_provenance"][2] = {"reward_joint_valid": False}
        person["keypoints_training"][3] = [-999, 70, 2]
        _, _, valid = sidecar_person_target(person)
        self.assertFalse(valid[2]); self.assertFalse(valid[3]); self.assertEqual(valid.sum().item(), 15)

    def test_crop_has_input_gradient(self):
        x = torch.randn(1, 3, 128, 128, requires_grad=True)
        crop_to_critic(x, fixed_crop_from_xywh([20, 20, 80, 80])).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all()); self.assertGreater(x.grad.norm().item(), 0)

    def test_simcc_target_and_losses_are_differentiable(self):
        coords = torch.full((1, 17, 2), 50.0); valid = torch.ones(1, 17, dtype=torch.bool)
        tx, ty = simcc_gaussian_targets(coords, valid)
        self.assertTrue(torch.allclose(tx.sum(-1), torch.ones_like(tx.sum(-1))))
        lx = torch.randn(1, 17, 384, requires_grad=True); ly = torch.randn(1, 17, 512, requires_grad=True)
        for kind in ("expectation_huber", "gaussian_cross_entropy"):
            loss = pose_loss(lx, ly, coords, valid, kind=kind); self.assertTrue(torch.isfinite(loss)); loss.backward(retain_graph=True)
        self.assertGreater(lx.grad.norm().item(), 0); self.assertEqual(simcc_statistics(lx, ly)["coords"].shape, (1, 17, 2))

    def test_frozen_critic_preserves_image_graph(self):
        critic = FrozenPoseCritic(_ToyPose()); image = torch.rand(1, 3, 256, 192, requires_grad=True)
        x, y = critic(image); (x.mean() + y.mean()).backward()
        self.assertIsNone(critic.model.scale.grad); self.assertIsNotNone(image.grad)


if __name__ == "__main__": unittest.main()
