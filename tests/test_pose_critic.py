import unittest

import torch
import torch.nn as nn

from pose_controlnet.pose_critic import (CriticSpec, FrozenPoseCritic, crop_to_critic,
    fixed_crop_from_xywh, image_to_crop_coords, pose_loss, sidecar_person_target,
    simcc_argmax_decode, simcc_gaussian_targets, simcc_statistics)


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

    def test_beta_softmax_sharpens_nonuniform_statistics(self):
        logits = torch.tensor([[[0.0, 0.2, 0.8, 0.1]]])
        y_logits = torch.tensor([[[0.0, 0.3, 0.7, 0.2]]])
        beta_one = simcc_statistics(logits, y_logits, CriticSpec(beta=1.0))
        beta_ten = simcc_statistics(logits, y_logits, CriticSpec(beta=10.0))
        self.assertLess(beta_ten["entropy"].item(), beta_one["entropy"].item())
        self.assertGreater(beta_ten["beta_softmax_confidence"].item(), beta_one["beta_softmax_confidence"].item())

    def test_official_argmax_coordinate_decode_is_detached(self):
        lx = torch.zeros(1, 17, 384, requires_grad=True); ly = torch.zeros(1, 17, 512, requires_grad=True)
        with torch.no_grad(): lx[:, :, 20] = 3.0; ly[:, :, 41] = 4.0
        coords = simcc_argmax_decode(lx, ly)
        self.assertTrue(torch.equal(coords, torch.tensor([[[10.0, 20.5]] * 17])))
        self.assertFalse(coords.requires_grad)

    def test_simcc_raw_target_and_official_kl_are_differentiable(self):
        coords = torch.full((1, 17, 2), 50.0); valid = torch.ones(1, 17, dtype=torch.bool)
        tx, ty = simcc_gaussian_targets(coords, valid)
        self.assertFalse(torch.allclose(tx.sum(-1), torch.ones_like(tx.sum(-1))))
        self.assertFalse(torch.allclose(ty.sum(-1), torch.ones_like(ty.sum(-1))))
        self.assertTrue(torch.allclose(tx[..., 100], torch.ones_like(tx[..., 100])))
        self.assertTrue(torch.allclose(ty[..., 100], torch.ones_like(ty[..., 100])))
        lx = torch.randn(1, 17, 384, requires_grad=True); ly = torch.randn(1, 17, 512, requires_grad=True)
        for kind in ("expectation_huber", "official_simcc_kl"):
            loss = pose_loss(lx, ly, coords, valid, kind=kind); self.assertTrue(torch.isfinite(loss)); loss.backward(retain_graph=True)
        self.assertGreater(lx.grad.norm().item(), 0); self.assertEqual(simcc_statistics(lx, ly)["coords"].shape, (1, 17, 2))

    def test_official_simcc_kl_matches_reference_and_masks_invalid_joints(self):
        spec = CriticSpec(); coords = torch.full((1, 17, 2), 50.0)
        valid = torch.ones(1, 17, dtype=torch.bool); valid[:, 0] = False
        lx = torch.randn(1, 17, 384, requires_grad=True); ly = torch.randn(1, 17, 512, requires_grad=True)
        tx, ty = simcc_gaussian_targets(coords, valid, spec)
        reference = (
            torch.nn.functional.kl_div(torch.log_softmax(lx.float() * spec.beta, -1), torch.softmax(tx.float() * spec.label_beta, -1), reduction="none").mean(-1)
            + torch.nn.functional.kl_div(torch.log_softmax(ly.float() * spec.beta, -1), torch.softmax(ty.float() * spec.label_beta, -1), reduction="none").mean(-1)
        ).masked_select(valid).mean()
        actual = pose_loss(lx, ly, coords, valid, kind="official_simcc_kl", spec=spec)
        self.assertTrue(torch.allclose(actual, reference))
        actual.backward(); self.assertGreater(lx.grad.norm().item(), 0); self.assertGreater(ly.grad.norm().item(), 0)

    def test_argmax_metric_is_not_part_of_reward_differentiation(self):
        coords = torch.full((1, 17, 2), 50.0); valid = torch.ones(1, 17, dtype=torch.bool)
        lx = torch.randn(1, 17, 384, requires_grad=True); ly = torch.randn(1, 17, 512, requires_grad=True)
        decoded = simcc_argmax_decode(lx, ly)
        self.assertFalse(decoded.requires_grad)
        pose_loss(lx, ly, coords, valid, kind="official_simcc_kl").backward()
        self.assertGreater(lx.grad.norm().item(), 0); self.assertGreater(ly.grad.norm().item(), 0)

    def test_frozen_critic_preserves_image_graph(self):
        critic = FrozenPoseCritic(_ToyPose()); image = torch.rand(1, 3, 256, 192, requires_grad=True)
        x, y = critic(image); (x.mean() + y.mean()).backward()
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))
        self.assertIsNotNone(image.grad)


if __name__ == "__main__": unittest.main()
