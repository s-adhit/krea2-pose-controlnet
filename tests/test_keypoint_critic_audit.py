"""CPU-only helper contracts for Gate B/Gate C audit scripts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from pose_controlnet.diffusion import make_flow_pair
from pose_controlnet.keypoint_critic_audit import (
    assert_authoritative_geometry_unchanged,
    assert_frozen_no_parameter_grad,
    authoritative_geometry_snapshot,
    deterministic_noise_like,
    metric_deltas,
    parse_timesteps,
    reconstruct_clean_latent,
    weighted_metric_mean,
)
from pose_controlnet.vae_preprocessing import decode_normalized_latents_autograd, qwen_decoded_to_unit_rgb


class _ToyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(z_dim=1, latents_mean=[0.0], latents_std=[1.0])
        self.scale = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def decode(self, raw: torch.Tensor):
        # Exact helper contract: B,C,T,H,W in and a one-frame RGB sample out.
        return SimpleNamespace(sample=raw[:, :1].expand(-1, 3, -1, -1, -1) * self.scale)


class KeypointCriticAuditTests(unittest.TestCase):
    def test_frozen_vae_decoder_propagates_to_latent_without_parameter_grads(self) -> None:
        vae = _ToyVAE().eval().requires_grad_(False)
        critic = nn.Conv2d(3, 1, kernel_size=1, bias=False).requires_grad_(False)
        latent = torch.full((1, 1, 2, 2), 0.2, requires_grad=True)
        decoded = decode_normalized_latents_autograd(vae, latent)
        loss = critic(qwen_decoded_to_unit_rgb(decoded)).square().mean()
        gradient, = torch.autograd.grad(loss, latent)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0.0)
        assert_frozen_no_parameter_grad(vae, critic)

    def test_metric_delta_bookkeeping_and_weighted_aggregation(self) -> None:
        baseline = {"joint_count": 2, "gaussian_heatmap_kl": 1.0, "soft_pck_005": 0.5}
        current = {"joint_count": 2, "gaussian_heatmap_kl": 1.25, "soft_pck_005": 0.75}
        self.assertEqual(metric_deltas(current, baseline), {"gaussian_heatmap_kl": 0.25, "soft_pck_005": 0.25})
        aggregate = weighted_metric_mean((baseline, {"joint_count": 4, "gaussian_heatmap_kl": 2.0, "soft_pck_005": 1.0}))
        self.assertEqual(aggregate["joint_count"], 6)
        self.assertAlmostEqual(float(aggregate["gaussian_heatmap_kl"]), 5.0 / 3.0)
        self.assertAlmostEqual(float(aggregate["soft_pck_005"]), 5.0 / 6.0)

    def test_authoritative_boxes_targets_and_mask_remain_identical(self) -> None:
        boxes = torch.tensor([[1.0, 2.0, 30.0, 40.0]])
        targets = torch.randn((1, 17, 2))
        valid = torch.ones((1, 17), dtype=torch.bool)
        snapshot = authoritative_geometry_snapshot(boxes, targets, valid)
        # A VAE round trip operates only on RGB/latents; this is the explicit
        # audit contract guarding the immutable Phase-1 tensors around it.
        assert_authoritative_geometry_unchanged(snapshot, boxes, targets, valid)
        targets[0, 0, 0] += 1.0
        with self.assertRaises(RuntimeError):
            assert_authoritative_geometry_unchanged(snapshot, boxes, targets, valid)

    def test_flow_identity_and_velocity_gradient_factor(self) -> None:
        x0 = torch.randn((1, 2, 3, 4))
        noise = torch.randn_like(x0)
        timestep = torch.tensor([0.3])
        noisy, target = make_flow_pair(x0, noise, timestep)
        self.assertTrue(torch.allclose(reconstruct_clean_latent(noisy, target, timestep), x0))
        velocity = torch.randn_like(x0, requires_grad=True)
        reconstructed = reconstruct_clean_latent(noisy, velocity, timestep)
        gradient, = torch.autograd.grad(reconstructed.sum(), velocity)
        self.assertTrue(torch.allclose(gradient, torch.full_like(velocity, -0.3)))

    def test_timestep_parsing_order_and_deterministic_noise(self) -> None:
        self.assertEqual(parse_timesteps((0.02, 0.05, 0.10)), (0.02, 0.05, 0.10))
        with self.assertRaises(ValueError):
            parse_timesteps((0.10, 0.05))
        clean = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
        first = deterministic_noise_like(clean, seed=42, stem="sample")
        self.assertTrue(torch.equal(first, deterministic_noise_like(clean, seed=42, stem="sample")))
        self.assertFalse(torch.equal(first, deterministic_noise_like(clean, seed=42, stem="other")))

    def test_metric_aggregation_is_independent_for_each_timestep(self) -> None:
        by_timestep = {
            0.02: ({"joint_count": 1, "gaussian_heatmap_kl": 1.0}, {"joint_count": 3, "gaussian_heatmap_kl": 3.0}),
            0.20: ({"joint_count": 1, "gaussian_heatmap_kl": 4.0}, {"joint_count": 3, "gaussian_heatmap_kl": 8.0}),
        }
        aggregate = {timestep: weighted_metric_mean(rows) for timestep, rows in by_timestep.items()}
        self.assertAlmostEqual(float(aggregate[0.02]["gaussian_heatmap_kl"]), 2.5)
        self.assertAlmostEqual(float(aggregate[0.20]["gaussian_heatmap_kl"]), 7.0)

    def test_frozen_boundary_rejects_parameter_gradient_or_trainability(self) -> None:
        module = nn.Linear(2, 2).requires_grad_(False)
        assert_frozen_no_parameter_grad(module)
        next(module.parameters()).requires_grad_(True)
        with self.assertRaises(RuntimeError):
            assert_frozen_no_parameter_grad(module)


if __name__ == "__main__":
    unittest.main()
