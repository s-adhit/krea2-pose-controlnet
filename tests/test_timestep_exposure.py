import unittest
from dataclasses import replace

import torch

from pose_controlnet.config import TrainConfig
from pose_controlnet.diffusion import (resolution_shift_mu, sample_flow_timestep,
    sample_pre_shift_flow_timestep, shift_timestep)


class TimestepExposureTest(unittest.TestCase):
    def cfg(self, **changes):
        return replace(TrainConfig(raw_ckpt="raw", shard_dir="shards"), **changes)

    def test_disabled_sampler_is_exact_historical_sigmoid_normal_and_generator_path(self):
        cfg = self.cfg()
        actual_generator = torch.Generator().manual_seed(981)
        expected_generator = torch.Generator().manual_seed(981)
        actual = sample_flow_timestep(64, 3952, cfg, "cpu", actual_generator)
        expected_pre_shift = torch.sigmoid(torch.randn(64, dtype=torch.float32, generator=expected_generator))
        expected = shift_timestep(expected_pre_shift, resolution_shift_mu(3952, cfg.mu_x1, cfg.mu_y1, cfg.mu_x2, cfg.mu_y2))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.rand(3, generator=actual_generator), torch.rand(3, generator=expected_generator)))

    def test_auxiliary_routes_about_twenty_percent_and_support_is_pre_shift_bounded(self):
        cfg = self.cfg(timestep_aux_prob=.20, timestep_aux_min=.04359494981207863,
                       timestep_aux_max=.3773562340267345)
        pre_shift, mask = sample_pre_shift_flow_timestep(200_000, cfg, "cpu", torch.Generator().manual_seed(42))
        self.assertAlmostEqual(float(mask.float().mean()), .20, delta=.003)
        self.assertTrue(torch.all(pre_shift[mask] >= cfg.timestep_aux_min))
        self.assertTrue(torch.all(pre_shift[mask] < cfg.timestep_aux_max))

    def test_normal_branch_is_sigmoid_normal_and_both_branches_share_unclamped_shift(self):
        cfg = self.cfg(timestep_aux_prob=.5, timestep_aux_min=.1, timestep_aux_max=.2)
        generator = torch.Generator().manual_seed(12)
        actual, mask = sample_flow_timestep(32, 4096, cfg, "cpu", generator, return_aux_mask=True)
        expected_generator = torch.Generator().manual_seed(12)
        normal = torch.sigmoid(torch.randn(32, dtype=torch.float32, generator=expected_generator))
        expected_mask = torch.rand(32, dtype=torch.float32, generator=expected_generator) < .5
        auxiliary = torch.empty(32, dtype=torch.float32).uniform_(.1, .2, generator=expected_generator)
        pre_shift = torch.where(expected_mask, auxiliary, normal)
        mu = resolution_shift_mu(4096, cfg.mu_x1, cfg.mu_y1, cfg.mu_x2, cfg.mu_y2)
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertTrue(torch.equal(actual, shift_timestep(pre_shift, mu)))
        self.assertTrue(torch.equal(pre_shift[~mask], normal[~mask]))
        self.assertTrue(torch.all((actual > 0) & (actual < 1)))
        self.assertTrue(torch.any(actual[mask] > .2))  # proves there is no post-shift clamp to pre-shift support.

    def test_invalid_auxiliary_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_flow_timestep(1, 1, self.cfg(timestep_aux_prob=.2, timestep_aux_min=.6, timestep_aux_max=.4), "cpu")


if __name__ == "__main__":
    unittest.main()
