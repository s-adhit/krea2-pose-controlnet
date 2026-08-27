import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pose_controlnet.turbo_evaluation import (TURBO_CFG, TURBO_MU, TURBO_STEPS, assert_exact_diagnostic_stems,
    assert_turbo_output_isolated, exact_turbo_checkpoints, raw_to_turbo_control_compatibility,
    sample_turbo_pose_image, turbo_schedule)


class TurboEvaluationTest(unittest.TestCase):
    def test_official_pinned_turbo_schedule_has_exactly_eight_steps_and_is_resolution_invariant(self):
        first = turbo_schedule(image_sequence_length=4096)
        second = turbo_schedule(image_sequence_length=16384)
        self.assertEqual(len(first), TURBO_STEPS + 1); self.assertEqual(first, second)
        self.assertEqual(first, [1.0, 0.956723690032959, 0.9045307636260986, 0.8403487801551819,
                                 0.7595109343528748, 0.6545668244361877, 0.5128440856933594,
                                 0.31090107560157776, 0.0])
        self.assertEqual(TURBO_MU, 1.15)
        with self.assertRaises(ValueError): turbo_schedule(image_sequence_length=1, steps=7)
        with self.assertRaises(ValueError): turbo_schedule(image_sequence_length=1, mu=1.14)

    def test_exact_checkpoint_resolution_only_uses_800_and_1500_with_existing_validator(self):
        with tempfile.TemporaryDirectory() as temp, patch("pose_controlnet.turbo_evaluation.ordered_checkpoints", return_value=[(800, Path("800.pt")), (1500, Path("1500.pt"))]) as resolved:
            self.assertEqual(exact_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id="private"), [(800, Path("800.pt")), (1500, Path("1500.pt"))])
        self.assertEqual(resolved.call_args.kwargs["steps"], (800, 1500))

    def test_canonical_output_namespace_is_rejected_and_diagnostic_contract_is_exact(self):
        with self.assertRaises(ValueError): assert_turbo_output_isolated("/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500/turbo")
        self.assertEqual(len(assert_exact_diagnostic_stems([str(i) for i in range(24)], [str(i) for i in range(24)])), 24)
        with self.assertRaises(ValueError): assert_exact_diagnostic_stems([str(i) for i in range(23)], [str(i) for i in range(23)])

    def test_cfg_zero_uses_one_controlled_forward_per_denoise_step(self):
        model = SimpleNamespace(config=SimpleNamespace(patch=1))
        sample = {"latent": torch.ones(1, 1, 1), "control": torch.ones(1, 1, 1), "context": torch.ones(1, 1, 1), "mask": torch.ones(1, dtype=torch.bool)}
        calls = []
        def forward(_model, image, control, *_args, **_kwargs):
            calls.append(control.clone()); return torch.zeros_like(image)
        with patch("pose_controlnet.turbo_evaluation.forward_pose_control", side_effect=forward):
            pixels = sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 1)
        self.assertEqual(len(calls), TURBO_STEPS); self.assertTrue(all(control.abs().max() > 0 for control in calls)); self.assertEqual(pixels.shape, (1, 1, 1))
        with self.assertRaises(ValueError): sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 1, guidance=1.0)

    def test_turbo_modules_are_evaluation_only(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts/turbo_benchmark.py").read_text().lower() + (root / "pose_controlnet/turbo_evaluation.py").read_text().lower()
        self.assertNotIn("torch.optim", source); self.assertNotIn("backward(", source); self.assertNotIn("optimizer.step", source)
        self.assertEqual(TURBO_CFG, 0.0)

    def test_raw_to_turbo_compatibility_requires_raw_provenance_and_exact_state(self):
        model = torch.nn.Linear(1, 1)
        with patch("pose_controlnet.turbo_evaluation.trainable_state_dict", return_value={"first.weight": torch.zeros(2, 2)}):
            result = raw_to_turbo_control_compatibility(model, {"config": {"raw_ckpt": "raw.safetensors"}, "model": {"first.weight": torch.zeros(2, 2)}})
        self.assertEqual(result["shape_mismatches"], 0)
        with patch("pose_controlnet.turbo_evaluation.trainable_state_dict", return_value={"first.weight": torch.zeros(2, 2)}):
            with self.assertRaises(ValueError): raw_to_turbo_control_compatibility(model, {"config": {}, "model": {"first.weight": torch.zeros(2, 2)}})


if __name__ == "__main__":
    unittest.main()
