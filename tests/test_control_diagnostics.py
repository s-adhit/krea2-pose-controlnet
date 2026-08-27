import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pose_controlnet.turbo_evaluation import (
    CONTROL_SCALE_TURBO_EVALUATION_ROOT, CONTROL_SCALE_VALUES,
    LR5E5_CHECKPOINT_ROOT, LR5E5_HF_REPO_ID, LR5E5_HF_RUN_NAME,
    LR5E5_TURBO_EVALUATION_ROOT, ORIGINAL_TURBO_EVALUATION_ROOT,
    TIMESTEP_TURBO_EVALUATION_ROOT, assert_control_scale_turbo_output_isolated,
    exact_lr5e5_step1500_local_checkpoint, sample_turbo_pose_image,
    scale_turbo_control_latent, turbo_metadata,
)
from scripts import audit_control_projection, turbo_control_scale_sweep


class _RecordingProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3, 4))
        self.calls = []

    def forward(self, value):
        self.calls.append(value.detach().clone())
        return torch.nn.functional.linear(value, self.weight)


class ControlDiagnosticsTest(unittest.TestCase):
    def test_default_scale_is_exact_identity_and_only_control_is_scaled(self):
        control = torch.tensor([[[1.0, -2.0]]], dtype=torch.bfloat16)
        image = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)
        self.assertIs(scale_turbo_control_latent(control), control)
        self.assertTrue(torch.equal(scale_turbo_control_latent(control, 1.0), control))
        self.assertTrue(torch.equal(scale_turbo_control_latent(control, 1.5), control * 1.5))
        self.assertTrue(torch.equal(image, torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)))
        with self.assertRaises(ValueError):
            scale_turbo_control_latent(control, 0.0)

    def test_turbo_scale_one_preserves_sampler_control_tokens_and_weights(self):
        model = SimpleNamespace(config=SimpleNamespace(patch=1), weight=torch.nn.Parameter(torch.tensor(5.0)))
        sample = {"latent": torch.ones(1, 1, 1), "control": torch.tensor([[[2.0]]]),
                  "context": torch.ones(1, 1, 1), "mask": torch.ones(1, dtype=torch.bool)}
        controls = []
        def forward(_model, _image, control, *_args, **_kwargs):
            controls.append(control.clone())
            return torch.zeros_like(control)
        before = model.weight.detach().clone()
        with patch("pose_controlnet.turbo_evaluation.forward_pose_control", side_effect=forward):
            sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 123)
        self.assertEqual(len(controls), 8)
        self.assertTrue(all(torch.equal(value, torch.tensor([[[2.0]]], dtype=torch.bfloat16)) for value in controls))
        self.assertTrue(torch.equal(before, model.weight))

    def test_nonidentity_scale_changes_only_control_tokens_for_same_seed(self):
        model = SimpleNamespace(config=SimpleNamespace(patch=1))
        sample = {"latent": torch.ones(1, 1, 1), "control": torch.tensor([[[2.0]]]),
                  "context": torch.ones(1, 1, 1), "mask": torch.ones(1, dtype=torch.bool)}
        traces = []
        def forward(_model, image, control, *_args, **_kwargs):
            traces.append((image.clone(), control.clone()))
            return torch.zeros_like(image)
        with patch("pose_controlnet.turbo_evaluation.forward_pose_control", side_effect=forward):
            sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 321, control_scale=1.0)
            baseline = list(traces); traces.clear()
            sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 321, control_scale=1.5)
            scaled = list(traces)
        self.assertEqual(len(baseline), len(scaled))
        self.assertTrue(all(torch.equal(image, scaled[index][0]) for index, (image, _control) in enumerate(baseline)))
        self.assertTrue(all(torch.equal(control * 1.5, scaled[index][1]) for index, (_image, control) in enumerate(baseline)))

    def test_fixed_sweep_accepts_only_required_scales_and_immutable_turbo_contract(self):
        self.assertEqual(CONTROL_SCALE_VALUES, (0.75, 1.0, 1.25, 1.5, 2.0))
        self.assertEqual(turbo_control_scale_sweep._scale_label(1.0), "1p00")
        with self.assertRaises(ValueError):
            turbo_control_scale_sweep._scale_label(3.0)
        args = turbo_control_scale_sweep.parser().parse_args(["preflight"])
        self.assertEqual(args.checkpoint_dir, str(LR5E5_CHECKPOINT_ROOT))
        self.assertFalse(hasattr(args, "control_scale"))
        self.assertEqual(turbo_metadata(), {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0,
                                            "mu": 1.15, "mu_resolution_dependent": False,
                                            "schedule_source": "https://github.com/krea-ai/krea-2/blob/main/sampling.py"})

    def test_scale_output_isolated_from_every_existing_turbo_tree(self):
        self.assertEqual(assert_control_scale_turbo_output_isolated(CONTROL_SCALE_TURBO_EVALUATION_ROOT),
                         CONTROL_SCALE_TURBO_EVALUATION_ROOT.resolve())
        for path in (ORIGINAL_TURBO_EVALUATION_ROOT, LR5E5_TURBO_EVALUATION_ROOT, TIMESTEP_TURBO_EVALUATION_ROOT):
            with self.subTest(path=path), self.assertRaises(ValueError):
                assert_control_scale_turbo_output_isolated(path)

    def test_projection_audit_output_isolated_from_existing_turbo_trees(self):
        self.assertEqual(audit_control_projection.assert_projection_audit_output_isolated(
            audit_control_projection.OUTPUT), audit_control_projection.OUTPUT.resolve())
        for path in (ORIGINAL_TURBO_EVALUATION_ROOT, LR5E5_TURBO_EVALUATION_ROOT, TIMESTEP_TURBO_EVALUATION_ROOT):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit_control_projection.assert_projection_audit_output_isolated(path)

    def test_source_checkpoint_is_only_exact_local_step_1500_with_matching_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            checkpoint = root / "step_001500.pt"
            with patch("pose_controlnet.turbo_evaluation.LR5E5_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_local_checkpoint_for_hf_step", return_value=checkpoint) as validate, \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", return_value={"global_step": 1500}):
                actual = exact_lr5e5_step1500_local_checkpoint(checkpoint_dir=root, hf_repo_id=LR5E5_HF_REPO_ID,
                                                                marker_download_dir=Path(temp) / "markers")
        self.assertEqual(actual, checkpoint)
        self.assertEqual(validate.call_args.kwargs["run_name"], LR5E5_HF_RUN_NAME)
        self.assertEqual(validate.call_args.kwargs["step"], 1500)
        self.assertEqual(validate.call_args.kwargs["checkpoint"], checkpoint)

    def test_projection_paths_use_verified_image_then_control_feature_order(self):
        first = _RecordingProjection()
        model = SimpleNamespace(first=first)
        image = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        control = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
        paths = audit_control_projection.projection_contributions(model, image, control)
        self.assertEqual(len(first.calls), 3)
        self.assertTrue(torch.equal(paths["image_only_input"], torch.cat([image, torch.zeros_like(control)], dim=-1)))
        self.assertTrue(torch.equal(paths["control_only_input"], torch.cat([torch.zeros_like(image), control], dim=-1)))
        self.assertTrue(torch.equal(paths["both_input"], torch.cat([image, control], dim=-1)))
        self.assertTrue(torch.equal(first.calls[0], paths["image_only_input"]))
        self.assertTrue(torch.equal(first.calls[1], paths["control_only_input"]))
        self.assertTrue(torch.equal(first.calls[2], paths["both_input"]))

    def test_projection_measurements_are_real_outputs_with_deterministic_fixed_timestep_grid(self):
        self.assertEqual(audit_control_projection.FIXED_TIMESTEPS, (0.1, 0.3, 0.5, 0.7, 0.9))
        self.assertEqual(audit_control_projection._stable_seed(7, "sample", .3),
                         audit_control_projection._stable_seed(7, "sample", .3))
        self.assertNotEqual(audit_control_projection._stable_seed(7, "sample", .3),
                            audit_control_projection._stable_seed(7, "sample", .5))
        first = _RecordingProjection(); model = SimpleNamespace(first=first)
        values = audit_control_projection._record_measurements(
            audit_control_projection.projection_contributions(model, torch.ones(1, 1, 2), torch.full((1, 1, 2), 2.0))
        )
        summary = audit_control_projection._summary([values])
        self.assertGreater(summary["control_only_projection_output_rms"], 0.0)
        self.assertGreater(summary["image_only_projection_output_rms"], 0.0)
        self.assertIn("control_to_image_projection_ratio", summary)

    def test_diagnostic_entrypoints_cannot_train_or_change_turbo_scoring_implementations(self):
        root = Path(__file__).resolve().parents[1]
        scale_source = (root / "scripts/turbo_control_scale_sweep.py").read_text().lower()
        audit_source = (root / "scripts/audit_control_projection.py").read_text().lower()
        for source in (scale_source, audit_source):
            for forbidden in ("torch.optim", "optimizer.", "backward(", ".backward(", "model.train("):
                self.assertNotIn(forbidden, source)
        self.assertIs(turbo_control_scale_sweep.score_authoritative_pck,
                      __import__("pose_controlnet.post1500_evaluation", fromlist=["score_authoritative_pck"]).score_authoritative_pck)
        self.assertIs(turbo_control_scale_sweep._clip_score,
                      __import__("scripts.turbo_benchmark", fromlist=["_clip_score"])._clip_score)


if __name__ == "__main__":
    unittest.main()
