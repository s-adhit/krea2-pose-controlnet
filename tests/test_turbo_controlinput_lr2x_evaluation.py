import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.turbo_evaluation import (
    CONTROLINPUT_LR2X_HF_REPO_ID,
    CONTROLINPUT_LR2X_HF_RUN_NAME,
    CONTROLINPUT_LR2X_TURBO_CHECKPOINT_STEPS,
    CONTROLINPUT_LR2X_TURBO_EVALUATION_ROOT,
    CONTROL_SCALE_TURBO_EVALUATION_ROOT,
    LR5E5_TURBO_EVALUATION_ROOT,
    ORIGINAL_TURBO_EVALUATION_ROOT,
    TIMESTEP_TURBO_EVALUATION_ROOT,
    assert_controlinput_lr2x_turbo_output_isolated,
    assert_turbo_diagnostic_contract,
    exact_controlinput_lr2x_turbo_checkpoints,
    turbo_metadata,
)
from scripts import turbo_benchmark, turbo_controlinput_lr2x_benchmark


def _spec():
    stems = ["alpha", "beta"]
    return {
        "format_version": 1, "kind": "turbo_fixed_pose", "split": "diagnostic_val", "seed": 420200,
        "stems": stems, "turbo": turbo_metadata(),
        "per_stem_seeds": {stem: {"sampling": index + 10, "noise": index + 20, "timestep": index + 30}
                           for index, stem in enumerate(stems)},
        "sample_identities": {stem: {"latent": f"latent-{stem}", "control": f"control-{stem}"} for stem in stems},
    }


def _row(step: int):
    return {"checkpoint_step": step, "pose": {"pck_005": 0.1}, "clip": {"mean_cosine_similarity": 0.3}}


class TurboControlinputLr2xEvaluationTest(unittest.TestCase):
    def test_exact_run_namespace_and_sparse_first_pass_are_pinned(self):
        self.assertEqual(CONTROLINPUT_LR2X_HF_RUN_NAME, "pose-learning-1500-controlinput-lr2x-to2800")
        self.assertEqual(CONTROLINPUT_LR2X_TURBO_CHECKPOINT_STEPS, (1800, 2200, 2600, 2800))
        self.assertEqual(turbo_controlinput_lr2x_benchmark.CONTROL_SCALE, 1.0)

    def test_exact_local_checkpoints_require_marker_backed_branch_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            markers = Path(temp) / "markers"
            recovered = [Path(temp) / f"step_{step:06d}.pt" for step in CONTROLINPUT_LR2X_TURBO_CHECKPOINT_STEPS]
            with patch("pose_controlnet.turbo_evaluation.CONTROLINPUT_LR2X_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_local_checkpoint_for_hf_step", side_effect=recovered) as validate, \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", side_effect=lambda path: {"global_step": int(path.stem.split("_")[1])}):
                actual = exact_controlinput_lr2x_turbo_checkpoints(
                    checkpoint_dir=root, hf_repo_id=CONTROLINPUT_LR2X_HF_REPO_ID, marker_download_dir=markers
                )
        self.assertEqual(actual, list(zip(CONTROLINPUT_LR2X_TURBO_CHECKPOINT_STEPS, recovered)))
        self.assertEqual(
            [(call.kwargs["repo_id"], call.kwargs["run_name"], call.kwargs["step"]) for call in validate.call_args_list],
            [(CONTROLINPUT_LR2X_HF_REPO_ID, CONTROLINPUT_LR2X_HF_RUN_NAME, step)
             for step in CONTROLINPUT_LR2X_TURBO_CHECKPOINT_STEPS],
        )
        self.assertEqual(validate.call_args_list[0].kwargs["marker_download_dir"], markers)

    def test_nearest_latest_and_other_steps_are_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch("pose_controlnet.turbo_evaluation.CONTROLINPUT_LR2X_CHECKPOINT_ROOT", Path(temp)), \
             patch("pose_controlnet.turbo_evaluation.validated_local_checkpoint_for_hf_step") as validate:
            for requested in ((1800,), (1800, 2200, 2600), (1800, 2200, 2600, 2700), (2800, 2600, 2200, 1800)):
                with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "requires exactly"):
                    exact_controlinput_lr2x_turbo_checkpoints(
                        checkpoint_dir=temp, hf_repo_id=CONTROLINPUT_LR2X_HF_REPO_ID,
                        marker_download_dir=Path(temp) / "markers", steps=requested,
                    )
        validate.assert_not_called()

    def test_wrong_root_repo_or_embedded_step_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "checkpoint root"):
            exact_controlinput_lr2x_turbo_checkpoints(
                checkpoint_dir=temp, hf_repo_id=CONTROLINPUT_LR2X_HF_REPO_ID, marker_download_dir=Path(temp) / "markers"
            )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            checkpoint = Path(temp) / "step_001800.pt"
            with patch("pose_controlnet.turbo_evaluation.CONTROLINPUT_LR2X_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_local_checkpoint_for_hf_step", return_value=checkpoint), \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", return_value={"global_step": 1799}):
                with self.assertRaisesRegex(ValueError, "embedded step mismatch"):
                    exact_controlinput_lr2x_turbo_checkpoints(
                        checkpoint_dir=root, hf_repo_id=CONTROLINPUT_LR2X_HF_REPO_ID, marker_download_dir=Path(temp) / "markers"
                    )

    def test_baseline_is_read_from_existing_machine_readable_results_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = _spec()
            (root / "turbo_spec.json").write_text(json.dumps(spec), encoding="utf-8")
            (root / "pck_clip_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "checkpoints": [_row(1500)]}), encoding="utf-8")
            baseline = turbo_controlinput_lr2x_benchmark._baseline_row(
                argparse.Namespace(baseline_output_dir=str(root)), spec
            )
        self.assertEqual(baseline["checkpoint_step"], 1500)
        source = Path(turbo_controlinput_lr2x_benchmark.__file__).read_text()
        generate_source = source[source.index("def generate"):source.index("def score")]
        self.assertNotIn("BASELINE_STEP", generate_source)
        self.assertNotIn("baseline_output_dir", generate_source)

    def test_diagnostics_seeds_controls_prompts_geometry_and_turbo_are_unchanged(self):
        original, branch = _spec(), _spec()
        self.assertIsNone(assert_turbo_diagnostic_contract(branch, original, branch_name="ControlInput-LR2x"))
        branch["sample_identities"]["alpha"]["control"] = "different-control"
        with self.assertRaisesRegex(ValueError, "inputs, or per-stem seeds"):
            assert_turbo_diagnostic_contract(branch, original, branch_name="ControlInput-LR2x")
        self.assertEqual(turbo_metadata(), {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0,
                                            "mu": 1.15, "mu_resolution_dependent": False,
                                            "schedule_source": "https://github.com/krea-ai/krea-2/blob/main/sampling.py"})

    def test_pck_clip_and_control_scale_contracts_are_the_established_ones(self):
        self.assertIs(turbo_controlinput_lr2x_benchmark.score_authoritative_pck, score_authoritative_pck)
        self.assertIs(turbo_controlinput_lr2x_benchmark._clip_score, turbo_benchmark._clip_score)
        source = Path(turbo_controlinput_lr2x_benchmark.__file__).read_text().lower()
        self.assertIn("confidence_threshold=.5", source)
        self.assertIn("turbo_scoring_geometry", source)
        self.assertIn("control_scale=control_scale", source)

    def test_output_root_is_exact_and_protected_from_all_existing_turbo_trees(self):
        self.assertEqual(
            assert_controlinput_lr2x_turbo_output_isolated(CONTROLINPUT_LR2X_TURBO_EVALUATION_ROOT),
            CONTROLINPUT_LR2X_TURBO_EVALUATION_ROOT.resolve(),
        )
        for protected in (ORIGINAL_TURBO_EVALUATION_ROOT, LR5E5_TURBO_EVALUATION_ROOT,
                          TIMESTEP_TURBO_EVALUATION_ROOT, CONTROL_SCALE_TURBO_EVALUATION_ROOT):
            with self.subTest(protected=protected), self.assertRaises(ValueError):
                assert_controlinput_lr2x_turbo_output_isolated(protected)

    def test_entrypoint_has_only_staged_evaluation_commands_and_no_training_operations(self):
        parser = turbo_controlinput_lr2x_benchmark.parser()
        args = parser.parse_args(["preflight"])
        self.assertEqual(args.checkpoint_dir, str(turbo_controlinput_lr2x_benchmark.CONTROLINPUT_LR2X_CHECKPOINT_ROOT))
        self.assertFalse(hasattr(args, "steps"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["train"])
        source = Path(turbo_controlinput_lr2x_benchmark.__file__).read_text().lower()
        for forbidden in ("torch.optim", "optimizer.", "backward(", ".backward(", "model.train("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
