import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.checkpointing import validated_local_checkpoint_for_hf_step
from pose_controlnet.turbo_evaluation import (
    LR5E5_TURBO_EVALUATION_ROOT,
    ORIGINAL_TURBO_EVALUATION_ROOT,
    TIMESTEP_HF_REPO_ID,
    TIMESTEP_HF_RUN_NAME,
    TIMESTEP_TURBO_CHECKPOINT_STEPS,
    assert_timestep_turbo_output_isolated,
    assert_turbo_diagnostic_contract,
    exact_timestep_turbo_checkpoints,
    turbo_metadata,
)
from scripts import turbo_benchmark, turbo_timestep_benchmark


def _spec():
    stems = ["alpha", "beta"]
    return {
        "format_version": 1,
        "kind": "turbo_fixed_pose",
        "split": "diagnostic_val",
        "seed": 420200,
        "stems": stems,
        "turbo": turbo_metadata(),
        "per_stem_seeds": {stem: {"sampling": index + 10, "noise": index + 20, "timestep": index + 30}
                           for index, stem in enumerate(stems)},
        "sample_identities": {stem: {"latent": f"latent-{stem}", "control": f"control-{stem}"}
                              for stem in stems},
    }


class TurboTimestepEvaluationTest(unittest.TestCase):
    def test_exact_timestep_checkpoints_resolve_only_from_completed_branch_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            recovery = Path(temp) / "recovery"
            recovered = [Path(temp) / f"step_{step:06d}.pt" for step in TIMESTEP_TURBO_CHECKPOINT_STEPS]
            with patch("pose_controlnet.turbo_evaluation.TIMESTEP_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_hf_checkpoint_for_step", side_effect=recovered[:2]) as fetch, \
                 patch("pose_controlnet.turbo_evaluation.validated_local_checkpoint_for_hf_step", return_value=recovered[2]) as local, \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", side_effect=lambda path: {"global_step": int(path.stem.split("_")[1])}):
                actual = exact_timestep_turbo_checkpoints(checkpoint_dir=root, hf_repo_id=TIMESTEP_HF_REPO_ID,
                                                           hf_recovery_dir=recovery)
        self.assertEqual(actual, list(zip(TIMESTEP_TURBO_CHECKPOINT_STEPS, recovered)))
        self.assertEqual([(call.kwargs["repo_id"], call.kwargs["run_name"], call.kwargs["step"])
                          for call in fetch.call_args_list],
                         [(TIMESTEP_HF_REPO_ID, TIMESTEP_HF_RUN_NAME, step) for step in (1600, 1700)])
        self.assertEqual(local.call_args.kwargs["checkpoint"], root / "step_001800.pt")
        self.assertEqual(local.call_args.kwargs["run_name"], TIMESTEP_HF_RUN_NAME)
        self.assertEqual(local.call_args.kwargs["step"], 1800)

    def test_only_exact_steps_are_accepted_without_nearest_or_latest_fallback(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch("pose_controlnet.turbo_evaluation.TIMESTEP_CHECKPOINT_ROOT", Path(temp)), \
             patch("pose_controlnet.turbo_evaluation.validated_hf_checkpoint_for_step") as fetch:
            for requested in ((1600,), (1600, 1700), (1600, 1700, 1801), (1800, 1700, 1600)):
                with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "requires exactly"):
                    exact_timestep_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id=TIMESTEP_HF_REPO_ID, steps=requested)
        fetch.assert_not_called()

    def test_wrong_local_root_or_hf_repo_are_rejected_before_checkpoint_resolution(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "checkpoint root"):
            exact_timestep_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id=TIMESTEP_HF_REPO_ID)
        with tempfile.TemporaryDirectory() as temp, \
             patch("pose_controlnet.turbo_evaluation.TIMESTEP_CHECKPOINT_ROOT", Path(temp)), \
             self.assertRaisesRegex(ValueError, "HF repo"):
            exact_timestep_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id="other/checkpoints")

    def test_embedded_global_step_validation_remains_active(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            checkpoint = Path(temp) / "step_001600.pt"
            with patch("pose_controlnet.turbo_evaluation.TIMESTEP_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_hf_checkpoint_for_step", return_value=checkpoint), \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", return_value={"global_step": 1599}):
                with self.assertRaisesRegex(ValueError, "embedded step mismatch"):
                    exact_timestep_turbo_checkpoints(checkpoint_dir=root, hf_repo_id=TIMESTEP_HF_REPO_ID)

    def test_local_step_1800_requires_its_exact_hf_marker_and_sha256(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "step_001800.pt"
            with patch("pose_controlnet.checkpointing._validated_hf_marker_for_step",
                       return_value=({"sha256": "expected"}, "pose-learning-1500-timestep-lowmid20-to1800/full/step_001800.pt")) as marker, \
                 patch("pose_controlnet.checkpointing._sha256", return_value="expected"), \
                 patch("pose_controlnet.checkpointing.load_training_state", return_value={"global_step": 1800}):
                self.assertEqual(validated_local_checkpoint_for_hf_step(
                    checkpoint=checkpoint, repo_id=TIMESTEP_HF_REPO_ID, run_name=TIMESTEP_HF_RUN_NAME,
                    step=1800, marker_download_dir=Path(temp) / "markers"), checkpoint)
            self.assertEqual(marker.call_args.kwargs["run_name"], TIMESTEP_HF_RUN_NAME)
            self.assertEqual(marker.call_args.kwargs["step"], 1800)
            with patch("pose_controlnet.checkpointing._validated_hf_marker_for_step",
                       return_value=({"sha256": "expected"}, "remote")), \
                 patch("pose_controlnet.checkpointing._sha256", return_value="wrong"):
                self.assertIsNone(validated_local_checkpoint_for_hf_step(
                    checkpoint=checkpoint, repo_id=TIMESTEP_HF_REPO_ID, run_name=TIMESTEP_HF_RUN_NAME,
                    step=1800, marker_download_dir=Path(temp) / "markers"))

    def test_timestep_output_cannot_overwrite_original_or_lr_only_roots(self):
        with self.assertRaisesRegex(ValueError, "original Turbo results"):
            assert_timestep_turbo_output_isolated(ORIGINAL_TURBO_EVALUATION_ROOT)
        with self.assertRaisesRegex(ValueError, "LR=5e-5 Turbo results"):
            assert_timestep_turbo_output_isolated(LR5E5_TURBO_EVALUATION_ROOT)

    def test_diagnostic_inputs_seeds_and_turbo_contract_are_unchanged(self):
        original = _spec()
        timestep = _spec()
        self.assertIsNone(assert_turbo_diagnostic_contract(timestep, original, branch_name="timestep-exposure"))
        timestep["sample_identities"]["alpha"]["control"] = "changed"
        with self.assertRaisesRegex(ValueError, "inputs, or per-stem seeds"):
            assert_turbo_diagnostic_contract(timestep, original, branch_name="timestep-exposure")
        self.assertEqual(turbo_metadata(), {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0,
                                            "mu": 1.15, "mu_resolution_dependent": False,
                                            "schedule_source": "https://github.com/krea-ai/krea-2/blob/main/sampling.py"})

    def test_pck_clip_and_training_safety_contracts_are_unchanged(self):
        self.assertIs(turbo_timestep_benchmark.score_authoritative_pck, score_authoritative_pck)
        self.assertIs(turbo_timestep_benchmark._clip_score, turbo_benchmark._clip_score)
        source = Path(turbo_timestep_benchmark.__file__).read_text().lower()
        self.assertIn("confidence_threshold=.5", source)
        self.assertIn("turbo_scoring_geometry", source)
        for forbidden in ("torch.optim", "optimizer.", "backward(", ".backward(", "model.train("):
            self.assertNotIn(forbidden, source)
        args = turbo_timestep_benchmark.parser().parse_args(["preflight"])
        self.assertEqual(args.checkpoint_dir, str(turbo_timestep_benchmark.TIMESTEP_CHECKPOINT_ROOT))
        self.assertFalse(hasattr(args, "steps"))


if __name__ == "__main__":
    unittest.main()
