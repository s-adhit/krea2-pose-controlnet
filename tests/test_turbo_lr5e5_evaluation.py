import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.turbo_evaluation import (
    LR5E5_HF_REPO_ID, LR5E5_HF_RUN_NAME, LR5E5_TURBO_CHECKPOINT_STEPS,
    ORIGINAL_TURBO_EVALUATION_ROOT, assert_lr5e5_diagnostic_contract,
    assert_lr5e5_turbo_output_isolated, exact_lr5e5_turbo_checkpoints,
    turbo_metadata,
)
from scripts import turbo_benchmark, turbo_lr5e5_benchmark


def _spec():
    stems = ["alpha", "beta"]
    return {
        "format_version": 1, "kind": "turbo_fixed_pose", "split": "diagnostic_val", "seed": 420200,
        "stems": stems, "turbo": turbo_metadata(),
        "per_stem_seeds": {stem: {"sampling": index + 10, "noise": index + 20, "timestep": index + 30}
                           for index, stem in enumerate(stems)},
        "sample_identities": {stem: {"latent": f"latent-{stem}", "control": f"control-{stem}"} for stem in stems},
    }


class TurboLr5e5EvaluationTest(unittest.TestCase):
    def test_lr_branch_checkpoints_resolve_only_from_exact_branch_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            recovery = Path(temp) / "recovery"
            recovered = [Path(temp) / f"step_{step:06d}.pt" for step in LR5E5_TURBO_CHECKPOINT_STEPS]
            with patch("pose_controlnet.turbo_evaluation.LR5E5_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_hf_checkpoint_for_step", side_effect=recovered) as fetch, \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", side_effect=lambda path: {"global_step": int(path.stem.split("_")[1])}):
                actual = exact_lr5e5_turbo_checkpoints(checkpoint_dir=root, hf_repo_id=LR5E5_HF_REPO_ID,
                                                        hf_recovery_dir=recovery)
        self.assertEqual(actual, list(zip(LR5E5_TURBO_CHECKPOINT_STEPS, recovered)))
        self.assertEqual([(call.kwargs["repo_id"], call.kwargs["run_name"], call.kwargs["step"])
                          for call in fetch.call_args_list],
                         [(LR5E5_HF_REPO_ID, LR5E5_HF_RUN_NAME, step) for step in LR5E5_TURBO_CHECKPOINT_STEPS])

    def test_original_and_lr_branch_checkpoints_cannot_be_confused(self):
        with tempfile.TemporaryDirectory() as temp, \
             self.assertRaisesRegex(ValueError, "checkpoint root"):
            exact_lr5e5_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id=LR5E5_HF_REPO_ID)
        with tempfile.TemporaryDirectory() as temp, \
             patch("pose_controlnet.turbo_evaluation.LR5E5_CHECKPOINT_ROOT", Path(temp)), \
             self.assertRaisesRegex(ValueError, "HF repo"):
            exact_lr5e5_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id="other/checkpoints")
        with self.assertRaisesRegex(ValueError, "original Turbo results"):
            assert_lr5e5_turbo_output_isolated(ORIGINAL_TURBO_EVALUATION_ROOT)

    def test_exact_embedded_global_step_validation_remains_active(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"; checkpoint = Path(temp) / "step_001000.pt"
            with patch("pose_controlnet.turbo_evaluation.LR5E5_CHECKPOINT_ROOT", root), \
                 patch("pose_controlnet.turbo_evaluation.validated_hf_checkpoint_for_step", return_value=checkpoint), \
                 patch("pose_controlnet.turbo_evaluation.load_training_state", return_value={"global_step": 999}):
                with self.assertRaisesRegex(ValueError, "embedded step mismatch"):
                    exact_lr5e5_turbo_checkpoints(checkpoint_dir=root, hf_repo_id=LR5E5_HF_REPO_ID)

    def test_diagnostic_seeds_inputs_and_turbo_contract_are_immutable(self):
        original = _spec(); branch = _spec()
        self.assertIsNone(assert_lr5e5_diagnostic_contract(branch, original))
        branch["per_stem_seeds"]["alpha"]["sampling"] += 1
        with self.assertRaisesRegex(ValueError, "inputs, or per-stem seeds"):
            assert_lr5e5_diagnostic_contract(branch, original)
        self.assertEqual(turbo_metadata(), {"model": "Krea-2 Turbo", "steps": 8, "cfg": 0.0,
                                            "mu": 1.15, "mu_resolution_dependent": False,
                                            "schedule_source": "https://github.com/krea-ai/krea-2/blob/main/sampling.py"})

    def test_pck_and_clip_reuse_the_unchanged_authoritative_implementations(self):
        self.assertIs(turbo_lr5e5_benchmark.score_authoritative_pck, score_authoritative_pck)
        self.assertIs(turbo_lr5e5_benchmark._clip_score, turbo_benchmark._clip_score)
        source = Path(turbo_lr5e5_benchmark.__file__).read_text().lower()
        self.assertIn("confidence_threshold=.5", source)
        self.assertIn("turbo_scoring_geometry", source)

    def test_branch_entrypoint_cannot_construct_optimizer_or_run_training(self):
        source = Path(turbo_lr5e5_benchmark.__file__).read_text().lower()
        for forbidden in ("torch.optim", "optimizer.", "backward(", ".backward(", "model.train("):
            self.assertNotIn(forbidden, source)
        args = turbo_lr5e5_benchmark.parser().parse_args(["preflight"])
        self.assertEqual(args.checkpoint_dir, str(turbo_lr5e5_benchmark.LR5E5_CHECKPOINT_ROOT))
        self.assertFalse(hasattr(args, "steps"))


if __name__ == "__main__":
    unittest.main()
