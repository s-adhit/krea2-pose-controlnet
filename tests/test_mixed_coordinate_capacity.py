import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pose_controlnet.capacity_pose import load_capacity_pose_records
from pose_controlnet.overfit_capacity import (
    CapacityScientificConfig, OVERFIT_CHECKPOINT_STEPS, OVERFIT_STEPS,
    capacity_experiment_name, validate_capacity_scientific_config, validate_manifest,
)
from pose_controlnet.reference_pose import load_exact_capacity_reference_sidecar
from scripts import audit_pose_gradient_balance as audit
from scripts import summarize_overfit_capacity as summary


SIDECAR = Path("data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl")


class _GeometryData:
    def __init__(self, records):
        self.items = [{"stem": row["stem"], "source_size": row.get("source_size", [768, 768]),
                       "resized_size": [768, 768], "crop_box": [0, 0, 768, 768], "bucket": [768, 768]}
                      for row in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


class MixedCoordinateCapacityTests(unittest.TestCase):
    def setUp(self):
        self.stems = validate_manifest("overfit32-mixed-r64-mse")

    def test_coordinate_huber_is_the_only_capacity_pose_choice_and_lambda_is_validated(self):
        selected = validate_capacity_scientific_config(CapacityScientificConfig(
            base_experiment="mixed32", resolution="768", pose_loss="normalized_coordinate_huber", lambda_pose=2e-5,
            pose_timestep_min=.10, pose_timestep_max=.20,
        ))
        self.assertEqual(selected.experiment_name, "overfit32-mixed-r64-coord-l2e-5-res768")
        self.assertEqual(capacity_experiment_name("mixed32", "768", "normalized_coordinate_huber", 2.5e-5),
                         "overfit32-mixed-r64-coord-l2.5e-5-res768")
        with self.assertRaises(ValueError):
            validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32", resolution="768", pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5, pose_timestep_min=.1, pose_timestep_max=.2))
        with self.assertRaises(ValueError): audit.validate_candidate_lambdas((0.0,))
        with self.assertRaises(ValueError): audit.validate_candidate_lambdas((1e-5, 1e-5))
        self.assertEqual(audit.validate_candidate_lambdas((1e-6, 1e-5)), (1e-6, 1e-5))

    def test_calibration_contract_is_768_coordinate_and_conservative_window(self):
        self.assertEqual(audit.validate_calibration_request(
            resolution="768", pose_loss="normalized_coordinate_huber", sidecar=SIDECAR,
            timestep_min=.10, timestep_max=.20, timesteps=(.10, .15, .20), candidates=(1e-6, 1e-5),
        ), (1e-6, 1e-5))
        with self.assertRaises(ValueError):
            audit.validate_calibration_request(resolution="native", pose_loss="normalized_coordinate_huber", sidecar=SIDECAR,
                                              timestep_min=.10, timestep_max=.20, timesteps=(.10,), candidates=(1e-5,))
        with self.assertRaises(ValueError):
            audit.validate_calibration_request(resolution="768", pose_loss="normalized_coordinate_huber", sidecar=SIDECAR,
                                              timestep_min=.10, timestep_max=.20, timesteps=(.30,), candidates=(1e-5,))

    def test_exact_mixed_identity_sidecar_reuse_and_danbooru_exclusion(self):
        before = SIDECAR.read_bytes()
        metadata, source_rows = load_exact_capacity_reference_sidecar(
            SIDECAR, experiment_name="overfit32-mixed-r64-coord-l2e-5-res768", expected_stems=self.stems,
        )
        data = _GeometryData(source_rows)
        projected_metadata, records = load_capacity_pose_records(
            sidecar=SIDECAR, experiment_name=audit.CALIBRATION_EXPERIMENT, data=data, stems=self.stems,
        )
        self.assertEqual(SIDECAR.read_bytes(), before)
        self.assertEqual(metadata["records_sha256"], projected_metadata["records_sha256"])
        self.assertEqual(tuple(records), self.stems)
        danbooru = [records[stem] for stem in self.stems if stem.startswith("danbooru_")]
        self.assertEqual(len(danbooru), 6)
        self.assertTrue(all(row["pose_reward_available"] is False for row in danbooru))
        self.assertTrue(all(row["pose_reward_available"] is True for stem, row in records.items() if not stem.startswith("danbooru_")))

    def test_audit_has_no_parameter_update_or_production_path_mutation(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("save_training_state", source)
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "step" for node in ast.walk(tree)))

    def test_checkpoint_schedule_and_baseline_mse_semantics_are_unchanged(self):
        self.assertEqual(OVERFIT_STEPS, (0, 50, 100, 200, 300, 400, 500))
        self.assertEqual(OVERFIT_CHECKPOINT_STEPS, (50, 100, 200, 300, 400, 500))
        baseline = validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32", resolution="768"))
        self.assertEqual((baseline.pose_loss, baseline.lambda_pose, baseline.experiment_name),
                         ("none", 0.0, "overfit32-mixed-r64-mse-res768"))

    def test_compact_native_only_comparison_never_declares_a_winner(self):
        checkpoints = [{"checkpoint_step": step, "pose": {"pck_010": step / 1000}} for step in OVERFIT_STEPS]
        summaries = {"baseline": {"quantitative_metrics": checkpoints}, "candidate": {"quantitative_metrics": checkpoints}}
        result = summary.compact_checkpoint_comparison(summaries, "baseline", "candidate")
        self.assertEqual(result["evaluation_resolution"], "native")
        self.assertFalse(result["winner_declared"])
        self.assertEqual([row["checkpoint_step"] for row in result["by_checkpoint"]], list(OVERFIT_STEPS))

    def test_compact_comparison_cli_loads_only_its_two_native_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output, checkpoints = root / "evaluation", root / "checkpoints"
            rows = [{"checkpoint_step": step, "pose": {"pck_010": step / 1000}} for step in OVERFIT_STEPS]
            for name in ("baseline", "candidate"):
                (output / name).mkdir(parents=True); (checkpoints / name).mkdir(parents=True)
                (output / name / "overfit_summary.json").write_text(json.dumps({"training_resolution": "768", "evaluation_resolution": "native", "checkpoints": rows}))
                (checkpoints / name / "experiment_metadata.json").write_text(json.dumps({"scientific_config": {"resolution": "768", "pose_loss": "none"}}))
            with patch.object(sys, "argv", ["summarize_overfit_capacity.py", "--output-root", str(output), "--checkpoint-root", str(checkpoints), "--compare", "baseline", "candidate"]):
                summary.main()
            result = json.loads((output / "capacity_comparison_summary.json").read_text())
            self.assertEqual(result["compact_checkpoint_comparison"]["baseline"], "baseline")
            self.assertFalse(result["compact_checkpoint_comparison"]["winner_declared"])


if __name__ == "__main__":
    unittest.main()
