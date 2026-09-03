import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prompting_guide_study.py"
SPEC = importlib.util.spec_from_file_location("prompting_guide_study", MODULE_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class PromptingGuideStudyContractTest(unittest.TestCase):
    def test_frozen_file_hash_and_exact_matrix_are_pinned(self):
        rows = study.load_study_rows()
        self.assertEqual(len(rows), 64)
        self.assertEqual(tuple(row["mode"] for row in rows[:8]), study.MODES)
        self.assertEqual(len({row["stem"] for row in rows}), 8)
        self.assertEqual(len({row["pose_class"] for row in rows}), 8)

    def test_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "prompting_study.jsonl"
            changed.write_text(study.STUDY_FILE.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                study.load_study_rows(changed)

    def test_duplicate_stem_mode_and_unexpected_mode_fail_closed(self):
        rows = study.load_study_rows()
        duplicated = [dict(row) for row in rows]
        duplicated[1]["mode"] = study.MODES[0]
        with mock.patch.object(study, "_read_jsonl", return_value=duplicated):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                study.load_study_rows()
        unexpected = [dict(row) for row in rows]
        unexpected[1]["mode"] = "P8_not_allowed"
        with mock.patch.object(study, "_read_jsonl", return_value=unexpected):
            with self.assertRaisesRegex(ValueError, "unexpected mode"):
                study.load_study_rows()

    def test_all_conditions_reuse_their_frozen_final_val_sampling_seed(self):
        rows = study.load_study_rows()
        final_spec, _ = study.final_val.load_final_spec(study.final_val.FINAL_SPEC)
        for stem in {row["stem"] for row in rows}:
            expected = final_spec["per_stem_seeds"][stem]["sampling"]
            self.assertIsInstance(expected, int)
            self.assertEqual({final_spec["per_stem_seeds"][row["stem"]]["sampling"] for row in rows if row["stem"] == stem}, {expected})

    def test_locked_turbo_runtime_contract_rejects_drift(self):
        study._validate_locked_runtime_contract()
        drifted = dict(study.TURBO, steps=7)
        with mock.patch.object(study, "TURBO", drifted):
            with self.assertRaisesRegex(ValueError, "locked Turbo"):
                study._validate_locked_runtime_contract()

    def test_incomplete_generation_artifacts_refuse_overwrite(self):
        row = {"stem": "one", "pose_class": "pose", "mode": study.MODES[0], "prompt": "exact prompt"}
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        contract = {
            "turbo": study.TURBO, "sampling_seeds": {"one": 1}, "buckets": {"one": [64, 64]},
            "control_sha256": {"one": "not-used"}, "prompt_file": {"sha256": study.STUDY_SHA256},
            "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None,
            "checkpoint_interpolation": {"alpha": .25},
        }
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(study._generation_status(Path(directory), [row], candidate, contract), "missing")
            output = Path(directory)
            output.joinpath("generation_results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study._generation_status(output, [row], candidate, contract)

    def test_orphaned_control_without_its_generation_refuses_overwrite(self):
        row = {"stem": "one", "pose_class": "pose", "mode": study.MODES[0], "prompt": "exact prompt"}
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        contract = {"turbo": study.TURBO, "sampling_seeds": {"one": 1}, "buckets": {"one": [64, 64]},
                    "control_sha256": {"one": "not-used"}, "prompt_file": {"sha256": study.STUDY_SHA256},
                    "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None,
                    "checkpoint_interpolation": {"alpha": .25}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            output.joinpath("controls").mkdir()
            output.joinpath("controls", "one.png").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study._generation_status(output, [row], candidate, contract)

    def test_scored_records_require_all_64_frozen_pairs_in_order(self):
        rows = study.load_study_rows()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            study._score_records({"per_generation": []}, rows)


if __name__ == "__main__":
    unittest.main()
