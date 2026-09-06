import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "control_scale_sweep.py"
SPEC = importlib.util.spec_from_file_location("control_scale_sweep", MODULE_PATH)
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


class ControlScaleSweepContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = sweep.load_sweep_spec()

    def test_exact_scale_order_generation_count_and_common_seed_per_pose(self):
        self.assertEqual(tuple(self.spec["control_scales"]), sweep.SCALES)
        self.assertEqual(self.spec["generation_count"], 35)
        pairs = sweep._expected_pairs(self.spec)
        self.assertEqual(len(pairs), 35)
        self.assertEqual(pairs, [(condition["stem"], scale) for scale in sweep.SCALES for condition in self.spec["conditions"]])
        for condition in self.spec["conditions"]:
            metadata = [sweep._generation_metadata(condition, scale, self._contract(), Path("/control.png")) for scale in sweep.SCALES]
            self.assertEqual({item["seed"] for item in metadata}, {condition["sampling_seed"]})
            self.assertEqual([item["control_scale"] for item in metadata], list(sweep.SCALES))

    def test_immutable_spec_hash_and_structural_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "sweep.json"
            changed.write_text(sweep.SWEEP_SPEC.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                sweep.load_sweep_spec(changed)
        self.assertEqual(sweep._sha256(sweep.SWEEP_SPEC), sweep.SWEEP_SPEC_SHA256)
        bad = copy.deepcopy(self.spec); bad["control_scales"] = list(reversed(sweep.SCALES))
        with self.assertRaisesRegex(ValueError, "five poses x seven ordered scales"):
            sweep._validate_locks(bad)

    def test_candidate_geometry_and_locked_turbo_runtime_refuse_drift(self):
        bad_candidate = copy.deepcopy(self.spec); bad_candidate["candidate"] = "mix-050"
        with self.assertRaisesRegex(ValueError, "mix-025"):
            sweep._validate_locks(bad_candidate)
        bad_geometry = copy.deepcopy(self.spec); bad_geometry["geometry"] = "square_crop"
        with self.assertRaisesRegex(ValueError, "native/aspect-preserving"):
            sweep._validate_locks(bad_geometry)
        bad_runtime = copy.deepcopy(self.spec); bad_runtime["runtime"]["steps"] = 7
        with self.assertRaisesRegex(ValueError, "8-step CFG-0 mu=1.15"):
            sweep._validate_locks(bad_runtime)
        self.assertEqual((self.spec["runtime"]["steps"], self.spec["runtime"]["cfg"], self.spec["runtime"]["mu"]), (8, 0.0, 1.15))

    def test_incomplete_output_refuses_overwrite(self):
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(sweep._generation_status(output, self.spec, candidate, self._contract()), "missing")
            output.joinpath("generation_results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                sweep._generation_status(output, self.spec, candidate, self._contract())

    def test_orphaned_control_refuses_overwrite(self):
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            output.joinpath("controls").mkdir()
            output.joinpath("controls", f"{self.spec['conditions'][0]['stem']}.png").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                sweep._generation_status(output, self.spec, candidate, self._contract())

    def test_provenance_metadata_binds_candidate_runtime_native_geometry_and_control(self):
        condition = self.spec["conditions"][0]
        metadata = sweep._generation_metadata(condition, 0.0, self._contract(), Path("/authoritative/control.png"))
        self.assertEqual(metadata["candidate"], "mix-025")
        self.assertEqual(metadata["checkpoint_interpolation"], self.spec["checkpoint_interpolation"])
        self.assertEqual(metadata["runtime"], self.spec["runtime"])
        self.assertEqual(metadata["geometry"], sweep.NATIVE_GEOMETRY)
        self.assertEqual(metadata["control_sha256"], condition["control_sha256"])
        self.assertEqual(metadata["seed"], condition["sampling_seed"])
        self.assertEqual(metadata["control_scale"], 0.0)

    def test_score_records_require_complete_exact_matrix_order(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            sweep._score_records({"per_generation": []}, self.spec)
        records = [{"stem": stem, "control_scale": scale} for stem, scale in sweep._expected_pairs(self.spec)]
        self.assertEqual(sweep._score_records({"per_generation": records}, self.spec), records)

    def _contract(self):
        return {
            "final_val_spec_sha256": self.spec["final_val_spec_sha256"],
            "runtime": self.spec["runtime"],
            "checkpoint_interpolation": self.spec["checkpoint_interpolation"],
        }


if __name__ == "__main__":
    unittest.main()
