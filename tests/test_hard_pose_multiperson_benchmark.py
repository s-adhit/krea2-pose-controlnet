import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hard_pose_multiperson_benchmark.py"
SPEC = importlib.util.spec_from_file_location("hard_pose_multiperson_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class HardPoseMultipersonContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = benchmark.load_spec()
        self.conditions = self.spec["conditions"]

    def test_exact_twelve_order_and_required_coverage(self):
        self.assertEqual(self.spec["generation_count"], 12)
        self.assertEqual(len(self.conditions), 12)
        self.assertEqual([row["order"] for row in self.conditions], list(range(1, 13)))
        self.assertEqual([row["condition_id"].split("_", 1)[1] for row in self.conditions], list(benchmark.SINGLE_CATEGORIES + benchmark.MULTI_CATEGORIES))
        self.assertEqual([row["person_group"] for row in self.conditions], ["single-person"] * 8 + ["multi-person"] * 4)

    def test_mix025_scale_native_geometry_and_turbo_runtime_are_locked(self):
        self.assertEqual((self.spec["candidate"], self.spec["control_scale"], self.spec["geometry"]), ("mix-025", 1.0, benchmark.NATIVE_GEOMETRY))
        self.assertEqual((self.spec["runtime"]["steps"], self.spec["runtime"]["cfg"], self.spec["runtime"]["mu"]), (8, 0.0, 1.15))
        for key, value in (("candidate", "parent-4000"), ("control_scale", .5), ("geometry", "dynamic_768_bucket")):
            bad = copy.deepcopy(self.spec); bad[key] = value
            with self.assertRaisesRegex(ValueError, "lock"):
                benchmark._validate_locks(bad)

    def test_every_condition_keeps_frozen_prompt_seed_control_identity_and_native_bucket(self):
        prompts = {row["stem"]: row for row in (json.loads(line) for line in Path(self.spec["frozen_sources"]["prompts"]["path"]).read_text().splitlines())}
        final_spec = json.loads(Path(self.spec["frozen_sources"]["final_spec"]["path"]).read_text())
        sidecar = {row["stem"]: row for row in (json.loads(line) for line in Path(self.spec["frozen_sources"]["authoritative_pose_sidecar"]["path"]).joinpath("records.jsonl").read_text().splitlines())}
        for condition in self.conditions:
            stem = condition["stem"]
            self.assertEqual(condition["prompt"], prompts[stem]["text"])
            self.assertEqual(condition["seed"], final_spec["per_stem_seeds"][stem]["sampling"])
            self.assertEqual(condition["native_bucket"], sidecar[stem]["bucket"])
            self.assertEqual(condition["expected_reference_people"], len(sidecar[stem]["people"]))
            self.assertRegex(self.spec["control_sha256"][stem], r"^[0-9a-f]{64}$")

    def test_authoritative_sidecar_and_historical_artifacts_are_required_unchanged(self):
        sidecar = self.spec["frozen_sources"]["authoritative_pose_sidecar"]
        metadata = json.loads(Path(sidecar["path"]).joinpath("metadata.json").read_text())
        self.assertEqual(metadata["sidecar_kind"], "final_val_benchmark_48_authoritative_pose_targets_v3")
        self.assertEqual(sidecar["records_sha256"], "3cc4defc282cb11e956ec06517eff4e8369622d4c0b3b567ab2247efb4a499a7")
        benchmark._validate_historical_artifacts(self.spec)

    def test_spec_sha_is_immutable(self):
        self.assertEqual(benchmark._sha256(benchmark.SPEC), benchmark.SPEC_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(benchmark.SPEC.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                benchmark.load_spec(changed)

    def test_incomplete_outputs_are_refused(self):
        contract = {"conditions": self.conditions}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(benchmark._generation_status(output, contract), "missing")
            (output / "controls").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                benchmark._generation_status(output, contract)

    def test_observable_failure_taxonomy_uses_only_measurements(self):
        flags = benchmark._failure_flags({"reference_available": True, "unmatched_reference_people": 1, "unmatched_predicted_people": 2, "pck_005": .2, "pck_020": .4})
        self.assertEqual(flags, ["missed_reference_person", "extra_predicted_people", "low_strict_pck", "low_coarse_pck"])
        self.assertEqual(benchmark._failure_flags({"reference_available": False}), [])


if __name__ == "__main__":
    unittest.main()
