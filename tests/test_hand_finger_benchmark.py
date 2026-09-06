import copy
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hand_finger_benchmark.py"
SPEC = importlib.util.spec_from_file_location("hand_finger_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class HandFingerBenchmarkContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = benchmark.load_spec()
        self.rows = benchmark._rows(self.spec)

    def test_exact_six_condition_selection_and_visible_authoritative_wrists(self):
        conditions = self.spec["conditions"]
        self.assertEqual(len(conditions), 6)
        self.assertEqual([row["order"] for row in conditions], list(range(1, 7)))
        self.assertEqual(
            [row["selection_class"] for row in conditions],
            ["simple_standing", "arms_raised_overhead", "bent_elbow_heavy", "foreshortened_arm_hand", "inversion_unusual_orientation", "two_person_interaction_visible_hands"],
        )
        sidecar, _ = benchmark._sidecar_records(
            list(json.loads(Path("docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json").read_text())["stems"]),
            conditions,
        )
        for record in sidecar["raw_records"]:
            self.assertTrue(any("coordinate" in wrist for wrist in benchmark._wrist_regions(record)))

    def test_exact_prompt_modes_candidates_and_generation_counts(self):
        self.assertEqual(tuple(self.spec["prompt_mode_order"]), benchmark.PROMPT_MODES)
        self.assertEqual(tuple(self.spec["candidate_order"]), benchmark.CANDIDATES)
        self.assertEqual((self.spec["primary_generation_count"], self.spec["control_scale_generation_count"], self.spec["generation_count"]), (72, 18, 90))
        primary = [row for row in self.rows if row["study"] == "primary"]
        scales = [row for row in self.rows if row["study"] == "control_scale"]
        self.assertEqual((len(primary), len(scales), len(self.rows)), (72, 18, 90))
        self.assertEqual([row["prompt_mode"] for row in primary[:12]], ["hand_neutral"] * 4 + ["hand_compatible"] * 4 + ["hand_conflicting"] * 4)
        self.assertEqual([row["candidate"] for row in primary[:4]], list(benchmark.CANDIDATES))
        first = self.spec["conditions"][0]
        self.assertEqual(benchmark._prompt(first, "hand_neutral", self.spec), first["source_prompt"])
        self.assertEqual(benchmark._prompt(first, "hand_compatible", self.spec), first["source_prompt"] + ", natural relaxed hands and natural fingers")
        self.assertEqual(benchmark._prompt(first, "hand_conflicting", self.spec), first["source_prompt"] + ", both arms folded tightly across the chest with hands hidden in pockets")

    def test_same_condition_prompt_seed_is_shared_across_candidates_and_scales(self):
        for condition in self.spec["conditions"]:
            for mode in benchmark.PROMPT_MODES:
                rows = [row for row in self.rows if row["study"] == "primary" and row["condition"]["stem"] == condition["stem"] and row["prompt_mode"] == mode]
                self.assertEqual([row["candidate"] for row in rows], list(benchmark.CANDIDATES))
                self.assertEqual({row["condition"]["seed"] for row in rows}, {condition["seed"]})
            neutral = [row for row in self.rows if row["study"] == "control_scale" and row["condition"]["stem"] == condition["stem"]]
            self.assertEqual([row["control_scale"] for row in neutral], list(benchmark.SCALES))
            self.assertEqual({row["condition"]["seed"] for row in neutral}, {condition["seed"]})

    def test_native_geometry_runtime_primary_scale_and_substudy_scale_order_are_locked(self):
        self.assertEqual(self.spec["geometry"], benchmark.NATIVE_GEOMETRY)
        self.assertEqual((self.spec["runtime"]["steps"], self.spec["runtime"]["cfg"], self.spec["runtime"]["mu"]), (8, 0.0, 1.15))
        self.assertEqual(self.spec["primary_control_scale"], 1.0)
        self.assertEqual(self.spec["control_scale_substudy"]["scales"], [0.75, 1.0, 1.5])
        bad = copy.deepcopy(self.spec); bad["runtime"]["steps"] = 7
        with self.assertRaisesRegex(ValueError, "runtime"):
            benchmark._validate_locks(bad)

    def test_turbo_base_is_unmodified_and_never_loads_pose_state(self):
        base = benchmark._candidate_contract(self.spec, "turbo-base")
        self.assertEqual((base["kind"], base["pose_lora_control_adapter_state"]), ("unmodified_krea2_turbo", "absent"))
        source = inspect.getsource(baseline_build := benchmark.baseline.build_unmodified_turbo_base_model)
        self.assertNotIn("build_turbo_pose_model", source)
        self.assertNotIn("load_trainable_state_dict", source)
        self.assertIsNotNone(baseline_build)

    def test_immutable_spec_historical_artifacts_and_incomplete_output_refusal(self):
        self.assertEqual(benchmark._sha256(benchmark.SPEC), benchmark.SPEC_SHA256)
        benchmark._validate_historical(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"; changed.write_text(benchmark.SPEC.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                benchmark.load_spec(changed)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); contract = {"control_sha256": {row["stem"]: row["control_sha256"] for row in self.spec["conditions"]}}
            self.assertEqual(benchmark._generation_status(output, contract, self.rows), "missing")
            (output / "controls").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                benchmark._generation_status(output, contract, self.rows)


if __name__ == "__main__":
    unittest.main()
