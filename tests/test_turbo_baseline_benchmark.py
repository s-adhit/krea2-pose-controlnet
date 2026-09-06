import copy
import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "turbo_baseline_benchmark.py"
SPEC = importlib.util.spec_from_file_location("turbo_baseline_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class TurboBaselineContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = benchmark.load_spec()

    def test_exact_candidate_order_and_192_generation_matrix(self):
        self.assertEqual(tuple(self.spec["candidate_order"]), benchmark.CANDIDATES)
        self.assertEqual(self.spec["generation_count"], 192)
        stems = [f"stem-{index}" for index in range(48)]
        self.assertEqual(len(benchmark._expected_pairs(stems)), 192)
        self.assertEqual(benchmark._expected_pairs(stems)[:4], [("stem-0", candidate) for candidate in benchmark.CANDIDATES])

    def test_turbo_base_is_unmodified_and_never_resolves_adapter_state(self):
        base = benchmark._candidate_config(self.spec, "turbo-base")
        self.assertEqual(base["kind"], "unmodified_krea2_turbo")
        self.assertEqual(base["pose_lora_control_adapter_state"], "absent")
        self.assertIsNone(base["control_scale"])
        with self.assertRaisesRegex(ValueError, "no Pose-LoRA/control adapter"):
            benchmark._resolve_controlled_candidate("turbo-base")
        source = inspect.getsource(benchmark.build_unmodified_turbo_base_model)
        self.assertIn("SingleStreamDiT", source)
        self.assertNotIn("build_turbo_pose_model", source)
        self.assertNotIn("load_trainable_state_dict", source)

    def test_parent_a4300_hashes_and_mix025_interpolation_are_pinned(self):
        parent = benchmark._candidate_config(self.spec, "parent-4000")
        finish = benchmark._candidate_config(self.spec, "finish-control-a4300")
        mix = benchmark._candidate_config(self.spec, "mix-025")
        self.assertEqual(parent["sha256"], "0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd")
        self.assertEqual(finish["sha256"], "17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff")
        self.assertEqual((mix["alpha"], mix["compute_dtype"], mix["endpoints"]), (.25, "float32", ["parent-4000", "finish-control-a4300"]))
        bad = copy.deepcopy(self.spec); bad["candidates"][2]["alpha"] = .5
        with self.assertRaisesRegex(ValueError, "mix-025"):
            benchmark._validate_locks(bad)

    def test_turbo_runtime_and_native_geometry_are_locked(self):
        self.assertEqual((self.spec["runtime"]["steps"], self.spec["runtime"]["cfg"], self.spec["runtime"]["mu"]), (8, 0.0, 1.15))
        self.assertEqual(self.spec["geometry"], benchmark.NATIVE_GEOMETRY)
        bad = copy.deepcopy(self.spec); bad["runtime"]["steps"] = 7
        with self.assertRaisesRegex(ValueError, "native geometry or locked Turbo"):
            benchmark._validate_locks(bad)

    def test_same_prompt_seed_and_geometry_are_recorded_for_every_candidate(self):
        condition = {"stem": "x", "prompt": "frozen caption", "seed": 42, "source": "coco", "orientation": "portrait", "control_sha256": "a" * 64, "geometry": {"bucket": [640, 960]}}
        contract = {"candidate_contracts": {candidate: benchmark._candidate_config(self.spec, candidate) for candidate in benchmark.CANDIDATES}, "final_val_spec_sha256": "b" * 64, "runtime": benchmark.RUNTIME, "turbo_checkpoint": {"path": "turbo", "sha256": "c" * 64}}
        rows = [benchmark._generation_metadata(condition, candidate, contract, Path("/control.png")) for candidate in benchmark.CANDIDATES]
        self.assertEqual({row["prompt"] for row in rows}, {"frozen caption"})
        self.assertEqual({row["seed"] for row in rows}, {42})
        self.assertEqual({tuple(row["geometry"]["bucket"]) for row in rows}, {(640, 960)})
        self.assertFalse(rows[0]["control_applied"])
        self.assertTrue(all(row["control_applied"] for row in rows[1:]))

    def test_incomplete_output_root_is_refused(self):
        conditions = [{"stem": f"stem-{index}", "geometry": {"bucket": [64, 64]}} for index in range(48)]
        contract = {"conditions": conditions}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(benchmark._generation_status(output, contract), "missing")
            output.joinpath("controls").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                benchmark._generation_status(output, contract)

    def test_frozen_spec_and_historical_artifacts_are_unchanged(self):
        self.assertEqual(benchmark._sha256(benchmark.SPEC), benchmark.SPEC_SHA256)
        benchmark._validate_historical_artifacts(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "spec.json"
            modified.write_text(benchmark.SPEC.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                benchmark.load_spec(modified)


if __name__ == "__main__":
    unittest.main()
