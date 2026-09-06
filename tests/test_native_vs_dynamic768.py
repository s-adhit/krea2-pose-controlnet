import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "native_vs_dynamic768.py"
SPEC = importlib.util.spec_from_file_location("native_vs_dynamic768", MODULE_PATH)
assert SPEC and SPEC.loader
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


class NativeVsDynamic768ContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = ablation.load_ablation_spec()

    def test_exact_two_mode_order_generation_count_and_shared_condition_inputs(self):
        self.assertEqual(tuple(self.spec["geometry_modes"]), ablation.GEOMETRY_MODES)
        self.assertEqual(self.spec["generation_count"], 10)
        pairs = ablation._expected_pairs(self.spec)
        self.assertEqual(pairs, [(row["stem"], mode) for row in self.spec["conditions"] for mode in ablation.GEOMETRY_MODES])
        self.assertEqual(len(pairs), 10)
        for condition in self.spec["conditions"]:
            native = ablation._generation_metadata(condition, ablation.GEOMETRY_MODES[0], {"bucket": condition["native_bucket"]}, self._contract(), Path("/control.png"))
            dynamic = ablation._generation_metadata(condition, ablation.GEOMETRY_MODES[1], {"bucket": condition["dynamic_768_bucket"]}, self._contract(), Path("/control.png"))
            self.assertEqual(native["seed"], dynamic["seed"])
            self.assertEqual(native["prompt"], dynamic["prompt"])

    def test_immutable_spec_hash_and_historical_artifacts_are_frozen(self):
        self.assertEqual(ablation._sha256(ablation.ABLATION_SPEC), ablation.ABLATION_SPEC_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "ablation.json"
            changed.write_text(ablation.ABLATION_SPEC.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ablation.load_ablation_spec(changed)
        ablation._validate_historical_artifacts()

    def test_mix_control_scale_and_turbo_locks_refuse_drift(self):
        for key, value, message in (("candidate", "mix-050", "mix-025"), ("control_scale", .5, "control scale 1.0")):
            bad = copy.deepcopy(self.spec); bad[key] = value
            with self.assertRaisesRegex(ValueError, message): ablation._validate_locks(bad)
        bad = copy.deepcopy(self.spec); bad["runtime"]["steps"] = 7
        with self.assertRaisesRegex(ValueError, "8-step CFG-0 mu=1.15"):
            ablation._validate_locks(bad)
        self.assertEqual((self.spec["runtime"]["steps"], self.spec["runtime"]["cfg"], self.spec["runtime"]["mu"]), (8, 0.0, 1.15))

    def test_native_and_dynamic_geometry_identity_is_explicit_and_ordered(self):
        expected = {
            "sculpture_humanart_14000000003803": ([1024, 1024], [768, 768]),
            "real_human_humanart_15000000000521": ([832, 1216], [640, 960]),
            "real_human_humanart_15000000000477": ([1472, 704], [1024, 576]),
            "sculpture_humanart_14000000000288": ([768, 1344], [576, 1024]),
            "real_human_humanart_17000000002207": ([1216, 832], [960, 640]),
        }
        self.assertEqual({row["stem"]: (row["native_bucket"], row["dynamic_768_bucket"]) for row in self.spec["conditions"]}, expected)
        for condition in self.spec["conditions"]:
            native = ablation._geometry_dict(tuple(condition["source_size"]), tuple(condition["native_bucket"]))
            dynamic = ablation._geometry_dict(tuple(condition["source_size"]), tuple(condition["dynamic_768_bucket"]))
            self.assertEqual(native["bucket"], condition["native_bucket"])
            self.assertEqual(dynamic["bucket"], condition["dynamic_768_bucket"])
            self.assertIn(tuple(dynamic["bucket"]), ablation.RESOLUTION_768_BUCKETS)

    def test_output_dimension_bucket_provenance(self):
        condition = self.spec["conditions"][2]
        geometry = ablation._geometry_dict(tuple(condition["source_size"]), tuple(condition["dynamic_768_bucket"]))
        metadata = ablation._generation_metadata(condition, ablation.GEOMETRY_MODES[1], geometry, self._contract(), Path("/authoritative/control.png"))
        self.assertEqual(metadata["geometry_mode"], "dynamic_768_bucket")
        self.assertEqual(metadata["output_dimensions"], condition["dynamic_768_bucket"])
        self.assertEqual(metadata["output_bucket"], condition["dynamic_768_bucket"])
        self.assertEqual(metadata["control_scale"], 1.0)

    def test_incomplete_output_refuses_overwrite(self):
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        geometries = {row["stem"]: {ablation.GEOMETRY_MODES[0]: {"bucket": row["native_bucket"]}, ablation.GEOMETRY_MODES[1]: {"bucket": row["dynamic_768_bucket"]}} for row in self.spec["conditions"]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(ablation._generation_status(output, self.spec, candidate, self._contract(), geometries), "missing")
            output.joinpath("generation_results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                ablation._generation_status(output, self.spec, candidate, self._contract(), geometries)

    def _contract(self):
        return {"final_val_spec_sha256": self.spec["final_val_spec_sha256"], "runtime": self.spec["runtime"], "checkpoint_interpolation": self.spec["checkpoint_interpolation"]}


if __name__ == "__main__":
    unittest.main()
