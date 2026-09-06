import hashlib
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DECISION = REPOSITORY_ROOT / "docs/evaluation/release/final_release_v1.json"
RELEASE_DECISION_SHA256 = "9c79e714b7d61a6cbc83e0ca2ba45dde61a8124b0340c062d2462a1f57e52a2b"


class FinalReleaseDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = RELEASE_DECISION.read_bytes()
        cls.decision = json.loads(cls.raw)

    def test_exact_mix_025_interpolation_contract(self):
        candidate = self.decision["candidate"]
        interpolation = candidate["interpolation"]
        self.assertEqual(candidate["id"], "mix-025")
        self.assertEqual(candidate["kind"], "trainable_tensor_interpolation")
        self.assertEqual(interpolation["alpha"], 0.25)
        self.assertEqual(interpolation["formula"], "(1 - alpha) * parent-4000 + alpha * finish-control-a4300")
        self.assertEqual(interpolation["tensor_scope"], "state['model'] trainable control/LoRA tensors only")
        self.assertEqual(interpolation["compute_dtype"], "float32")

    def test_exact_endpoint_hashes(self):
        endpoints = {endpoint["id"]: endpoint for endpoint in self.decision["candidate"]["interpolation"]["endpoints"]}
        self.assertEqual(endpoints["parent-4000"]["sha256"], "0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd")
        self.assertEqual(endpoints["finish-control-a4300"]["sha256"], "17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff")

    def test_runtime_geometry_and_control_defaults(self):
        runtime = self.decision["runtime_defaults"]
        self.assertEqual((runtime["model"], runtime["steps"], runtime["cfg"], runtime["mu"]), ("Krea-2 Turbo", 8, 0.0, 1.15))
        self.assertFalse(runtime["mu_resolution_dependent"])
        self.assertEqual(runtime["geometry"]["default"], "native_aspect_preserving_cached_latent_bucket")
        self.assertEqual(runtime["geometry"]["optional_alternative"], "dynamic_768_bucket")
        self.assertEqual(runtime["control_scale"]["default"], 1.0)
        self.assertEqual(runtime["control_scale"]["optional_stronger_control_range"], [1.25, 1.5])

    def test_exact_style_lora_scope_and_defaults(self):
        scope = self.decision["style_lora_release_scope"]
        self.assertEqual(scope["maximum_active_adapters"], 1)
        self.assertFalse(scope["multi_style_lora_composition"])
        self.assertTrue(scope["pose_and_style_tensors_separate"])
        self.assertFalse(scope["permanent_merge"])
        self.assertEqual(scope["runtime_application"], "reversible")
        self.assertTrue(scope["style_strength_configurable"])
        self.assertEqual(scope["defaults"], {
            "darkbrush": {"default": 0.75, "useful_range": [0.5, 0.75]},
            "rainywindow": {"default": 0.5, "useful_range": [0.25, 0.75]},
            "retroanime": {"default": 0.5, "useful_range": [0.25, 0.5]},
            "realism": {"default": 0.25, "useful_range": [0.25, 0.75]},
        })

    def test_required_evidence_references_exist_and_match_available_hashes(self):
        required = {"final_val_spec", "final_val_results", "turbo_baseline_spec", "native_dynamic_spec", "native_dynamic_results", "control_scale_spec", "control_scale_results", "hard_pose_spec", "hard_pose_results", "hands_spec", "hands_results", "style_strength_spec", "style_strength_results"}
        references = {reference["id"]: reference for reference in self.decision["evidence_references"]}
        self.assertEqual(set(references), required)
        for reference in references.values():
            path = REPOSITORY_ROOT / reference["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), reference["sha256"])

    def test_immutable_json_hash_is_pinned(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), RELEASE_DECISION_SHA256)


if __name__ == "__main__":
    unittest.main()
