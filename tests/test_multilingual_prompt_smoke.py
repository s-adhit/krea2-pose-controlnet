import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "multilingual_prompt_smoke.py"
SPEC = importlib.util.spec_from_file_location("multilingual_prompt_smoke", MODULE_PATH)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class MultilingualPromptSmokeContractTest(unittest.TestCase):
    def test_frozen_hash_order_and_exact_utf8_prompts(self):
        rows = smoke.load_smoke_rows()
        self.assertEqual(smoke._sha256(smoke.SMOKE_FILE), smoke.SMOKE_SHA256)
        self.assertEqual([row["language"] for row in rows], ["en", "zh", "te"])
        self.assertEqual([row["prompt"] for row in rows], [smoke.PROMPTS[language] for language in smoke.LANGUAGES])
        self.assertEqual(rows[2]["prompt"], "ఒక వయోజన మహిళ సరళమైన క్రీమ్ రంగు దుస్తులు ధరించి, నిశ్శబ్దమైన బొటానికల్ ప్రాంగణంలో ఉంది, మృదువైన మేఘావృత దినకాంతి, సహజమైన వాస్తవిక పదార్థాల స్పర్శ.")
        self.assertEqual({row["stem"] for row in rows}, {smoke.STEM})

    def test_immutable_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "multilingual.jsonl"
            changed.write_text(smoke.SMOKE_FILE.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                smoke.load_smoke_rows(changed)

    def test_v2_binds_all_languages_to_identical_seed_control_candidate_and_runtime(self):
        rows = smoke.load_smoke_rows()
        contract = {
            "sampling_seed": 8675987726486463627,
            "control_sha256": "a" * 64,
            "bucket": [1024, 1024],
            "final_val_spec_sha256": "b" * 64,
            "turbo": smoke.legacy.guide.TURBO,
            "candidate_kind": "trainable_tensor_interpolation",
            "checkpoint_step": None,
            "checkpoint_interpolation": {"alpha": 0.25},
        }
        with smoke._v2_contract():
            metadata = [smoke.legacy._metadata(row, contract, Path("/authoritative/control.png")) for row in rows]
        for key, expected in (("seed", {contract["sampling_seed"]}), ("control_sha256", {contract["control_sha256"]}), ("candidate", {"mix-025"}), ("geometry", {smoke.legacy.guide.NATIVE_GEOMETRY})):
            self.assertEqual({item[key] for item in metadata}, expected)
        self.assertTrue(all(item["steps"] == 8 and item["cfg"] == 0.0 and item["mu"] == 1.15 and item["control_scale"] == 1.0 for item in metadata))

    def test_output_completeness_rejects_partial_or_wrong_language_records(self):
        rows = smoke.load_smoke_rows()
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        contract = {"turbo": smoke.legacy.guide.TURBO, "sampling_seed": 1, "bucket": [64, 64], "control_sha256": "unused",
                    "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None, "checkpoint_interpolation": {"alpha": 0.25}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(smoke._generation_status(output, rows, candidate, contract), "missing")
            output.joinpath("generation_results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                smoke._generation_status(output, rows, candidate, contract)
        partial = {"per_generation": [{"language": "en"}, {"language": "zh"}]}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            smoke._score_records(partial, rows)
        wrong_order = {"per_generation": [{"language": language, "stem": smoke.STEM, "prompt": smoke.PROMPTS[language]} for language in ("en", "te", "zh")]}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            smoke._score_records(wrong_order, rows)

    def test_legacy_en_zh_contract_remains_unchanged(self):
        legacy = smoke.legacy
        rows = legacy.load_smoke_rows()
        self.assertEqual(legacy._sha256(legacy.SMOKE_FILE), "c782d6fecff1bc6393f9175a52cb9b66f11185dcf0a3a3c8cccf1ab3a095769e")
        self.assertEqual([row["language"] for row in rows], ["en", "zh"])
        self.assertEqual(json.loads(legacy.SMOKE_FILE.read_text(encoding="utf-8").splitlines()[1])["prompt"], smoke.PROMPTS["zh"])


if __name__ == "__main__":
    unittest.main()
