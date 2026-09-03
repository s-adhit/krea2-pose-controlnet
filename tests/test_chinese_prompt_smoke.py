import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "chinese_prompt_smoke.py"
SPEC = importlib.util.spec_from_file_location("chinese_prompt_smoke", MODULE_PATH)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class ChinesePromptSmokeContractTest(unittest.TestCase):
    def test_frozen_utf8_file_hash_and_exact_rows_are_pinned(self):
        rows = smoke.load_smoke_rows()
        self.assertEqual([row["language"] for row in rows], ["en", "zh"])
        self.assertEqual({row["stem"] for row in rows}, {smoke.STEM})
        self.assertEqual(smoke._sha256(smoke.SMOKE_FILE), smoke.SMOKE_SHA256)

    def test_hash_drift_and_wrong_languages_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "smoke.jsonl"
            changed.write_text(smoke.SMOKE_FILE.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                smoke.load_smoke_rows(changed)
        rows = smoke.load_smoke_rows()
        wrong = [dict(row) for row in rows]
        wrong[1]["language"] = "fr"
        with mock.patch.object(smoke, "_sha256", return_value=smoke.SMOKE_SHA256), mock.patch.object(Path, "read_text", return_value="\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in wrong)):
            with self.assertRaisesRegex(ValueError, "ordered en and zh"):
                smoke.load_smoke_rows(smoke.SMOKE_FILE)

    def test_wrong_stem_fails_closed(self):
        rows = smoke.load_smoke_rows()
        wrong = [dict(row) for row in rows]
        wrong[0]["stem"] = "wrong"
        with mock.patch.object(smoke, "_sha256", return_value=smoke.SMOKE_SHA256), mock.patch.object(Path, "read_text", return_value="\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in wrong)):
            with self.assertRaisesRegex(ValueError, "frozen stem"):
                smoke.load_smoke_rows(smoke.SMOKE_FILE)

    def test_incomplete_generation_refuses_overwrite(self):
        rows = smoke.load_smoke_rows()
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        contract = {"turbo": smoke.guide.TURBO, "sampling_seed": 1, "bucket": [64, 64], "control_sha256": "unused",
                    "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None, "checkpoint_interpolation": {"alpha": .25}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(smoke._generation_status(output, rows, candidate, contract), "missing")
            output.joinpath("generation_results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                smoke._generation_status(output, rows, candidate, contract)

    def test_score_records_require_exact_language_order(self):
        rows = smoke.load_smoke_rows()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            smoke._score_records({"per_generation": [{"language": "en"}]}, rows)

    def test_metadata_binds_both_languages_to_identical_seed_and_control_contract(self):
        rows = smoke.load_smoke_rows()
        contract = {"sampling_seed": 420300, "control_sha256": "a" * 64, "bucket": [768, 512],
                    "final_val_spec_sha256": "b" * 64, "turbo": smoke.guide.TURBO,
                    "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None,
                    "checkpoint_interpolation": {"alpha": .25}}
        metadata = [smoke._metadata(row, contract, Path("/authoritative/control.png")) for row in rows]
        self.assertEqual({item["seed"] for item in metadata}, {420300})
        self.assertEqual({item["control_sha256"] for item in metadata}, {"a" * 64})
        self.assertEqual({item["candidate"] for item in metadata}, {"mix-025"})
        self.assertTrue(all(item["geometry"] == smoke.guide.NATIVE_GEOMETRY for item in metadata))
        self.assertTrue(all(item["steps"] == 8 and item["cfg"] == 0.0 and item["mu"] == 1.15 and item["control_scale"] == 1.0 for item in metadata))


if __name__ == "__main__":
    unittest.main()
