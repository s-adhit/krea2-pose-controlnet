import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "frozen_prompt_turbo.py"
SPEC = importlib.util.spec_from_file_location("frozen_prompt_turbo", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FrozenPromptTurboContractTest(unittest.TestCase):
    def test_prompt_injection_file_is_pinned_and_exactly_matches_final_stem_order(self):
        rows = benchmark.load_prompt_injection_rows()
        spec, _ = benchmark.final_val.load_final_spec(benchmark.final_val.FINAL_SPEC)
        self.assertEqual([row["stem"] for row in rows], spec["stems"])
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row["stem"] for row in rows}), 48)

    def test_prompt_injection_rejects_wrong_order_even_when_json_rows_are_otherwise_valid(self):
        rows = benchmark.load_prompt_injection_rows()
        swapped = [rows[1], rows[0], *rows[2:]]
        with mock.patch.object(benchmark, "_read_jsonl", return_value=swapped):
            with self.assertRaisesRegex(ValueError, "order"):
                benchmark.load_prompt_injection_rows()

    def test_prompt_injection_rejects_duplicate_stems(self):
        rows = benchmark.load_prompt_injection_rows()
        duplicate = [dict(row) for row in rows]
        duplicate[1]["stem"] = duplicate[0]["stem"]
        with mock.patch.object(benchmark, "_read_jsonl", return_value=duplicate):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                benchmark.load_prompt_injection_rows()

    def test_prompt_injection_rejects_any_prompt_file_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            drifted = Path(directory) / "prompt_injection_48.jsonl"
            drifted.write_text(benchmark.PROMPT_INJECTION_FILE.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                benchmark.load_prompt_injection_rows(drifted)

    def test_hero_file_is_pinned_to_one_six_prompt_pose(self):
        rows = benchmark.load_hero_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["stem"] for row in rows}, {benchmark.HERO_STEM})
        self.assertEqual(len({benchmark._hero_seed(row["hero_id"]) for row in rows}), 6)
        self.assertEqual(benchmark._hero_seed(rows[0]["hero_id"]), benchmark._hero_seed(rows[0]["hero_id"]))

    def test_generation_status_requires_every_recorded_artifact(self):
        candidate = {"label": "mix-025", "kind": "trainable_tensor_interpolation"}
        row = {"stem": "one", "prompt_id": "p1", "prompt": "prompt"}
        contract = {"turbo": {**benchmark.turbo_metadata(), "control_scale": 1.0},
                    "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None,
                    "checkpoint_interpolation": {"alpha": .25}, "prompt_file": {"sha256": "p"},
                    "sampling_seeds": {"one": 123}, "control_sha256": {"one": None}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(benchmark._generation_status(output, [row], candidate, contract, hero=False), "missing")
            artifact = output / "fixed_pose" / "one" / "mix-025.png"
            artifact.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2), "white").save(artifact)
            Image.new("RGB", (2, 2), "black").save(artifact.parent / "control.png")
            contract["control_sha256"]["one"] = benchmark._sha256(artifact.parent / "control.png")
            (artifact.parent / "metadata.json").write_text(json.dumps({
                "stem": "one", "prompt_id": "p1", "prompt": "prompt", "candidate": "mix-025",
                "geometry": benchmark.NATIVE_GEOMETRY, **contract["turbo"],
                "prompt_file_sha256": "p", "seed": 123,
                "control_sha256": contract["control_sha256"]["one"],
                "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None,
                "checkpoint_interpolation": {"alpha": .25},
            }))
            (output / "generation_results.json").write_text(json.dumps({
                "candidate": "mix-025", "prompt_file": {"sha256": "p"}, "prompt_mapping": [row],
                "generated_artifacts": {"one": ["mix-025.png"]},
            }))
            (output / "generation_results.json").write_text(json.dumps({
                "candidate": "mix-025", "prompt_file": {"sha256": "p"}, "prompt_mapping": [row],
                "generated_artifacts": {},
            }))
            with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                benchmark._generation_status(output, [row], candidate, contract, hero=False)


if __name__ == "__main__":
    unittest.main()
