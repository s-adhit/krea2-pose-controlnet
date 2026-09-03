import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from pose_controlnet.evaluation import make_evaluation_spec
from pose_controlnet.turbo_runtime import turbo_metadata
from scripts import final_val_turbo_benchmark as evaluator


class _Dataset:
    def __init__(self, stems):
        self.records = [("shard.pt", index, (4, 4), stem) for index, stem in enumerate(stems)]
        self.samples = {
            stem: {"stem": stem, "latent": torch.full((16, 4, 4), index + 1.0),
                   "control": torch.ones(16, 4, 4), "context": torch.ones(2, 1, 1),
                   "mask": torch.ones(2, dtype=torch.bool), "prompt": f"prompt {stem}"}
            for index, stem in enumerate(stems)
        }

    def __getitem__(self, index):
        return self.samples[self.records[index][3]]


class FinalValTurboBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.stems = [*(f"coco_{index:02d}" for index in range(16)),
                      *(f"painting_humanart_{index:02d}" for index in range(12)),
                      *(f"real_human_humanart_{index:02d}" for index in range(12)),
                      *(f"sculpture_humanart_{index:02d}" for index in range(8))]
        self.dataset = _Dataset(self.stems)
        self.spec = make_evaluation_spec(self.dataset, split="val", count=48, seed=420300,
                                         kind="final_val_turbo_fixed_pose", stems=self.stems)
        self.spec["turbo"] = {**turbo_metadata(), "control_scale": 1.0}
        self.spec["benchmark"] = {"name": "final_val_benchmark_48",
            "source_counts": {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8},
            "orientation_counts": {"landscape": 16, "near_square": 17, "portrait": 15}}

    def test_frozen_spec_validation_and_cached_identity_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.json"; path.write_text(json.dumps(self.spec))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded, observed = evaluator.load_final_spec(path, expected_sha256=digest)
            self.assertEqual(loaded, self.spec); self.assertEqual(observed, digest)
            evaluator.validate_cached_contract(self.dataset, loaded)
            changed = copy.deepcopy(loaded); changed["per_stem_seeds"][self.stems[0]]["sampling"] += 1
            with self.assertRaisesRegex(ValueError, "cache identity/seed"):
                evaluator.validate_cached_contract(self.dataset, changed)

    def test_rejects_nonfinal_contract_and_unselected_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.json"; bad = copy.deepcopy(self.spec); bad["turbo"]["steps"] = 7
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "locked 8-step"):
                evaluator.load_final_spec(path, expected_sha256=None)
        with self.assertRaisesRegex(ValueError, "supports only"):
            evaluator.candidate_checkpoint("turbo-base")

    def test_final_sidecar_requires_exact_frozen_order_and_non_diagnostic_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            records = [{"stem": stem, "source": "coco" if stem.startswith("coco_") else "humanart", "status": "unavailable"}
                       for stem in self.stems]
            path.write_text(json.dumps({"records": records}))
            loaded, _ = evaluator._load_final_sidecar(path, self.stems)
            self.assertEqual(loaded["records"], records)
            records.reverse(); path.write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(ValueError, "in order"):
                evaluator._load_final_sidecar(path, self.stems)

    def test_generation_artifacts_require_complete_matching_final_contract(self):
        candidate = {"label": "parent-4000", "step": 4000}
        stems = ["first", "second"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); digest = "frozen-digest"
            for stem in stems:
                directory = root / "fixed_pose" / stem; directory.mkdir(parents=True)
                Image.new("RGB", (2, 2), "white").save(directory / "step_004000.png")
                metadata = {"stem": stem, "candidate": candidate["label"], "checkpoint_step": 4000,
                            "final_spec_sha256": digest, "control_scale": 1.0, **turbo_metadata()}
                (directory / "metadata.json").write_text(json.dumps(metadata))
            (root / "generation_results.json").write_text(json.dumps({
                "stems": stems, "candidate": candidate["label"], "final_spec_sha256": digest,
                "turbo": {**turbo_metadata(), "control_scale": 1.0},
                "generated_steps": {stem: [4000] for stem in stems},
            }))
            self.assertEqual(evaluator._generation_status(root, stems, candidate, digest), "complete")
            bad = json.loads((root / "fixed_pose" / "second" / "metadata.json").read_text())
            bad["checkpoint_step"] = 4300
            (root / "fixed_pose" / "second" / "metadata.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "contract-inconsistent"):
                evaluator._generation_status(root, stems, candidate, digest)


if __name__ == "__main__":
    unittest.main()
