import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.create_final_val_benchmark_spec import (
    FINAL_VAL_SEED, FinalValBenchmarkSpecError, build_final_val_benchmark_spec, write_immutable_spec,
)


class _Dataset:
    def __init__(self, stems):
        self.records = [("absolute/shard.pt", index, (4, 4), stem) for index, stem in enumerate(stems)]
        self.samples = {
            stem: {"stem": stem, "latent": torch.full((16, 4, 4), index + 1.0), "control": torch.ones(16, 4, 4),
                   "context": torch.ones(2, 1, 1), "mask": torch.ones(2, dtype=torch.bool)}
            for index, stem in enumerate(stems)
        }

    def __getitem__(self, index):
        return self.samples[self.records[index][3]]


class FinalValBenchmarkSpecTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.stems = [*(f"coco_{index:02d}" for index in range(16)), *(f"painting_{index:02d}" for index in range(12)),
                      *(f"real_human_{index:02d}" for index in range(12)), *(f"sculpture_{index:02d}" for index in range(8))]
        self.selection = self.root / "final.jsonl"; self.pool = self.root / "pool.jsonl"
        pool_rows = [{"stem": stem} for stem in self.stems]
        self.pool.write_text("\n".join(json.dumps(row) for row in pool_rows) + "\n")
        digest = hashlib.sha256(self.pool.read_bytes()).hexdigest()
        orientations = ("landscape", "near_square", "portrait")
        rows = [{"stem": stem, "source": stem.rsplit("_", 1)[0], "orientation": orientations[index % 3],
                 "candidate_pool_sha256": digest} for index, stem in enumerate(self.stems)]
        self.selection.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.val, self.diagnostic = self.root / "val.jsonl", self.root / "diagnostic.jsonl"
        self.val.write_text("\n".join(json.dumps({"file_name": f"{stem}.jpg"}) for stem in self.stems) + "\n")
        self.diagnostic.write_text(json.dumps({"file_name": "diagnostic.jpg"}) + "\n")

    def tearDown(self): self.temp.cleanup()

    def test_builds_deterministic_cache_identity_spec_without_absolute_paths(self):
        dataset = _Dataset(self.stems)
        first = build_final_val_benchmark_spec(dataset, frozen_selection=self.selection, val_manifest=self.val,
                                               diagnostic_manifest=self.diagnostic, candidate_pool=self.pool)
        second = build_final_val_benchmark_spec(dataset, frozen_selection=self.selection, val_manifest=self.val,
                                                diagnostic_manifest=self.diagnostic, candidate_pool=self.pool)
        self.assertEqual(first, second); self.assertEqual(first["seed"], FINAL_VAL_SEED)
        self.assertEqual(first["stems"], self.stems); self.assertEqual(first["benchmark"]["source_counts"],
                                                                         {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8})
        self.assertEqual(first["turbo"]["steps"], 8); self.assertEqual(first["turbo"]["cfg"], 0.0)
        self.assertNotIn("absolute/shard.pt", json.dumps(first)); self.assertNotIn(str(self.root), json.dumps(first))
        self.assertEqual(set(first["benchmark"]["provenance"]), {"final_val_benchmark_48", "val_manifest", "diagnostic_val_manifest", "candidate_pool_96"})
        destination = self.root / "spec.json"
        self.assertEqual(write_immutable_spec(destination, first), destination)
        self.assertEqual(write_immutable_spec(destination, second), destination)
        changed = dict(first); changed["seed"] += 1
        with self.assertRaisesRegex(FinalValBenchmarkSpecError, "conflicts"):
            write_immutable_spec(destination, changed)

    def test_rejects_selection_that_overlaps_diagnostic_manifest(self):
        self.diagnostic.write_text(json.dumps({"file_name": f"{self.stems[0]}.jpg"}) + "\n")
        with self.assertRaisesRegex(FinalValBenchmarkSpecError, "overlaps diagnostic"):
            build_final_val_benchmark_spec(_Dataset(self.stems), frozen_selection=self.selection, val_manifest=self.val,
                                           diagnostic_manifest=self.diagnostic, candidate_pool=self.pool)


if __name__ == "__main__":
    unittest.main()
