import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.freeze_final_val_benchmark import BenchmarkFreezeError, QUOTAS, REVIEW_FIELDS, freeze_benchmark


class FreezeFinalValBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool = self.root / "candidate_pool_96.jsonl"
        self.review = self.root / "candidate_review.csv"
        self.val = self.root / "val.jsonl"
        self.diagnostic = self.root / "diagnostic_val.jsonl"
        self.output = self.root / "final.jsonl"
        self.rows = []
        for source, count in QUOTAS.items():
            for index in range(count):
                stem = f"{source}_{index:02d}"
                self.rows.append({"stem": stem, "source": source, "text": f"caption {stem}", "selection_key": stem})
        self._write_jsonl(self.pool, self.rows)
        self._write_jsonl(self.val, [{"file_name": f"{row['stem']}.jpg", "text": row["text"]} for row in self.rows])
        self._write_jsonl(self.diagnostic, [{"file_name": "diagnostic_only.jpg", "text": "other"}])
        self._write_review()
        self.pool_hash = hashlib.sha256(self.pool.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _write_review(self, rows=None):
        rows = self.rows if rows is None else rows
        with self.review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for index, candidate in enumerate(rows):
                row = {field: "" for field in REVIEW_FIELDS}
                row.update({"keep": "yes", "difficulty": f"d{index}", "pose_type": "action",
                            "multi_person": "no", "notes": f"note {index}", **candidate})
                writer.writerow({field: row[field] for field in REVIEW_FIELDS})

    def _freeze(self):
        return freeze_benchmark(candidate_pool=self.pool, candidate_review=self.review, val_manifest=self.val,
                                diagnostic_manifest=self.diagnostic, output=self.output,
                                expected_pool_sha256=self.pool_hash)

    def test_writes_deterministic_selection_with_review_fields_and_quotas(self):
        first = self._freeze()
        first_bytes = self.output.read_bytes()
        second = self._freeze()
        self.assertEqual(len(first), 48)
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, self.output.read_bytes())
        self.assertEqual(Counter(row["source"] for row in first), QUOTAS)
        self.assertEqual(first[0]["difficulty"], "d0")
        self.assertEqual(first[0]["candidate_pool_sha256"], self.pool_hash)

    def test_rejects_hash_count_quota_duplicate_non_candidate_non_val_overlap_and_caption_errors(self):
        with self.assertRaisesRegex(BenchmarkFreezeError, "SHA256 mismatch"):
            freeze_benchmark(candidate_pool=self.pool, candidate_review=self.review, val_manifest=self.val,
                             diagnostic_manifest=self.diagnostic, output=self.output, expected_pool_sha256="0" * 64)
        self._write_review(self.rows[:-1])
        with self.assertRaisesRegex(BenchmarkFreezeError, "exactly 48"):
            self._freeze()
        invalid = list(self.rows); invalid[-1] = {**invalid[-1], "source": "coco"}
        self._write_jsonl(self.pool, invalid)
        self.pool_hash = hashlib.sha256(self.pool.read_bytes()).hexdigest()
        self._write_review(invalid)
        with self.assertRaisesRegex(BenchmarkFreezeError, "source quotas"):
            self._freeze()
        self._write_jsonl(self.pool, self.rows)
        self.pool_hash = hashlib.sha256(self.pool.read_bytes()).hexdigest()
        self._write_review(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(BenchmarkFreezeError, "duplicate stem"):
            self._freeze()
        invalid = list(self.rows); invalid[-1] = {**invalid[-1], "stem": "not_a_candidate", "text": "caption"}
        self._write_review(invalid)
        with self.assertRaisesRegex(BenchmarkFreezeError, "non-candidate"):
            self._freeze()
        self._write_jsonl(self.val, [{"file_name": f"{row['stem']}.jpg", "text": row["text"]} for row in self.rows[:-1]])
        self._write_review()
        with self.assertRaisesRegex(BenchmarkFreezeError, "not in val"):
            self._freeze()
        self._write_jsonl(self.val, [{"file_name": f"{row['stem']}.jpg", "text": row["text"]} for row in self.rows])
        self._write_jsonl(self.diagnostic, [{"file_name": f"{self.rows[0]['stem']}.jpg", "text": self.rows[0]["text"]}])
        with self.assertRaisesRegex(BenchmarkFreezeError, "overlap"):
            self._freeze()
        self._write_jsonl(self.diagnostic, [{"file_name": "diagnostic_only.jpg", "text": "other"}])
        bad = list(self.rows); bad[0] = {**bad[0], "text": "wrong caption"}
        self._write_review(bad)
        with self.assertRaisesRegex(BenchmarkFreezeError, "caption mismatch"):
            self._freeze()


if __name__ == "__main__":
    unittest.main()
