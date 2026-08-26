import json
import tempfile
import unittest
from pathlib import Path

import torch

import train
from pose_controlnet.data import collate
from pose_controlnet.text_conditioning import CachedTextConditioning, FORMAT_VERSION, METADATA_NAME, TextConditioningError


def _entry(stem: str, length: int = 2) -> dict:
    return {"stem": stem, "context": torch.ones(length, 12, 4, dtype=torch.bfloat16), "mask": torch.ones(length, dtype=torch.bool)}


class CachedTextConditioningTest(unittest.TestCase):
    def test_cached_loader_collate_and_seeded_unconditional_dropout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / METADATA_NAME).write_text(json.dumps({"format_version": FORMAT_VERSION, "complete": True,
                "context_dimensions": [12, 4], "expected_counts": {"train": 1, "val": 1, "diagnostic_val": 1},
                "total_samples": 3, "shard_samples": 64, "context_dtype": "bfloat16", "mask_dtype": "bool"}))
            (root / "train").mkdir()
            torch.save({"format_version": FORMAT_VERSION, "split": "train", "samples": [_entry("stem")]}, root / "train/train-00000.pt")
            torch.save({"format_version": FORMAT_VERSION, "context": torch.full((1, 12, 4), 3.0, dtype=torch.bfloat16), "mask": torch.ones(1, dtype=torch.bool)}, root / "unconditional.pt")
            cache = CachedTextConditioning(root, "train")
            item = cache.get("stem")
            batch = collate([{"latent": torch.ones(1, 2, 2), "control": torch.ones(1, 2, 2), "prompt": "caption", **item}])
            train.apply_cached_caption_dropout(batch, cache.unconditional, 1.0, 42, 0)
            self.assertEqual(batch["context"].dtype, torch.bfloat16)
            self.assertEqual(batch["text_mask"].dtype, torch.bool)
            self.assertEqual(batch["context"].shape, (1, 1, 12, 4))
            self.assertEqual(batch["context"][0, 0, 0, 0].item(), 3.0)

    def test_cached_loader_rejects_nonfinite_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / METADATA_NAME).write_text(json.dumps({"format_version": FORMAT_VERSION, "complete": True,
                "context_dimensions": [12, 4], "expected_counts": {}, "total_samples": 0, "shard_samples": 64,
                "context_dtype": "bfloat16", "mask_dtype": "bool"}))
            (root / "train").mkdir()
            bad = _entry("stem"); bad["context"][0, 0, 0] = float("nan")
            torch.save({"format_version": FORMAT_VERSION, "split": "train", "samples": [bad]}, root / "train/train-00000.pt")
            torch.save({"format_version": FORMAT_VERSION, "context": torch.ones(1, 12, 4, dtype=torch.bfloat16), "mask": torch.ones(1, dtype=torch.bool)}, root / "unconditional.pt")
            with self.assertRaises(TextConditioningError): CachedTextConditioning(root, "train")


if __name__ == "__main__": unittest.main()
