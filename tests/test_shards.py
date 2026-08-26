import tempfile
import unittest
from pathlib import Path

import torch

from prepare_shards import ShardError, validate_shard, validate_shard_file, write_shard_atomically


def _sample(stem: str = "sample") -> dict:
    return {
        "stem": stem,
        "file_name": f"{stem}.jpg",
        "text": "a posed person",
        "split": "train",
        "bucket": [16, 24],
        "source_size": [16, 24],
        "resized_size": [16, 24],
        "crop_box": [0, 0, 16, 24],
        "image_latent": torch.ones(16, 3, 2, dtype=torch.float32),
        "control_latent": torch.full((16, 3, 2), 2.0, dtype=torch.float32),
    }


class ShardFormatTest(unittest.TestCase):
    def test_atomic_write_is_loadable_and_float32(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train" / "train-00000.pt"
            write_shard_atomically(path, "train", [_sample()])
            self.assertEqual(validate_shard_file(path, expected_split="train", expected_stems=["sample"]), ["sample"])
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_rejects_nonfinite_and_mismatched_metadata(self) -> None:
        sample = _sample()
        sample["control_latent"] = torch.full((16, 3, 2), float("nan"), dtype=torch.float32)
        with self.assertRaisesRegex(ShardError, "nonfinite"):
            validate_shard({"format_version": 1, "split": "train", "samples": [sample]})
        sample = _sample()
        sample["file_name"] = "different.jpg"
        with self.assertRaisesRegex(ShardError, "file_name/stem mismatch"):
            validate_shard({"format_version": 1, "split": "train", "samples": [sample]})

    def test_rejects_duplicate_samples_and_wrong_dtype(self) -> None:
        duplicate = _sample()
        with self.assertRaisesRegex(ShardError, "Duplicate"):
            validate_shard({"format_version": 1, "split": "train", "samples": [_sample(), duplicate]})
        sample = _sample()
        sample["image_latent"] = sample["image_latent"].bfloat16()
        with self.assertRaisesRegex(ShardError, "float32"):
            validate_shard({"format_version": 1, "split": "train", "samples": [sample]})


if __name__ == "__main__":
    unittest.main()
