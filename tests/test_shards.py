import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from prepare_shards import (
    SHARD_FORMAT_VERSION,
    ShardError,
    prepare_shards,
    validate_shard,
    validate_shard_file,
    write_shard_atomically,
)
from scripts.verify_shards import verify_shards


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


class CompletionSemanticsTest(unittest.TestCase):
    def test_complete_metadata_with_zero_shards_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shards.json").write_text(
                """{
  "format_version": 1,
  "complete": true,
  "expected_counts": {"train": 16503, "val": 889, "diagnostic_val": 24},
  "shard_samples": 256,
  "total_samples": 17416
}
""",
                encoding="utf-8",
            )
            snapshot = SimpleNamespace(
                records_by_split={"train": (), "val": (), "diagnostic_val": ()}
            )
            with patch("scripts.verify_shards.validate_posebridge_snapshot", return_value=snapshot):
                with self.assertRaises(ShardError):
                    verify_shards(dataset_root=root, output_root=root)

    def test_interruption_leaves_explicit_incomplete_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = SimpleNamespace(
                split="train",
                stem="one",
                file_name="one.jpg",
                text="a posed person",
                rgb_path=root / "images" / "one.jpg",
            )
            snapshot = SimpleNamespace(
                records_by_split={"train": (record,), "val": (record,), "diagnostic_val": (record,)}
            )
            # This reproduces an interrupt before VAE loading/first shard write.
            with patch("prepare_shards.validate_posebridge_snapshot", return_value=snapshot), patch(
                "prepare_shards.load_krea_vae", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    prepare_shards(dataset_root=root, output_root=root / "latents", device="cpu")
            metadata = __import__("json").loads((root / "latents" / "shards.json").read_text())
            self.assertEqual(metadata["format_version"], SHARD_FORMAT_VERSION)
            self.assertFalse(metadata["complete"])
            self.assertFalse(list((root / "latents").glob("*/*.pt")))

    def test_restart_reuses_valid_final_shards_without_loading_vae(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_by_split = {}
            for split in ("train", "val", "diagnostic_val"):
                stem = f"{split}-one"
                record = SimpleNamespace(
                    split=split,
                    stem=stem,
                    file_name=f"{stem}.jpg",
                    text="a posed person",
                    rgb_path=root / "images" / f"{stem}.jpg",
                )
                records_by_split[split] = (record,)
                sample = _sample(stem)
                sample["split"] = split
                write_shard_atomically(root / "latents" / split / f"{split}-00000.pt", split, [sample])
            snapshot = SimpleNamespace(records_by_split=records_by_split)
            with patch("prepare_shards.validate_posebridge_snapshot", return_value=snapshot), patch(
                "prepare_shards.load_krea_vae"
            ) as load_vae:
                self.assertEqual(
                    prepare_shards(dataset_root=root, output_root=root / "latents", device="cpu"),
                    {"train": 1, "val": 1, "diagnostic_val": 1},
                )
            load_vae.assert_not_called()
            metadata = __import__("json").loads((root / "latents" / "shards.json").read_text())
            self.assertTrue(metadata["complete"])


if __name__ == "__main__":
    unittest.main()
