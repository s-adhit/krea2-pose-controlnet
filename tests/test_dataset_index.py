import json
from pathlib import Path
import tempfile
import unittest

from pose_controlnet.dataset_index import DatasetIndex, DatasetIndexError


def write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_pair(root: Path, stem: str, shard: str = "nested/shard_00") -> None:
    write_file(root / "images" / shard / f"{stem}.jpg")
    write_file(root / "conditioning_images" / shard / f"{stem}.png")


class DatasetIndexTest(unittest.TestCase):
    def test_recursive_discovery_resolves_physical_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_pair(root, "one", "deep/rgb/a")
            make_pair(root, "two", "shard_08/more")

            index = DatasetIndex.discover(root)

            self.assertEqual(set(index.rgb_by_stem), {"one", "two"})
            rgb, control = index.resolve("one.jpg")
            self.assertEqual(rgb, (root / "images/deep/rgb/a/one.jpg").resolve())
            self.assertEqual(control, (root / "conditioning_images/deep/rgb/a/one.png").resolve())

    def test_duplicate_stem_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_pair(root, "same")
            write_file(root / "images" / "other" / "same.jpg")

            with self.assertRaisesRegex(DatasetIndexError, "Duplicate RGB JPG stem"):
                DatasetIndex.discover(root)

    def test_missing_counterpart_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_file(root / "images" / "shard_00" / "unpaired.jpg")
            (root / "conditioning_images").mkdir()

            with self.assertRaisesRegex(DatasetIndexError, "lack a control counterpart"):
                DatasetIndex.discover(root)

    def test_manifest_resolution_requires_bare_filename_and_nonempty_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_pair(root, "valid")
            index = DatasetIndex.discover(root)
            manifest = root / "train.jsonl"
            write_manifest(manifest, [{"file_name": "valid.jpg", "text": "a person"}])

            result = index.validate_manifests(
                {"train": manifest}, expected_counts={"train": 1}, expected_total=1
            )

            record = result.records_by_split["train"][0]
            self.assertEqual(record.stem, "valid")
            self.assertEqual(record.rgb_path.name, "valid.jpg")
            write_manifest(manifest, [{"file_name": "nested/valid.jpg", "text": "a person"}])
            with self.assertRaisesRegex(DatasetIndexError, "must be bare"):
                index.validate_manifests({"train": manifest})
            write_manifest(manifest, [{"file_name": "valid.jpg", "text": "  "}])
            with self.assertRaisesRegex(DatasetIndexError, "empty caption"):
                index.validate_manifests({"train": manifest})

    def test_split_disjointness_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_pair(root, "shared")
            index = DatasetIndex.discover(root)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_manifest(train, [{"file_name": "shared.jpg", "text": "caption"}])
            write_manifest(val, [{"file_name": "shared.jpg", "text": "caption"}])

            with self.assertRaisesRegex(DatasetIndexError, "appears in both"):
                index.validate_manifests({"train": train, "val": val})
