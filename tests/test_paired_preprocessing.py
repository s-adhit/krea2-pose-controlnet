import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from pose_controlnet.dataset_index import DatasetIndex, DatasetIndexError
from pose_controlnet.paired_preprocessing import (
    REFERENCE_KREA_BUCKETS,
    PairedPreprocessingError,
    choose_bucket,
    inspect_resolved_samples,
    preprocess_pair,
    resize_center_crop_geometry,
)


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_pair(root: Path, stem: str, rgb_size: tuple[int, int], control_size: tuple[int, int] | None = None) -> None:
    control_size = control_size or rgb_size
    write_image(root / "images/shard_00" / f"{stem}.jpg", rgb_size, (255, 0, 0))
    write_image(root / "conditioning_images/shard_00" / f"{stem}.png", control_size, (0, 255, 0))


def resolved_record(root: Path, stem: str, rgb_size: tuple[int, int], control_size: tuple[int, int] | None = None):
    make_pair(root, stem, rgb_size, control_size)
    manifest = root / "train.jsonl"
    write_manifest(manifest, [{"file_name": f"{stem}.jpg", "text": "a pose"}])
    return DatasetIndex.discover(root).validate_manifests({"train": manifest}).records_by_split["train"][0]


class PairedPreprocessingTest(unittest.TestCase):
    def test_bucket_selection_square_portrait_and_landscape(self) -> None:
        self.assertEqual(choose_bucket((1000, 1000)), (1024, 1024))
        self.assertEqual(choose_bucket((800, 1200)), (832, 1216))
        self.assertEqual(choose_bucket((1200, 800)), (1216, 832))
        self.assertEqual(set(REFERENCE_KREA_BUCKETS), {
            (1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832),
            (768, 1344), (1344, 768), (704, 1472), (1472, 704),
        })

    def test_extreme_aspect_ratios_choose_outer_buckets(self) -> None:
        self.assertEqual(choose_bucket((100, 1000)), (704, 1472))
        self.assertEqual(choose_bucket((1000, 100)), (1472, 704))

    def test_reference_geometry_is_deterministic(self) -> None:
        geometry = resize_center_crop_geometry((2000, 1000), (1024, 1024))
        self.assertEqual(geometry.resized_size, (2048, 1024))
        self.assertEqual(geometry.crop_box, (512, 0, 1536, 1024))
        self.assertEqual(geometry, resize_center_crop_geometry((2000, 1000), (1024, 1024)))

    def test_identical_geometry_and_exact_output_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = resolved_record(Path(directory), "portrait", (1000, 1500))
            pair = preprocess_pair(record)
            self.assertEqual(pair.geometry.bucket, (832, 1216))
            self.assertEqual(pair.rgb.size, pair.control.size)
            self.assertEqual(pair.rgb.size, pair.geometry.bucket)
            self.assertEqual(pair.geometry.resized_size, (832, 1248))
            self.assertEqual(pair.geometry.crop_box, (0, 16, 832, 1232))

    def test_dimension_mismatch_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = resolved_record(Path(directory), "mismatch", (1000, 1500), (999, 1500))
            with self.assertRaisesRegex(PairedPreprocessingError, "dimensions disagree"):
                preprocess_pair(record)

    def test_missing_pair_is_rejected_by_shared_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_image(root / "images/shard_00/missing.jpg", (16, 16), (0, 0, 0))
            (root / "conditioning_images").mkdir()
            with self.assertRaisesRegex(DatasetIndexError, "lack a control counterpart"):
                DatasetIndex.discover(root)

    def test_malformed_resolved_image_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_pair(root, "broken", (32, 32))
            (root / "conditioning_images/shard_00/broken.png").write_bytes(b"not a PNG")
            manifest = root / "train.jsonl"
            write_manifest(manifest, [{"file_name": "broken.jpg", "text": "a pose"}])
            record = DatasetIndex.discover(root).validate_manifests(
                {"train": manifest}
            ).records_by_split["train"][0]
            with self.assertRaisesRegex(PairedPreprocessingError, "Unable to read resolved pair"):
                preprocess_pair(record)

    def test_inspection_reports_resolved_pair_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = resolved_record(Path(directory), "inspect", (1800, 1000))
            reports = inspect_resolved_samples([record], limit=1)
            self.assertEqual(reports, [{
                "stem": "inspect",
                "source_dimensions": [1800, 1000],
                "bucket": [1344, 768],
                "resize_dimensions": [1382, 768],
                "crop_box": [19, 0, 1363, 768],
                "output_dimensions": [1344, 768],
            }])


if __name__ == "__main__":
    unittest.main()
