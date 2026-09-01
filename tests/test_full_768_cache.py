import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from unittest.mock import patch

from pose_controlnet.full_768_cache import (
    FULL_TRAIN_COUNT, POLICY, Full768CacheError, _cache_metadata, _identity,
    _load_complete_cache, _unavailable, _validate_sample_768,
)
from pose_controlnet.paired_preprocessing import resize_center_crop_geometry
from pose_controlnet.pose_targets import transform_person


class Full768CacheContractTests(unittest.TestCase):
    @staticmethod
    def sample(*, bucket=(768, 768)):
        geometry = resize_center_crop_geometry((1000, 700), bucket)
        h, w = bucket[1] // 8, bucket[0] // 8
        return {
            "stem": "coco_1_1", "bucket": list(bucket), "source_size": list(geometry.source_size),
            "resized_size": list(geometry.resized_size), "crop_box": list(geometry.crop_box),
            "image_latent": torch.ones((16, h, w), dtype=torch.float32),
            "control_latent": torch.ones((16, h, w), dtype=torch.float32),
        }

    def test_deterministic_paired_768_geometry_and_latent_shape(self):
        sample = self.sample()
        _validate_sample_768(sample, stem="coco_1_1")
        sample["crop_box"][0] += 1
        with self.assertRaisesRegex(Full768CacheError, "geometry"):
            _validate_sample_768(sample, stem="coco_1_1")

    def test_rejects_native_geometry_contamination(self):
        sample = self.sample(bucket=(1024, 1024))
        with self.assertRaisesRegex(Full768CacheError, "contamination"):
            _validate_sample_768(sample, stem="coco_1_1")

    def test_pose_projection_uses_same_768_geometry(self):
        geometry = {key: self.sample()[key] for key in ("source_size", "resized_size", "crop_box", "bucket")}
        person = {"annotation_id": 1, "bbox_xywh": [100, 100, 200, 300],
                  "keypoints_source": [[500, 350, 2]] + [[0, 0, 0]] * 16,
                  "_authoritative_source_size": geometry["source_size"]}
        projected = transform_person(person, **geometry)
        x, y, _ = projected["keypoints_training"][0]
        self.assertTrue(0 <= x < 768 and 0 <= y < 768)
        self.assertTrue(projected["joint_provenance"][0]["reward_joint_valid"])

    def test_danbooru_is_explicitly_unavailable(self):
        geometry = {key: self.sample()[key] for key in ("source_size", "resized_size", "crop_box", "bucket")}
        record = _unavailable("danbooru_1", "danbooru", geometry)
        self.assertFalse(record["pose_reward_available"])
        self.assertIsNone(record["people"])

    @staticmethod
    def manifest_rows():
        return [
            {"file_name": f"coco_{number}_1.jpg", "conditioning_image": f"conditioning_images/coco_{number}_1.png",
             "text": f"caption {number}"}
            for number in range(FULL_TRAIN_COUNT)
        ]

    @staticmethod
    def records_for(rows):
        return [SimpleNamespace(stem=Path(row["file_name"]).stem, file_name=row["file_name"], text=row["text"])
                for row in rows]

    @staticmethod
    def write_manifest(path, rows, *, sort_keys=False, compact=False):
        separators = (",", ":") if compact else None
        path.write_text("".join(json.dumps(row, sort_keys=sort_keys, separators=separators) + "\n" for row in rows),
                        encoding="utf-8")

    def test_identical_parsed_full_manifests_accept_different_raw_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, snapshot = root / "project.jsonl", root / "snapshot.jsonl"
            rows = self.manifest_rows()
            self.write_manifest(project, rows)
            self.write_manifest(snapshot, rows, sort_keys=True, compact=True)
            with patch("pose_controlnet.full_768_cache.load_krea_vae") as load_vae:
                identity = _identity(self.records_for(rows), project, snapshot)
            load_vae.assert_not_called()
            self.assertEqual(identity["sample_count"], FULL_TRAIN_COUNT)
            self.assertEqual(identity["manifest_records_sha256"], hashlib.sha256(
                (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest())
            self.assertNotEqual(identity["authoritative_train_manifest_raw_sha256"],
                                identity["snapshot_train_manifest_raw_sha256"])

    def test_manifest_identity_requires_exact_ordered_records_and_stems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, snapshot = root / "project.jsonl", root / "snapshot.jsonl"
            rows = self.manifest_rows()
            records = self.records_for(rows)
            self.write_manifest(project, rows)

            self.write_manifest(snapshot, list(reversed(rows)))
            with self.assertRaisesRegex(Full768CacheError, "ordered records"):
                _identity(records, project, snapshot)

            changed_stem = list(rows)
            changed_stem[0] = {**changed_stem[0], "file_name": "coco_changed_1.jpg"}
            self.write_manifest(snapshot, changed_stem)
            with self.assertRaisesRegex(Full768CacheError, "ordered records"):
                _identity(records, project, snapshot)

            changed_scientific_field = list(rows)
            changed_scientific_field[0] = {**changed_scientific_field[0],
                                           "conditioning_image": "conditioning_images/different.png"}
            self.write_manifest(snapshot, changed_scientific_field)
            with self.assertRaisesRegex(Full768CacheError, "ordered records"):
                _identity(records, project, snapshot)

    def test_manifest_identity_rejects_snapshot_resolved_order_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, snapshot = root / "project.jsonl", root / "snapshot.jsonl"
            rows = self.manifest_rows()
            self.write_manifest(project, rows)
            self.write_manifest(snapshot, rows)
            with self.assertRaisesRegex(Full768CacheError, "resolved train records"):
                _identity(list(reversed(self.records_for(rows))), project, snapshot)

    def test_partial_cache_and_mismatched_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {"sample_count": FULL_TRAIN_COUNT, "ordered_stems": [f"coco_{i}_1" for i in range(FULL_TRAIN_COUNT)],
                        "ordered_stems_sha256": "identity", "manifest_records_sha256": "manifest"}
            (root / "train_manifest_identity.json").write_text(json.dumps(identity))
            metadata = _cache_metadata(identity, dataset_root=root, shard_samples=256, complete=False)
            (root / "shards.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(Full768CacheError, "completion marker"):
                _load_complete_cache(root)
            metadata["complete"] = True
            metadata["ordered_stems_sha256"] = "wrong"
            (root / "shards.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(Full768CacheError, "identity metadata"):
                _load_complete_cache(root)

    def test_benchmark_fails_closed_before_gpu_work_when_verification_fails(self):
        from scripts import benchmark_production_trainer as benchmark
        argv = ["benchmark_production_trainer.py", "--raw-ckpt", "raw", "--dataset-root", "dataset",
                "--latent-root", "latents", "--text-conditioning-root", "text", "--pose-sidecar", "pose",
                "--output-json", "result.json", "--label", "test"]
        with patch.object(sys, "argv", argv), patch.object(benchmark, "verify_full_768_cache",
                                                            side_effect=Full768CacheError("unverified")):
            with self.assertRaisesRegex(Full768CacheError, "unverified"):
                benchmark.main()


if __name__ == "__main__":
    unittest.main()
