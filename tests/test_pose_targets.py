import json
import inspect
import tempfile
import unittest
from pathlib import Path

import pose_controlnet.pose_targets as pose_targets
from pose_controlnet.pose_targets import (
    PoseTargetError, build_sidecar_records, common_body_mapping, coverage_summary,
    load_sidecar, pose_reward_target_for_stem, transform_person, write_sidecar,
)
from pose_controlnet.control_reconstruction import compare_control, render_record, select_reconstruction_records, summarize_reconstruction


class PoseTargetGeometryTest(unittest.TestCase):
    @staticmethod
    def person(x=5.0, y=4.0, score=2.0, box=None):
        return {
            "person_id": 1, "annotation_id": 1, "bbox_xywh": box,
            "keypoints_source": [[x, y, score]] + [[0.0, 0.0, 0.0] for _ in range(16)],
        }

    def test_resize_and_crop_translation(self):
        result = transform_person(self.person(), source_size=(10, 10), resized_size=(20, 20), crop_box=(2, 3, 18, 17), bucket=(16, 14))
        self.assertEqual(result["keypoints_training"][0], [8.0, 5.0, 2.0])
        self.assertTrue(result["reward_visible_mask"][0])

    def test_aspect_ratio_scale_uses_persisted_resize_dimensions(self):
        result = transform_person(self.person(50, 25), source_size=(100, 50), resized_size=(200, 100), crop_box=(10, 5, 190, 95), bucket=(180, 90))
        self.assertEqual(result["keypoints_training"][0], [90.0, 45.0, 2.0])

    def test_crop_clips_and_masks_partially_visible_person(self):
        result = transform_person(self.person(1, 9, box=[-2, 4, 8, 8]), source_size=(10, 10), resized_size=(20, 20), crop_box=(5, 5, 15, 15), bucket=(10, 10))
        self.assertEqual(result["keypoints_training"][0], [0.0, 9.0, 2.0])
        self.assertFalse(result["keypoints_training_in_frame"][0])
        self.assertFalse(result["reward_visible_mask"][0])
        self.assertEqual(result["bbox_training_xywh"], [0.0, 3.0, 7.0, 6.0])

    def test_multiple_people_remain_distinct(self):
        first = transform_person(self.person(3, 3), source_size=(10, 10), resized_size=(10, 10), crop_box=(0, 0, 10, 10), bucket=(10, 10))
        second = transform_person(self.person(7, 7), source_size=(10, 10), resized_size=(10, 10), crop_box=(0, 0, 10, 10), bucket=(10, 10))
        self.assertNotEqual(first["keypoints_training"], second["keypoints_training"])

    def test_common_mapping_has_only_physical_body_joints(self):
        mapping = common_body_mapping("coco17")
        self.assertEqual(len(mapping["common_joints"]), 17)
        self.assertEqual(mapping["source_indices"], list(range(17)))


class PoseTargetSidecarTest(unittest.TestCase):
    @staticmethod
    def geometry():
        geometry = {"source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90]}
        return {"coco_12_34": geometry, "danbooru_1": geometry}

    def test_missing_source_is_fail_closed(self):
        with self.assertRaisesRegex(PoseTargetError, "No authoritative source"):
            build_sidecar_records(self.geometry(), {})

    @staticmethod
    def coco_spec(annotations):
        return {"pose_reward_available": True, "target_provenance": "original_annotation", "annotation_source": "synthetic-coco", "joint_schema": "coco17", "format": "coco_keypoints", "annotation_paths": [str(annotations)], "provenance_metadata": {"renderer": {"identifier": "historical-test", "sha256": "a" * 64}}}

    @staticmethod
    def unavailable_spec():
        return {"pose_reward_available": False, "target_provenance": "unavailable", "format": "unavailable"}

    def test_partial_coverage_allows_unavailable_danbooru_and_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); annotations = root / "person_keypoints.json"
            annotations.write_text(json.dumps({"images": [{"id": 12, "width": 100, "height": 50}], "annotations": [{"id": 34, "image_id": 12, "category_id": 1, "iscrowd": 0, "num_keypoints": 1, "bbox": [1, 2, 3, 4], "keypoints": [item for i in range(17) for item in (i + 1, i + 2, 2 if i == 0 else 0)]}]}))
            spec = {"coco": self.coco_spec(annotations), "danbooru": self.unavailable_spec()}
            records, summary = build_sidecar_records(self.geometry(), spec)
            by_stem = {record["stem"]: record for record in records}
            self.assertEqual(records[0]["target_provenance"], "original_annotation")
            self.assertTrue(by_stem["coco_12_34"]["pose_reward_available"])
            self.assertFalse(by_stem["danbooru_1"]["pose_reward_available"])
            self.assertEqual(by_stem["danbooru_1"]["target_provenance"], "unavailable")
            self.assertIsNone(by_stem["danbooru_1"]["people"])
            self.assertIsNotNone(pose_reward_target_for_stem(by_stem, "coco_12_34"))
            self.assertIsNone(pose_reward_target_for_stem(by_stem, "danbooru_1"))
            self.assertEqual(summary["coverage"]["total"], {"total": 2, "available": 1, "unavailable": 1, "available_percent": 50.0, "unavailable_percent": 50.0})
            output = root / "sidecar"; write_sidecar(records, output, build_metadata=summary)
            metadata, loaded = load_sidecar(output)
            self.assertEqual(metadata["record_count"], 2); self.assertEqual({row["stem"] for row in loaded}, {"coco_12_34", "danbooru_1"})

    def test_claimed_coco_corruption_or_geometry_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); annotations = root / "person_keypoints.json"
            malformed = {"images": [{"id": 12, "width": 100, "height": 50}], "annotations": [{"id": 34, "image_id": 12, "category_id": 1, "iscrowd": 0, "num_keypoints": 1, "keypoints": [1, 2, 2]}]}
            annotations.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(PoseTargetError, "malformed keypoints"):
                build_sidecar_records(self.geometry(), {"coco": self.coco_spec(annotations), "danbooru": self.unavailable_spec()})
            annotations.write_text(json.dumps({"images": [{"id": 12, "width": 99, "height": 50}], "annotations": [{"id": 34, "image_id": 12, "category_id": 1, "iscrowd": 0, "num_keypoints": 1, "keypoints": [item for i in range(17) for item in (1, 2, 2 if i == 0 else 0)]}]}))
            with self.assertRaisesRegex(PoseTargetError, "source size"):
                build_sidecar_records(self.geometry(), {"coco": self.coco_spec(annotations), "danbooru": self.unavailable_spec()})

    def test_humanart_adapter_preserves_grouping_and_fails_closed_on_missing_stem(self):
        geometry = {"painting_humanart_7": {"source_size": [10, 10], "resized_size": [10, 10], "crop_box": [0, 0, 10, 10], "bucket": [10, 10]}}
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp) / "humanart.jsonl"
            adapter.write_text(json.dumps({"stem": "painting_humanart_7", "source": "humanart_painting", "source_size": [10, 10], "people": [{"person_id": "a", "keypoints": [[2, 3, 1]] * 17}, {"person_id": "b", "bbox_xywh": [1, 1, 2, 2], "keypoints": [[4, 5, 1]] * 17}]}) + "\n")
            spec = {"humanart_painting": {"pose_reward_available": True, "target_provenance": "original_annotation", "annotation_source": "adapter", "joint_schema": "coco17", "format": "humanart_pose_adapter_jsonl", "adapter_path": str(adapter), "provenance_metadata": {"renderer": {"identifier": "test"}}}}
            records, _ = build_sidecar_records(geometry, spec)
            self.assertEqual([person["person_id"] for person in records[0]["people"]], ["a", "b"])
            adapter.write_text(json.dumps({"stem": "painting_humanart_7", "source_size": [11, 10], "people": []}) + "\n")
            with self.assertRaisesRegex(PoseTargetError, "source size"):
                build_sidecar_records(geometry, spec)

    def test_coverage_is_deterministic_and_no_detector_or_raster_fallback_exists(self):
        records = [
            {"source": "danbooru", "pose_reward_available": False},
            {"source": "coco", "pose_reward_available": True},
            {"source": "danbooru", "pose_reward_available": False},
        ]
        self.assertEqual(coverage_summary(records), coverage_summary(reversed(records)))
        source = inspect.getsource(pose_targets)
        self.assertNotIn("historical_dwpose_jsonl", source)
        self.assertNotIn("pseudolabel_path", source)
        self.assertNotIn("Image.open", source)

    def test_reconstruction_selection_skips_explicitly_unavailable_danbooru(self):
        selected = select_reconstruction_records([
            {"stem": "danbooru_1", "source": "danbooru", "pose_reward_available": False},
            {"stem": "coco_1_1", "source": "coco", "pose_reward_available": True},
        ], per_source=16)
        self.assertEqual(list(selected), ["coco"])
        self.assertEqual(selected["coco"][0]["stem"], "coco_1_1")

    def test_reconstruction_requires_verified_renderer_and_reports_metrics(self):
        record = {
            "stem": "coco_1_1", "bucket": [20, 20], "renderer": {"validated_historical_renderer": True, "topology": "openpose_body18", "line_rgb": [255, 255, 255], "line_width": 2},
            "people": [{"keypoints_training": [[5.0, 5.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 2.0], [15.0, 10.0, 2.0]] + [[0.0, 0.0, 0.0] for _ in range(10)]}],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control.png"; render_record(record).save(path)
            metrics, _, _, _ = compare_control(record, path)
            metrics["source"] = "coco"
            report = summarize_reconstruction([metrics], min_foreground_iou=1.0, max_mae=0.0)
            self.assertEqual(report["status"], "PASS")
            record["renderer"]["validated_historical_renderer"] = False
            with self.assertRaisesRegex(PoseTargetError, "has not been verified"):
                render_record(record)
