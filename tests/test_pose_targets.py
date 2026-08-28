import json
import tempfile
import unittest
from pathlib import Path

from pose_controlnet.pose_targets import (
    OPENPOSE18_TO_COCO17, PoseTargetError, build_sidecar_records, common_body_mapping,
    load_sidecar, transform_person, write_sidecar,
)
from pose_controlnet.control_reconstruction import compare_control, render_record, summarize_reconstruction


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
        mapping = common_body_mapping("openpose18")
        self.assertEqual(len(mapping["common_joints"]), 17)
        self.assertNotIn(1, mapping["source_indices"])
        self.assertEqual(mapping["source_indices"], list(OPENPOSE18_TO_COCO17))


class PoseTargetSidecarTest(unittest.TestCase):
    @staticmethod
    def geometry():
        return {"coco_12_34": {"source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90]}}

    def test_missing_source_is_fail_closed(self):
        with self.assertRaisesRegex(PoseTargetError, "No authoritative source"):
            build_sidecar_records(self.geometry(), {})

    def test_coco_build_and_readonly_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); annotations = root / "person_keypoints.json"
            annotations.write_text(json.dumps({"images": [{"id": 12, "width": 100, "height": 50}], "annotations": [{"id": 34, "image_id": 12, "category_id": 1, "iscrowd": 0, "num_keypoints": 1, "bbox": [1, 2, 3, 4], "keypoints": [item for i in range(17) for item in (i + 1, i + 2, 2 if i == 0 else 0)]}]}))
            spec = {"coco": {"target_provenance": "original_annotation", "annotation_source": "synthetic-coco", "joint_schema": "coco17", "format": "coco_keypoints", "annotation_paths": [str(annotations)], "provenance_metadata": {"renderer": {"identifier": "historical-test", "sha256": "a" * 64}}}}
            records, summary = build_sidecar_records(self.geometry(), spec)
            self.assertEqual(records[0]["target_provenance"], "original_annotation")
            self.assertEqual(records[0]["people"][0]["keypoints_training"][0], [0.0, 0.0, 2.0])
            self.assertFalse(records[0]["people"][0]["reward_visible_mask"][0])
            output = root / "sidecar"; write_sidecar(records, output, build_metadata=summary)
            metadata, loaded = load_sidecar(output)
            self.assertEqual(metadata["record_count"], 1); self.assertEqual(loaded[0]["stem"], "coco_12_34")

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
