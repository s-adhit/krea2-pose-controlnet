import json
import tempfile
import unittest
from pathlib import Path

from pose_controlnet.reference_pose import ReferencePoseError, build_coco_reference_records, parse_coco_stem, transform_keypoints, write_reference_jsonl


class ReferencePoseTest(unittest.TestCase):
    def annotation_file(self, root: Path) -> Path:
        path = root / "person_keypoints_val2017.json"
        path.write_text(json.dumps({"images": [{"id": 12, "width": 100, "height": 50}], "annotations": [
            {"id": 34, "image_id": 12, "category_id": 1, "iscrowd": 0, "num_keypoints": 17, "bbox": [1, 2, 3, 4], "area": 12, "keypoints": [value for joint in range(17) for value in (joint + 1, joint + 2, 2)]},
            {"id": 35, "image_id": 12, "category_id": 1, "iscrowd": 0, "num_keypoints": 17, "bbox": [1, 2, 3, 4], "area": 12, "keypoints": [value for joint in range(17) for value in (joint + 2, joint + 3, 2)]},
        ]}))
        return path

    def sample(self, stem="coco_12_34"):
        return {"stem": stem, "source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90]}

    def test_stem_mapping_and_coco17_geometry(self):
        self.assertEqual(parse_coco_stem("coco_12_34"), (12, 34)); self.assertEqual(parse_coco_stem("coco_12_crowd"), (12, None))
        self.assertEqual(transform_keypoints([value for _ in range(17) for value in (5, 4, 2)], source_size=(10, 10), resized_size=(20, 20), crop_box=(2, 3, 18, 17))[0], [8.0, 5.0, 2.0])
        with self.assertRaises(ReferencePoseError): parse_coco_stem("coco_bad")

    def test_annotation_join_multi_person_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); annotations = self.annotation_file(root)
            single = build_coco_reference_records([self.sample()], [annotations])
            self.assertEqual(len(single), 1); self.assertEqual([person["person_id"] for person in single[0]["people"]], [34])
            self.assertEqual(single[0]["people"][0]["keypoints_bucket"][0], [-8.0, -1.0, 2.0])
            crowd = build_coco_reference_records([self.sample("coco_12_crowd")], [annotations])
            self.assertEqual([person["person_id"] for person in crowd[0]["people"]], [34, 35])
            output = root / "reference_pose.jsonl"; digest = write_reference_jsonl(crowd, output)
            self.assertEqual(len(digest), 64); self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_mismatched_source_dimensions_fail_loudly(self):
        with tempfile.TemporaryDirectory() as temp:
            annotations = self.annotation_file(Path(temp)); sample = self.sample(); sample["source_size"] = [99, 50]
            with self.assertRaisesRegex(ReferencePoseError, "does not match shard source_size"):
                build_coco_reference_records([sample], [annotations])


if __name__ == "__main__":
    unittest.main()
