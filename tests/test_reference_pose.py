import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from pose_controlnet.reference_pose import (
    ReferencePoseError, build_coco_reference_records, parse_coco_stem,
    pck_person_from_source, renderer_joint_states, renderer_includes_person,
    transform_keypoints, write_reference_jsonl,
)


class ReferencePoseTest(unittest.TestCase):
    @staticmethod
    def joints(visible=()):
        return [[float(index), float(index), 2.0 if index in visible else 0.0] for index in range(17)]

    def test_visible_joint_with_only_invisible_neighbor_is_not_rendered(self):
        states, _, _ = renderer_joint_states(self.joints((15,)))
        self.assertTrue(states[15]["source_visible"])
        self.assertFalse(states[15]["rendered_in_control"])
        self.assertFalse(states[15]["pck_eligible"])

    def test_exact_left_ankle_case_is_not_pck_eligible(self):
        joints = self.joints((15,)); joints[15] = [224.7366, 217.9995, 2.0]
        states, unified, _ = renderer_joint_states(joints)
        self.assertEqual(states[15]["unified_index"], 13)
        self.assertEqual(unified[12][2], 0.0)
        self.assertFalse(states[15]["rendered_in_control"])
        self.assertFalse(states[15]["pck_eligible"])

    def test_visible_joint_in_rendered_limb_is_pck_eligible(self):
        states, _, _ = renderer_joint_states(self.joints((13, 15)))
        self.assertTrue(states[13]["pck_eligible"])
        self.assertTrue(states[15]["pck_eligible"])

    def test_synthesized_neck_changes_topology_but_is_not_pck_joint(self):
        states, unified, limbs = renderer_joint_states(self.joints((5, 6, 7)))
        self.assertGreater(unified[1][2], 0.0)
        self.assertGreater(limbs, 0)
        self.assertEqual(len(states), 17)
        self.assertNotIn(1, [state["unified_index"] for state in states])

    def test_renderer_qualification_uses_core_and_minimum_visible_limbs(self):
        included, details = renderer_includes_person(self.joints((5, 6, 7, 9, 11, 13)))
        self.assertTrue(included)
        self.assertGreaterEqual(details["visible_limb_count"], 5)

    def test_geometry_validation_checks_only_renderer_represented_joints(self):
        states, _, _ = renderer_joint_states(self.joints((15,)))
        self.assertFalse(any(state["rendered_in_control"] for state in states))
        # This is the eligibility predicate the raster gate applies; the visible
        # ankle has no rendered limb and must not reach a pixel-distance check.
        self.assertEqual([state for state in states if state["rendered_in_control"]], [])

    @unittest.skipUnless(Path("/lambda/nfs/adhit/krea2-pose/posebridge_hf/conditioning_images/shard_07/painting_humanart_10000000000838.png").is_file(), "GH200 diagnostic raster unavailable")
    def test_painting_humanart_corrected_geometry_validation_passes(self):
        spec = importlib.util.spec_from_file_location("reference_pose_gate", Path(__file__).resolve().parents[1] / "scripts" / "reference_pose_gate.py")
        module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
        sidecar = json.loads(Path("data/manifests/diagnostic_reference_pose.json").read_text())
        record = next(row for row in sidecar["records"] if row["stem"] == "painting_humanart_10000000000838")
        import torch
        shard = torch.load("/lambda/nfs/adhit/krea2-pose/posebridge_latents/diagnostic_val/diagnostic_val-00000.pt", map_location="cpu", weights_only=False)
        geometry = next(row for row in shard["samples"] if row["stem"] == record["stem"])
        result = module.validate_record(record, Path("/lambda/nfs/adhit/krea2-pose/posebridge_hf/conditioning_images/shard_07/painting_humanart_10000000000838.png"), geometry)
        self.assertEqual(result["status"], "PASS")
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
