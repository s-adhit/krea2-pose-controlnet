import ast
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from pose_controlnet.overfit_capacity import validate_manifest
from pose_controlnet.reference_pose import (
    ReferencePoseError,
    build_exact_manifest_reference_records,
    load_exact_capacity_reference_sidecar,
    reference_people_from_sidecar,
    write_exact_manifest_reference_jsonl,
)
from scripts import build_overfit_capacity_reference_pose as builder
from scripts import evaluate_overfit_capacity as evaluator


class MixedCapacityReferencePoseTests(unittest.TestCase):
    source_sha = "dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f"
    compatible = ("overfit32-mixed-r64-mse", "overfit32-mixed-r64-mse-res768")

    @staticmethod
    def _stems():
        return (
            *(f"coco_{index}_{index + 100}" for index in range(6)),
            *(f"painting_humanart_{index}" for index in range(7)),
            *(f"real_human_humanart_{index}" for index in range(7)),
            *(f"sculpture_humanart_{index}" for index in range(6)),
            *(f"danbooru_anime_{index}" for index in range(6)),
        )

    def _manifest(self, root: Path):
        stems = self._stems(); path = root / "mixed.jsonl"
        path.write_text("".join(json.dumps({"file_name": f"{stem}.jpg", "text": "caption"}) + "\n" for stem in stems), encoding="utf-8")
        return path, stems

    @staticmethod
    def _record(stem: str):
        source = (
            "coco" if stem.startswith("coco_") else "humanart_painting" if stem.startswith("painting_")
            else "humanart_real_human" if stem.startswith("real_human_") else "humanart_sculpture"
        )
        sample_type = "crowd" if stem.startswith("coco_") else None
        record = {
            "stem": stem, "source": source, "pose_reward_available": True,
            "target_provenance": "original_annotation", "annotation_source": "reviewed-source",
            "source_size": [100, 50], "joint_schema": "coco17",
            "source_image_id": 3, "source_image_name": "source.jpg", "source_annotation_split": "train",
            "provenance_metadata": {"source": source}, "renderer": {"identifier": "historical"},
            "people": [{"person_id": 3, "annotation_id": 3, "bbox_source_xywh": [1, 2, 3, 4],
                        "keypoints_source": [[50.0, 25.0, 2.0]] + [[0.0, 0.0, 0.0] for _ in range(16)]}],
        }
        if sample_type is not None:
            record["sample_type"] = sample_type
        return record

    def _build(self, root: Path, records=None):
        manifest, stems = self._manifest(root)
        records = records if records is not None else [self._record(stem) for stem in stems if not stem.startswith("danbooru_")]
        return manifest, stems, build_exact_manifest_reference_records(
            manifest_path=manifest, authoritative_records=records,
            authoritative_metadata={"records_sha256": self.source_sha, "records_file": "records.jsonl", "schema_version": 3},
            compatible_experiments=self.compatible,
        )

    def test_exact_mixed_order_resolution_unavailable_and_source_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest, stems, (records, metadata) = self._build(root)
            output = root / "reference.jsonl"; written = write_exact_manifest_reference_jsonl(records, output, metadata=metadata)
            self.assertEqual([record["stem"] for record in records], list(stems))
            self.assertEqual(metadata["authoritative_source"]["records_sha256"], self.source_sha)
            self.assertEqual(written["records_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            loaded, rows = load_exact_capacity_reference_sidecar(output, experiment_name=self.compatible[1], expected_stems=stems)
            self.assertEqual([row["stem"] for row in rows], list(stems))
            self.assertEqual(loaded["coverage"], {"total": 32, "eligible_available": 26, "explicitly_unavailable": 6})
            unavailable = [row for row in rows if row["source"] == "danbooru"]
            self.assertEqual(len(unavailable), 6)
            self.assertTrue(all(row["status"] == "unavailable" and row["people"] is None for row in unavailable))
            self.assertTrue(all("keypoints_training" not in json.dumps(row) for row in rows))

    def test_missing_duplicate_and_mismatched_eligible_targets_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest, stems = self._manifest(root)
            records = [self._record(stem) for stem in stems if not stem.startswith("danbooru_")]
            with self.assertRaisesRegex(ReferencePoseError, "missing"):
                build_exact_manifest_reference_records(manifest_path=manifest, authoritative_records=records[1:], authoritative_metadata={"records_sha256": self.source_sha}, compatible_experiments=self.compatible)
            with self.assertRaisesRegex(ReferencePoseError, "duplicate"):
                build_exact_manifest_reference_records(manifest_path=manifest, authoritative_records=[*records, records[0]], authoritative_metadata={"records_sha256": self.source_sha}, compatible_experiments=self.compatible)
            records[0]["source"] = "humanart"
            with self.assertRaisesRegex(ReferencePoseError, "does not match"):
                build_exact_manifest_reference_records(manifest_path=manifest, authoritative_records=records, authoritative_metadata={"records_sha256": self.source_sha}, compatible_experiments=self.compatible)

    def test_source_space_transform_uses_native_generation_geometry_not_768_training_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, stems, (records, metadata) = self._build(root)
            output = root / "reference.jsonl"; write_exact_manifest_reference_jsonl(records, output, metadata=metadata)
            geometry = {stem: {"source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90]} for stem in stems}
            _, loaded = load_exact_capacity_reference_sidecar(output, experiment_name=self.compatible[1], expected_stems=stems, geometry_by_stem=geometry)
            available = next(row for row in loaded if row["source"] == "coco")
            person = reference_people_from_sidecar(available, source_size=(100, 50), resized_size=(200, 100), crop_box=(10, 5, 190, 95))[0]
            self.assertEqual(person["keypoints_bucket"][0], [90.0, 45.0, 2.0])
            generated = {"evaluation_provenance": {"native_geometry": {"samples": geometry}}}
            self.assertEqual(evaluator._native_generation_geometry(generated, stems), geometry)
            source = inspect.getsource(evaluator.score)
            self.assertIn("_native_generation_geometry(generated, stems)", source)
            self.assertNotIn("turbo_scoring_geometry", source)
            self.assertNotIn("AlternateResolutionDataset", Path(evaluator.__file__).read_text(encoding="utf-8"))

    def test_same_sidecar_declares_both_native_and_768_trained_mixed_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, stems, (records, metadata) = self._build(root)
            output = root / "reference.jsonl"; write_exact_manifest_reference_jsonl(records, output, metadata=metadata)
            for experiment in self.compatible:
                with self.subTest(experiment=experiment):
                    load_exact_capacity_reference_sidecar(output, experiment_name=experiment, expected_stems=stems)

    def test_builder_never_uses_detector_and_existing_coco_sidecar_stays_loadable(self):
        source = Path(builder.__file__).read_text(encoding="utf-8")
        self.assertNotIn("KeypointRCNN", source)
        self.assertNotIn("post500_evaluation", source)
        self.assertNotIn("Image.open", source)
        coco_stems = validate_manifest("overfit32-coco-r64-mse")
        coco = Path("data/manifests/overfit_capacity_reference_pose/overfit32-coco-r64-mse.jsonl")
        metadata, records = load_exact_capacity_reference_sidecar(coco, experiment_name="overfit32-coco-r64-mse", expected_stems=coco_stems)
        self.assertEqual(metadata["source"], "coco")
        self.assertEqual(len(records), 32)

    def test_score_only_is_evaluation_only(self):
        source = ast.get_source_segment(Path(evaluator.__file__).read_text(encoding="utf-8"), next(
            node for node in ast.walk(ast.parse(Path(evaluator.__file__).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "score"
        ))
        self.assertNotIn("sample_turbo_pose_image", source)
        self.assertNotIn("load_training_state", source)
        self.assertNotIn("backward(", source)
        self.assertNotIn("optimizer", source)
        self.assertIn("score_authoritative_pck", source)


if __name__ == "__main__":
    unittest.main()
