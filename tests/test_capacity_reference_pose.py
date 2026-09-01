import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pose_controlnet.reference_pose import (
    ReferencePoseError, build_exact_coco_capacity_reference_sidecar,
    load_exact_capacity_reference_sidecar, resolve_exact_capacity_latent_samples,
)
from scripts import evaluate_overfit_capacity as evaluator


class CapacityReferencePoseTests(unittest.TestCase):
    experiment = "overfit32-coco-r64-mse"

    def _manifest(self, root: Path, stems: list[str]) -> Path:
        path = root / "overfit32-coco-r64-mse.jsonl"
        path.write_text("".join(json.dumps({"file_name": f"{stem}.jpg", "text": "exact caption"}) + "\n" for stem in stems), encoding="utf-8")
        return path

    @staticmethod
    def _sample(stem: str) -> dict:
        image_id, annotation_id = (int(value) for value in stem.split("_")[1:])
        return {"stem": stem, "file_name": f"{stem}.jpg", "text": "exact caption",
                "source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90],
                "image_latent": torch.ones(16, 11, 22), "control_latent": torch.ones(16, 11, 22),
                "_fixture_image_id": image_id, "_fixture_annotation_id": annotation_id}

    def _shard(self, root: Path, name: str, samples: list[dict], *, format_version: int = 1) -> Path:
        path = root / name
        torch.save({"format_version": format_version, "split": "train", "samples": samples}, path)
        return path

    def _annotations(self, root: Path, stems: list[str]) -> Path:
        images, annotations = [], []
        for stem in stems:
            image_id, annotation_id = (int(value) for value in stem.split("_")[1:])
            images.append({"id": image_id, "width": 100, "height": 50})
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                                "num_keypoints": 17, "bbox": [1, 2, 30, 40], "area": 1200,
                                "keypoints": [value for joint in range(17) for value in (joint + 1, joint + 2, 2)]})
        path = root / "person_keypoints_train2017.json"
        path.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
        return path

    def _fixture(self, root: Path) -> tuple[list[str], Path, Path, Path]:
        stems = [f"coco_{1000 + index}_{2000 + index}" for index in range(32)]
        manifest = self._manifest(root, stems); latent_root = root / "latents"; latent_root.mkdir()
        self._shard(latent_root, "train-00000.pt", [self._sample(stem) for stem in stems[:16]])
        self._shard(latent_root, "train-00001.pt", [self._sample(stem) for stem in stems[16:]])
        return stems, manifest, latent_root, self._annotations(root, stems)

    def test_exact_32_manifest_resolves_only_direct_train_shards_and_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems, manifest, latent_root, annotations = self._fixture(root)
            # These must never be globbed: they duplicate a requested stem but are
            # not direct train shards under the explicit root.
            ignored = latent_root / "text_conditioning_v1_backup"; ignored.mkdir()
            self._shard(ignored, "train-ignored.pt", [self._sample(stems[0])])
            checkpoint = latent_root / "checkpoints"; checkpoint.mkdir()
            self._shard(checkpoint, "train-checkpoint.pt", [self._sample(stems[1])])
            resolved, samples, shards = resolve_exact_capacity_latent_samples(
                experiment_name=self.experiment, latent_root=latent_root, manifest_path=manifest,
            )
            self.assertEqual(resolved, tuple(stems)); self.assertEqual(len(samples), 32); self.assertEqual(len(shards), 2)
            self.assertEqual(samples[0]["crop_box"], [10, 5, 190, 95])
            output = root / "capacity_reference.jsonl"
            metadata = build_exact_coco_capacity_reference_sidecar(
                experiment_name=self.experiment, latent_root=latent_root, annotation_paths=[annotations], output=output, manifest_path=manifest,
            )
            self.assertEqual(metadata["record_count"], 32); self.assertEqual(metadata["output_record_count"], 32)
            loaded_metadata, records = load_exact_capacity_reference_sidecar(
                output, experiment_name=self.experiment, expected_stems=stems,
                geometry_by_stem={stem: {"source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90]} for stem in stems},
            )
            self.assertEqual(loaded_metadata["records_sha256"], metadata["records_sha256"])
            self.assertEqual({record["stem"] for record in records}, set(stems)); self.assertEqual(records[0]["geometry"]["crop_box"], [10, 5, 190, 95])

    def test_missing_duplicate_and_malformed_requested_shards_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems, manifest, latent_root, _ = self._fixture(root)
            (latent_root / "train-00001.pt").unlink()
            with self.assertRaisesRegex(ReferencePoseError, "missing"):
                resolve_exact_capacity_latent_samples(experiment_name=self.experiment, latent_root=latent_root, manifest_path=manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems, manifest, latent_root, _ = self._fixture(root)
            self._shard(latent_root, "train-00002.pt", [self._sample(stems[0])])
            with self.assertRaisesRegex(ReferencePoseError, "Ambiguous requested stem"):
                resolve_exact_capacity_latent_samples(experiment_name=self.experiment, latent_root=latent_root, manifest_path=manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems, manifest, latent_root, _ = self._fixture(root)
            self._shard(latent_root, "train-00002.pt", [], format_version=99)
            with self.assertRaisesRegex(ReferencePoseError, "Invalid verified v1"):
                resolve_exact_capacity_latent_samples(experiment_name=self.experiment, latent_root=latent_root, manifest_path=manifest)

    def test_exact_manifest_and_official_annotations_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems = ["coco_1_1"] * 32; manifest = self._manifest(root, stems)
            latent_root = root / "latents"; latent_root.mkdir()
            with self.assertRaisesRegex(ValueError, "exactly 32 unique"):
                resolve_exact_capacity_latent_samples(experiment_name=self.experiment, latent_root=latent_root, manifest_path=manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems, manifest, latent_root, _ = self._fixture(root)
            with self.assertRaisesRegex(ReferencePoseError, "require official COCO"):
                build_exact_coco_capacity_reference_sidecar(
                    experiment_name=self.experiment, latent_root=latent_root, annotation_paths=[root / "not-official.json"],
                    output=root / "out.jsonl", manifest_path=manifest,
                )

    def test_explicit_sidecar_rejects_old_or_incomplete_coverage_and_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stems, manifest, latent_root, annotations = self._fixture(root); output = root / "out.jsonl"
            build_exact_coco_capacity_reference_sidecar(experiment_name=self.experiment, latent_root=latent_root, annotation_paths=[annotations], output=output, manifest_path=manifest)
            exact_geometry = {stem: {"source_size": [100, 50], "resized_size": [200, 100], "crop_box": [10, 5, 190, 95], "bucket": [180, 90]} for stem in stems}
            # Exact four-field geometry must reconcile; all fields remain part
            # of the immutable sidecar contract.
            load_exact_capacity_reference_sidecar(output, experiment_name=self.experiment, expected_stems=stems,
                                                  geometry_by_stem=exact_geometry)
            with self.assertRaisesRegex(ReferencePoseError, "exactly cover"):
                load_exact_capacity_reference_sidecar(output, experiment_name=self.experiment, expected_stems=stems[:-1])
            mismatches = {
                "source_size": [99, 50],
                "resized_size": [199, 100],
                "crop_box": [9, 5, 189, 95],
                "bucket": [179, 90],
            }
            for field, value in mismatches.items():
                with self.subTest(field=field), self.assertRaisesRegex(ReferencePoseError, "cannot be reconciled"):
                    geometry = {stem: dict(values) for stem, values in exact_geometry.items()}
                    geometry[stems[0]][field] = value
                    load_exact_capacity_reference_sidecar(output, experiment_name=self.experiment, expected_stems=stems,
                                                          geometry_by_stem=geometry)
            with self.assertRaisesRegex(ReferencePoseError, "required"):
                load_exact_capacity_reference_sidecar(root / "diagnostic_reference_pose.json", experiment_name=self.experiment, expected_stems=stems)

    def test_score_only_reuses_existing_images_and_has_no_sampler_or_training_path(self):
        stems = tuple(f"coco_{index}_1" for index in range(32))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); (output / "training_set").mkdir()
            (output / "generation_results.json").write_text(json.dumps({"experiment": self.experiment, "training_set_overfit": True, "stems": list(stems), "steps": [0, 50, 100, 200, 300, 400, 500]}))
            for stem in stems:
                directory = output / "training_set" / stem; directory.mkdir()
                for name in ("control.png", "target.png"):
                    (directory / name).touch()
                for step in (0, 50, 100, 200, 300, 400, 500): (directory / f"step_{step:06d}.png").touch()
            before = {path: path.stat().st_mtime_ns for path in output.rglob("*.png")}
            evaluator._verify_existing_generation(output, experiment_name=self.experiment, stems=stems)
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in output.rglob("*.png")})
        score_source = ast.get_source_segment(Path(evaluator.__file__).read_text(), next(node for node in ast.walk(ast.parse(Path(evaluator.__file__).read_text())) if isinstance(node, ast.FunctionDef) and node.name == "score"))
        self.assertNotIn("sample_turbo_pose_image", score_source); self.assertNotIn("load_training_state", score_source)
        self.assertNotIn("build_turbo_pose_model", score_source); self.assertNotIn("backward(", score_source)
        self.assertNotIn(".train(", score_source); self.assertNotIn("optimizer", score_source)
        self.assertNotIn("torch.optim", score_source); self.assertIn("score_authoritative_pck", score_source); self.assertIn("_clip_score", score_source)
        with patch.object(evaluator, "_inputs", return_value=(Path("checkpoints"), Path("output"), stems, object())):
            with self.assertRaisesRegex(ValueError, "--reference-sidecar"):
                evaluator.score(SimpleNamespace(reference_sidecar=None, experiment=self.experiment))


if __name__ == "__main__":
    unittest.main()
