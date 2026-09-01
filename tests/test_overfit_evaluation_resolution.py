import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from pose_controlnet.capacity_resolution import preprocess_native_evaluation_pair
from pose_controlnet.dataset_index import ManifestRecord
from pose_controlnet.overfit_capacity import OVERFIT_STEPS, validate_manifest
from scripts import evaluate_overfit_capacity as evaluator
from scripts import summarize_overfit_capacity as summarizer


class _NativeData:
    def __init__(self, stems):
        self.items = [{
            "stem": stem, "prompt": f"caption {stem}", "latent": torch.zeros(16, 96, 96),
            "control": torch.ones(16, 96, 96), "source_size": [1000, 800],
            "resized_size": [960, 768], "crop_box": [96, 0, 864, 768], "bucket": [768, 768],
        } for stem in stems]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


class OverfitEvaluationResolutionTests(unittest.TestCase):
    experiment = "overfit32-mixed-r64-mse-res768"

    def setUp(self):
        self.stems = validate_manifest("overfit32-mixed-r64-mse")
        self.data = _NativeData(self.stems)

    def _checkpoint(self, root: Path, resolution: str) -> Path:
        checkpoint = root / self.experiment
        checkpoint.mkdir(parents=True)
        (checkpoint / "experiment_metadata.json").write_text(json.dumps({
            "scientific_config": {"resolution": resolution, "pose_loss": "none"},
            "resolution_policy": resolution,
        }))
        return checkpoint

    def _provenance(self, checkpoint: Path):
        return evaluator._evaluation_provenance(checkpoint, self.data, self.stems)

    def _complete_generation(self, output: Path, provenance: dict):
        output.mkdir(parents=True, exist_ok=True)
        (output / "generation_results.json").write_text(json.dumps({
            "experiment": self.experiment, "training_set_overfit": True,
            "stems": list(self.stems), "steps": list(OVERFIT_STEPS),
            "training_resolution": provenance["training_resolution"],
            "evaluation_resolution": "native", "evaluation_provenance": provenance,
        }))
        for stem in self.stems:
            sample = output / "training_set" / stem
            sample.mkdir(parents=True)
            for name in ("control.png", "target.png", *(f"step_{step:06d}.png" for step in OVERFIT_STEPS)):
                (sample / name).touch()

    def test_native_and_768_training_both_evaluate_native_with_preserved_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = self._provenance(self._checkpoint(root / "native", "native"))
            trained_768 = self._provenance(self._checkpoint(root / "res768", "768"))
        self.assertEqual(native["training_resolution"], "native")
        self.assertEqual(trained_768["training_resolution"], "768")
        self.assertEqual(native["evaluation_resolution"], "native")
        self.assertEqual(trained_768["evaluation_resolution"], "native")
        self.assertEqual(native["native_geometry"], trained_768["native_geometry"])

    def test_evaluation_cli_has_no_768_mode_or_resolution_cache(self):
        with self.assertRaises(SystemExit):
            evaluator.parser().parse_args(["--experiment", self.experiment, "--resolution", "768"])
        source = Path("scripts/evaluate_overfit_capacity.py").read_text(encoding="utf-8")
        self.assertNotIn("AlternateResolutionDataset", source)
        self.assertNotIn("load_alternate_resolution_cache", source)
        self.assertNotIn("resolution-cache-root", source)

    def test_exact_mixed_order_and_checkpoint_schedule_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = self._checkpoint(root / "checkpoints", "768")
            provenance = self._provenance(checkpoint); output = root / "evaluation" / self.experiment
            self._complete_generation(output, provenance)
            evaluator._verify_existing_generation(output, experiment_name=self.experiment, stems=self.stems, provenance=provenance)
            payload = json.loads((output / "generation_results.json").read_text()); payload["stems"] = list(reversed(self.stems))
            (output / "generation_results.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "exact immutable"):
                evaluator._verify_existing_generation(output, experiment_name=self.experiment, stems=self.stems, provenance=provenance)

    def test_native_rgb_and_control_rebuild_from_the_same_persisted_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rgb, control = root / "rgb.jpg", root / "control.png"
            Image.new("RGB", (1000, 800), (20, 30, 40)).save(rgb)
            Image.new("RGB", (1000, 800), (220, 40, 60)).save(control)
            record = ManifestRecord(split="train", stem="paired", file_name="paired.jpg", text="caption", rgb_path=rgb, control_path=control)
            pair = preprocess_native_evaluation_pair(record, self.data[0])
        self.assertEqual(pair.rgb.size, (768, 768))
        self.assertEqual(pair.control.size, pair.rgb.size)
        self.assertEqual(pair.geometry.resized_size, (960, 768))
        self.assertEqual(pair.geometry.crop_box, (96, 0, 864, 768))

    def test_incompatible_partial_768_generation_fails_closed_with_archive_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = self._checkpoint(root / "checkpoints", "768")
            provenance = self._provenance(checkpoint); output = root / "evaluation" / self.experiment; output.mkdir(parents=True)
            (output / "generation_results.json").write_text(json.dumps({
                "experiment": self.experiment, "training_set_overfit": True,
                "stems": list(self.stems), "steps": list(OVERFIT_STEPS),
                "training_resolution": "768", "evaluation_resolution": "768",
            }))
            with self.assertRaisesRegex(ValueError, "archive it first: mv --"):
                evaluator._verify_existing_generation(output, experiment_name=self.experiment, stems=self.stems, provenance=provenance)

    def test_existing_complete_native_evaluation_is_reused_without_gpu_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = self._checkpoint(root / "checkpoints", "768")
            provenance = self._provenance(checkpoint); output = root / "evaluation" / self.experiment
            self._complete_generation(output, provenance)
            before = (output / "generation_results.json").read_bytes()
            args = type("Args", (), {"experiment": self.experiment})()
            with patch.object(evaluator, "_inputs", return_value=(checkpoint, output, self.stems, self.data)), \
                 patch.object(evaluator.torch.cuda, "is_available") as cuda:
                evaluator.generate(args)
            cuda.assert_not_called()
            self.assertEqual((output / "generation_results.json").read_bytes(), before)

    def test_report_records_768_train_native_eval_and_never_constructs_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = self._checkpoint(root / "checkpoints", "768")
            provenance = self._provenance(checkpoint); output = root / "evaluation" / self.experiment
            self._complete_generation(output, provenance)
            args = type("Args", (), {"experiment": self.experiment})()
            with patch.object(evaluator, "_inputs", return_value=(checkpoint, output, self.stems, self.data)), \
                 patch.object(evaluator, "make_contact_sheet"):
                evaluator.report(args)
            summary = json.loads((output / "overfit_summary.json").read_text())
        self.assertEqual((summary["training_resolution"], summary["evaluation_resolution"]), ("768", "native"))
        source = Path("scripts/evaluate_overfit_capacity.py").read_text(encoding="utf-8")
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)

    def test_comparison_labels_native_only_and_excludes_non_native_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint_root = root / "checkpoints"; output_root = root / "evaluation"
            cases = {
                "overfit32-mixed-r64-mse-resnative": ("native", "none", "native"),
                "overfit32-mixed-r64-mse-res768": ("768", "none", "native"),
                "overfit32-mixed-r64-coord-l1e-5-res768": ("768", "normalized_coordinate_huber", "native"),
                "old-768-eval": ("768", "none", "768"),
            }
            for name, (training, pose_loss, evaluation) in cases.items():
                checkpoint = checkpoint_root / name; checkpoint.mkdir(parents=True)
                (checkpoint / "experiment_metadata.json").write_text(json.dumps({"scientific_config": {"resolution": training, "pose_loss": pose_loss}}))
                destination = output_root / name; destination.mkdir(parents=True)
                (destination / "overfit_summary.json").write_text(json.dumps({"training_resolution": training, "evaluation_resolution": evaluation, "checkpoints": []}))
            names = list(cases)
            with patch.object(sys, "argv", ["summarize_overfit_capacity.py", "--output-root", str(output_root), "--checkpoint-root", str(checkpoint_root), *names]):
                summarizer.main()
            result = json.loads((output_root / "capacity_comparison_summary.json").read_text())
        self.assertEqual(result["evaluation_resolution"], "native")
        self.assertEqual(result["experiments"]["overfit32-mixed-r64-mse-res768"]["comparison_label"], "768 train / Native eval")
        self.assertEqual(result["experiments"]["overfit32-mixed-r64-coord-l1e-5-res768"]["comparison_label"], "768+pose train / Native eval")
        self.assertIn("old-768-eval", result["excluded_experiments"])

    def test_training_cli_still_owns_768_while_evaluator_does_not(self):
        runner = Path("scripts/run_overfit_capacity.py").read_text(encoding="utf-8")
        trainer = Path("scripts/train_overfit_capacity.py").read_text(encoding="utf-8")
        self.assertIn('"--resolution", default="native", choices=("native", "current", "768")', runner)
        self.assertIn("prepare_alternate_resolution_cache", trainer)


class LegacyNativeProvenanceCompatibilityTests(unittest.TestCase):
    experiment = "overfit32-mixed-r64-mse"

    def setUp(self):
        self.stems = validate_manifest(self.experiment)
        self.data = _NativeData(self.stems)

    def _generation(self, provenance, stems=None):
        stems = self.stems if stems is None else stems
        return {
            "experiment": self.experiment, "training_set_overfit": True,
            "stems": list(stems), "steps": list(OVERFIT_STEPS),
            "training_resolution": "native", "evaluation_resolution": "native",
            "evaluation_provenance": provenance,
        }

    @staticmethod
    def _metadata(*, scientific_resolution="none", training_resolution=None, resolution_policy="none"):
        metadata = {"scientific_config": {"resolution": scientific_resolution}, "resolution_policy": resolution_policy}
        if training_resolution is not None:
            metadata["training_resolution"] = training_resolution
        return metadata

    def _checkpoint(self, root: Path, metadata: dict, *, steps=OVERFIT_STEPS) -> Path:
        root.mkdir(parents=True)
        (root / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        for step in steps:
            (root / f"step_{step:06d}.pt").touch()
        return root

    def _provenance(self, checkpoint: Path, *, stems=None, generated=None,
                    require_persisted_generation=False, data=None):
        stems = self.stems if stems is None else stems
        return evaluator._evaluation_provenance(
            checkpoint, self.data if data is None else data, stems, experiment_name=self.experiment,
            generated=generated, require_persisted_generation=require_persisted_generation,
        )

    def test_generation_establishes_legacy_native_without_preexisting_generation_metadata_or_checkpoint_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata())
            before = (checkpoint / "experiment_metadata.json").read_bytes()
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint):
                provenance = self._provenance(checkpoint)
            self.assertEqual((checkpoint / "experiment_metadata.json").read_bytes(), before)
        self.assertEqual(provenance["training_resolution"], "native")
        self.assertEqual(provenance["evaluation_resolution"], "native")
        self.assertEqual(provenance["training_resolution_source"], "legacy_native_compatibility")

    def test_arbitrary_none_resolution_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "other", self._metadata())
            with self.assertRaisesRegex(ValueError, "supported native or 768"):
                evaluator._evaluation_provenance(
                    checkpoint, self.data, self.stems, experiment_name="other-experiment",
                    require_persisted_generation=False,
                )

    def test_contradictory_legacy_resolution_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata(scientific_resolution="768"))
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 self.assertRaisesRegex(ValueError, "contradictory"):
                self._provenance(checkpoint)

    def test_wrong_legacy_checkpoint_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self._checkpoint(root / "expected", self._metadata())
            wrong = self._checkpoint(root / "wrong", self._metadata())
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", expected), \
                 self.assertRaisesRegex(ValueError, "expected legacy checkpoint root"):
                self._provenance(wrong)

    def test_wrong_mixed32_stem_order_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata())
            wrong_stems = tuple(reversed(self.stems))
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 self.assertRaisesRegex(ValueError, "stem order"):
                self._provenance(checkpoint, stems=wrong_stems, generated=self._generation(wrong_stems))

    def test_legacy_metadata_checkpoint_schedule_if_present_must_be_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self._metadata(); metadata["checkpoint_steps"] = [0, 100, 500]
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, metadata)
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 self.assertRaisesRegex(ValueError, "checkpoint schedule"):
                self._provenance(checkpoint)

    def test_legacy_checkpoint_files_must_match_the_exact_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata(), steps=(0, 50, 100, 200, 300, 400))
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 self.assertRaisesRegex(ValueError, "exact checkpoint set"):
                self._provenance(checkpoint)

    def test_resolution_manifest_or_alternate_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root / "checkpoints" / self.experiment, self._metadata())
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint):
                (checkpoint / "resolution_manifest.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "alternate-resolution"):
                    self._provenance(checkpoint)
                (checkpoint / "resolution_manifest.json").unlink()
                cache = checkpoint.parent.parent / "resolution_cache" / self.experiment
                cache.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "alternate-resolution"):
                    self._provenance(checkpoint)

    def test_missing_native_latent_geometry_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata())
            incomplete = _NativeData(self.stems[:-1])
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 self.assertRaisesRegex(ValueError, "native latent geometry is missing"):
                self._provenance(checkpoint, data=incomplete)

    def test_paired_native_rgb_control_geometry_must_be_recoverable_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rgb, control = root / "rgb.jpg", root / "control.png"
            Image.new("RGB", (1000, 800)).save(rgb)
            Image.new("RGB", (999, 800)).save(control)
            physical = {
                stem: ManifestRecord(split="train", stem=stem, file_name=f"{stem}.jpg", text="caption",
                                     rgb_path=rgb, control_path=control)
                for stem in self.stems
            }
            with self.assertRaisesRegex(ValueError, "Persisted source geometry disagrees"):
                evaluator._recover_native_evaluation_pairs(self.data, self.stems, physical)

    def test_regenerated_generation_provenance_records_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata())
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint):
                provenance = self._provenance(checkpoint)
        generated = evaluator._generation_results(
            experiment_name=self.experiment, stems=self.stems, provenance=provenance,
            turbo={}, compatibility={},
        )
        self.assertEqual(generated["training_resolution"], "native")
        self.assertEqual(generated["evaluation_resolution"], "native")
        self.assertEqual(generated["evaluation_provenance"]["training_resolution_source"], "legacy_native_compatibility")

    def test_report_and_score_require_persisted_regenerated_native_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = self._checkpoint(root / "checkpoints" / self.experiment, self._metadata())
            output = root / "evaluation" / self.experiment; args = type("Args", (), {
                "experiment": self.experiment, "reference_sidecar": root / "sidecar.jsonl",
            })()
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 patch.object(evaluator, "_inputs", return_value=(checkpoint, output, self.stems, self.data)), \
                 self.assertRaisesRegex(ValueError, "persisted regenerated native generation metadata"):
                evaluator.report(args)
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint), \
                 patch.object(evaluator, "_inputs", return_value=(checkpoint, output, self.stems, self.data)), \
                 self.assertRaisesRegex(ValueError, "persisted regenerated native generation metadata"):
                evaluator.score(args)

    def test_report_and_score_accept_only_regenerated_native_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoints" / self.experiment, self._metadata())
            with patch.object(evaluator, "LEGACY_NATIVE_CHECKPOINT_ROOT", checkpoint):
                generation_provenance = self._provenance(checkpoint)
                provenance = self._provenance(
                    checkpoint, generated=self._generation(generation_provenance),
                    require_persisted_generation=True,
                )
        self.assertEqual(provenance["training_resolution_source"], "legacy_native_compatibility")

    def test_score_metric_definitions_are_not_in_the_legacy_provenance_path(self):
        source = Path("scripts/evaluate_overfit_capacity.py").read_text(encoding="utf-8")
        self.assertIn("score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=image_for, detector=detector, confidence_threshold=.5, require_images=True)", source)
        self.assertIn("values = aggregate([row[\"cosine_similarity\"] for row in clip_rows])", source)


if __name__ == "__main__":
    unittest.main()
