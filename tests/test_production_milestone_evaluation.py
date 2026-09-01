import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from pose_controlnet import paired_preprocessing
from pose_controlnet.overfit_capacity import RESOLUTION_768_BUCKETS
from pose_controlnet.production_milestone_evaluation import (
    EVALUATION_MODES,
    ProductionMilestoneEvaluationError,
    SUMMARY_COLUMNS,
    assert_mode_metadata,
    cross_checkpoint_summary,
    geometry_for_mode,
    mode_metadata,
    mode_output_root,
)
from pose_controlnet.reference_pose import reference_person_from_sidecar
from pose_controlnet.evaluation_geometry import persisted_scoring_geometry
from scripts import evaluate_production_milestones as harness


class ProductionMilestoneEvaluationTests(unittest.TestCase):
    native_sample = {
        "source_size": [640, 427],
        "resized_size": [1247, 832],
        "crop_box": [15, 0, 1231, 832],
        "bucket": [1216, 832],
    }

    def test_dynamic_768_reuses_training_bucket_selector_and_exact_policy(self):
        self.assertIs(harness.RESOLUTION_768_BUCKETS, RESOLUTION_768_BUCKETS)
        from pose_controlnet import production_milestone_evaluation as milestone
        self.assertIs(milestone.choose_bucket, paired_preprocessing.choose_bucket)
        self.assertIs(milestone.resize_center_crop_geometry, paired_preprocessing.resize_center_crop_geometry)
        expected = {
            (1000, 1000): (768, 768), (780, 1000): (704, 896), (1280, 1000): (896, 704),
            (660, 1000): (640, 960), (1500, 1000): (960, 640), (560, 1000): (576, 1024),
            (1780, 1000): (1024, 576), (440, 1000): (512, 1152), (2250, 1000): (1152, 512),
        }
        self.assertEqual(RESOLUTION_768_BUCKETS, (
            (768, 768), (704, 896), (896, 704), (640, 960), (960, 640),
            (576, 1024), (1024, 576), (512, 1152), (1152, 512),
        ))
        for source_size, bucket in expected.items():
            geometry = geometry_for_mode(mode="dynamic-768", native_sample=self.native_sample, source_size=source_size)
            self.assertEqual(tuple(geometry["bucket"]), bucket)
            shared = paired_preprocessing.resize_center_crop_geometry(source_size, paired_preprocessing.choose_bucket(source_size, RESOLUTION_768_BUCKETS))
            self.assertEqual(geometry, {"source_size": list(shared.source_size), "resized_size": list(shared.resized_size),
                                        "crop_box": list(shared.crop_box), "bucket": list(shared.bucket)})

    def test_dynamic_reference_coordinates_use_exact_dynamic_eval_geometry(self):
        geometry = geometry_for_mode(mode="dynamic-768", native_sample=self.native_sample, source_size=(1000, 500))
        self.assertEqual(geometry, {"source_size": [1000, 500], "resized_size": [1152, 576],
                                    "crop_box": [64, 0, 1088, 576], "bucket": [1024, 576]})
        person = {"annotation_id": 7, "keypoints": [[500, 250, 2] for _ in range(17)]}
        transformed = reference_person_from_sidecar(person, source_size=tuple(geometry["source_size"]),
                                                     resized_size=tuple(geometry["resized_size"]),
                                                     crop_box=tuple(geometry["crop_box"]), requires_renderer_qualification=False)
        self.assertEqual(transformed["keypoints_bucket"][0], [512.0, 288.0, 2.0])
        self.assertEqual(transformed["keypoints"][0], [512.0, 288.0, 1.0])

    def test_native_geometry_remains_the_locked_persisted_geometry(self):
        expected = persisted_scoring_geometry(self.native_sample, label="Turbo")
        actual = geometry_for_mode(mode="native", native_sample=self.native_sample, source_size=(640, 427))
        self.assertEqual(actual, expected)
        self.assertEqual(actual["bucket"], [1216, 832])
        self.assertNotEqual(actual, geometry_for_mode(mode="dynamic-768", native_sample=self.native_sample, source_size=(640, 427)))

    def test_checkpoint_and_mode_roots_and_metadata_cannot_mix(self):
        native = mode_output_root("/tmp/evaluation", 500, "native")
        dynamic = mode_output_root("/tmp/evaluation", 500, "dynamic-768")
        self.assertEqual(native, Path("/tmp/evaluation/step_000500/native"))
        self.assertEqual(dynamic, Path("/tmp/evaluation/step_000500/dynamic-768"))
        self.assertNotEqual(native, dynamic)
        self.assertEqual(mode_output_root("/tmp/evaluation", 3500, "native"), Path("/tmp/evaluation/step_003500/native"))
        metadata = mode_metadata(mode="native", stem="sample", prompt="caption", seed=42,
                                 geometry=geometry_for_mode(mode="native", native_sample=self.native_sample, source_size=(640, 427)))
        assert_mode_metadata(metadata, mode="native", stem="sample")
        with self.assertRaisesRegex(ProductionMilestoneEvaluationError, "another mode"):
            assert_mode_metadata(metadata, mode="dynamic-768", stem="sample")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "step_000500" / "native"
            (root / "fixed_pose" / "sample").mkdir(parents=True)
            with self.assertRaisesRegex(ProductionMilestoneEvaluationError, "Partial production milestone"):
                harness._generation_complete(root, ("sample",), 500, "native")

    def test_cross_checkpoint_summary_requires_explicit_mode(self):
        summary = cross_checkpoint_summary((
            {"checkpoint_step": 500, "mode": "dynamic-768", "pose": {"pck_020": .2}},
            {"checkpoint_step": 500, "mode": "native", "pose": {"pck_020": .3}},
        ))
        self.assertEqual(summary["modes"], list(EVALUATION_MODES))
        self.assertIn("mode", SUMMARY_COLUMNS)
        self.assertEqual([(row["checkpoint_step"], row["mode"]) for row in summary["checkpoints"]],
                         [(500, "native"), (500, "dynamic-768")])
        with self.assertRaisesRegex(ProductionMilestoneEvaluationError, "explicit mode"):
            cross_checkpoint_summary(({"checkpoint_step": 500},))

    def test_cli_defaults_to_exact_dual_mode_milestones(self):
        args = harness.parser().parse_args([
            "report", "--checkpoint-root", "/tmp/checkpoints", "--output-root", "/tmp/evaluation",
            "--dataset-root", "/tmp/dataset", "--latent-root", "/tmp/latents", "--text-conditioning-root", "/tmp/text",
            "--turbo-ckpt", "/tmp/turbo.safetensors", "--reference-sidecar", "/tmp/sidecar.json",
        ])
        self.assertEqual(tuple(args.steps), (500, 1000, 1500, 2000, 2500, 3000))
        self.assertEqual(tuple(args.modes), ("native", "dynamic-768"))

    def _contact_inputs(self, root: Path, *, steps=(500, 1750), modes=("native", "dynamic-768"), stems=("first", "second")):
        rgb = root / "source-rgb.png"; control = root / "source-control.png"
        Image.new("RGB", (6, 4), "red").save(rgb); Image.new("RGB", (6, 4), "blue").save(control)
        for step in steps:
            for mode in modes:
                mode_root = root / "evaluation" / f"step_{step:06d}" / mode
                mode_root.mkdir(parents=True)
                (mode_root / "generation_results.json").write_text(json.dumps({
                    "checkpoint_step": step, "mode": mode, "stems": list(stems),
                }))
                for stem in stems:
                    generated = mode_root / "fixed_pose" / stem / "generated.png"
                    generated.parent.mkdir(parents=True)
                    Image.new("RGB", (6, 4), "green").save(generated)
        snapshot = SimpleNamespace(records_by_split={"diagnostic_val": tuple(
            SimpleNamespace(stem=stem, rgb_path=rgb, control_path=control) for stem in stems
        )})
        return SimpleNamespace(evaluation_root=root / "evaluation", dataset_root=root / "dataset", steps=list(steps),
                               modes=list(modes), output_dir=root / "sheets"), snapshot

    def test_contact_sheet_accepts_arbitrary_steps_and_modes_with_isolated_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, snapshot = self._contact_inputs(Path(temporary), steps=(250, 1250, 3000), modes=("dynamic-768",))
            with mock.patch("pose_controlnet.dataset_index.validate_posebridge_snapshot", return_value=snapshot):
                harness.contact_sheet(args)
            self.assertTrue((args.output_dir / "dynamic768_full_contact_sheet.png").is_file())
            self.assertFalse((args.output_dir / "native_full_contact_sheet.png").exists())
            self.assertEqual(harness.contact_sheet_filename("dynamic-768"), "dynamic768_full_contact_sheet.png")

    def test_contact_sheet_preserves_stem_order_and_fails_closed_for_bad_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, snapshot = self._contact_inputs(Path(temporary))
            calls = []
            with mock.patch("pose_controlnet.dataset_index.validate_posebridge_snapshot", return_value=snapshot), \
                 mock.patch("pose_controlnet.evaluation.make_contact_sheet", side_effect=lambda rows, *_args, **kwargs: calls.append((rows, kwargs))):
                harness.contact_sheet(args)
            self.assertEqual([stem for stem, _ in calls[0][0]], ["first", "second"])
            self.assertEqual(calls[0][1]["column_labels"], ("Pose control", "Target RGB", "Step 500", "Step 1750"))
            mismatch = args.evaluation_root / "step_001750" / "native" / "generation_results.json"
            mismatch.write_text(json.dumps({"checkpoint_step": 1750, "mode": "native", "stems": ["second", "first"]}))
            with mock.patch("pose_controlnet.dataset_index.validate_posebridge_snapshot", return_value=snapshot):
                with self.assertRaisesRegex(ProductionMilestoneEvaluationError, "stem order differs"):
                    harness.contact_sheet(args)
            mismatch.write_text(json.dumps({"checkpoint_step": 1750, "mode": "native", "stems": ["first", "second"]}))
            (args.evaluation_root / "step_000500" / "native" / "fixed_pose" / "first" / "generated.png").unlink()
            with mock.patch("pose_controlnet.dataset_index.validate_posebridge_snapshot", return_value=snapshot):
                with self.assertRaisesRegex(FileNotFoundError, "Missing generated"):
                    harness.contact_sheet(args)

    def test_contact_sheet_cli_is_dynamic_without_changing_evaluation_defaults(self):
        args = harness.parser().parse_args([
            "contact-sheet", "--evaluation-root", "/tmp/evaluation", "--dataset-root", "/tmp/dataset",
            "--steps", "250", "3000", "--modes", "native", "--output-dir", "/tmp/sheets",
        ])
        self.assertEqual((args.command_name, args.steps, args.modes), ("contact-sheet", [250, 3000], ["native"]))


if __name__ == "__main__":
    unittest.main()
