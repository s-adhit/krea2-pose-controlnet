import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pose_controlnet.data import PreparedLatentShardDataset
from pose_controlnet.paired_preprocessing import resize_center_crop_geometry
from pose_controlnet.turbo_evaluation import (DEFAULT_TURBO_CHECKPOINT_STEPS, TURBO_CFG, TURBO_MU, TURBO_STEPS, TURBO_CHECKPOINT_STEPS, assert_exact_diagnostic_stems,
    assert_turbo_output_isolated, exact_turbo_checkpoints, raw_to_turbo_control_compatibility,
    sample_turbo_pose_image, turbo_schedule, turbo_scoring_geometry)


class TurboEvaluationTest(unittest.TestCase):
    def test_official_pinned_turbo_schedule_has_exactly_eight_steps_and_is_resolution_invariant(self):
        first = turbo_schedule(image_sequence_length=4096)
        second = turbo_schedule(image_sequence_length=16384)
        self.assertEqual(len(first), TURBO_STEPS + 1); self.assertEqual(first, second)
        self.assertEqual(first, [1.0, 0.956723690032959, 0.9045307636260986, 0.8403487801551819,
                                 0.7595109343528748, 0.6545668244361877, 0.5128440856933594,
                                 0.31090107560157776, 0.0])
        self.assertEqual(TURBO_MU, 1.15)
        with self.assertRaises(ValueError): turbo_schedule(image_sequence_length=1, steps=7)
        with self.assertRaises(ValueError): turbo_schedule(image_sequence_length=1, mu=1.14)

    def test_exact_checkpoint_resolution_accepts_900_and_1200_with_existing_validator(self):
        with tempfile.TemporaryDirectory() as temp, patch("pose_controlnet.turbo_evaluation.ordered_checkpoints", return_value=[(900, Path("900.pt")), (1200, Path("1200.pt"))]) as resolved:
            self.assertEqual(exact_turbo_checkpoints(checkpoint_dir=temp, hf_repo_id="adhit-420/Krea-2-PoseControl-LoRA-checkpoints", steps=(900, 1200)), [(900, Path("900.pt")), (1200, Path("1200.pt"))])
        self.assertEqual(resolved.call_args.kwargs["steps"], (900, 1200))
        self.assertEqual(resolved.call_args.kwargs["hf_repo_id"], "adhit-420/Krea-2-PoseControl-LoRA-checkpoints")
        self.assertEqual(DEFAULT_TURBO_CHECKPOINT_STEPS, (800, 1500))
        self.assertEqual(TURBO_CHECKPOINT_STEPS, (800, 900, 1200, 1500))

    def test_cli_steps_and_hf_archive_namespace_are_exact(self):
        from scripts import turbo_benchmark
        from pose_controlnet.evaluation import HF_ARCHIVE_RUNS

        args = turbo_benchmark.parser().parse_args(["preflight", "--steps", "900", "1200"])
        self.assertEqual(tuple(args.steps), (900, 1200))
        self.assertEqual(HF_ARCHIVE_RUNS[900], "pose-learning-1500")
        self.assertEqual(HF_ARCHIVE_RUNS[1200], "pose-learning-1500")
        self.assertEqual("pose-learning-1500/full/step_000900.pt", f"{HF_ARCHIVE_RUNS[900]}/full/step_{900:06d}.pt")
        self.assertEqual("pose-learning-1500/full/step_001200.pt", f"{HF_ARCHIVE_RUNS[1200]}/full/step_{1200:06d}.pt")

    def test_canonical_output_namespace_is_rejected_and_diagnostic_contract_is_exact(self):
        with self.assertRaises(ValueError): assert_turbo_output_isolated("/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500/turbo")
        self.assertEqual(len(assert_exact_diagnostic_stems([str(i) for i in range(24)], [str(i) for i in range(24)])), 24)
        with self.assertRaises(ValueError): assert_exact_diagnostic_stems([str(i) for i in range(23)], [str(i) for i in range(23)])

    def test_cfg_zero_uses_one_controlled_forward_per_denoise_step(self):
        model = SimpleNamespace(config=SimpleNamespace(patch=1))
        sample = {"latent": torch.ones(1, 1, 1), "control": torch.ones(1, 1, 1), "context": torch.ones(1, 1, 1), "mask": torch.ones(1, dtype=torch.bool)}
        calls = []
        def forward(_model, image, control, *_args, **_kwargs):
            calls.append(control.clone()); return torch.zeros_like(image)
        with patch("pose_controlnet.turbo_evaluation.forward_pose_control", side_effect=forward):
            pixels = sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 1)
        self.assertEqual(len(calls), TURBO_STEPS); self.assertTrue(all(control.abs().max() > 0 for control in calls)); self.assertEqual(pixels.shape, (1, 1, 1))
        with self.assertRaises(ValueError): sample_turbo_pose_image(model, lambda latent: latent, sample, torch.device("cpu"), 1, guidance=1.0)

    def test_turbo_modules_are_evaluation_only(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts/turbo_benchmark.py").read_text().lower() + (root / "pose_controlnet/turbo_evaluation.py").read_text().lower()
        self.assertNotIn("torch.optim", source); self.assertNotIn("backward(", source); self.assertNotIn("optimizer.step", source)
        self.assertEqual(TURBO_CFG, 0.0)

    def test_raw_to_turbo_compatibility_requires_raw_provenance_and_exact_state(self):
        model = torch.nn.Linear(1, 1)
        with patch("pose_controlnet.turbo_evaluation.trainable_state_dict", return_value={"first.weight": torch.zeros(2, 2)}):
            result = raw_to_turbo_control_compatibility(model, {"config": {"raw_ckpt": "raw.safetensors"}, "model": {"first.weight": torch.zeros(2, 2)}})
        self.assertEqual(result["shape_mismatches"], 0)
        with patch("pose_controlnet.turbo_evaluation.trainable_state_dict", return_value={"first.weight": torch.zeros(2, 2)}):
            with self.assertRaises(ValueError): raw_to_turbo_control_compatibility(model, {"config": {}, "model": {"first.weight": torch.zeros(2, 2)}})

    def test_turbo_scoring_geometry_matches_canonical_paired_preprocessing_for_all_aspects(self):
        examples = {"portrait": ((1000, 1500), (832, 1216)), "landscape": ((1800, 1000), (1344, 768)),
                    "square": ((1000, 1000), (1024, 1024))}
        for stem, (source_size, bucket) in examples.items():
            with self.subTest(stem=stem):
                canonical = resize_center_crop_geometry(source_size, bucket)
                sample = {"stem": stem, "bucket": list(bucket), "source_size": list(source_size),
                          "resized_size": list(canonical.resized_size), "crop_box": list(canonical.crop_box)}
                self.assertEqual(turbo_scoring_geometry(sample), {
                    "source_size": list(canonical.source_size), "resized_size": list(canonical.resized_size),
                    "crop_box": list(canonical.crop_box),
                })

    def test_turbo_scoring_geometry_requires_complete_persisted_fields(self):
        with self.assertRaisesRegex(ValueError, "missing persisted paired fields: crop_box"):
            turbo_scoring_geometry({"stem": "missing", "source_size": [100, 100],
                                    "resized_size": [1024, 1024], "bucket": [1024, 1024]})

    def test_existing_turbo_outputs_score_without_regeneration(self):
        from scripts import turbo_benchmark

        examples = {"portrait": ((1000, 1500), (832, 1216)), "landscape": ((1800, 1000), (1344, 768)),
                    "square": ((1000, 1000), (1024, 1024))}
        samples = []
        for stem, (source_size, bucket) in examples.items():
            canonical = resize_center_crop_geometry(source_size, bucket)
            samples.append({"stem": stem, "bucket": list(bucket), "source_size": list(source_size),
                            "resized_size": list(canonical.resized_size), "crop_box": list(canonical.crop_box)})

        class Dataset:
            records = [("unused", index, (1, 1), sample["stem"]) for index, sample in enumerate(samples)]
            def __getitem__(self, index): return samples[index]

        sidecar = {"records": [{"stem": sample["stem"], "source": "humanart", "status": "available", "people": []} for sample in samples]}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "generation_results.json").write_text("{}")
            for stem in examples:
                directory = output / "fixed_pose" / stem; directory.mkdir(parents=True)
                (directory / "metadata.json").write_text('{"prompt": "pose"}')
                for step in (800, 1500): (directory / f"step_{step:06d}.png").touch()
            args = SimpleNamespace(output_dir=output, reference_sidecar=output / "sidecar.json", clip_model_id="clip")
            args.reference_sidecar.write_text(__import__("json").dumps(sidecar))
            fake_clip = SimpleNamespace(to=lambda _device: fake_clip, eval=lambda: fake_clip)
            with patch.object(turbo_benchmark, "_dataset_and_spec", return_value=(Dataset(), tuple(examples), {})), \
                 patch.object(turbo_benchmark, "KeypointRCNNEstimator"), \
                 patch.object(turbo_benchmark.CLIPProcessor, "from_pretrained", return_value=object()), \
                 patch.object(turbo_benchmark.CLIPModel, "from_pretrained", return_value=fake_clip), \
                 patch.object(turbo_benchmark, "score_authoritative_pck", return_value={"pck_005": 1.0}), \
                 patch.object(turbo_benchmark, "_clip_score", return_value=.5), \
                 patch.object(turbo_benchmark, "sample_turbo_pose_image") as generate:
                turbo_benchmark.score(args)
            generate.assert_not_called()
            self.assertTrue((output / "pck_clip_results.json").is_file())

    def test_incremental_result_merging_preserves_legacy_steps_and_canonical_order(self):
        from scripts import turbo_benchmark

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "generation_results.json").write_text('{"generated_steps": {"a": [800, 1500]}}')
            self.assertEqual(turbo_benchmark._merged_generation_results(output, generated={"a": [900, 1200]}),
                             {"a": [800, 900, 1200, 1500]})
            (output / "pck_clip_results.json").write_text('{"checkpoints": [{"checkpoint_step": 1500}, {"checkpoint_step": 800}]}')
            rows = turbo_benchmark._existing_result_rows(output)
            rows[900] = {"checkpoint_step": 900}; rows[1200] = {"checkpoint_step": 1200}
            self.assertEqual([rows[step]["checkpoint_step"] for step in TURBO_CHECKPOINT_STEPS], [800, 900, 1200, 1500])

    def test_existing_800_and_1500_images_are_reused_and_only_900_1200_are_missing(self):
        from scripts import turbo_benchmark

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "step_000800.png").touch(); (directory / "step_001500.png").touch()
            checkpoints = [(step, Path(f"{step}.pt")) for step in TURBO_CHECKPOINT_STEPS]
            self.assertEqual(turbo_benchmark._missing_generation_checkpoints(directory, checkpoints),
                             [(900, Path("900.pt")), (1200, Path("1200.pt"))])

    def test_prepared_dataset_exposes_persisted_paired_geometry_to_turbo_scoring(self):
        canonical = resize_center_crop_geometry((1000, 1500), (832, 1216))
        sample = {"stem": "portrait", "text": "pose", "bucket": [832, 1216], "source_size": [1000, 1500],
                  "resized_size": list(canonical.resized_size), "crop_box": list(canonical.crop_box),
                  "image_latent": torch.ones(16, 152, 104), "control_latent": torch.ones(16, 152, 104)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "diagnostic_val").mkdir()
            (root / "shards.json").write_text('{"format_version": 1, "complete": true}')
            torch.save({"format_version": 1, "split": "diagnostic_val", "samples": [sample]}, root / "diagnostic_val/diagnostic_val-00000.pt")
            item = PreparedLatentShardDataset(root, "diagnostic_val")[0]
        self.assertEqual(turbo_scoring_geometry(item), {"source_size": [1000, 1500],
                                                         "resized_size": list(canonical.resized_size),
                                                         "crop_box": list(canonical.crop_box)})


if __name__ == "__main__":
    unittest.main()
