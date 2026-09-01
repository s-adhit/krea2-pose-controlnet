import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from pose_controlnet.keypoint_critic import (
    differentiable_pose_loss,
    normalized_coordinate_distances,
    normalized_coordinate_huber,
    soft_coordinates,
)
from pose_controlnet.turbo_evaluation import turbo_metadata
from scripts import turbo_benchmark


def _record(stem: str, *, available: bool = True) -> dict:
    return {
        "stem": stem, "source": "coco" if available else "danbooru", "pose_reward_available": available,
        "bucket": [8, 8], "people": ([{
            "bbox_training_xywh": [1.0, 1.0, 6.0, 6.0],
            "joint_provenance": [{"training_coordinate": [4.0, 4.0], "reward_joint_valid": joint < 2} for joint in range(17)],
        }] if available else None),
    }


def _score(step: int, stems: tuple[str, ...], pck: float) -> dict:
    return {
        "checkpoint_step": step,
        "pose": {"pck_005": pck, "pck_010": pck + .1, "pck_020": pck + .2,
                 "detection_coverage": .8, "joint_evaluation_coverage": .7,
                 "per_image": [{"stem": stem, "source": "coco", "pck_005": pck} for stem in stems if stem == "a"]},
        "clip": {"mean_cosine_similarity": .3},
    }


def _write_artifacts(root: Path, *, steps: tuple[int, ...], stems: tuple[str, ...], experiment: str | None,
                     resolved: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    spec = {"kind": "turbo_fixed_pose", "seed": 420200, "stems": list(stems), "per_stem_seeds": {},
            "sample_identities": {}, "turbo": turbo_metadata(), "control_scale": 1.0}
    if experiment is not None:
        spec["experiment_name"] = experiment
    if resolved is not None:
        spec["resolved_experiment"] = resolved
    (root / "turbo_spec.json").write_text(json.dumps(spec))
    generated = {stem: list(steps) for stem in stems}
    (root / "generation_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0,
                                                                  "stems": list(stems), "checkpoints": list(steps), "generated_steps": generated}))
    scores = [_score(step, stems, .1 + index * .01) for index, step in enumerate(steps)]
    payload = {"metadata": turbo_metadata(), "control_scale": 1.0, "checkpoints": scores}
    if experiment is not None:
        payload["experiment_name"] = experiment
    (root / "pck_clip_results.json").write_text(json.dumps(payload))
    for stem in stems:
        directory = root / "fixed_pose" / stem; directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps({"stem": stem, "control_scale": 1.0, **turbo_metadata()}))
        for step in steps:
            Image.new("RGB", (8, 8), (10, 20, 30)).save(directory / f"step_{step:06d}.png")


class _FakeCritic(torch.nn.Module):
    identifier = "torchvision/keypointrcnn_resnet50_fpn:COCO_V1"
    def to(self, device):
        return self

    def eval(self):
        return self

    def forward(self, rgb, boxes):
        supplied = boxes[0]
        logits = torch.zeros((len(supplied), 17, 2, 2), dtype=rgb.dtype, device=rgb.device)
        return SimpleNamespace(logits=logits, boxes_training=supplied)


class CriticAlignmentTest(unittest.TestCase):
    def test_cli_parses_arbitrary_steps(self):
        args = turbo_benchmark.parser().parse_args([
            "critic-alignment", "--output-root", "evaluation", "--sidecar", "sidecar", "--steps", "1525", "1550",
        ])
        self.assertEqual(args.steps, [1525, 1550])
        self.assertEqual(args.device, "cuda")

    def test_shared_normalized_huber_and_euclidean_metric_preserve_masking(self):
        logits = torch.zeros((1, 17, 2, 2), dtype=torch.float32)
        boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        target = torch.zeros((1, 17, 2), dtype=torch.float32)
        valid = torch.zeros((1, 17), dtype=torch.bool); valid[0, 0] = True
        coordinates = soft_coordinates(logits, boxes, 1.0)
        expected = differentiable_pose_loss("normalized_coordinate_huber", logits, target, boxes, valid)
        self.assertTrue(torch.allclose(normalized_coordinate_huber(coordinates, target, boxes, valid), expected))
        distances = normalized_coordinate_distances(coordinates, target, boxes, valid)
        self.assertAlmostEqual(float(distances[0, 0]), 2 ** -.5, places=6)
        target[:, 1] = 1000.0  # Invalid/OOB sidecar joints cannot affect the metric.
        self.assertTrue(torch.allclose(normalized_coordinate_huber(coordinates, target, boxes, valid), expected))

    def test_checkpoint_aggregate_deltas_and_negative_correlation(self):
        external = _score(1500, ("a",), .4)
        baseline = turbo_benchmark._alignment_aggregate(1500, "baseline", [{"critic_loss": .4, "normalized_coordinate_error": .8, "valid_joint_count": 2}], external)
        candidate = turbo_benchmark._alignment_aggregate(1525, "candidate", [{"critic_loss": .2, "normalized_coordinate_error": .4, "valid_joint_count": 3}], _score(1525, ("a",), .6))
        self.assertEqual(baseline["critic_loss"], {"mean": .4, "median": .4})
        self.assertEqual(candidate["total_valid_joints"], 3)
        self.assertAlmostEqual(turbo_benchmark._alignment_delta(candidate["pck_005"], baseline["pck_005"])["absolute"], .2)
        self.assertAlmostEqual(turbo_benchmark._alignment_pearson([.4, .2, .1], [.2, .6, .8]), -1.0)
        correlations = turbo_benchmark._alignment_correlations([baseline, candidate])
        self.assertIn("negative", correlations["expected_direction"])
        self.assertIn("Internal critic improves", turbo_benchmark._alignment_report({"checkpoints": [baseline, candidate], "correlations": correlations}))

    def test_completed_artifacts_include_baseline_and_exclude_phase1_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); branch, baseline = root / "branch", root / "baseline"; stems = ("a", "danbooru")
            resolved = {"steps": [1525, 1550], "baseline": {"output_root": str(baseline), "checkpoint_step": 1500, "label": "baseline"},
                        "labels": {"checkpoint_template": "candidate {step}"},
                        "training_metadata": {"pose_loss": "normalized_coordinate_huber", "pose_loss_temperature": 1.0}}
            _write_artifacts(branch, steps=(1525, 1550), stems=stems, experiment="arbitrary", resolved=resolved)
            _write_artifacts(baseline, steps=(1500,), stems=stems, experiment=None)
            args = SimpleNamespace(output_root=branch, sidecar=root / "sidecar", steps=[1525, 1550], baseline_output_root=None,
                                   baseline_step=None, experiment_name="arbitrary", expected_sidecar_records_sha256=None, device="cpu")
            with patch.object(turbo_benchmark, "load_sidecar", return_value=({"records_sha256": "sidecar"}, [_record("a"), _record("danbooru", available=False)])), \
                 patch.object(turbo_benchmark, "FixedBoxKeypointRCNNCritic", _FakeCritic):
                turbo_benchmark.critic_alignment(args)
            summary = json.loads((branch / "critic_alignment_summary.json").read_text())
            samples = json.loads((branch / "critic_alignment_samples.json").read_text())["samples"]
            self.assertEqual([row["checkpoint_step"] for row in summary["checkpoints"]], [1500, 1525, 1550])
            self.assertEqual(summary["phase1_excluded_stems"], ["danbooru"])
            self.assertEqual(len(samples), 3)
            self.assertTrue((branch / "critic_alignment_report.md").is_file())
            args.expected_sidecar_records_sha256 = "wrong"
            with patch.object(turbo_benchmark, "load_sidecar", return_value=({"records_sha256": "sidecar"}, [_record("a"), _record("danbooru", available=False)])), \
                 patch.object(turbo_benchmark, "FixedBoxKeypointRCNNCritic") as critic:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    turbo_benchmark.critic_alignment(args)
                critic.assert_not_called()

    def test_missing_or_partial_images_and_provenance_fail_closed_before_critic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); branch, baseline = root / "branch", root / "baseline"; stems = ("a",)
            resolved = {"steps": [1525], "baseline": {"output_root": str(baseline), "checkpoint_step": 1500}, "labels": {},
                        "training_metadata": {"pose_loss": "normalized_coordinate_huber", "pose_loss_temperature": 1.0}}
            _write_artifacts(branch, steps=(1525,), stems=stems, experiment="arbitrary", resolved=resolved)
            _write_artifacts(baseline, steps=(1500,), stems=stems, experiment=None)
            args = SimpleNamespace(output_root=branch, sidecar=root / "sidecar", steps=None, baseline_output_root=None,
                                   baseline_step=None, experiment_name=None, expected_sidecar_records_sha256=None, device="cpu")
            (branch / "fixed_pose" / "a" / "step_001525.png").unlink()
            with patch.object(turbo_benchmark, "load_sidecar") as sidecar, patch.object(turbo_benchmark, "FixedBoxKeypointRCNNCritic") as critic:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    turbo_benchmark.critic_alignment(args)
                sidecar.assert_not_called(); critic.assert_not_called()
            _write_artifacts(branch, steps=(1525,), stems=stems, experiment="arbitrary", resolved=resolved)
            generation = json.loads((branch / "generation_results.json").read_text()); generation["checkpoints"] = [1525, 1550]
            (branch / "generation_results.json").write_text(json.dumps(generation))
            with self.assertRaisesRegex(ValueError, "requested checkpoints differ"):
                turbo_benchmark.critic_alignment(args)

    def test_critic_configuration_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); branch = root / "branch"
            (branch).mkdir()
            (branch / "turbo_spec.json").write_text(json.dumps({"kind": "turbo_fixed_pose", "turbo": turbo_metadata(), "control_scale": 1.0,
                "experiment_name": "arbitrary", "stems": ["a"], "resolved_experiment": {"steps": [1525],
                "training_metadata": {"pose_loss": "gaussian_heatmap_kl", "pose_loss_temperature": 1.0}}}))
            args = SimpleNamespace(output_root=branch, sidecar=root / "sidecar", steps=None, baseline_output_root=None, baseline_step=None,
                                   experiment_name=None, expected_sidecar_records_sha256=None, device="cpu")
            with self.assertRaisesRegex(ValueError, "normalized_coordinate_huber"):
                turbo_benchmark.critic_alignment(args)

    def test_critic_alignment_source_has_no_parameter_update_path(self):
        source = Path(turbo_benchmark.__file__).read_text().lower()
        start = source.index("def critic_alignment")
        alignment = source[start:source.index("def parser", start)]
        for forbidden in ("torch.optim", "optimizer.", "backward(", ".backward(", "model.train("):
            self.assertNotIn(forbidden, alignment)


if __name__ == "__main__":
    unittest.main()
