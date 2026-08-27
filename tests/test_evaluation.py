import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pose_controlnet.config import TrainConfig
from pose_controlnet.evaluation import (
    CHECKPOINT_STEPS, evaluate_fixed_pose, fixed_flow_inputs, fixed_flow_loss,
    load_comparison_state, make_contact_sheet, make_evaluation_spec, ordered_checkpoints,
)


class _Dataset:
    def __init__(self):
        self.records = [("x", 0, (4, 4), "alpha"), ("x", 1, (4, 4), "beta")]
        self.text_conditioning = SimpleNamespace(unconditional={"context": torch.ones(2, 1, 1, dtype=torch.bfloat16), "mask": torch.ones(2, dtype=torch.bool)})
        self.samples = {
            stem: {"stem": stem, "latent": torch.full((16, 4, 4), number, dtype=torch.float32),
                   "control": torch.ones(16, 4, 4), "prompt": f"prompt {stem}",
                   "context": torch.ones(2, 1, 1, dtype=torch.bfloat16), "mask": torch.ones(2, dtype=torch.bool)}
            for number, stem in enumerate(("alpha", "beta"), 1)
        }
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.samples[self.records[index][3]]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.first = torch.nn.Linear(1, 1); self.config = SimpleNamespace(patch=2)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.dataset, self.model = _Dataset(), _Model()
        self.cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards")
        self.spec = make_evaluation_spec(self.dataset, split="val", count=2, seed=420100, kind="fixed_flow")

    def test_fixed_flow_inputs_are_checkpoint_independent(self):
        sample = self.dataset[0]
        first = fixed_flow_inputs(sample, self.cfg, self.model, seed=self.spec["seed"], device=torch.device("cpu"))
        second = fixed_flow_inputs(sample, self.cfg, self.model, seed=self.spec["seed"], device=torch.device("cpu"))
        self.assertTrue(torch.equal(first[0], second[0])); self.assertTrue(torch.equal(first[1], second[1]))

    def test_fixed_validation_is_exactly_repeatable_and_restores_eval_mode(self):
        def zero_forward(model, image, control, context, timestep, pos, mask, **kwargs):
            return torch.zeros_like(image)
        self.model.train()
        with patch("pose_controlnet.evaluation.forward_pose_control", side_effect=zero_forward):
            one = fixed_flow_loss(self.model, self.dataset, self.spec, self.cfg, torch.device("cpu"))
            two = fixed_flow_loss(self.model, self.dataset, self.spec, self.cfg, torch.device("cpu"))
        self.assertEqual(one, two); self.assertTrue(self.model.training); self.assertIsNone(self.model.first.weight.grad)

    def test_checkpoint_order_and_embedded_step_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("pose_controlnet.evaluation.load_training_state", side_effect=lambda path: {"global_step": int(path.stem.split("_")[1])}):
                self.assertEqual([step for step, _ in ordered_checkpoints(root)], list(CHECKPOINT_STEPS))
            with patch("pose_controlnet.evaluation.load_training_state", return_value={"global_step": 99}):
                with self.assertRaisesRegex(ValueError, "mismatch"): ordered_checkpoints(root, (20,))

    def test_checkpoint_resolution_uses_two_roots_for_post_100_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            early, late = Path(temp) / "early", Path(temp) / "late"
            with patch("pose_controlnet.evaluation.load_training_state", side_effect=lambda path: {"global_step": int(path.stem.split("_")[1])}):
                resolved = ordered_checkpoints(early, later_checkpoint_dir=late)
        self.assertEqual(resolved[6][1], late / "step_000200.pt")

    def test_missing_later_archives_recover_from_their_exact_hf_namespaces(self):
        with tempfile.TemporaryDirectory() as temp:
            early, late, archive, recovery = Path(temp) / "early", Path(temp) / "late", Path(temp) / "archive", Path(temp) / "recovery"
            def state(path):
                return {"global_step": int(path.stem.split("_")[1])}
            with patch("pose_controlnet.evaluation.load_training_state", side_effect=state), patch(
                "pose_controlnet.evaluation.validated_hf_checkpoint_for_step",
                side_effect=lambda **kwargs: Path(kwargs["download_dir"]) / f"step_{kwargs['step']:06d}.pt",
            ) as fetch:
                resolved = ordered_checkpoints(early, steps=(225, 600), later_checkpoint_dir=late, archive_checkpoint_dir=archive,
                                               hf_repo_id="user/private", hf_recovery_dir=recovery)
        self.assertEqual([path for _, path in resolved], [recovery / "pose-learning-500" / "step_000225.pt",
                                                           recovery / "pose-learning-1500" / "step_000600.pt"])
        self.assertEqual([(call.kwargs["step"], call.kwargs["run_name"], Path(call.kwargs["download_dir"]).name)
                          for call in fetch.call_args_list],
                         [(225, "pose-learning-500", "pose-learning-500"),
                          (600, "pose-learning-1500", "pose-learning-1500")])

    def test_valid_local_mid_checkpoint_is_preferred_over_hf_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            early, mid, archive = Path(temp) / "early", Path(temp) / "mid", Path(temp) / "archive"
            local = mid / "step_000500.pt"; local.parent.mkdir(); local.touch()
            with patch("pose_controlnet.evaluation.load_training_state", return_value={"global_step": 500}), patch(
                "pose_controlnet.evaluation.validated_hf_checkpoint_for_step") as fetch:
                resolved = ordered_checkpoints(early, steps=(500,), later_checkpoint_dir=mid,
                                               archive_checkpoint_dir=archive, hf_repo_id="user/private")
        self.assertEqual(resolved, [(500, local)])
        fetch.assert_not_called()

    def test_baseline_and_checkpoint_loading_use_trainable_state_interface(self):
        with patch("pose_controlnet.evaluation.load_training_state", return_value={"global_step": 20, "model": {"x": torch.tensor(1)}}), patch("pose_controlnet.evaluation.load_trainable_state_dict") as load:
            self.assertEqual(load_comparison_state(self.model, Path("step_000020.pt")), 20)
        load.assert_called_once_with(self.model, {"x": torch.tensor(1)})
        self.assertEqual(load_comparison_state(self.model, None), 0)

    def test_pose_metadata_and_filenames_are_deterministic(self):
        pose_spec = make_evaluation_spec(self.dataset, split="diagnostic_val", count=1, seed=420200, kind="fixed_pose", stems=["alpha"])
        with tempfile.TemporaryDirectory() as temp, patch("pose_controlnet.evaluation.load_trainable_state_dict"), patch("pose_controlnet.evaluation.sample_eval_image", return_value=torch.zeros((4, 4, 3), dtype=torch.uint8).numpy()):
            output = Path(temp); control = output / "source.png"; __import__("PIL").Image.new("RGB", (4, 4)).save(control)
            evaluate_fixed_pose(self.model, self.dataset, pose_spec, self.cfg, torch.device("cpu"), [(0, None), (20, None)], object(), {"alpha": control}, output)
            sample_dir = output / "fixed_pose" / "alpha"
            self.assertTrue((sample_dir / "step_000000.png").is_file()); self.assertTrue((sample_dir / "step_000020.png").is_file())
            metadata = json.loads((sample_dir / "metadata.json").read_text())
            self.assertEqual(metadata["stem"], "alpha"); self.assertEqual(metadata["prompt"], "prompt alpha"); self.assertEqual(metadata["seed"], pose_spec["per_stem_seeds"]["alpha"]["sampling"])

    def test_pose_extension_reuses_an_existing_checkpoint_image(self):
        pose_spec = make_evaluation_spec(self.dataset, split="diagnostic_val", count=1, seed=420200, kind="fixed_pose", stems=["alpha"])
        with tempfile.TemporaryDirectory() as temp, patch("pose_controlnet.evaluation.load_trainable_state_dict"), patch("pose_controlnet.evaluation.sample_eval_image") as sample:
            output = Path(temp); control = output / "source.png"; __import__("PIL").Image.new("RGB", (4, 4)).save(control)
            directory = output / "fixed_pose" / "alpha"; directory.mkdir(parents=True); __import__("PIL").Image.new("RGB", (4, 4)).save(directory / "step_000600.png")
            result = evaluate_fixed_pose(self.model, self.dataset, pose_spec, self.cfg, torch.device("cpu"), [(600, Path("step_000600.pt"))], object(), {"alpha": control}, output)
        sample.assert_not_called(); self.assertEqual(result["reused_steps"], {"alpha": [600]})

    def test_comparison_grid_is_compact_preserves_aspect_ratio_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for index in range(7):
                image = __import__("PIL").Image.new("RGB", (80, 20) if index == 0 else (20, 80), (index + 1, 0, 0))
                item = root / f"image_{index}.png"; image.save(item); paths.append(item)
            first, second = root / "first.png", root / "second.png"
            make_contact_sheet([("alpha", paths)], first, thumbnail_width=100, thumbnail_height=60)
            make_contact_sheet([("alpha", paths)], second, thumbnail_width=100, thumbnail_height=60)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            grid = __import__("PIL").Image.open(first)
            self.assertEqual(grid.size, (700, 84))
            self.assertEqual(grid.getpixel((50, 54)), (1, 0, 0))
            self.assertEqual(grid.getpixel((150, 54)), (2, 0, 0))


if __name__ == "__main__": unittest.main()
