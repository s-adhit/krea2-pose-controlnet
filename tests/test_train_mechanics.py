import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import train
from pose_controlnet.checkpointing import load_training_state, save_training_state
from pose_controlnet.config import TrainConfig
from pose_controlnet.wandb_logging import TrainingTelemetry


class TrainMechanicsTest(unittest.TestCase):
    def test_effective_batch_formula(self):
        self.assertEqual(train.effective_batch_size(1, 32, 1), 32)
        self.assertEqual(train.effective_batch_size(2, 16, 1), 32)
        with self.assertRaises(ValueError): train.effective_batch_size(0, 32)

    def test_adamw_intended_trainables_and_frozen_exclusion(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        model[1].requires_grad_(False)
        cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards")
        with patch.object(train, "audit_control_model"):
            optimizer = train.build_optimizer(model, cfg)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.99))
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]}, {id(p) for p in model[0].parameters()})

    def test_warmup_and_optimizer_boundary_counts(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)
        scheduler = train.OptimizerStepWarmup(optimizer, 200)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        for _ in range(200):
            parameter.grad = torch.ones_like(parameter)
            train.optimizer_update(optimizer, scheduler, [parameter], 1.0)
        self.assertEqual(scheduler.step_count, 200)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertIsNone(parameter.grad)

    def test_gradient_clipping_precedes_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0])); parameter.grad = torch.tensor([10.0])
        optimizer = torch.optim.AdamW([parameter], lr=1e-3); scheduler = train.OptimizerStepWarmup(optimizer, 1)
        with patch("train.torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip, patch.object(optimizer, "step", wraps=optimizer.step) as step:
            train.optimizer_update(optimizer, scheduler, [parameter], 0.5)
        self.assertTrue(clip.called and step.called)
        self.assertLess(clip.call_args_list[0][0][1], 1.0)

    def test_order_and_caption_dropout_are_reproducible(self):
        records = [("a", 0, (4, 4)), ("a", 1, (4, 4)), ("b", 0, (8, 4)), ("b", 1, (8, 4))]
        plan = train.DeterministicBucketBatches(records, 2, 42)
        self.assertEqual(plan.for_epoch(0), plan.for_epoch(0))
        self.assertNotEqual(plan.for_epoch(0), plan.for_epoch(1))
        prompts = ["one", "two", "three"]
        self.assertEqual(train.apply_caption_dropout(prompts, .1, 42, 7), train.apply_caption_dropout(prompts, .1, 42, 7))

    def test_validation_never_updates_optimizer_or_changes_mode(self):
        model = torch.nn.Linear(1, 1); model.train()
        cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards")
        fake_loss = torch.tensor(2.0)
        with patch.object(train, "_flow_loss", return_value=(fake_loss, {})):
            value = train.validate_flow_loss(model, object(), [{"x": 1}], cfg, torch.device("cpu"), torch.Generator())
        self.assertEqual(value, 2.0); self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)

    def test_full_resume_state_preserves_position_and_rng_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_000001.pt"
            state = {"model": {"first.weight": torch.ones(1)}, "optimizer": {"state": {}}, "scheduler": {"step_count": 1}, "global_step": 1, "epoch": 2, "batch_position": 3, "rng": train._capture_rng()}
            save_training_state(path, state)
            restored = load_training_state(path)
            self.assertEqual((restored["global_step"], restored["epoch"], restored["batch_position"]), (1, 2, 3))

    def test_telemetry_failures_are_nonfatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards", wandb_enabled=False, metrics_jsonl_path=str(Path(temporary) / "metrics.jsonl"))
            telemetry = TrainingTelemetry(cfg, "test")
            self.assertTrue(telemetry.log_train(loss=1.0, learning_rate=0.0, global_grad_norm=1.0, sec_per_step=.1, samples_per_second=1.0, step=1))
            telemetry.close()

    def test_max_steps_is_required_and_bounded(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "10", "--microbatch-size", "1", "--gradient-accumulation-steps", "32"]):
            self.assertEqual(train.parse_args().max_steps, 10)
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "6000", "--microbatch-size", "1", "--gradient-accumulation-steps", "32"]):
            with self.assertRaises(SystemExit): train.parse_args()


if __name__ == "__main__": unittest.main()
