import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import train
from pose_controlnet.checkpointing import load_training_state, save_training_state
from pose_controlnet.config import TrainConfig
from pose_controlnet.diffusion import checkpointed_main_block_indices, forward_pose_control
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

    def test_diagnostic_gradients_are_captured_before_optimizer_clears_them(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(4, 2, bias=False)
                self.adapter = torch.nn.Module()
                self.adapter.A = torch.nn.Parameter(torch.ones(2, 2))
                self.adapter.B = torch.nn.Parameter(torch.ones(2, 2))

        model = Model()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = train.OptimizerStepWarmup(optimizer, 1)
        captured = {}

        def capture() -> None:
            captured["control"], captured["lora"] = train._diagnostic_grad_norms(model)

        train.optimizer_update(optimizer, scheduler, list(model.parameters()), 100.0, before_step=capture)

        self.assertGreater(captured["control"]["full"], 0.0)
        self.assertGreater(captured["control"]["control_half"], 0.0)
        self.assertEqual(set(captured["lora"]), {"adapter.A", "adapter.B"})
        self.assertTrue(all(norm > 0.0 for norm in captured["lora"].values()))
        cleared_control, cleared_lora = train._diagnostic_grad_norms(model)
        self.assertEqual(cleared_control, {"full": 0.0, "control_half": 0.0})
        self.assertEqual(cleared_lora, {})

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

    def test_runtime_defaults_disable_compile_and_gradient_checkpointing(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "1", "--microbatch-size", "1", "--gradient-accumulation-steps", "32"]):
            cfg = train.config_from_args(train.parse_args())
        self.assertFalse(cfg.compile)
        self.assertFalse(cfg.gradient_checkpointing)
        self.assertEqual(cfg.gradient_checkpointing_blocks, 0)

    def test_runtime_flags_propagate_to_config(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "1", "--microbatch-size", "1", "--gradient-accumulation-steps", "32", "--compile", "--gradient-checkpointing"]):
            cfg = train.config_from_args(train.parse_args())
        self.assertTrue(cfg.compile)
        self.assertTrue(cfg.gradient_checkpointing)
        self.assertEqual(cfg.gradient_checkpointing_blocks, 28)

        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "1", "--microbatch-size", "1", "--gradient-accumulation-steps", "32", "--compile", "--no-compile", "--gradient-checkpointing", "--no-gradient-checkpointing"]):
            cfg = train.config_from_args(train.parse_args())
        self.assertFalse(cfg.compile)
        self.assertFalse(cfg.gradient_checkpointing)
        self.assertEqual(cfg.gradient_checkpointing_blocks, 0)

    def test_selective_gradient_checkpointing_cli_propagates(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "1", "--microbatch-size", "2", "--gradient-accumulation-steps", "16", "--gradient-checkpointing-blocks", "8"]):
            cfg = train.config_from_args(train.parse_args())
        self.assertEqual(cfg.gradient_checkpointing_blocks, 8)
        self.assertTrue(cfg.gradient_checkpointing)

    def test_invalid_gradient_checkpointing_block_count_rejected(self):
        for count in ("-1", "29"):
            with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "1", "--microbatch-size", "1", "--gradient-accumulation-steps", "32", "--gradient-checkpointing-blocks", count]):
                with self.assertRaises(SystemExit): train.parse_args()
        with self.assertRaises(ValueError):
            checkpointed_main_block_indices(28, 29)

    def test_selective_checkpointing_uses_exact_prefix_and_preserves_forward_shape(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, combined, t_vec, freqs, full_mask):
                self.calls += 1
                return combined + 1

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(2, 1, bias=False)
                self.first.weight.data.fill_(1)
                self.tmlp = torch.nn.Identity()
                self.tproj = torch.nn.Identity()
                self.txtfusion = type("TextFusion", (torch.nn.Module,), {
                    "forward": lambda self, context, mask: context
                })()
                self.txtmlp = torch.nn.Identity()
                self.posemb = torch.nn.Identity()
                self.blocks = torch.nn.ModuleList([Block() for _ in range(4)])
                self.config = type("Config", (), {"tdim": 1})()

            def last(self, combined, t_raw):
                return combined

        model = Model()
        noisy = pose = context = torch.ones(1, 1, 1)
        t = torch.ones(1)
        pos = torch.zeros(1, 2, 3)
        mask = torch.ones(1, 2, dtype=torch.bool)
        checkpointed_blocks = []
        def run_checkpoint(function, *args, **kwargs):
            checkpointed_blocks.append(function)
            return function(*args)

        with patch("pose_controlnet.diffusion.checkpoint", side_effect=run_checkpoint) as checkpoint_mock:
            output = forward_pose_control(model, noisy, pose, context, t, pos, mask, gradient_checkpointing_blocks=2)
        self.assertEqual(checkpoint_mock.call_count, 2)
        self.assertEqual(checkpointed_blocks, list(model.blocks[:2]))
        self.assertEqual([block.calls for block in model.blocks], [1, 1, 1, 1])
        self.assertEqual(output.shape, (1, 1, 1))
        self.assertTrue(torch.equal(output, torch.full((1, 1, 1), 6.0)))

        with patch("pose_controlnet.diffusion.checkpoint") as checkpoint_mock:
            output_zero = forward_pose_control(model, noisy, pose, context, t, pos, mask, gradient_checkpointing_blocks=0)
        checkpoint_mock.assert_not_called()
        self.assertEqual(output_zero.shape, output.shape)
        self.assertTrue(torch.equal(output_zero, output))

    def test_no_compile_runtime_leaves_text_mlp_unwrapped(self):
        model = torch.nn.Module()
        model.txtmlp = torch.nn.Linear(1, 1)
        original_forward = model.txtmlp.forward
        with patch("train.torch.compile") as compile_mock:
            train.configure_runtime(model, compile_enabled=False)
        compile_mock.assert_not_called()
        self.assertIs(model.txtmlp.forward.__self__, original_forward.__self__)

    def test_flow_loss_propagates_no_gradient_checkpointing(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"patch": 1})()

        model = Model()
        cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards", gradient_checkpointing_blocks=0)
        batch = {"latent": torch.ones(1, 1, 2, 2), "control": torch.ones(1, 1, 2, 2), "prompts": ["pose"]}
        context = torch.ones(1, 1, 1, 1)
        with patch("train.sample_flow_timestep", return_value=torch.tensor([0.5])), \
             patch("train.make_flow_pair", side_effect=lambda clean, noise, timestep: (clean, clean)), \
             patch("train.patchify_and_position", side_effect=[(torch.ones(1, 4, 1), torch.zeros(1, 5, 3), torch.ones(1, 5, dtype=torch.bool)), (torch.ones(1, 4, 1), None, None), (torch.ones(1, 4, 1), None, None)]), \
             patch("train.forward_pose_control", return_value=torch.ones(1, 4, 1)) as forward:
            loss, _ = train._flow_loss(model, lambda prompts: (context, torch.ones(1, 1, dtype=torch.bool)), batch, cfg, torch.device("cpu"), torch.Generator(), gradient_checkpointing_blocks=cfg.gradient_checkpointing_blocks)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(forward.call_args.kwargs["gradient_checkpointing_blocks"], 0)


if __name__ == "__main__": unittest.main()
