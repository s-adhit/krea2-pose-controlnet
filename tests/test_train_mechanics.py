import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import train
from pose_controlnet.checkpointing import (
    HFTrainingCheckpointMirror, load_training_state, resolve_auto_resume,
    save_training_state,
)
from pose_controlnet.config import TrainConfig
from pose_controlnet.diffusion import checkpointed_main_block_indices, forward_pose_control
from pose_controlnet.wandb_logging import TrainingTelemetry


class TrainMechanicsTest(unittest.TestCase):
    def full_state(self, step: int = 1) -> dict:
        return {"model": {"first.weight": torch.ones(1)}, "optimizer": {"state": {}},
                "scheduler": {"step_count": step, "base_lrs": [1e-4], "warmup_steps": 200},
                "global_step": step, "epoch": 2, "batch_position": 3,
                "rng": train._capture_rng(), "flow_generator_state": torch.Generator().get_state(),
                "config": {"seed": 42}}
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

    def test_warmup_uses_lr_for_current_optimizer_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)
        scheduler = train.OptimizerStepWarmup(optimizer, 200)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-7)
        used = []
        for _ in range(200):
            parameter.grad = torch.ones_like(parameter)
            used.append(optimizer.param_groups[0]["lr"])
            train.optimizer_update(optimizer, scheduler, [parameter], 1.0)
        self.assertEqual(scheduler.step_count, 200)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(used[0], 5e-7)
        self.assertAlmostEqual(used[9], 5e-6)
        self.assertAlmostEqual(used[199], 1e-4)
        self.assertIsNone(parameter.grad)

    def test_resumed_warmup_installs_next_update_lr(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0])); optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = train.OptimizerStepWarmup(optimizer, 200)
        scheduler.load_state_dict({"step_count": 10, "base_lrs": [1e-4], "warmup_steps": 200})
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5.5e-6)

    def test_telemetry_lr_matches_lr_used_by_optimizer_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0])); optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = train.OptimizerStepWarmup(optimizer, 200); used = scheduler.current_update_learning_rates[0]
        parameter.grad = torch.ones_like(parameter); observed = []
        original_step = optimizer.step
        with patch.object(optimizer, "step", side_effect=lambda: (observed.append(optimizer.param_groups[0]["lr"]), original_step())[1]):
            train.optimizer_update(optimizer, scheduler, [parameter], 1.0)
        self.assertAlmostEqual(used, 5e-7)
        self.assertEqual(observed, [used])
        self.assertAlmostEqual(scheduler.current_update_learning_rates[0], 1e-6)

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
            state = self.full_state()
            save_training_state(path, state)
            restored = load_training_state(path)
            self.assertEqual((restored["global_step"], restored["epoch"], restored["batch_position"]), (1, 2, 3))

    def test_hf_mirror_failure_is_nonfatal_and_retries(self):
        class FailingApi:
            def __init__(self): self.uploads = 0
            def create_repo(self, *args, **kwargs): pass
            def upload_file(self, **kwargs): self.uploads += 1; raise OSError("offline")
        with tempfile.TemporaryDirectory() as temporary:
            path = save_training_state(Path(temporary) / "step_000001.pt", self.full_state())
            api, sleeps = FailingApi(), []
            mirror = HFTrainingCheckpointMirror(repo_id="user/private", run_name="run", max_attempts=3,
                                                api=api, sleep=sleeps.append)
            self.assertFalse(mirror._upload(path)); self.assertEqual(api.uploads, 3)
            self.assertEqual(sleeps, [5.0, 10.0]); self.assertIn("offline", mirror.last_error)

    def test_hf_mirror_and_auto_recovery_prefer_local_then_valid_remote(self):
        class MemoryApi:
            def __init__(self): self.files = {}
            def create_repo(self, *args, **kwargs): pass
            def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
                self.files[path_in_repo] = (Path(path_or_fileobj).read_bytes() if isinstance(path_or_fileobj, str)
                                            else path_or_fileobj.read())
            def list_repo_files(self, *args, **kwargs): return list(self.files)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run_dir = root / "run"; local = save_training_state(run_dir / "step_000005.pt", self.full_state(5))
            api = MemoryApi(); mirror = HFTrainingCheckpointMirror(repo_id="user/private", run_name="run", api=api)
            self.assertTrue(mirror._upload(local)); self.assertIn("run/full/step_000005.pt.complete.json", api.files)
            remote = root / "remote.pt"; remote.write_bytes(local.read_bytes())
            with patch("pose_controlnet.checkpointing.newest_valid_hf_checkpoint", return_value=remote):
                self.assertEqual(resolve_auto_resume(checkpoint_dir=root, run_name="run", repo_id="user/private", remote_download_dir=root / "recovery"), local)
                shutil.rmtree(run_dir)
                recovered = resolve_auto_resume(checkpoint_dir=root, run_name="run", repo_id="user/private", remote_download_dir=root / "recovery")
            restored = load_training_state(recovered)
            self.assertEqual(restored["global_step"], 5); self.assertEqual(restored["scheduler"]["step_count"], 5)
            self.assertEqual(restored["epoch"], 2); self.assertEqual(restored["batch_position"], 3)

    def test_corrupt_newest_hf_candidate_is_skipped(self):
        class Api:
            def __init__(self): self.files = {}
            def list_repo_files(self, *args, **kwargs): return list(self.files)
        with tempfile.TemporaryDirectory() as temporary:
            root, api = Path(temporary), Api()
            valid = save_training_state(root / "step_000005.pt", self.full_state(5))
            api.files["run/full/step_000005.pt"] = valid.read_bytes()
            from pose_controlnet.checkpointing import _sha256
            api.files["run/full/step_000005.pt.complete.json"] = json.dumps({"checkpoint": "run/full/step_000005.pt", "sha256": _sha256(valid), "global_step": 5}).encode()
            api.files["run/full/step_000010.pt"] = b"corrupt"
            api.files["run/full/step_000010.pt.complete.json"] = json.dumps({"checkpoint": "run/full/step_000010.pt", "sha256": __import__("hashlib").sha256(b"corrupt").hexdigest(), "global_step": 10}).encode()
            def download(**kwargs):
                destination = Path(kwargs["local_dir"]) / kwargs["filename"]; destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(api.files[kwargs["filename"]]); return str(destination)
            from pose_controlnet.checkpointing import newest_valid_hf_checkpoint
            recovered = newest_valid_hf_checkpoint(repo_id="user/private", run_name="run", download_dir=root / "dl", api=api, download_fn=download)
            self.assertEqual(load_training_state(recovered)["global_step"], 5)

    def test_telemetry_failures_are_nonfatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards", wandb_enabled=False, metrics_jsonl_path=str(Path(temporary) / "metrics.jsonl"))
            telemetry = TrainingTelemetry(cfg, "test")
            self.assertTrue(telemetry.log_train(loss=1.0, learning_rate=0.0, global_grad_norm=1.0, sec_per_step=.1, samples_per_second=1.0, step=1))
            telemetry.close()

    def test_max_steps_100_works_without_extended_training_opt_in(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "100", "--microbatch-size", "1", "--gradient-accumulation-steps", "32"]):
            args = train.parse_args()
        self.assertEqual(args.max_steps, 100)
        self.assertFalse(args.allow_extended_training)

    def test_max_steps_101_is_rejected_without_extended_training_opt_in(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "101", "--microbatch-size", "1", "--gradient-accumulation-steps", "32"]):
            with self.assertRaises(SystemExit):
                train.parse_args()

    def test_max_steps_500_works_with_explicit_extended_training_opt_in(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "500", "--allow-extended-training", "--microbatch-size", "1", "--gradient-accumulation-steps", "32"]):
            cfg = train.config_from_args(train.parse_args())
        self.assertEqual(cfg.max_steps, 500)
        self.assertTrue(cfg.allow_extended_training)

    def test_resume_from_step_100_to_step_500_is_accepted_with_opt_in(self):
        with patch.object(sys, "argv", ["train.py", "--run-name", "x", "--max-steps", "500", "--allow-extended-training", "--resume", "step_000100.pt", "--microbatch-size", "2", "--gradient-accumulation-steps", "16"]):
            args = train.parse_args()
        self.assertEqual(args.resume, "step_000100.pt")
        self.assertEqual(args.max_steps, 500)
        self.assertTrue(args.allow_extended_training)

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
