import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import train
from pose_controlnet.checkpointing import HFTrainingCheckpointMirror, save_training_state
from pose_controlnet.config import TrainConfig


class TinyControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(2, 2)
        self.block = torch.nn.Module()
        self.block.A = torch.nn.Parameter(torch.ones(2, 2))
        self.block.B = torch.nn.Parameter(torch.ones(2, 2))


class ControlInputLr2xContinuationTest(unittest.TestCase):
    def source_state(self) -> tuple[TinyControlModel, dict]:
        model = TinyControlModel()
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=5e-5, betas=(.9, .99), eps=1e-8, weight_decay=0.0)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        optimizer.step()
        cfg = TrainConfig(
            raw_ckpt="raw", shard_dir="shards", ckpt_dir=train.LR_BRANCH_CHECKPOINT_DIR,
            run_name=train.CONTROLINPUT_BRANCH_SOURCE_RUN, lr=5e-5,
            max_steps=1500, allow_extended_training=True, save_every=25,
            hf_repo_id=train.LR_BRANCH_SOURCE_HF_REPO, hf_mirror_every_steps=100,
            metrics_jsonl_path=str(Path(train.LR_BRANCH_CHECKPOINT_DIR) / train.CONTROLINPUT_BRANCH_SOURCE_RUN / "metrics.jsonl"),
        )
        return model, {
            "model": {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
            "optimizer": optimizer.state_dict(),
            "scheduler": {"step_count": 1500, "base_lrs": [5e-5], "warmup_steps": 200},
            "global_step": 1500, "epoch": 7, "batch_position": 13,
            "rng": {**train._capture_rng(), "cuda": [torch.Generator().get_state()]},
            "flow_generator_state": torch.Generator().manual_seed(99).get_state(),
            "config": asdict(cfg),
        }

    def branch_config(self, source: dict) -> TrainConfig:
        return train.controlinput_branch_config_from_source_state(source)

    def test_exact_source_and_one_variable_lr_metadata(self):
        _, source = self.source_state()
        cfg = self.branch_config(source)
        self.assertEqual(cfg.run_name, "pose-learning-1500-controlinput-lr2x-to2800")
        self.assertEqual(cfg.source_step, 1500)
        self.assertEqual(cfg.target_step, 2800)
        self.assertEqual(cfg.lr, 5e-5)
        self.assertEqual(cfg.control_input_lr, 1e-4)
        self.assertEqual(cfg.control_input_lr_multiplier, 2.0)
        self.assertEqual(cfg.timestep_aux_prob, 0.0)
        self.assertEqual((cfg.timestep_aux_min, cfg.timestep_aux_max), (0.0, 1.0))
        self.assertEqual(cfg.control_dropout, 0.0)
        self.assertEqual(cfg.caption_dropout, 0.1)

    def test_optimizer_groups_are_exact_disjoint_and_complete(self):
        model, source = self.source_state()
        cfg = self.branch_config(source)
        with patch.object(train, "audit_control_model"):
            optimizer = train.build_optimizer(model, cfg)
        self.assertEqual([group["group_name"] for group in optimizer.param_groups], ["lora", "control_input"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [5e-5, 1e-4])
        self.assertEqual([group["betas"] for group in optimizer.param_groups], [(0.9, 0.99), (0.9, 0.99)])
        self.assertEqual([group["eps"] for group in optimizer.param_groups], [1e-8, 1e-8])
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.0, 0.0])
        assigned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(assigned), len({id(parameter) for parameter in assigned}))
        self.assertEqual({id(parameter) for parameter in assigned}, {id(parameter) for parameter in model.parameters()})
        self.assertTrue(all(name.endswith((".A", ".B")) for name, _ in train._controlinput_named_groups(model.named_parameters())[0]))

    def test_optimizer_migration_preserves_moments_steps_scheduler_rng_and_flow_generator(self):
        _, source = self.source_state()
        model = TinyControlModel()
        cfg = self.branch_config(source)
        with patch.object(train, "audit_control_model"), patch.object(train, "load_trainable_state_dict", wraps=lambda target, state: target.load_state_dict(state)):
            optimizer = train.build_optimizer(model, cfg)
            scheduler = train.OptimizerStepWarmup(optimizer, 200)
            random.seed(12); np.random.seed(12); torch.manual_seed(12)
            resumed = train._restore_controlinput_source_state(model, optimizer, scheduler, source)
        self.assertEqual(resumed[:3], (1500, 7, 13))
        self.assertTrue(torch.equal(resumed[3], source["flow_generator_state"]))
        self.assertEqual(scheduler.step_count, 1500)
        self.assertEqual(scheduler.base_lrs, [5e-5, 1e-4])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [5e-5, 1e-4])
        source_ids = source["optimizer"]["param_groups"][0]["params"]
        for parameter, source_id in zip(model.parameters(), source_ids):
            self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"], source["optimizer"]["state"][source_id]["exp_avg"]))
            self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg_sq"], source["optimizer"]["state"][source_id]["exp_avg_sq"]))
            self.assertEqual(float(optimizer.state[parameter]["step"]), float(source["optimizer"]["state"][source_id]["step"]))

    def test_recovery_rejects_another_run_and_accepts_only_its_own_configuration(self):
        _, source = self.source_state()
        cfg = self.branch_config(source)
        model = TinyControlModel()
        with patch.object(train, "audit_control_model"), patch.object(train, "load_trainable_state_dict", wraps=lambda target, state: target.load_state_dict(state)):
            optimizer = train.build_optimizer(model, cfg)
            scheduler = train.OptimizerStepWarmup(optimizer, 200)
            train._restore_controlinput_source_state(model, optimizer, scheduler, source)
        state = dict(source)
        state.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "global_step": 1600,
            "config": asdict(cfg), "model": {name: value.clone() for name, value in source["model"].items()},
        })
        state["scheduler"]["step_count"] = 1600
        self.assertEqual(train.validate_controlinput_branch_recovery_state(Path("step_001600.pt"), state, source).run_name, cfg.run_name)
        wrong = dict(state); wrong["config"] = dict(state["config"], run_name="another-run")
        with self.assertRaisesRegex(ValueError, "config"):
            train.validate_controlinput_branch_recovery_state(Path("step_001600.pt"), wrong, source)

    def test_milestones_and_local_retention_are_exact_and_protected(self):
        self.assertEqual(train.CONTROLINPUT_BRANCH_REQUIRED_CHECKPOINT_STEPS, (1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, source = self.source_state()
            for step in (1600, 1700, 1800):
                state = dict(source); state["global_step"] = step; state["scheduler"] = dict(source["scheduler"], step_count=step)
                save_training_state(root / f"step_{step:06d}.pt", state)
            mirror = HFTrainingCheckpointMirror(repo_id="user/private", run_name="run", protected_milestone_steps=(1600, 1700, 1800))
            mirror.prune_local(root)
            self.assertEqual(sorted(path.name for path in root.glob("step_*.pt")), ["step_001600.pt", "step_001700.pt", "step_001800.pt"])

    def test_cli_preflight_and_source_validation_are_read_only_selectors(self):
        with patch.object(__import__("sys"), "argv", ["train.py", "--controlinput-lr2x-1500-to2800"]):
            self.assertTrue(train.parse_args().controlinput_lr2x_1500_to2800)
        with patch.object(__import__("sys"), "argv", ["train.py", "--recover-controlinput-lr2x-1500-to2800", "--resume", "other.pt"]):
            with self.assertRaises(SystemExit):
                train.parse_args()
        source_file = Path("/tmp/not-the-real-source.pt")
        with patch.object(train, "CONTROLINPUT_BRANCH_SOURCE_CHECKPOINT", source_file):
            with self.assertRaises(FileNotFoundError):
                train.resolve_controlinput_branch_source_checkpoint()
        script = Path("scripts/preflight_controlinput_lr2x.py").read_text(encoding="utf-8")
        self.assertIn("preflight_starts_training", script)
        self.assertNotIn("train.main(", script)


if __name__ == "__main__":
    unittest.main()
