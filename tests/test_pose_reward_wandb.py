"""No-network contracts for optional Gate-E W&B mirroring."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pose_controlnet.checkpointing import load_training_state, save_training_state
from pose_controlnet.config import TrainConfig
from pose_controlnet.wandb_logging import OptionalWandbMirror
from scripts.train_pose_reward_smoke import (
    _gate_e_metadata,
    gate_e_wandb_config,
    gate_e_wandb_run_id,
    gate_e_wandb_step_metrics,
)
from pose_controlnet.pose_consistency import aggregate_step_diagnostics, update_cumulative_counters


class FakeRun:
    def __init__(self, *, run_id: str = "new-run", fail_log: bool = False) -> None:
        self.id = run_id
        self.fail_log = fail_log
        self.records: list[tuple[dict, int]] = []
        self.finished = False

    def log(self, metrics, *, step):
        if self.fail_log:
            raise OSError("simulated W&B outage")
        self.records.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self, *, fail_init: bool = False, fail_log: bool = False) -> None:
        self.fail_init = fail_init
        self.run = FakeRun(fail_log=fail_log)
        self.kwargs = None

    def init(self, **kwargs):
        self.kwargs = kwargs
        if self.fail_init:
            raise ConnectionError("simulated W&B init outage")
        return self.run


class PoseRewardWandbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TrainConfig(raw_ckpt="/models/raw.safetensors", shard_dir="/latents",
                               run_name="pose-reward-kl-exposure5pct-l2e5-t010-020",
                               microbatch_size=1, gradient_accumulation_steps=32,
                               max_steps=1700, save_every=50,
                               hf_repo_id="owner/private-checkpoints")
        self.parent = {"global_step": 1500, "sha256": "parent-sha", "filename": "step_001500.pt"}

    def config(self) -> dict:
        return gate_e_wandb_config(
            cfg=self.cfg, immutable_parent=self.parent, pose_loss="normalized_coordinate_huber", lambda_pose=2e-5,
            timestep_min=.10, timestep_max=.20, forced_exposure_probability=.05,
            target_global_step=1700, hf_subdir=self.cfg.run_name,
            sidecar_metadata={"records_sha256": "sidecar-sha"},
        )

    def step_metrics(self) -> dict:
        metrics = aggregate_step_diagnostics([{
            "flow_loss": 1.0, "pose_loss": None, "total_loss": 1.0,
            "pose_active_count": 0, "pose_eligible_count": 1,
            "pose_forced_count": 0, "pose_natural_active_count": 0,
            "timesteps": [.15], "active_timesteps": [],
        }])
        metrics.update({"global_step": 1501, "global_grad_norm": .5, "sec_per_step": 2.0})
        metrics["pose_cumulative_counters"] = update_cumulative_counters(
            {"eligible_samples_seen": 0, "forced_samples": 0,
             "naturally_active_samples": 0, "total_active_samples": 0}, metrics)
        return metrics

    def test_no_initialization_without_project(self) -> None:
        fake = FakeWandb()
        mirror = OptionalWandbMirror(project=None, run_name="run", config=self.config(), wandb_module=fake)
        self.assertFalse(mirror.enabled)
        self.assertIsNone(fake.kwargs)

    def test_initialization_config_and_aggregated_metrics(self) -> None:
        fake = FakeWandb()
        mirror = OptionalWandbMirror(
            project="krea2-pose-controlnet", entity="team", run_name="visible-run",
            group="gate-e", tags=["pose", "kl"], config=self.config(), wandb_module=fake,
        )
        self.assertTrue(mirror.enabled)
        self.assertEqual(fake.kwargs["project"], "krea2-pose-controlnet")
        self.assertEqual(fake.kwargs["name"], "visible-run")
        self.assertEqual(fake.kwargs["config"]["parent_checkpoint"], self.parent)
        self.assertEqual(fake.kwargs["config"]["model_base"], "Krea-2 Raw")
        self.assertEqual(fake.kwargs["config"]["effective_batch_size"], 32)
        self.assertEqual(fake.kwargs["config"]["canonical_sidecar_records_sha256"], "sidecar-sha")
        self.assertEqual(fake.kwargs["config"]["pose_loss"], "normalized_coordinate_huber")
        metrics = gate_e_wandb_step_metrics(self.step_metrics())
        mirror.log(metrics, step=1501)
        logged, step = fake.run.records[0]
        self.assertEqual(step, 1501)
        for key in ("flow_loss_mean_step", "total_loss_mean_step", "global_grad_norm", "sec_per_step",
                    "pose_loss_mean_active", "pose_loss_max_active", "pose_eligible_samples_step",
                    "pose_forced_samples_step", "pose_natural_active_samples_step", "pose_active_samples_step",
                    "pose_active_microbatches_step", "pose_forced_fraction_of_eligible_step",
                    "pose_total_active_fraction_of_eligible_step", "timestep_min_step", "timestep_mean_step",
                    "timestep_max_step", "active_timestep_min_step", "active_timestep_mean_step",
                    "active_timestep_max_step", "cumulative_eligible", "cumulative_forced",
                    "cumulative_natural_active", "cumulative_total_active"):
            self.assertIn(key, logged)
        self.assertIsNone(logged["pose_loss_mean_active"])
        self.assertIsNone(logged["active_timestep_mean_step"])
        mirror.close()
        self.assertTrue(fake.run.finished)

    def test_wandb_failure_leaves_jsonl_and_checkpoint_path_usable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics_path = root / "metrics.jsonl"
            local_metrics = self.step_metrics()
            metrics_path.write_text(json.dumps(local_metrics, sort_keys=True) + "\n", encoding="utf-8")
            mirror = OptionalWandbMirror(project="project", run_name="run", config=self.config(),
                                         wandb_module=FakeWandb(fail_log=True))
            mirror.log(gate_e_wandb_step_metrics(local_metrics), step=1501)
            self.assertFalse(mirror.enabled)
            self.assertEqual(json.loads(metrics_path.read_text()), local_metrics)
            checkpoint = root / "step_001501.pt"
            save_training_state(checkpoint, {"model": {"first.weight": torch.ones(1)}, "optimizer": {},
                                             "scheduler": {"step_count": 1501}, "global_step": 1501, "epoch": 0,
                                             "batch_position": 0,
                                             "rng": {"python": None, "numpy": None,
                                                     "torch": torch.get_rng_state(), "cuda": None},
                                             "flow_generator_state": torch.Generator().get_state(),
                                             "config": asdict(self.cfg)})
            self.assertEqual(load_training_state(checkpoint)["global_step"], 1501)

    def test_checkpoint_run_id_resume_and_legacy_metadata(self) -> None:
        metadata = _gate_e_metadata(
            self.cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5, timestep_min=.10, timestep_max=.20,
            forced_exposure_probability=.05, hf_subdir=self.cfg.run_name,
            immutable_parent=self.parent,
            cumulative_counters={"eligible_samples_seen": 0, "forced_samples": 0,
                                 "naturally_active_samples": 0, "total_active_samples": 0},
            model_state={"first.weight": torch.ones(1)}, wandb_run_id="original-run-id",
        )
        self.assertEqual(gate_e_wandb_run_id({"gate_e": metadata}), "original-run-id")
        fake = FakeWandb()
        mirror = OptionalWandbMirror(project="project", run_name="continuation", config=self.config(),
                                     resume_run_id=gate_e_wandb_run_id({"gate_e": metadata}), wandb_module=fake)
        self.assertEqual(mirror.run_id, "new-run")
        self.assertEqual(fake.kwargs["id"], "original-run-id")
        self.assertEqual(fake.kwargs["resume"], "allow")
        self.assertIsNone(gate_e_wandb_run_id({"gate_e": {"format": 2}}))

    def test_config_and_metadata_do_not_serialize_secrets(self) -> None:
        metadata = _gate_e_metadata(
            self.cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5, timestep_min=.10, timestep_max=.20,
            forced_exposure_probability=.05, hf_subdir=self.cfg.run_name,
            immutable_parent=self.parent,
            cumulative_counters={"eligible_samples_seen": 0, "forced_samples": 0,
                                 "naturally_active_samples": 0, "total_active_samples": 0},
            model_state={"first.weight": torch.ones(1)}, wandb_run_id="safe-id",
        )
        serialized = json.dumps({"config": self.config(), "metadata": metadata}, sort_keys=True)
        self.assertNotIn("hf_token", serialized)
        self.assertNotIn("wandb_api_key", serialized)
        self.assertNotIn("secret", serialized.lower())


if __name__ == "__main__":
    unittest.main()
