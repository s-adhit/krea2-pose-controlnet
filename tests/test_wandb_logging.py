"""Targeted tests for failure-isolated experiment telemetry."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pose_controlnet.wandb_logging import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    TrainingTelemetry,
)


@dataclass
class LoggingConfig:
    wandb_enabled: bool = True
    wandb_entity: str = DEFAULT_WANDB_ENTITY
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_mode: str = "online"
    metrics_jsonl_path: str = "unused.jsonl"
    api_key: str = "must-not-be-serialized"


class FakeRun:
    def __init__(self, fail_log: bool = False) -> None:
        self.fail_log = fail_log
        self.records: list[tuple[dict, int | None]] = []
        self.finished = False

    def log(self, metrics, step=None):
        if self.fail_log:
            raise OSError("simulated network outage")
        self.records.append((metrics, step))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self, *, fail_init: bool = False, fail_log: bool = False) -> None:
        self.fail_init = fail_init
        self.run = FakeRun(fail_log)
        self.kwargs = None

    def init(self, **kwargs):
        self.kwargs = kwargs
        if self.fail_init:
            raise ConnectionError("simulated init outage")
        return self.run


class WandbLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.metrics_path = Path(self.tempdir.name) / "metrics.jsonl"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_logger(self, fake, **config_overrides):
        cfg = LoggingConfig(**config_overrides)
        return TrainingTelemetry(cfg, "test-run", metrics_path=self.metrics_path, wandb_module=fake)

    def test_successful_initialization_uses_project_defaults_and_safe_config(self) -> None:
        fake = FakeWandb()
        logger = self.make_logger(fake)
        self.assertIs(logger.run, fake.run)
        self.assertEqual(fake.kwargs["entity"], DEFAULT_WANDB_ENTITY)
        self.assertEqual(fake.kwargs["project"], DEFAULT_WANDB_PROJECT)
        self.assertNotIn("api_key", fake.kwargs["config"])
        logger.close()

    def test_init_failure_is_nonfatal_and_local_metrics_continue(self) -> None:
        logger = self.make_logger(FakeWandb(fail_init=True))
        self.assertIsNone(logger.run)
        self.assertTrue(logger.log({"train/loss": 0.25}, step=3))
        self.assertTrue(logger.wandb_errors)
        logger.close()

    def test_logging_failure_is_nonfatal_and_local_metrics_continue(self) -> None:
        logger = self.make_logger(FakeWandb(fail_log=True))
        self.assertTrue(logger.log_train(loss=0.25, learning_rate=1e-4, global_grad_norm=0.5,
                                         sec_per_step=2.0, samples_per_second=16.0, step=3))
        self.assertTrue(logger.wandb_errors)
        logger.close()

    def test_jsonl_fallback_writes_named_metric_interfaces(self) -> None:
        logger = self.make_logger(FakeWandb())
        logger.log_validation_flow_loss(0.1, step=4)
        logger.log_control_diagnostics(control_latent_rms=1.0, control_latent_std=0.2,
                                       control_input_grad_norms={"control_half": 0.3},
                                       lora_grad_norms={"q_proj": 0.4}, step=4)
        logger.log_cuda_memory(allocated_bytes=1, reserved_bytes=2, peak_allocated_bytes=3, step=4)
        logger.log_checkpoint(checkpoint_step=4, checkpoint_time="2026-08-26T00:00:00Z", step=4)
        logger.log_hf_upload(success=True, remote_checkpoint_age_seconds=60.0, step=4,
                             uploaded_checkpoint_step=4, error_status=None)
        logger.close()
        rows = [json.loads(line) for line in self.metrics_path.read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["validation/flow_loss"], 0.1)
        self.assertEqual(rows[1]["diagnostics/control_input_grad_norm/control_half"], 0.3)
        self.assertEqual(rows[2]["cuda/peak_allocated_bytes"], 3)
        self.assertTrue(rows[4]["hf/upload_success"])
        self.assertEqual(rows[4]["hf/uploaded_checkpoint_step"], 4)

    def test_disabled_and_offline_modes(self) -> None:
        disabled = FakeWandb()
        logger = self.make_logger(disabled, wandb_enabled=False)
        self.assertIsNone(logger.run)
        self.assertIsNone(disabled.kwargs)
        logger.close()
        offline = FakeWandb()
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False):
            logger = self.make_logger(offline)
        self.assertEqual(offline.kwargs["mode"], "offline")
        logger.close()

    def test_environment_overrides_entity_and_project(self) -> None:
        fake = FakeWandb()
        with patch.dict(os.environ, {"WANDB_ENTITY": "override-entity",
                                     "WANDB_PROJECT": "override-project"}, clear=False):
            logger = self.make_logger(fake)
        self.assertEqual(fake.kwargs["entity"], "override-entity")
        self.assertEqual(fake.kwargs["project"], "override-project")
        logger.close()

    def test_credentials_are_not_written_to_project_telemetry(self) -> None:
        logger = self.make_logger(FakeWandb())
        logger.log({"safe": 1}, step=1)
        logger.close()
        self.assertNotIn("must-not-be-serialized", self.metrics_path.read_text())


if __name__ == "__main__":
    unittest.main()
