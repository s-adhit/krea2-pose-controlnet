import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import train
from pose_controlnet.checkpointing import save_training_state
from pose_controlnet.config import TrainConfig
from scripts import train_overfit_capacity as harness
from scripts import evaluate_overfit_capacity as evaluator

from pose_controlnet.overfit_capacity import (
    MIXED_COMPOSITION, OVERFIT_ACCUMULATION, OVERFIT_CHECKPOINT_STEPS, OVERFIT_EXPERIMENTS, OVERFIT_LR,
    OVERFIT_MAX_STEPS, OVERFIT_MICROBATCH, OVERFIT_STEPS, OVERFIT_WARMUP, SelectedDeterministicBatches,
    SelectedLatentShardDataset, assert_fresh_initialization, assert_overfit_contract, parameter_audit,
    is_overfit_checkpoint_step, per_step_exposures, should_continue_overfit, validate_all_manifests, validate_manifest,
)
from pose_controlnet.pose_targets import source_for_stem


class _Base:
    def __init__(self, stems):
        self.records = [("unused", i, (64, 64), stem) for i, stem in enumerate(stems)]
        self.text_conditioning = object()
    def __getitem__(self, index): return {"stem": self.records[index][3], "latent": index, "control": index, "prompt": "caption"}


class OverfitCapacityTests(unittest.TestCase):
    def _resume_fixture(self, root: Path, *, step: int = 100) -> tuple[object, TrainConfig, tuple[str, ...], Path, dict]:
        args = harness.parser().parse_args([
            "--experiment", "overfit32-coco-r64-mse", "--checkpoint-root", str(root),
        ])
        cfg = harness.build_config(args)
        checkpoint_dir = root / args.experiment; checkpoint_dir.mkdir(parents=True)
        stems = tuple(f"coco_{index}_1" for index in range(32))
        metadata = {
            "experiment": args.experiment, "checkpoint_dir": str(checkpoint_dir), "stems": list(stems),
            "checkpoint_steps": list(OVERFIT_STEPS), "config": asdict(cfg),
            "objective": "flow_mse_only", "pose_reward_enabled": False, "critic_enabled": False,
            "fresh_lora_checkpoint_loaded": False, "parameter_audit": parameter_audit(),
            "actual_model_audit": {"lora_rank": 64, "lora_target_modules": 224},
        }
        (checkpoint_dir / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        presentations = step * OVERFIT_ACCUMULATION
        state = {
            "model": {"first.weight": torch.ones(1)}, "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"step_count": step, "base_lrs": [OVERFIT_LR], "warmup_steps": 0},
            "global_step": step, "epoch": 0 if step == 0 else (presentations - 1) // 32,
            "batch_position": 0 if step == 0 else (presentations - 1) % 32 + 1,
            "rng": train._capture_rng(), "flow_generator_state": torch.Generator().get_state(),
            "config": asdict(cfg), "overfit_capacity": {"fresh_lora_checkpoint_loaded": False, **per_step_exposures(step)},
        }
        path = save_training_state(checkpoint_dir / f"step_{step:06d}.pt", state)
        return args, cfg, stems, path, state

    def test_all_committed_manifests_are_exact_train_subsets_and_mixed_reuses(self):
        values = validate_all_manifests()
        self.assertEqual(set(values), set(OVERFIT_EXPERIMENTS))
        mixed = values["overfit32-mixed-r64-mse"]
        self.assertEqual({source: sum(source_for_stem(stem) == source for stem in mixed) for source in MIXED_COMPOSITION}, MIXED_COMPOSITION)
        for name, source in OVERFIT_EXPERIMENTS.items():
            self.assertEqual(len(values[name]), 32); self.assertEqual(len(set(values[name])), 32)
            if source != "mixed": self.assertTrue({stem for stem in mixed if source_for_stem(stem) == source} <= set(values[name]))

    def test_selected_dataset_and_repeated_shuffles_cannot_escape_32_stems(self):
        stems = tuple(f"coco_{i}_1" for i in range(32)); data = SelectedLatentShardDataset(_Base(stems + ("coco_outside_1",)), stems)
        plan = SelectedDeterministicBatches(data, 1)
        for epoch in range(4):
            batches = plan.for_epoch(epoch)
            self.assertEqual({data[index]["stem"] for batch in batches for index in batch}, set(stems))
        with self.assertRaises(IndexError): data[32]

    def test_fresh_objective_schedule_and_batch_contract_are_locked(self):
        assert_overfit_contract(rank=64, warmup_steps=0, max_steps=500, lr=1e-4, microbatch_size=1, accumulation_steps=8)
        with self.assertRaises(ValueError): assert_fresh_initialization(resume="trained.pt")
        with self.assertRaises(ValueError): assert_overfit_contract(rank=64, warmup_steps=1, max_steps=500, lr=OVERFIT_LR, microbatch_size=1, accumulation_steps=8)
        with self.assertRaises(ValueError): assert_overfit_contract(rank=64, warmup_steps=0, max_steps=500, lr=OVERFIT_LR, microbatch_size=1, accumulation_steps=8, critic_enabled=True)
        self.assertEqual(OVERFIT_STEPS, (0, 50, 100, 200, 300, 400, 500)); self.assertEqual(OVERFIT_CHECKPOINT_STEPS, (50, 100, 200, 300, 400, 500))
        self.assertEqual((OVERFIT_MICROBATCH, OVERFIT_ACCUMULATION, OVERFIT_MAX_STEPS, OVERFIT_WARMUP), (1, 8, 500, 0))
        with self.assertRaises(ValueError):
            assert_overfit_contract(rank=64, warmup_steps=0, max_steps=500, lr=1e-4, microbatch_size=1,
                                    accumulation_steps=8, checkpoint_steps=(50, 100, 150, 200, 300, 400, 500))

    def test_authoritative_save_schedule_skips_non_scientific_50_step_boundaries(self):
        self.assertEqual(tuple(step for step in range(1, OVERFIT_MAX_STEPS + 1) if is_overfit_checkpoint_step(step)), OVERFIT_CHECKPOINT_STEPS)
        for step in (150, 250, 350, 450): self.assertFalse(is_overfit_checkpoint_step(step))
        for step in OVERFIT_CHECKPOINT_STEPS: self.assertTrue(is_overfit_checkpoint_step(step))
        self.assertEqual(OVERFIT_MAX_STEPS, 500); self.assertTrue(should_continue_overfit(499)); self.assertFalse(should_continue_overfit(500))

    def test_exact_architecture_parameter_audit(self):
        audit = parameter_audit()
        self.assertEqual(audit["rank"], 64); self.assertEqual(audit["alpha"], 64)
        self.assertEqual(audit["lora_target_modules"], 224); self.assertEqual(audit["trainable_parameter_count"], 215_488_512)
        self.assertEqual(audit["total_model_parameter_count"], 13_035_162_188)

    def test_trainer_and_evaluator_are_isolated_from_production_and_backward_contracts(self):
        trainer = Path("scripts/train_overfit_capacity.py").read_text(); evaluator = Path("scripts/evaluate_overfit_capacity.py").read_text()
        self.assertIn("train._flow_loss", trainer); self.assertIn("build_pose_model", trainer); self.assertIn("fresh_lora_checkpoint_loaded", trainer)
        self.assertIn("--resume", trainer); self.assertIn("is_overfit_checkpoint_step(global_step)", trainer); self.assertIn("sample_turbo_pose_image", evaluator)
        self.assertNotIn(".backward(", evaluator); self.assertNotIn("torch.optim", evaluator)
        self.assertIn("turbo_metadata", evaluator); self.assertIn("score_authoritative_pck", evaluator)

    def test_manifest_fail_closed_when_not_32_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); train = root / "train.jsonl"; train.write_text('{"file_name":"coco_1_1.jpg","text":"x"}\n')
            manifests = root / "manifests"; manifests.mkdir()
            # A malformed fixture is sufficient to prove the shared validator rejects it before data access.
            path = manifests / "overfit32-coco-r64-mse.jsonl"; path.write_text('{"file_name":"coco_1_1.jpg","text":"x"}\n' * 32)
            from pose_controlnet.overfit_capacity import validate_manifest
            with self.assertRaises(ValueError): validate_manifest("overfit32-coco-r64-mse", manifest_root=manifests, train_manifest=train)

    def test_explicit_resume_validates_identity_manifest_and_rejects_non_overfit_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args, cfg, stems, path, state = self._resume_fixture(root)
            restored_path, restored = harness.validate_overfit_resume_checkpoint(
                path, args=args, cfg=cfg, stems=stems, checkpoint_dir=path.parent,
            )
            self.assertEqual(restored_path, path); self.assertEqual(restored["global_step"], 100)
            metadata_path = path.parent / "experiment_metadata.json"; metadata = json.loads(metadata_path.read_text())
            metadata["experiment"] = "overfit32-mixed-r64-mse"; metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "another experiment"):
                harness.validate_overfit_resume_checkpoint(path, args=args, cfg=cfg, stems=stems, checkpoint_dir=path.parent)
            metadata["experiment"] = args.experiment; metadata["stems"] = list(reversed(stems)); metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "exact current COCO-32 manifest"):
                harness.validate_overfit_resume_checkpoint(path, args=args, cfg=cfg, stems=stems, checkpoint_dir=path.parent)
            metadata["stems"] = list(stems); metadata_path.write_text(json.dumps(metadata))
            invalid_step = dict(state); invalid_step["global_step"] = 150; invalid_step["scheduler"] = dict(state["scheduler"], step_count=150)
            invalid_path = save_training_state(path.parent / "step_000150.pt", invalid_step)
            with self.assertRaisesRegex(ValueError, "embedded global step"):
                harness.validate_overfit_resume_checkpoint(invalid_path, args=args, cfg=cfg, stems=stems, checkpoint_dir=path.parent)
            outside = root / "production" / path.name; outside.parent.mkdir(); save_training_state(outside, state)
            with self.assertRaisesRegex(ValueError, "explicit file in this experiment"):
                harness.validate_overfit_resume_checkpoint(outside, args=args, cfg=cfg, stems=stems, checkpoint_dir=path.parent)

    def test_explicit_resume_restores_trainable_optimizer_scheduler_and_flow_state(self):
        source = torch.nn.Module(); source.register_parameter("weight", torch.nn.Parameter(torch.tensor([3.0])))
        source_optimizer = torch.optim.AdamW([source.weight], lr=OVERFIT_LR, betas=(.9, .99), weight_decay=0.0)
        source.weight.grad = torch.tensor([.25]); source_optimizer.step()
        flow = torch.Generator().manual_seed(9876)
        state = {
            "model": {"weight": source.weight.detach().clone()}, "optimizer": source_optimizer.state_dict(),
            "scheduler": {"step_count": 100, "base_lrs": [OVERFIT_LR], "warmup_steps": 0},
            "global_step": 100, "epoch": 24, "batch_position": 32, "rng": train._capture_rng(),
            "flow_generator_state": flow.get_state(), "config": {},
        }
        target = torch.nn.Module(); target.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
        optimizer = torch.optim.AdamW([target.weight], lr=OVERFIT_LR, betas=(.9, .99), weight_decay=0.0)
        scheduler = train.OptimizerStepWarmup(optimizer, 0); generator = torch.Generator().manual_seed(1)
        def load_model(model, saved): model.weight.data.copy_(saved["weight"])
        with patch("train.load_trainable_state_dict", side_effect=load_model) as load:
            progress = harness.restore_overfit_resume_state(target, optimizer, scheduler, generator, state)
        self.assertEqual(progress, (100, 24, 32)); load.assert_called_once_with(target, state["model"])
        self.assertTrue(torch.equal(target.weight, source.weight)); self.assertEqual(scheduler.step_count, 100)
        self.assertTrue(torch.equal(optimizer.state[target.weight]["exp_avg"], source_optimizer.state[source.weight]["exp_avg"]))
        self.assertTrue(torch.equal(generator.get_state(), state["flow_generator_state"]))

    def test_resume_reconciles_only_uncheckpointed_metrics_and_normal_preflight_is_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            original = b'{"global_step": 99}\n{"global_step": 100}\n{"global_step": 101}\n'
            path.write_bytes(original)
            backup = harness.reconcile_metrics_for_resume(path, 100)
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(path.read_text(), '{"global_step": 99}\n{"global_step": 100}\n')
        args = harness.parser().parse_args(["--experiment", "overfit32-coco-r64-mse"])
        with patch.object(harness, "assert_fresh_initialization") as fresh:
            proof = harness.preflight(args)
        fresh.assert_called_once_with(); self.assertIsNone(proof["resume_checkpoint"])


class OverfitCapacityReportTests(unittest.TestCase):
    """Report-only contract: existing generation is enough for qualitative review."""

    experiment = "overfit32-mixed-r64-mse"
    provenance = {"training_resolution": "native", "evaluation_resolution": "native",
                  "native_geometry": {"format_version": 1, "fixture": True}}

    def _generation_fixture(self, root: Path) -> tuple[Path, tuple[str, ...]]:
        stems = validate_manifest(self.experiment)
        output = root / self.experiment
        (output / "generation_results.json").parent.mkdir(parents=True)
        (output / "generation_results.json").write_text(json.dumps({
            "experiment": self.experiment,
            "training_set_overfit": True,
            "stems": list(stems),
            "steps": list(OVERFIT_STEPS),
            "training_resolution": "native",
            "evaluation_resolution": "native",
            "evaluation_provenance": self.provenance,
        }), encoding="utf-8")
        for stem in stems:
            directory = output / "training_set" / stem; directory.mkdir(parents=True)
            for name in ("control.png", "target.png", *(f"step_{step:06d}.png" for step in OVERFIT_STEPS)):
                (directory / name).touch()
        return output, stems

    @contextmanager
    def _inputs(self, output: Path, stems: tuple[str, ...]):
        with patch.object(evaluator, "_inputs", return_value=(Path("checkpoints"), output, stems, object())), \
             patch.object(evaluator, "_evaluation_provenance", return_value=self.provenance):
            yield

    def test_report_without_metrics_creates_qualitative_only_summary_and_never_generates_or_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, stems = self._generation_fixture(Path(temporary)); args = type("Args", (), {"experiment": self.experiment})()
            with self._inputs(output, stems), patch.object(evaluator, "make_contact_sheet") as contact, \
                 patch.object(evaluator, "generate") as generate, patch.object(evaluator, "score") as score, \
                 patch.object(evaluator, "score_authoritative_pck") as pck, \
                 patch.object(sys, "argv", ["evaluate_overfit_capacity.py", "--experiment", self.experiment, "--stage", "report"]):
                evaluator.main()
            generate.assert_not_called(); score.assert_not_called(); pck.assert_not_called()
            self.assertEqual(contact.call_count, 2)
            selection_rows = contact.call_args_list[0].args[0]; all_rows = contact.call_args_list[1].args[0]
            labels = contact.call_args_list[1].kwargs["column_labels"]
            self.assertEqual(labels, ("Pose control", "Target training RGB", "Step 0", "Step 50", "Step 100", "Step 200", "Step 300", "Step 400", "Step 500"))
            self.assertEqual([stem for stem, _ in all_rows], list(stems)); self.assertEqual([stem for stem, _ in selection_rows], list(stems[:4]))
            self.assertEqual([path.name for path in all_rows[0][1]], ["control.png", "target.png", "step_000000.png", "step_000050.png", "step_000100.png", "step_000200.png", "step_000300.png", "step_000400.png", "step_000500.png"])
            summary = json.loads((output / "overfit_summary.json").read_text(encoding="utf-8"))
            self.assertFalse((output / "training_set_overfit_metrics.json").exists())
            self.assertTrue(summary["training_set_equals_evaluation_set"]); self.assertEqual(summary["sample_count"], 32)
            self.assertEqual(summary["checkpoints"], list(OVERFIT_STEPS)); self.assertEqual(summary["provenance"]["immutable_manifest_stems"], list(stems))
            self.assertEqual(summary["quantitative_scoring"], "not_yet_available")
            self.assertEqual(summary["qualitative_grids"], {"checkpoint_selection": "checkpoint_selection_grid.png", "full_contact_sheet": "full_training_set_contact_sheet.png"})
            self.assertNotIn("pose", summary); self.assertNotIn("clip", summary)

    def test_report_preserves_existing_score_metrics_and_compatible_summary_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, stems = self._generation_fixture(Path(temporary)); args = type("Args", (), {"experiment": self.experiment})()
            metrics = {"experiment": self.experiment, "training_set_equals_evaluation_set": True,
                       "evaluable_sample_count": 32, "metric_provenance": "existing-score-artifact",
                       "training_resolution": "native", "evaluation_resolution": "native",
                       "evaluation_provenance": self.provenance,
                       "checkpoints": [{"checkpoint_step": step, "pose": {"pck_010": .5}} for step in OVERFIT_STEPS]}
            metrics_path = output / "training_set_overfit_metrics.json"; metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            original_metrics = metrics_path.read_bytes()
            (output / "overfit_summary.json").write_text(json.dumps({"compatible_prior_field": "preserve-me", "old_only": True}), encoding="utf-8")
            with self._inputs(output, stems), patch.object(evaluator, "make_contact_sheet"):
                evaluator.report(args)
            self.assertEqual(metrics_path.read_bytes(), original_metrics)
            summary = json.loads((output / "overfit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["compatible_prior_field"], "preserve-me")
            self.assertEqual(summary["metric_provenance"], "existing-score-artifact")
            self.assertEqual(summary["checkpoints"], metrics["checkpoints"])
            self.assertEqual(tuple(row["checkpoint_step"] for row in summary["checkpoints"]), OVERFIT_STEPS)
            self.assertEqual(summary["qualitative_grids"]["full_contact_sheet"], "full_training_set_contact_sheet.png")

    def test_report_fails_closed_for_missing_generated_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, stems = self._generation_fixture(Path(temporary)); args = type("Args", (), {"experiment": self.experiment})()
            (output / "training_set" / stems[0] / "step_000200.png").unlink()
            with self._inputs(output, stems), patch.object(evaluator, "make_contact_sheet") as contact:
                with self.assertRaisesRegex(FileNotFoundError, "step_000200.png"):
                    evaluator.report(args)
            contact.assert_not_called(); self.assertFalse((output / "overfit_summary.json").exists())

    def test_report_fails_closed_for_malformed_generation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, stems = self._generation_fixture(Path(temporary)); args = type("Args", (), {"experiment": self.experiment})()
            (output / "generation_results.json").write_text(json.dumps({"experiment": self.experiment, "training_set_overfit": True,
                                                                            "stems": list(reversed(stems)), "steps": list(OVERFIT_STEPS)}), encoding="utf-8")
            with self._inputs(output, stems), patch.object(evaluator, "make_contact_sheet") as contact:
                with self.assertRaisesRegex(ValueError, "exact immutable"):
                    evaluator.report(args)
            contact.assert_not_called(); self.assertFalse((output / "overfit_summary.json").exists())


if __name__ == "__main__": unittest.main()
