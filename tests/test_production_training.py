import contextlib
import copy
import inspect
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import train
from pose_controlnet.checkpointing import load_training_state, save_training_state
from pose_controlnet.pose_reward_tools import combine_flow_and_pose_loss
from pose_controlnet.production_training import (
    PRODUCTION_METADATA_KEY,
    CooldownContinuation,
    CosineContinuationScheduler,
    FINISH_PARENT_RUN_NAME,
    PoseLambdaContinuationScheduler,
    ProductionRecipe,
    _production_checkpoint_metadata,
    build_train_config,
    build_arg_parser,
    checkpoint_due,
    cooldown_continuation_from_args,
    empty_pose_cumulative_counters,
    pose_cumulative_counters_from_checkpoint,
    production_hf_milestone_steps,
    production_wandb_mirror,
    optimizer_update_steps,
    recipe_from_args,
    run_metadata,
    run,
    planned_microbatches,
    resolve_resume,
    restore_cooldown_continuation_state,
    validate_cooldown_parent_identity,
    validate_resume_identity,
    verify_production_artifacts,
)
from pose_controlnet.checkpointing import HFTrainingCheckpointMirror
from scripts.train_pose_reward_smoke import update_cumulative_counters


class ProductionTrainingTests(unittest.TestCase):
    def args(self, *extra: str):
        return build_arg_parser().parse_args(["--run-name", "production-test", "--max-steps", "3000", *extra])

    def identities(self):
        return {
            "dataset_root": "/dataset",
            "train_manifest": {"raw_sha256": "manifest", "records_sha256": "records", "ordered_stems_sha256": "order"},
            "latent_cache": {"cache_contract_sha256": "cache"},
            "pose_sidecar": {"records_sha256": "pose"},
            "text_conditioning": {"metadata_sha256": "text"},
            "raw_checkpoint": {"sha256": "raw"},
        }

    def full_state(self, step: int, metadata: dict):
        return {"model": {}, "optimizer": {}, "scheduler": {"step_count": step}, "global_step": step,
                "epoch": 0, "batch_position": 0, "rng": train._capture_rng(),
                "flow_generator_state": torch.Generator().get_state(), "config": {},
                PRODUCTION_METADATA_KEY: metadata}

    def finishing_args(self, branch: str, *extra: str):
        pose_args = ("--pose-lambda-schedule", "constant") if branch == "control" else (
            "--pose-lambda-schedule", "linear", "--pose-lambda-final", "0",
        )
        return build_arg_parser().parse_args([
            "--run-name", f"finish-{branch}", "--max-steps", "4500", "--save-every", "100",
            "--continue-from", "/parent/step_004000.pt", "--continue-from-step", "4000",
            "--lr-schedule", "cosine", "--lr-start", "2e-5", "--lr-final", "5e-6",
            *pose_args, *extra,
        ])

    def finishing_parent(self, identities: dict):
        parent_args = self.args("--run-name", FINISH_PARENT_RUN_NAME, "--max-steps", "5000")
        cooldown = CooldownContinuation(Path("/parent/step_003000.pt"), 3000, "cosine", 1e-4, 1e-5, 2000)
        metadata = _production_checkpoint_metadata(
            args=parent_args, recipe=ProductionRecipe(), identities=identities, current_step=4000,
            continuation_metadata=cooldown.metadata(parent_run_name="pose-control-production-3000", parent_sha256="sha"),
        )
        metadata["data_position"] = {"epoch": 0, "batch_position": 0, "sample_position": 0}
        return self.full_state(4000, metadata)

    def test_cli_defaults_are_the_locked_loader4_recipe(self):
        args = self.args()
        self.assertIsNone(args.wandb_entity)
        recipe = recipe_from_args(args)
        self.assertIsNone(build_train_config(args, recipe).wandb_entity)
        self.assertEqual(recipe.effective_batch_size, 32)
        self.assertEqual((recipe.learning_rate, recipe.warmup_steps), (1e-4, 200))
        self.assertEqual((recipe.lambda_pose, recipe.pose_timestep_min, recipe.pose_timestep_max), (.04, .10, .20))
        self.assertEqual((recipe.data_loader_workers, recipe.persistent_workers, recipe.pin_memory, recipe.prefetch_factor),
                         (4, True, True, 4))
        self.assertFalse(recipe.compile)
        self.assertFalse(recipe.fused_adamw)
        self.assertEqual(recipe.gradient_checkpointing_blocks, 0)

    def test_wandb_is_disabled_by_default_and_never_initializes_a_remote_module(self):
        class FakeWandb:
            def init(self, **_kwargs):
                raise AssertionError("W&B must not initialize when disabled")
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        self.assertFalse(args.wandb)
        self.assertFalse(build_train_config(args, recipe).wandb_enabled)
        mirror = production_wandb_mirror(args=args, recipe=recipe, identities=identities,
                                         wandb_module=FakeWandb())
        self.assertFalse(mirror.enabled)

    def test_wandb_cli_enablement_and_local_checkpoint_resume_identity(self):
        class FakeRun:
            id = "continued-wandb-id"
            def log(self, *_args, **_kwargs): pass
            def finish(self): pass
        class FakeWandb:
            def __init__(self): self.kwargs = None
            def init(self, **kwargs): self.kwargs = kwargs; return FakeRun()
        args = self.args("--wandb", "--wandb-project", "project", "--wandb-entity", "adhit-420",
                         "--wandb-name", "visible-name")
        recipe, identities = ProductionRecipe(), self.identities()
        self.assertTrue(build_train_config(args, recipe).wandb_enabled)
        self.assertEqual(build_train_config(args, recipe).wandb_entity, "adhit-420")
        fake = FakeWandb()
        first = production_wandb_mirror(args=args, recipe=recipe, identities=identities, wandb_module=fake)
        self.assertTrue(first.enabled)
        self.assertEqual((fake.kwargs["project"], fake.kwargs["entity"], fake.kwargs["name"]),
                         ("project", "adhit-420", "visible-name"))
        self.assertEqual(run_metadata(args=args, recipe=recipe, identities=identities, current_step=0)
                         ["observability"]["wandb"]["entity"], "adhit-420")
        state = self.full_state(5, _production_checkpoint_metadata(
            args=args, recipe=recipe, identities=identities, current_step=5,
            wandb_run_id=first.run_id,
        ))
        resumed_fake = FakeWandb()
        resumed = production_wandb_mirror(args=args, recipe=recipe, identities=identities,
                                          resume_state=state, wandb_module=resumed_fake)
        self.assertEqual(resumed.run_id, "continued-wandb-id")
        self.assertEqual(resumed_fake.kwargs["entity"], "adhit-420")
        self.assertEqual(resumed_fake.kwargs["id"], "continued-wandb-id")
        self.assertEqual(resumed_fake.kwargs["resume"], "allow")

    def test_requires_explicit_max_steps_and_accepts_3000_without_gate_f_limit(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_arg_parser().parse_args(["--run-name", "missing-max"])
        self.assertEqual(recipe_from_args(self.args()).effective_batch_size, 32)

    def test_locked_runtime_overrides_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Production recipe is locked"):
            recipe_from_args(self.args("--compile"))
        with self.assertRaisesRegex(ValueError, "Production recipe is locked"):
            recipe_from_args(self.args("--data-loader-workers", "0"))

    def test_verifier_runs_before_any_cuda_training_work(self):
        args = self.args()
        invoked = []
        def verifier(_):
            invoked.append(True)
            return self.identities()
        with mock.patch("pose_controlnet.production_training.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "GH200"):
                run(args, verifier=verifier)
        self.assertEqual(invoked, [True])

    def test_bad_cache_contract_fails_closed(self):
        args = self.args()
        with mock.patch("pose_controlnet.production_training.verify_full_768_cache",
                        return_value={"cache_samples": 16503, "resolution_policy": "native"}):
            with self.assertRaisesRegex(ValueError, "did not prove"):
                verify_production_artifacts(args)

    def test_resume_refuses_changed_pose_sidecar_identity(self):
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities, current_step=12)
        state = {"global_step": 12, PRODUCTION_METADATA_KEY: metadata}
        changed = self.identities(); changed["pose_sidecar"] = {"records_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "artifact_identity"):
            validate_resume_identity(state, args=args, recipe=recipe, identities=changed)

    def test_resume_accepts_matching_identity_and_position(self):
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities, current_step=12)
        metadata["data_position"] = {"epoch": 3, "batch_position": 5, "sample_position": 5}
        validate_resume_identity({"global_step": 12, "epoch": 3, "batch_position": 5,
                                  PRODUCTION_METADATA_KEY: metadata}, args=args, recipe=recipe, identities=identities)

    def test_resume_rng_restoration_is_deterministic(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        state = train._capture_rng()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        random.random(); np.random.rand(); torch.rand(())
        train._restore_rng(state)
        observed = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(observed, expected)

    def test_resume_position_reconstructs_the_same_deterministic_microbatches(self):
        records = [("shard", index, (96, 96), f"stem-{index}") for index in range(8)]
        recipe = ProductionRecipe()
        full = planned_microbatches(records, recipe=recipe, epoch=0, batch_position=0, count=8)
        resumed = planned_microbatches(records, recipe=recipe, epoch=0, batch_position=3, count=5)
        self.assertEqual(resumed, full[3:])

    def test_auto_resume_is_local_only_and_requires_no_network(self):
        with mock.patch("pose_controlnet.production_training.newest_valid_local_checkpoint", return_value=Path("local.pt")) as latest:
            self.assertEqual(resolve_resume("auto", Path("/checkpoint/run")), Path("local.pt"))
        latest.assert_called_once_with(Path("/checkpoint/run"))

    def test_hf_mirroring_is_disabled_by_default_and_cadence_is_explicit(self):
        args, recipe = self.args(), ProductionRecipe()
        cfg = build_train_config(args, recipe)
        self.assertEqual((cfg.hf_repo_id, cfg.hf_mirror_every_steps), ("", 0))
        mirror = HFTrainingCheckpointMirror(repo_id=cfg.hf_repo_id, run_name=args.run_name)
        with mock.patch.object(mirror, "_get_api", side_effect=AssertionError("HF must not initialize when disabled")):
            mirror.start()
        self.assertIsNone(mirror._thread)
        enabled = self.args("--hf-repo-id", "owner/private", "--hf-mirror-every-steps", "500")
        enabled_cfg = build_train_config(enabled, recipe_from_args(enabled))
        self.assertEqual((enabled_cfg.hf_repo_id, enabled_cfg.hf_mirror_every_steps), ("owner/private", 500))
        with self.assertRaisesRegex(ValueError, "requires --hf-repo-id"):
            recipe_from_args(self.args("--hf-mirror-every-steps", "500"))
        with self.assertRaisesRegex(ValueError, "divisible"):
            recipe_from_args(self.args("--hf-repo-id", "owner/private", "--hf-mirror-every-steps", "300"))

    def test_production_trainer_uses_draining_hf_mirror_shutdown(self):
        self.assertIn("mirror.stop(drain=True, timeout=None)", inspect.getsource(run))

    def test_3000_step_hf_milestones_are_exact_completed_local_checkpoints(self):
        self.assertEqual(production_hf_milestone_steps(max_steps=3000, mirror_every_steps=500),
                         (500, 1000, 1500, 2000, 2500, 3000))
        self.assertEqual(production_hf_milestone_steps(max_steps=3000, mirror_every_steps=0), ())

    def test_cooldown_scheduler_has_no_second_warmup_and_reaches_final_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        scheduler = CosineContinuationScheduler(optimizer, parent_step=3000, continuation_steps=2000,
                                                start_lr=1e-4, final_lr=1e-5)
        rates = []
        for _ in range(2000):
            rates.append(scheduler.current_update_learning_rates[0]); scheduler.step()
        self.assertEqual(rates[0], 1e-4)
        self.assertAlmostEqual(rates[-1], 1e-5, places=14)
        self.assertTrue(all(next_rate < rate for rate, next_rate in zip(rates, rates[1:])))
        self.assertEqual(scheduler.step_count, 5000)

    def test_cooldown_restores_adam_moments_and_records_provenance(self):
        old_parameter = torch.nn.Parameter(torch.tensor(1.0))
        old_optimizer = torch.optim.AdamW([old_parameter], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        (old_parameter.square()).backward(); old_optimizer.step()
        saved_optimizer = copy.deepcopy(old_optimizer.state_dict())
        new_parameter = torch.nn.Parameter(torch.tensor(2.0))
        new_optimizer = torch.optim.AdamW([new_parameter], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        state = {"model": {}, "optimizer": saved_optimizer, "global_step": 3000, "epoch": 2,
                 "batch_position": 9, "rng": train._capture_rng(), "flow_generator_state": torch.Generator().get_state()}
        with mock.patch("pose_controlnet.production_training.train.load_trainable_state_dict"):
            restored = restore_cooldown_continuation_state(object(), new_optimizer, state)
        self.assertEqual(restored[:3], (3000, 2, 9))
        self.assertTrue(torch.equal(new_optimizer.state[new_parameter]["exp_avg"], old_optimizer.state[old_parameter]["exp_avg"]))
        self.assertTrue(torch.equal(new_optimizer.state[new_parameter]["exp_avg_sq"], old_optimizer.state[old_parameter]["exp_avg_sq"]))
        continuation = CooldownContinuation(Path("/parent/step_003000.pt"), 3000, "cosine", 1e-4, 1e-5, 2000)
        provenance = continuation.metadata(parent_run_name="pose-control-production-3000", parent_sha256="parent-sha")
        self.assertEqual((provenance["parent_global_step"], provenance["end_global_step"], provenance["kind"]),
                         (3000, 5000, "scientific_continuation"))
        metadata = _production_checkpoint_metadata(args=self.args("--max-steps", "5000"), recipe=ProductionRecipe(),
            identities=self.identities(), current_step=3250, continuation_metadata=provenance)
        with tempfile.TemporaryDirectory() as directory:
            path = save_training_state(Path(directory) / "step_003250.pt", self.full_state(3250, metadata))
            self.assertEqual(load_training_state(path)[PRODUCTION_METADATA_KEY]["continuation"], provenance)

    def test_cooldown_contract_parent_identity_and_global_hf_schedule_fail_closed(self):
        args = self.args("--run-name", "cooldown", "--max-steps", "5000", "--continue-from", "/parent/step_003000.pt",
                         "--lr-schedule", "cosine", "--lr-final", "1e-5")
        recipe = recipe_from_args(args); continuation = cooldown_continuation_from_args(args, recipe)
        self.assertEqual((continuation.parent_step, continuation.continuation_steps), (3000, 2000))
        self.assertEqual(production_hf_milestone_steps(max_steps=5000, mirror_every_steps=500, start_step=3000),
                         (3500, 4000, 4500, 5000))
        metadata = _production_checkpoint_metadata(args=self.args(), recipe=recipe, identities=self.identities(), current_step=3000)
        metadata["data_position"] = {"epoch": 0, "batch_position": 0, "sample_position": 0}
        parent = self.full_state(3000, metadata)
        self.assertEqual(validate_cooldown_parent_identity(parent, continuation=continuation, recipe=recipe,
                                                           identities=self.identities())["run_name"], "production-test")
        bad = copy.deepcopy(parent); bad[PRODUCTION_METADATA_KEY]["artifact_identity"]["pose_sidecar"] = {"records_sha256": "bad"}
        with self.assertRaisesRegex(ValueError, "immutable science identity"):
            validate_cooldown_parent_identity(bad, continuation=continuation, recipe=recipe, identities=self.identities())
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            cooldown_continuation_from_args(self.args("--continue-from", "/parent/step_003000.pt"), recipe)

    def test_cooldown_wandb_starts_a_new_run_not_the_parent_run(self):
        class FakeRun:
            id = "new-cooldown-run"
            def log(self, *_args, **_kwargs): pass
            def finish(self): pass
        class FakeWandb:
            def __init__(self): self.kwargs = None
            def init(self, **kwargs): self.kwargs = kwargs; return FakeRun()
        args = self.args("--wandb", "--wandb-project", "Krea-2-PoseControl-Lora", "--wandb-entity", "adhit-projects",
                         "--wandb-name", "pose-control-production-cooldown-3000-to5000")
        continuation = CooldownContinuation(Path("/parent/step_003000.pt"), 3000, "cosine", 1e-4, 1e-5, 2000)
        provenance = continuation.metadata(parent_run_name="pose-control-production-3000", parent_sha256="sha")
        parent_metadata = _production_checkpoint_metadata(args=self.args(), recipe=ProductionRecipe(), identities=self.identities(),
                                                          current_step=3000, wandb_run_id="parent-wandb-run")
        fake = FakeWandb()
        mirror = production_wandb_mirror(args=args, recipe=ProductionRecipe(), identities=self.identities(),
                                         resume_state=self.full_state(3000, parent_metadata), wandb_module=fake,
                                         continuation_metadata=provenance)
        self.assertEqual(mirror.run_id, "new-cooldown-run")
        self.assertNotIn("id", fake.kwargs)
        self.assertNotIn("resume", fake.kwargs)
        self.assertEqual(fake.kwargs["config"]["continuation"]["parent_run_name"], "pose-control-production-3000")

    def test_finishing_ab_contract_parent_provenance_and_global_checkpoint_cadence(self):
        identities = self.identities()
        parent = self.finishing_parent(identities)
        control_args = self.finishing_args("control")
        control = cooldown_continuation_from_args(control_args, recipe_from_args(control_args))
        self.assertEqual((control.parent_step, control.end_step, control.branch_type), (4000, 4500, "finish-control"))
        self.assertEqual(validate_cooldown_parent_identity(parent, continuation=control, recipe=ProductionRecipe(),
                                                           identities=identities)["current_step"], 4000)
        provenance = control.metadata(parent_run_name=FINISH_PARENT_RUN_NAME, parent_sha256="parent-sha")
        self.assertEqual(provenance["parent_checkpoint_sha256"], "parent-sha")
        self.assertEqual(provenance["pose_lambda_schedule"], {"name": "constant", "start": .04, "final": .04})
        self.assertEqual(production_hf_milestone_steps(max_steps=4500, mirror_every_steps=100, start_step=4000),
                         (4100, 4200, 4300, 4400, 4500))
        self.assertEqual(tuple(f"step_{step:06d}.pt" for step in range(4100, 4501, 100)),
                         ("step_004100.pt", "step_004200.pt", "step_004300.pt", "step_004400.pt", "step_004500.pt"))
        wrong_step = self.finishing_args("control", "--continue-from-step", "3999")
        with self.assertRaisesRegex(ValueError, "4000"):
            cooldown_continuation_from_args(wrong_step, recipe_from_args(wrong_step))
        bad = copy.deepcopy(parent)
        bad[PRODUCTION_METADATA_KEY]["artifact_identity"]["pose_sidecar"] = {"records_sha256": "bad"}
        with self.assertRaisesRegex(ValueError, "immutable science identity"):
            validate_cooldown_parent_identity(bad, continuation=control, recipe=ProductionRecipe(), identities=identities)

    def test_finishing_schedules_have_exact_endpoints_and_resume_position(self):
        args = self.finishing_args("anneal")
        continuation = cooldown_continuation_from_args(args, recipe_from_args(args))
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        pose = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                               start_value=.04, final_value=0.)
        scheduler = CosineContinuationScheduler(optimizer, parent_step=4000, continuation_steps=500,
                                                start_lr=2e-5, final_lr=5e-6, pose_lambda_scheduler=pose)
        rates, lambdas = [], []
        for _ in range(500):
            rates.append(scheduler.current_update_learning_rates[0]); lambdas.append(scheduler.current_pose_lambda); scheduler.step()
        self.assertEqual((rates[0], rates[-1]), (2e-5, 5e-6))
        self.assertEqual((lambdas[0], lambdas[-1]), (.04, 0.))
        self.assertTrue(all(next_rate < rate for rate, next_rate in zip(rates, rates[1:])))
        self.assertTrue(all(next_value < value for value, next_value in zip(lambdas, lambdas[1:])))
        expected = {4100: .04 * (1 - 99 / 499), 4200: .04 * (1 - 199 / 499),
                    4300: .04 * (1 - 299 / 499), 4400: .04 * (1 - 399 / 499), 4500: 0.}
        for step, value in expected.items():
            self.assertAlmostEqual(lambdas[step - 4001], value, places=14)
        control_pose = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="constant",
                                                       start_value=.04, final_value=.04)
        control_lambdas = []
        for _ in range(500):
            control_lambdas.append(control_pose.current_value); control_pose.step()
        self.assertTrue(all(value == .04 for value in control_lambdas))
        replay_pose = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                                      start_value=.04, final_value=0.)
        replay_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        replay = CosineContinuationScheduler(replay_optimizer, parent_step=4000, continuation_steps=500,
                                             start_lr=2e-5, final_lr=5e-6, pose_lambda_scheduler=replay_pose)
        saved = None
        for _ in range(100):
            replay.step()
        saved = replay.state_dict()
        resumed_pose = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                                       start_value=.04, final_value=0.)
        resumed_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        resumed = CosineContinuationScheduler(resumed_optimizer, parent_step=4000, continuation_steps=500,
                                              start_lr=2e-5, final_lr=5e-6, pose_lambda_scheduler=resumed_pose)
        resumed.load_state_dict(saved)
        self.assertEqual((resumed.step_count, resumed.current_update_learning_rates[0], resumed.current_pose_lambda),
                         (4100, rates[100], lambdas[100]))
        self.assertEqual(continuation.branch_type, "finish-pose-anneal")

    def test_finishing_execution_sequence_includes_final_global_update_and_resume_tail(self):
        """The production loop consumes this inclusive sequence directly."""
        updates = optimizer_update_steps(completed_global_step=4000, max_steps=4500)
        self.assertEqual(len(updates), 500)
        self.assertEqual((updates[0], updates[-1]), (4001, 4500))
        self.assertNotIn(4501, updates)
        self.assertEqual(optimizer_update_steps(completed_global_step=4400, max_steps=4500), tuple(range(4401, 4501)))

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        anneal = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                                 start_value=.04, final_value=0.)
        scheduler = CosineContinuationScheduler(optimizer, parent_step=4000, continuation_steps=500,
                                                start_lr=2e-5, final_lr=5e-6, pose_lambda_scheduler=anneal)
        observed = []
        for step in updates:
            observed.append((step, scheduler.current_update_learning_rates[0], scheduler.current_pose_lambda))
            scheduler.step()
        self.assertEqual(observed[-1], (4500, 5e-6, 0.))

        # Recreate a real step-4400 scheduler state rather than loading the
        # completed state above, then prove its exact-resume tail is 4401..4500.
        resumed_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-4,
                                              betas=(.9, .99), weight_decay=0.)
        resumed_anneal = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                                         start_value=.04, final_value=0.)
        resumed_scheduler = CosineContinuationScheduler(resumed_optimizer, parent_step=4000, continuation_steps=500,
                                                        start_lr=2e-5, final_lr=5e-6,
                                                        pose_lambda_scheduler=resumed_anneal)
        for _ in range(400):
            resumed_scheduler.step()
        checkpoint_4400 = resumed_scheduler.state_dict()
        self.assertEqual(checkpoint_4400["step_count"], 4400)
        resumed_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-4,
                                              betas=(.9, .99), weight_decay=0.)
        resumed_anneal = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                                         start_value=.04, final_value=0.)
        resumed_scheduler = CosineContinuationScheduler(resumed_optimizer, parent_step=4000, continuation_steps=500,
                                                        start_lr=2e-5, final_lr=5e-6,
                                                        pose_lambda_scheduler=resumed_anneal)
        resumed_scheduler.load_state_dict(checkpoint_4400)
        resumed_updates = optimizer_update_steps(completed_global_step=4400, max_steps=4500)
        resumed_observed = []
        for step in resumed_updates:
            resumed_observed.append((step, resumed_scheduler.current_update_learning_rates[0],
                                     resumed_scheduler.current_pose_lambda))
            resumed_scheduler.step()
        self.assertEqual((resumed_observed[0][0], resumed_observed[-1]), (4401, (4500, 5e-6, 0.)))

        control = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="constant",
                                                  start_value=.04, final_value=.04)
        for _ in updates:
            final_control_lambda = control.current_value
            control.step()
        self.assertEqual(final_control_lambda, .04)
        self.assertEqual(tuple(step for step in updates if step % 100 == 0),
                         (4100, 4200, 4300, 4400, 4500))
        self.assertEqual(production_hf_milestone_steps(max_steps=4500, mirror_every_steps=100, start_step=4000),
                         (4100, 4200, 4300, 4400, 4500))

        args = self.finishing_args("anneal")
        continuation = cooldown_continuation_from_args(args, recipe_from_args(args))
        provenance = continuation.metadata(parent_run_name=FINISH_PARENT_RUN_NAME, parent_sha256="parent-sha")
        metadata = _production_checkpoint_metadata(args=args, recipe=ProductionRecipe(), identities=self.identities(),
                                                   current_step=4400, continuation_metadata=provenance)
        metadata["data_position"] = {"epoch": 0, "batch_position": 0, "sample_position": 0}
        exact_resume = self.full_state(4400, metadata)
        validate_resume_identity(exact_resume, args=args, recipe=ProductionRecipe(), identities=self.identities(),
                                 continuation_metadata=provenance)
        incompatible = copy.deepcopy(exact_resume)
        incompatible[PRODUCTION_METADATA_KEY]["scheduler"]["final_lr"] = 1e-5
        with self.assertRaisesRegex(ValueError, "scheduler"):
            validate_resume_identity(incompatible, args=args, recipe=ProductionRecipe(), identities=self.identities(),
                                     continuation_metadata=provenance)

    def test_finishing_zero_endpoint_reaches_optimizer_and_checkpoint_path(self):
        """CPU regression for the production loop's exact step-4500 boundary."""
        args = self.finishing_args("anneal")
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(.9, .99), weight_decay=0.)
        pose = PoseLambdaContinuationScheduler(parent_step=4000, continuation_steps=500, schedule="linear",
                                               start_value=.04, final_value=0.)
        scheduler = CosineContinuationScheduler(optimizer, parent_step=4000, continuation_steps=500,
                                                start_lr=2e-5, final_lr=5e-6, pose_lambda_scheduler=pose)
        for _ in range(499):
            scheduler.step()
        self.assertEqual((scheduler.step_count, scheduler.current_pose_lambda), (4499, 0.0))

        flow_loss, pose_loss = parameter.square(), parameter * 3
        total_loss = combine_flow_and_pose_loss(flow_loss, pose_loss, active_count=1,
                                                lambda_pose=scheduler.current_pose_lambda)
        self.assertIs(total_loss, flow_loss)
        total_loss.backward()
        train.optimizer_update(optimizer, scheduler, [parameter], max_grad_norm=1.0)
        self.assertEqual(scheduler.step_count, 4500)
        self.assertNotEqual(float(parameter.detach()), 1.0)

        global_step = scheduler.step_count
        self.assertTrue(checkpoint_due(global_step=global_step, save_every=args.save_every,
                                       max_steps=args.max_steps, stopped=False))
        continuation = cooldown_continuation_from_args(args, recipe_from_args(args))
        provenance = continuation.metadata(parent_run_name=FINISH_PARENT_RUN_NAME, parent_sha256="parent-sha")
        metadata = _production_checkpoint_metadata(args=args, recipe=ProductionRecipe(), identities=self.identities(),
                                                   current_step=global_step, continuation_metadata=provenance)
        metadata["data_position"] = {"epoch": 0, "batch_position": 0, "sample_position": 0}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_training_state(Path(directory) / "step_004500.pt", self.full_state(global_step, metadata),
                                             overwrite=False)
            self.assertEqual((checkpoint.name, load_training_state(checkpoint)["global_step"]),
                             ("step_004500.pt", 4500))

    def test_ordinary_and_cooldown_update_sequences_remain_global_max_inclusive(self):
        ordinary = optimizer_update_steps(completed_global_step=0, max_steps=3000)
        cooldown = optimizer_update_steps(completed_global_step=3000, max_steps=5000)
        self.assertEqual((ordinary[0], ordinary[-1], len(ordinary)), (1, 3000, 3000))
        self.assertEqual((cooldown[0], cooldown[-1], len(cooldown)), (3001, 5000, 2000))

    def test_finishing_wandb_starts_new_branch_runs_and_exact_resume_stays_strict(self):
        class FakeRun:
            def __init__(self, identifier): self.id = identifier
            def log(self, *_args, **_kwargs): pass
            def finish(self): pass
        class FakeWandb:
            def __init__(self): self.calls = []
            def init(self, **kwargs): self.calls.append(kwargs); return FakeRun(f"branch-{len(self.calls)}")
        identities, parent = self.identities(), self.finishing_parent(self.identities())
        for branch in ("control", "anneal"):
            args = self.finishing_args(branch, "--wandb")
            continuation = cooldown_continuation_from_args(args, recipe_from_args(args))
            provenance = continuation.metadata(parent_run_name=FINISH_PARENT_RUN_NAME, parent_sha256="sha")
            fake = FakeWandb()
            mirror = production_wandb_mirror(args=args, recipe=ProductionRecipe(), identities=identities,
                                             resume_state=parent, continuation_metadata=provenance, wandb_module=fake)
            self.assertEqual(mirror.run_id, "branch-1")
            self.assertNotIn("id", fake.calls[0])
            self.assertNotIn("resume", fake.calls[0])
            self.assertEqual(fake.calls[0]["config"]["continuation"]["branch_type"], continuation.branch_type)
            child_metadata = _production_checkpoint_metadata(args=args, recipe=ProductionRecipe(), identities=identities,
                                                              current_step=4100, continuation_metadata=provenance)
            child_metadata["data_position"] = {"epoch": 0, "batch_position": 0, "sample_position": 0}
            validate_resume_identity(self.full_state(4100, child_metadata), args=args, recipe=ProductionRecipe(),
                                     identities=identities, continuation_metadata=provenance)
            wrong_provenance = dict(provenance)
            wrong_provenance["branch_type"] = ("finish-pose-anneal"
                                                if continuation.branch_type == "finish-control" else "finish-control")
            with self.assertRaisesRegex(ValueError, "continuation provenance"):
                validate_resume_identity(self.full_state(4100, child_metadata), args=args, recipe=ProductionRecipe(),
                                         identities=identities, continuation_metadata=wrong_provenance)

    def test_hf_failure_keeps_atomic_local_checkpoint_authoritative(self):
        class FailingApi:
            def create_repo(self, *_args, **_kwargs): pass
            def upload_file(self, **_kwargs): raise OSError("simulated offline")
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities, current_step=500)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_training_state(Path(directory) / "step_000500.pt", self.full_state(500, metadata))
            mirror = HFTrainingCheckpointMirror(repo_id="owner/private", run_name="run", api=FailingApi(),
                                                max_attempts=1)
            self.assertFalse(mirror._upload(checkpoint, "step"))
            self.assertTrue(checkpoint.exists())
            self.assertEqual(load_training_state(checkpoint)["global_step"], 500)

    def test_incomplete_or_temp_checkpoints_are_never_mirrored(self):
        class Api:
            def __init__(self): self.uploads = 0
            def create_repo(self, *_args, **_kwargs): pass
            def upload_file(self, **_kwargs): self.uploads += 1
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities, current_step=500)
        with tempfile.TemporaryDirectory() as directory:
            temporary = save_training_state(Path(directory) / "step_000500.pt.publish.tmp", self.full_state(500, metadata))
            api = Api(); mirror = HFTrainingCheckpointMirror(repo_id="owner/private", run_name="run", api=api)
            self.assertFalse(mirror.submit(temporary, reason="step"))
            self.assertEqual(api.uploads, 0)

    def test_pose_cumulative_counters_persist_and_continue_after_resume(self):
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        before = {"eligible_samples_seen": 141, "forced_samples": 0,
                  "naturally_active_samples": 1, "total_active_samples": 1}
        metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities, current_step=5,
                                                   cumulative_counters=before)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_training_state(Path(directory) / "step_000005.pt", self.full_state(5, metadata))
            restored = pose_cumulative_counters_from_checkpoint(load_training_state(checkpoint))
        self.assertEqual(restored, before)
        continued = update_cumulative_counters(restored, {
            "pose_eligible_samples_step": 30, "pose_forced_samples_step": 0,
            "pose_natural_active_samples_step": 1, "pose_active_samples_step": 1,
        })
        self.assertEqual(continued, {"eligible_samples_seen": 171, "forced_samples": 0,
                                     "naturally_active_samples": 2, "total_active_samples": 2})

    def test_atomic_checkpoint_metadata_contains_all_required_identities(self):
        args, recipe, identities = self.args(), ProductionRecipe(), self.identities()
        metadata = _production_checkpoint_metadata(args=args, recipe=recipe, identities=identities, current_step=7)
        self.assertEqual(metadata["scientific_recipe"]["effective_batch_size"], 32)
        self.assertEqual(metadata["artifact_identity"]["latent_cache"]["cache_contract_sha256"], "cache")
        self.assertEqual(metadata["artifact_identity"]["pose_sidecar"]["records_sha256"], "pose")
        self.assertEqual(metadata["artifact_identity"]["train_manifest"]["ordered_stems_sha256"], "order")
        state = {"model": {}, "optimizer": {}, "scheduler": {"step_count": 7}, "global_step": 7,
                 "epoch": 0, "batch_position": 0, "rng": train._capture_rng(),
                 "flow_generator_state": torch.Generator().get_state(), "config": {},
                 PRODUCTION_METADATA_KEY: metadata}
        with tempfile.TemporaryDirectory() as directory:
            path = save_training_state(Path(directory) / "step_000007.pt", state, overwrite=False)
            self.assertEqual(load_training_state(path)[PRODUCTION_METADATA_KEY]["run_name"], "production-test")


if __name__ == "__main__":
    unittest.main()
