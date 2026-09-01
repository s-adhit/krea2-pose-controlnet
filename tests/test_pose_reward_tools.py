import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from pose_controlnet.pose_reward_tools import (
    combine_flow_and_pose_loss, combined_gradient_diagnostics, gradient_interaction,
    lambda_calibration, pose_active_mask, select_trainable_named_parameters, validate_smoke_invocation,
)
from pose_controlnet.checkpointing import HFTrainingCheckpointMirror, load_training_state, save_training_state
from pose_controlnet.keypoint_critic import gaussian_heatmap_kl
from pose_controlnet.config import TrainConfig
from scripts.train_pose_reward_smoke import (
    GATE_E_METADATA_KEY, _gate_e_metadata, _validate_parent,
    aggregate_step_diagnostics, load_gate_e_microbatch, pose_active_window,
    checkpoint_publication_steps, resolve_target_global_step, should_build_pose_graph,
    update_cumulative_counters, validate_gate_e_destination, validate_gate_e_resume_checkpoint,
    _validate_hf_branch_args, build_arg_parser, prepare_gate_e_run_setup,
)
from pose_controlnet.keypoint_critic_audit import deterministic_noise_like


class PoseRewardToolsTest(unittest.TestCase):
    def _state(self, *, global_step=1610, config=None, metadata=None):
        state = {"model": {"first.weight": torch.ones(1)}, "optimizer": {"state": {}},
                 "scheduler": {"step_count": global_step}, "global_step": global_step,
                 "epoch": 3, "batch_position": 4,
                 "rng": {"python": None, "numpy": None, "torch": torch.get_rng_state(), "cuda": None},
                 "flow_generator_state": torch.Generator().get_state(),
                 "config": config or asdict(TrainConfig(raw_ckpt="raw", shard_dir="shards", run_name="gate-e", max_steps=1700,
                                                          allow_extended_training=True, microbatch_size=1,
                                                          gradient_accumulation_steps=32))}
        if metadata is not None:
            state[GATE_E_METADATA_KEY] = metadata
        return state

    def _gate_e_cfg(self, root: Path, *, max_steps=1700):
        return TrainConfig(raw_ckpt="raw", shard_dir="shards", ckpt_dir=str(root.parent), run_name=root.name,
                           max_steps=max_steps, allow_extended_training=True, microbatch_size=1,
                           gradient_accumulation_steps=32, metrics_jsonl_path=str(root / "metrics.jsonl"))

    @staticmethod
    def _immutable_parent():
        return {"global_step": 1500,
                "sha256": "6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0",
                "filename": "step_001500.pt"}

    def _metadata(self, cfg, model_state):
        return _gate_e_metadata(
            cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5, timestep_min=.10, timestep_max=.20,
            forced_exposure_probability=.05, hf_subdir=cfg.run_name,
            immutable_parent=self._immutable_parent(), cumulative_counters={
                "eligible_samples_seen": 4, "forced_samples": 1,
                "naturally_active_samples": 2, "total_active_samples": 3,
            }, model_state=model_state,
        )

    def test_gate_e_microbatch_fetch_uses_dataset_getitem_and_collates_stems(self):
        class IndexOnlyDataset:
            def __init__(self):
                self.indices = []
                self.items = [
                    {"latent": torch.full((2, 2, 2), 1.0), "control": torch.full((2, 2, 2), 2.0), "prompt": "first", "stem": "one"},
                    {"latent": torch.full((2, 2, 2), 3.0), "control": torch.full((2, 2, 2), 4.0), "prompt": "second", "stem": "two"},
                ]

            def __getitem__(self, index):
                self.indices.append(index)
                return self.items[index]

        data = IndexOnlyDataset()
        batch = load_gate_e_microbatch(data, [1, 0])
        self.assertEqual(data.indices, [1, 0])
        self.assertEqual(batch["prompts"], ["second", "first"])
        self.assertEqual(batch["stems"], ["two", "one"])
        self.assertEqual(tuple(batch["latent"].shape), (2, 2, 2, 2))

    def test_incremental_gradient_interaction_and_lambda_calibration(self):
        flow = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
        pose = [torch.tensor([0.0, 10.0]), torch.tensor([0.0])]
        stats = gradient_interaction(flow, pose)
        self.assertAlmostEqual(stats["flow_grad_norm"], 5.0)
        self.assertAlmostEqual(stats["pose_grad_norm"], 10.0)
        self.assertAlmostEqual(stats["ratio"], 2.0)
        self.assertAlmostEqual(stats["dot"], 40.0)
        self.assertAlmostEqual(stats["cosine"], .8)
        lambdas = lambda_calibration(stats)
        self.assertAlmostEqual(lambdas["lambda_5pct"], .025)
        combined = combined_gradient_diagnostics(stats, lambdas)["lambda_5pct"]
        self.assertGreater(combined["total_over_flow"], 1.0)
        self.assertGreater(combined["cosine_total_flow"], 0.0)

    def test_zero_gradient_is_safe(self):
        stats = gradient_interaction([torch.zeros(2)], [torch.ones(2)])
        self.assertIsNone(stats["ratio"])
        self.assertIsNone(stats["cosine"])
        self.assertIsNone(lambda_calibration(stats)["lambda_1pct"])

    def test_pose_masks_and_loss_combination(self):
        t = torch.tensor([.10, .20, .30])
        available = torch.tensor([True, False, True])
        self.assertTrue(torch.equal(pose_active_mask(t, available, (.10, .20)), torch.tensor([True, False, False])))
        self.assertTrue(torch.equal(pose_active_window(t, available, .10, .20), torch.tensor([True, False, False])))
        flow, pose = torch.tensor(2.0), torch.tensor(3.0)
        self.assertEqual(float(combine_flow_and_pose_loss(flow, None, 0, .1)), 2.0)
        self.assertTrue(torch.equal(combine_flow_and_pose_loss(flow, pose, 1, .04), flow + .04 * pose))
        self.assertIs(combine_flow_and_pose_loss(flow, pose, 1, 0.0), flow)
        self.assertIs(combine_flow_and_pose_loss(flow, None, 0, 0.0), flow)
        with self.assertRaises(ValueError): combine_flow_and_pose_loss(flow, None, 1, .1)
        for invalid in (-.01, float("nan"), float("inf")):
            with self.subTest(lambda_pose=invalid):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    combine_flow_and_pose_loss(flow, pose, 1, invalid)
        self.assertFalse(should_build_pose_graph(torch.empty(0, dtype=torch.long)))
        self.assertTrue(should_build_pose_graph(torch.tensor([0])))

    def test_pose_loss_cli_parsing_defaults_to_compatible_kl_and_selects_coordinate_huber(self):
        common = [
            "--parent-checkpoint", "parent.pt", "--raw-ckpt", "raw", "--latent-root", "latents",
            "--text-conditioning-root", "text", "--sidecar", "sidecar", "--checkpoint-dir", "checkpoints",
            "--run-name", "isolated", "--lambda-pose", "1e-5", "--pose-timestep-min", ".10",
            "--pose-timestep-max", ".20", "--forced-pose-exposure-probability", ".10",
            "--hf-repo-id", "owner/private", "--hf-subdir", "isolated", "--hf-mirror-every-steps", "25",
            "--target-global-step", "1600", "--save-every", "25", "--microbatch-size", "1",
            "--gradient-accumulation-steps", "32",
        ]
        parser = build_arg_parser()
        self.assertEqual(parser.parse_args(common).pose_loss, "gaussian_heatmap_kl")
        self.assertEqual(parser.parse_args([*common, "--pose-loss", "normalized_coordinate_huber"]).pose_loss,
                         "normalized_coordinate_huber")

    def test_gate_d_independent_graphs_use_identical_deterministic_noise(self):
        clean = torch.zeros(1, 2, 3, 4)
        first = deterministic_noise_like(clean, seed=42, stem="same-sample", label="gate-d-noise")
        second = deterministic_noise_like(clean, seed=42, stem="same-sample", label="gate-d-noise")
        self.assertTrue(torch.equal(first, second))

    def test_required_inputs_and_isolated_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent.pt"; parent.touch()
            with self.assertRaises(ValueError):
                validate_smoke_invocation(lambda_pose=None, pose_timesteps=(.1,), run_name="x", checkpoint_dir=temporary, parent_checkpoint=parent, verify_parent=True)
            value, timesteps, destination = validate_smoke_invocation(lambda_pose=.01, pose_timesteps=(.1, .2), run_name="isolated", checkpoint_dir=temporary, parent_checkpoint=parent, verify_parent=True)
            self.assertEqual((value, timesteps), (.01, (.1, .2)))
            self.assertEqual(destination, Path(temporary) / "isolated")
            destination.mkdir(); (destination / "old.pt").touch()
            with self.assertRaises(FileExistsError):
                validate_smoke_invocation(lambda_pose=.01, pose_timesteps=(.1,), run_name="isolated", checkpoint_dir=temporary, parent_checkpoint=parent, verify_parent=True)
            resume_parent = destination / "step_001610.pt"; resume_parent.touch()
            self.assertEqual(
                validate_smoke_invocation(lambda_pose=.01, pose_timesteps=(.1,), run_name="isolated",
                                          checkpoint_dir=temporary, parent_checkpoint=resume_parent,
                                          verify_parent=True, allow_existing_destination=True)[2],
                destination,
            )

    def test_explicit_parent_sha_is_optional_but_enforced_when_supplied(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_001500.pt"
            save_training_state(path, self._state(global_step=1500))
            with patch("scripts.train_pose_reward_smoke._sha256", return_value="expected"):
                self.assertEqual(_validate_parent(path, None)["global_step"], 1500)
                self.assertEqual(_validate_parent(path, "expected")["global_step"], 1500)
            with patch("scripts.train_pose_reward_smoke._sha256", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    _validate_parent(path, "expected")

    def test_gate_e_intermediate_resume_and_target_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "gate-e"; destination.mkdir()
            cfg = self._gate_e_cfg(destination, max_steps=1700)
            state = self._state(global_step=1537, config=asdict(cfg))
            state[GATE_E_METADATA_KEY] = self._metadata(cfg, state["model"])
            checkpoint = destination / "step_001537.pt"; save_training_state(checkpoint, state)
            loaded = _validate_parent(checkpoint, None)
            counters = validate_gate_e_resume_checkpoint(checkpoint, loaded, cfg=cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5,
                                                         timestep_min=.10, timestep_max=.20,
                                                         forced_exposure_probability=.05, hf_subdir=cfg.run_name,
                                                         immutable_parent=self._immutable_parent())
            self.assertEqual(counters["total_active_samples"], 3)
            validate_gate_e_destination(checkpoint, destination, loaded_global_step=1537, target_global_step=1700)
            self.assertEqual(resolve_target_global_step(1537, target_global_step=1700, max_steps=None), 1700)
            self.assertEqual(resolve_target_global_step(1537, target_global_step=None, max_steps=163), 1700)
            with self.assertRaisesRegex(ValueError, "strictly greater"):
                resolve_target_global_step(1537, target_global_step=1537, max_steps=None)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                resolve_target_global_step(1537, target_global_step=1700, max_steps=163)

    def test_non_gpu_setup_keeps_checkpoint_path_and_state_separate_for_new_and_resume(self):
        """Exercise main's pre-GPU lifecycle without loading a model or contacting W&B/HF."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_parent = root / "canonical" / "step_001500.pt"
            canonical_parent.parent.mkdir()
            save_training_state(canonical_parent, self._state(global_step=1500))
            destination = root / "gate-e"
            common = {
                "raw_ckpt": "raw", "latent_root": "shards", "checkpoint_dir": str(root),
                "run_name": "gate-e", "microbatch_size": 1,
                "gradient_accumulation_steps": 32, "save_every": 25,
                "hf_repo_id": "user/private", "hf_subdir": "gate-e",
                "hf_mirror_every_steps": 25, "pose_loss": "gaussian_heatmap_kl", "lambda_pose": 2e-5,
                "timestep_min": .10, "timestep_max": .20,
                "forced_exposure_probability": .05, "target_global_step": 1700,
                "max_steps": None, "destination": destination,
            }
            new_setup = prepare_gate_e_run_setup(
                parent_path=canonical_parent, expected_parent_sha256=None, **common,
            )
            self.assertIsInstance(new_setup.parent_path, Path)
            self.assertIsInstance(new_setup.parent_state, dict)
            self.assertEqual(new_setup.parent_path, canonical_parent)
            self.assertEqual(new_setup.parent_state["global_step"], 1500)
            self.assertEqual(new_setup.destination, destination)
            self.assertFalse(destination.exists())

            destination.mkdir()
            resumed_state = self._state(global_step=1537, config=asdict(new_setup.cfg))
            resumed_state[GATE_E_METADATA_KEY] = _gate_e_metadata(
                new_setup.cfg, pose_loss=common["pose_loss"], lambda_pose=common["lambda_pose"],
                timestep_min=common["timestep_min"], timestep_max=common["timestep_max"],
                forced_exposure_probability=common["forced_exposure_probability"],
                hf_subdir=common["hf_subdir"], immutable_parent=new_setup.immutable_parent,
                cumulative_counters={"eligible_samples_seen": 4, "forced_samples": 1,
                                     "naturally_active_samples": 2, "total_active_samples": 3},
                model_state=resumed_state["model"],
            )
            resume_path = save_training_state(destination / "step_001537.pt", resumed_state)
            resumed_setup = prepare_gate_e_run_setup(
                parent_path=resume_path, expected_parent_sha256=None, **common,
            )
            self.assertIsInstance(resumed_setup.parent_path, Path)
            self.assertIsInstance(resumed_setup.parent_state, dict)
            self.assertEqual(resumed_setup.parent_path, resume_path)
            self.assertEqual(resumed_setup.parent_state["global_step"], 1537)
            self.assertEqual(resumed_setup.cumulative_counters["total_active_samples"], 3)

    def test_intermediate_expected_sha_is_optional_but_enforced_when_supplied(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step_001537.pt"; save_training_state(path, self._state(global_step=1537))
            with patch("scripts.train_pose_reward_smoke._sha256", return_value="expected"):
                self.assertEqual(_validate_parent(path, "expected")["global_step"], 1537)
            with patch("scripts.train_pose_reward_smoke._sha256", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    _validate_parent(path, "expected")

    def test_target_and_publication_cadence_are_not_branch_constants(self):
        self.assertEqual(resolve_target_global_step(1500, target_global_step=1650, max_steps=None), 1650)
        self.assertEqual(resolve_target_global_step(1500, target_global_step=None, max_steps=137), 1637)
        self.assertEqual(checkpoint_publication_steps(1500, 1650, 25),
                         (1525, 1550, 1575, 1600, 1625, 1650))
        self.assertEqual(checkpoint_publication_steps(1537, 1601, 25), (1550, 1575, 1600, 1601))
        source = Path("scripts/train_pose_reward_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn("pose-reward-coord-exposure10pct-l1e5-t010-020", source)

    def test_legacy_gate_e_intermediate_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "gate-e"; destination.mkdir()
            cfg = self._gate_e_cfg(destination)
            checkpoint = destination / "step_001610.pt"; save_training_state(checkpoint, self._state(config=asdict(cfg)))
            with self.assertRaisesRegex(ValueError, "metadata"):
                validate_gate_e_resume_checkpoint(
                    checkpoint, _validate_parent(checkpoint, None), cfg=cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5,
                    timestep_min=.10, timestep_max=.20, forced_exposure_probability=.05,
                    hf_subdir=cfg.run_name, immutable_parent=self._immutable_parent(),
                )

    def test_incompatible_gate_e_resume_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "gate-e"; destination.mkdir()
            cfg = self._gate_e_cfg(destination)
            state = self._state(config=asdict(cfg))
            state[GATE_E_METADATA_KEY] = self._metadata(cfg, state["model"])
            checkpoint = destination / "step_001610.pt"; save_training_state(checkpoint, state)
            with self.assertRaisesRegex(ValueError, "lambda_pose"):
                validate_gate_e_resume_checkpoint(checkpoint, state, cfg=cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=3e-5,
                                                  timestep_min=.10, timestep_max=.20,
                                                  forced_exposure_probability=.05, hf_subdir=cfg.run_name,
                                                  immutable_parent=self._immutable_parent())
            mismatched_cfg = replace(cfg, gradient_accumulation_steps=16)
            with self.assertRaisesRegex(ValueError, "critical_train_config"):
                validate_gate_e_resume_checkpoint(checkpoint, state, cfg=mismatched_cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5,
                                                  timestep_min=.10, timestep_max=.20,
                                                  forced_exposure_probability=.05, hf_subdir=cfg.run_name,
                                                  immutable_parent=self._immutable_parent())
            with self.assertRaisesRegex(ValueError, "forced_exposure_probability"):
                validate_gate_e_resume_checkpoint(checkpoint, state, cfg=cfg, pose_loss="gaussian_heatmap_kl", lambda_pose=2e-5,
                                                  timestep_min=.10, timestep_max=.20,
                                                  forced_exposure_probability=.0, hf_subdir=cfg.run_name,
                                                  immutable_parent=self._immutable_parent())
            with self.assertRaisesRegex(ValueError, "pose_loss"):
                validate_gate_e_resume_checkpoint(checkpoint, state, cfg=cfg,
                                                  pose_loss="normalized_coordinate_huber", lambda_pose=2e-5,
                                                  timestep_min=.10, timestep_max=.20,
                                                  forced_exposure_probability=.05, hf_subdir=cfg.run_name,
                                                  immutable_parent=self._immutable_parent())

    def test_checkpoint_metadata_records_selected_pose_loss(self):
        cfg = self._gate_e_cfg(Path("/tmp") / "gate-e")
        metadata = _gate_e_metadata(
            cfg, pose_loss="normalized_coordinate_huber", lambda_pose=1e-5,
            timestep_min=.10, timestep_max=.20, forced_exposure_probability=.10,
            hf_subdir=cfg.run_name, immutable_parent=self._immutable_parent(),
            cumulative_counters={"eligible_samples_seen": 0, "forced_samples": 0,
                                 "naturally_active_samples": 0, "total_active_samples": 0},
            model_state={"first.weight": torch.ones(1)},
        )
        self.assertEqual(metadata["pose_loss"], "normalized_coordinate_huber")

    def test_controlled_branch_hf_namespace_is_fail_closed_and_upload_preserves_local(self):
        self.assertEqual(_validate_hf_branch_args(hf_repo_id="user/private", hf_subdir="isolated",
                                                  run_name="isolated", save_every=50,
                                                  mirror_every_steps=50), "isolated")
        for arguments in (("", "isolated", "isolated", 50, 50),
                          ("user/private", "old-branch", "isolated", 50, 50),
                          ("user/private", "isolated", "isolated", 50, 100)):
            with self.assertRaises(ValueError):
                _validate_hf_branch_args(hf_repo_id=arguments[0], hf_subdir=arguments[1], run_name=arguments[2],
                                         save_every=arguments[3], mirror_every_steps=arguments[4])

        class MemoryApi:
            def __init__(self): self.files = {}
            def create_repo(self, *args, **kwargs): pass
            def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
                self.files[path_in_repo] = (Path(path_or_fileobj).read_bytes() if isinstance(path_or_fileobj, str)
                                            else path_or_fileobj.read())

        class FailingApi:
            def create_repo(self, *args, **kwargs): pass
            def upload_file(self, **kwargs): raise OSError("offline")

        with tempfile.TemporaryDirectory() as temporary:
            local = save_training_state(Path(temporary) / "step_001550.pt", self._state(global_step=1550))
            original = local.read_bytes(); api = MemoryApi()
            mirror = HFTrainingCheckpointMirror(repo_id="user/private", run_name="isolated", api=api,
                                                prune_local_after_success=False)
            self.assertTrue(mirror._upload(local, "step"))
            self.assertEqual(local.read_bytes(), original)
            self.assertIn("isolated/full/step_001550.pt", api.files)
            self.assertNotIn("old-branch/full/step_001550.pt", api.files)
            failure = HFTrainingCheckpointMirror(repo_id="user/private", run_name="isolated", api=FailingApi(),
                                                 max_attempts=1, prune_local_after_success=False)
            self.assertFalse(failure._upload(local, "step"))
            self.assertEqual(local.read_bytes(), original)

    def test_destination_and_checkpoint_publish_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); canonical_parent = root / "source" / "step_001500.pt"; canonical_parent.parent.mkdir()
            save_training_state(canonical_parent, self._state(global_step=1500))
            existing = root / "existing"; existing.mkdir()
            with self.assertRaises(FileExistsError):
                validate_gate_e_destination(canonical_parent, existing, loaded_global_step=1500, target_global_step=1700)
            target = existing / "step_001700.pt"; target.touch()
            with self.assertRaises(FileExistsError):
                validate_gate_e_destination(existing / "step_001610.pt", existing, loaded_global_step=1610, target_global_step=1700)
            with self.assertRaises(FileExistsError):
                save_training_state(target, self._state(), overwrite=False)

    def test_accumulation_diagnostics_include_earlier_active_microbatch(self):
        metrics = aggregate_step_diagnostics([
            {"flow_loss": 1.0, "pose_loss": 2.0, "total_loss": 1.00004, "pose_active_count": 1,
             "pose_eligible_count": 1, "timesteps": [.15]},
            {"flow_loss": 3.0, "pose_loss": None, "total_loss": 3.0, "pose_active_count": 0,
             "pose_eligible_count": 1, "timesteps": [.55]},
        ])
        self.assertEqual(metrics["pose_active_samples_step"], 1)
        self.assertEqual(metrics["pose_active_microbatches_step"], 1)
        self.assertEqual(metrics["pose_eligible_samples_step"], 2)
        self.assertEqual(metrics["pose_active_fraction"], .5)
        self.assertEqual(metrics["pose_loss_mean_active"], 2.0)
        self.assertEqual(metrics["pose_loss_max_active"], 2.0)
        self.assertEqual(metrics["flow_loss_mean_step"], 2.0)
        self.assertAlmostEqual(metrics["total_loss_mean_step"], 2.00002)
        self.assertAlmostEqual(metrics["timestep_min_step"], .15)
        self.assertAlmostEqual(metrics["timestep_max_step"], .55)
        self.assertAlmostEqual(metrics["timestep_mean_step"], .35)

    def test_accumulation_diagnostics_represent_all_zero_activity(self):
        metrics = aggregate_step_diagnostics([
            {"flow_loss": 1.0, "pose_loss": None, "total_loss": 1.0, "pose_active_count": 0,
             "pose_eligible_count": 1, "timesteps": [.30]},
            {"flow_loss": 3.0, "pose_loss": None, "total_loss": 3.0, "pose_active_count": 0,
             "pose_eligible_count": 0, "timesteps": [.40]},
        ])
        self.assertEqual(metrics["pose_active_samples_step"], 0)
        self.assertEqual(metrics["pose_active_microbatches_step"], 0)
        self.assertEqual(metrics["pose_eligible_samples_step"], 1)
        self.assertIsNone(metrics["pose_loss_mean_active"])
        self.assertIsNone(metrics["pose_loss_max_active"])

    def test_controlled_accumulation_and_cumulative_accounting(self):
        metrics = aggregate_step_diagnostics([
            {"flow_loss": 1.0, "pose_loss": 2.0, "total_loss": 1.00004, "pose_active_count": 1,
             "pose_eligible_count": 1, "pose_forced_count": 1, "pose_natural_active_count": 0,
             "timesteps": [.15], "active_timesteps": [.15]},
            {"flow_loss": 3.0, "pose_loss": 4.0, "total_loss": 3.00008, "pose_active_count": 1,
             "pose_eligible_count": 2, "pose_forced_count": 0, "pose_natural_active_count": 1,
             "timesteps": [.55], "active_timesteps": [.55]},
        ])
        self.assertEqual(metrics["pose_forced_samples_step"], 1)
        self.assertEqual(metrics["pose_natural_active_samples_step"], 1)
        self.assertEqual(metrics["pose_active_samples_step"], 2)
        self.assertAlmostEqual(metrics["pose_forced_fraction_of_eligible_step"], 1 / 3)
        self.assertAlmostEqual(metrics["pose_total_active_fraction_of_eligible_step"], 2 / 3)
        self.assertAlmostEqual(metrics["active_timestep_mean_step"], .35)
        cumulative = update_cumulative_counters({"eligible_samples_seen": 7, "forced_samples": 2,
                                                 "naturally_active_samples": 3, "total_active_samples": 5}, metrics)
        self.assertEqual(cumulative, {"eligible_samples_seen": 10, "forced_samples": 3,
                                      "naturally_active_samples": 4, "total_active_samples": 7})

    def test_diagnostic_aggregation_preserves_backward_scaling_and_rng(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        ((parameter * 2 + parameter * 4) / 2).backward()
        self.assertEqual(float(parameter.grad), 3.0)
        state = torch.get_rng_state(); expected_next = torch.rand(1)
        torch.set_rng_state(state)
        aggregate_step_diagnostics([
            {"flow_loss": 1.0, "pose_loss": None, "total_loss": 1.0, "pose_active_count": 0,
             "pose_eligible_count": 0, "timesteps": [.3]},
        ])
        self.assertTrue(torch.equal(torch.rand(1), expected_next))

    def test_no_optimizer_step_is_in_gradient_math(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0])); optimizer = torch.optim.SGD([parameter], lr=1.0)
        _ = gradient_interaction([torch.tensor([2.0])], [torch.tensor([3.0])])
        self.assertEqual(float(parameter.item()), 1.0)
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_trainable_selection_and_frozen_boundary(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        model[1].requires_grad_(False)
        selected = select_trainable_named_parameters(model)
        self.assertEqual({name for name, _ in selected}, {"0.weight", "0.bias"})
        model[1].weight.grad = torch.ones_like(model[1].weight)
        with self.assertRaises(RuntimeError): select_trainable_named_parameters(model)

    def test_invalid_joints_contribute_zero_pose_loss(self):
        logits = torch.randn(1, 17, 4, 4)
        targets = torch.zeros(1, 17, 2)
        boxes = torch.tensor([[0.0, 0.0, 32.0, 32.0]])
        valid = torch.zeros(1, 17, dtype=torch.bool); valid[0, 0] = True
        first = gaussian_heatmap_kl(logits, targets, boxes, valid)
        targets[0, 1:] = 1e6
        second = gaussian_heatmap_kl(logits, targets, boxes, valid)
        self.assertTrue(torch.allclose(first, second))

    def test_smoke_checkpoint_schema_uses_existing_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "isolated" / "step_001501.pt"
            state = {"model": {"first.weight": torch.ones(1)}, "optimizer": {"state": {}},
                     "scheduler": {"step_count": 1501}, "global_step": 1501, "epoch": 0, "batch_position": 0,
                     "rng": {"python": None, "numpy": None, "torch": torch.get_rng_state(), "cuda": None},
                     "flow_generator_state": torch.Generator().get_state(), "config": {"run_name": "isolated"}}
            save_training_state(path, state)
            self.assertEqual(load_training_state(path)["global_step"], 1501)


if __name__ == "__main__":
    unittest.main()
