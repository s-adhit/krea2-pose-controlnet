import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from pose_controlnet.post1500_evaluation import score_authoritative_pck
from pose_controlnet.turbo_evaluation import (
    assert_exact_diagnostic_stems,
    discover_turbo_checkpoint_steps,
    exact_direct_local_turbo_checkpoints,
    exact_local_turbo_checkpoints,
    controlled_branch_metadata,
    load_turbo_experiment_spec,
    normalize_turbo_steps,
    turbo_metadata,
)
from scripts import turbo_benchmark


ROOT = Path(__file__).resolve().parents[1]


def make_spec(root: Path) -> Path:
    value = {
        "experiment_name": "arbitrary-run", "checkpoint_root": str(root / "checkpoints"),
        "hf_repo_id": "org/private", "hf_namespace": "arbitrary-run/full/", "output_root": str(root / "evaluation"),
        "labels": {"checkpoint_template": "candidate {step}"}, "training_metadata": {"lora_learning_rate": 0.00005},
        "turbo_contract": {**turbo_metadata(), "control_scale": 1.0},
        "diagnostics": {"canonical_manifest": str(root / "diagnostic.jsonl"), "canonical_reference_spec": str(root / "canonical.json"), "expected_count": 2, "seed": 420200},
        "paths": {"latent_root": str(root / "latents"), "text_conditioning_root": str(root / "text"), "turbo_ckpt": str(root / "turbo.safetensors"), "reference_sidecar": str(root / "sidecar.json")},
    }
    path = root / "experiment.json"; path.write_text(json.dumps(value)); return path


def score_row(step: int) -> dict:
    pose = {"detection_coverage": .9, "joint_evaluation_coverage": .8, "pck_005": .1, "pck_010": .2, "pck_020": .3,
            "per_source": {source: {key: value for key, value in (("pck_005", .1), ("pck_010", .2), ("pck_020", .3))} for source in ("COCO", "Human-Art")},
            "multi_person": {"pck_005": .1, "pck_010": .2, "pck_020": .3}, "single_person": {"pck_005": .1, "pck_010": .2, "pck_020": .3},
            "matched_people": 2, "unmatched_reference_people": 1, "predicted_people": 3, "unmatched_predicted_people": 1}
    return {"checkpoint_step": step, "pose": pose, "clip": {"mean_cosine_similarity": .4}}


class TurboEvaluationTest(unittest.TestCase):
    @staticmethod
    def _controlled_state(step: int, *, pose_loss: str = "gaussian_heatmap_kl",
                          lambda_pose: float = 1e-5, microbatch: int = 1) -> dict:
        return {
            "global_step": step,
            "config": {"microbatch_size": microbatch, "gradient_accumulation_steps": 32,
                       "save_every": 25, "max_steps": 1650},
            "gate_e": {
                "pose_loss": pose_loss, "temperature": 1.0,
                "lambda_pose": lambda_pose, "pose_timestep_window": [.1, .2],
                "forced_exposure_probability": .05, "forced_sampler_policy": "policy-v1",
                "immutable_parent": {"global_step": 1500, "sha256": "a" * 64, "filename": "step_001500.pt"},
                "hf_subdir": "arbitrary-run",
                "cumulative_counters": {"eligible_samples_seen": step, "forced_samples": 1,
                                        "naturally_active_samples": 2, "total_active_samples": 3},
            },
        }

    def test_dynamic_experiment_cli_derives_arbitrary_branch_metadata_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoints = root / "checkpoints"; checkpoints.mkdir()
            paths = []
            for step in (1525, 1550, 1600):
                path = checkpoints / f"step_{step:06d}.pt"; path.write_bytes(f"checkpoint-{step}".encode()); paths.append(path)
            digests = [f"{step}={hashlib.sha256(path.read_bytes()).hexdigest()}" for step, path in zip((1525, 1550, 1600), paths)]
            args = turbo_benchmark.parser().parse_args([
                "experiment", "--checkpoint-root", str(checkpoints), "--steps", "1525", "1550", "1600",
                "--output-root", str(root / "isolated-output"), "--experiment-name", "arbitrary-run",
                "--checkpoint-label-template", "candidate {step}", "--expected-sha256", *digests,
            ])
            with patch("pose_controlnet.turbo_evaluation.load_training_state", side_effect=lambda path: self._controlled_state(int(Path(path).stem.split("_")[1]))):
                config = turbo_benchmark._config(args)
            self.assertEqual(config.steps, (1525, 1550, 1600))
            self.assertEqual(config.labels["checkpoint_template"], "candidate {step}")
            self.assertEqual(config.output_root, root / "isolated-output")
            self.assertEqual(config.training_metadata["lambda_pose"], 1e-5)
            self.assertEqual(config.training_metadata["per_checkpoint"]["1550"]["microbatch_size"], 1)

    def test_controlled_branch_metadata_rejects_inconsistent_runtime_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); first = root / "step_001525.pt"; second = root / "step_001550.pt"
            first.write_bytes(b"first"); second.write_bytes(b"second")
            def state(path):
                step = int(Path(path).stem.split("_")[1])
                return self._controlled_state(step, microbatch=2 if step == 1550 else 1)
            with patch("pose_controlnet.turbo_evaluation.load_training_state", side_effect=state):
                with self.assertRaisesRegex(ValueError, "runtime metadata is inconsistent"):
                    controlled_branch_metadata(((1525, first), (1550, second)))

    def test_dynamic_provenance_extracts_selected_coordinate_pose_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = root / "step_001525.pt"
            checkpoint.write_bytes(b"coordinate")
            with patch("pose_controlnet.turbo_evaluation.load_training_state",
                       return_value=self._controlled_state(1525, pose_loss="normalized_coordinate_huber")):
                metadata = controlled_branch_metadata(((1525, checkpoint),))
        self.assertEqual(metadata["pose_loss"], "normalized_coordinate_huber")

    def test_current_spec_pins_contract_and_established_24_diagnostics(self):
        config = load_turbo_experiment_spec(ROOT / "configs/evaluation/controlinput_lr2x_turbo.json")
        self.assertEqual(config.experiment_name, "pose-learning-1500-controlinput-lr2x-to2800")
        self.assertEqual(len(turbo_benchmark._manifest_stems(ROOT / config.diagnostics["canonical_manifest"])), 24)
        self.assertEqual(config.diagnostics["expected_count"], 24)
        self.assertEqual({**turbo_metadata(), "control_scale": 1.0}["steps"], 8)

    def test_gate_e_spec_pins_exact_local_checkpoints_training_facts_and_step1700_sha(self):
        config = load_turbo_experiment_spec(ROOT / "configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json")
        self.assertEqual(config.steps, (1550, 1600, 1650, 1700))
        self.assertEqual(config.checkpoint_validation["mode"], "direct_local")
        self.assertEqual(config.checkpoint_validation["expected_sha256"]["1700"], "b454cfff01e6c2608415abc54d910682be9705d1ea337b342511fe1586828415")
        self.assertEqual(config.training_metadata["resumed_exposure"]["active_fraction_percent"], .7987)
        deltas = turbo_benchmark._deltas(score_row(1700), score_row(1500), config)
        self.assertEqual(deltas["coco_pck_pck_020"], 0.0)
        self.assertEqual(deltas["human_art_pck_pck_010"], 0.0)
        self.assertEqual(deltas["single_person_pck_pck_005"], 0.0)
        self.assertEqual(deltas["multi_person_pck_pck_020"], 0.0)

    def test_arbitrary_steps_duplicates_and_cli_selection(self):
        args = turbo_benchmark.parser().parse_args(["preflight", "--spec", "example.json", "--steps", "1600", "1700", "1900"])
        self.assertEqual(normalize_turbo_steps(args.steps), (1600, 1700, 1900))
        with self.assertRaisesRegex(ValueError, "unique"): normalize_turbo_steps((1600, 1600))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            turbo_benchmark.parser().parse_args(["preflight", "--spec", "x.json", "--steps", "1600", "--all-checkpoints"])

    def test_all_checkpoints_discovers_configured_direct_root_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); root.mkdir(exist_ok=True)
            for step in (2200, 1600, 1900): (root / f"step_{step:06d}.pt").touch()
            (root / "nested").mkdir(); (root / "nested/step_009999.pt").touch()
            with patch("pose_controlnet.turbo_evaluation.load_training_state", side_effect=lambda path: {"global_step": int(Path(path).stem.split("_")[1])}):
                self.assertEqual(discover_turbo_checkpoint_steps(root), (1600, 1900, 2200))

    def test_exact_local_marker_sha_schema_step_validation_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoints = root / "checkpoints"; checkpoints.mkdir(); point = checkpoints / "step_001600.pt"; point.touch()
            with patch("pose_controlnet.turbo_evaluation.validated_local_checkpoint_for_hf_step", return_value=point) as validate, patch("pose_controlnet.turbo_evaluation.load_training_state", return_value={"global_step": 1600}):
                self.assertEqual(exact_local_turbo_checkpoints(checkpoint_root=checkpoints, hf_repo_id="org/repo", hf_namespace="run/full/", marker_download_dir=root / "markers", steps=(1600,)), [(1600, point)])
            self.assertEqual(validate.call_args.kwargs["run_name"], "run")
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                exact_local_turbo_checkpoints(checkpoint_root=checkpoints, hf_repo_id="org/repo", hf_namespace="run/full/", marker_download_dir=root / "markers", steps=(1700,))
        source = Path("pose_controlnet/turbo_evaluation.py").read_text(); resolver = source[source.index("def exact_local_turbo_checkpoints"):source.index("def turbo_schedule")]
        self.assertNotIn("validated_hf_checkpoint_for_step", resolver); self.assertNotIn("newest_valid", resolver)

    def test_direct_local_checkpoint_selection_has_no_marker_fallback_and_honors_supplied_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = root / "step_001700.pt"; checkpoint.write_bytes(b"gate-e")
            with patch("pose_controlnet.turbo_evaluation.load_training_state", return_value={"global_step": 1700}):
                self.assertEqual(exact_direct_local_turbo_checkpoints(checkpoint_root=root, steps=(1700,)), [(1700, checkpoint)])
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    exact_direct_local_turbo_checkpoints(checkpoint_root=root, steps=(1700,), expected_sha256={"1700": "0" * 64})

    def test_existing_complete_generation_and_scores_are_dynamic_and_partial_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = load_turbo_experiment_spec(make_spec(root)); output = config.output_root; output.mkdir()
            spec = {"kind": "turbo_fixed_pose", "seed": 420200, "stems": ["a", "b"], "per_stem_seeds": {}, "sample_identities": {}, "turbo": turbo_metadata()}
            for stem in spec["stems"]:
                directory = output / "fixed_pose" / stem; directory.mkdir(parents=True); Image.new("RGB", (1, 1)).save(directory / "step_001600.png")
                (directory / "metadata.json").write_text(json.dumps({"stem": stem, "control_scale": 1.0, **turbo_metadata()}))
            (output / "generation_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0, "stems": ["a", "b"], "generated_steps": {"a": [1600], "b": [1600]}}))
            self.assertEqual(turbo_benchmark._generation_status(output, spec["stems"], 1600), "complete")
            self.assertEqual(turbo_benchmark._merged_generation_steps(output, spec["stems"], {"a": [1700], "b": [1700]}), {"a": [1600, 1700], "b": [1600, 1700]})
            (output / "pck_clip_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0, "checkpoints": [score_row(1600)]}))
            self.assertEqual(list(turbo_benchmark._existing_scored_rows(output, spec, config)), [1600])
            (output / "fixed_pose/b/step_001600.png").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                turbo_benchmark._generation_status(output, spec["stems"], 1600)

    def test_report_discovers_sorts_and_renders_canonical_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); spec_path = make_spec(root); config = load_turbo_experiment_spec(spec_path); output = config.output_root; output.mkdir()
            spec = {"kind": "turbo_fixed_pose", "seed": 420200, "stems": ["second", "first"], "per_stem_seeds": {}, "sample_identities": {}, "turbo": turbo_metadata(), "experiment_name": config.experiment_name}
            (output / "turbo_spec.json").write_text(json.dumps(spec)); (output / "generation_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0, "stems": spec["stems"], "generated_steps": {stem: [1600, 2200] for stem in spec["stems"]}}))
            for stem in spec["stems"]:
                directory = output / "fixed_pose" / stem; directory.mkdir(parents=True); Image.new("RGB", (1, 1)).save(directory / "control.png")
                (directory / "metadata.json").write_text(json.dumps({"stem": stem, "control_scale": 1.0, **turbo_metadata()}))
                for step in (1600, 2200): Image.new("RGB", (1, 1)).save(directory / f"step_{step:06d}.png")
            (output / "pck_clip_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0, "experiment_name": config.experiment_name, "checkpoints": [score_row(2200), score_row(1600)]}))
            args = SimpleNamespace(spec=str(spec_path), checkpoint_root=None, hf_repo_id=None, hf_namespace=None, output_root=None, diagnostic_manifest=None, latent_root=None, text_conditioning_root=None, turbo_ckpt=None, dataset_root=None, reference_sidecar=None, clip_model_id=None)
            calls = []
            with patch.object(turbo_benchmark, "_dataset_and_spec", return_value=(object(), tuple(spec["stems"]), spec)), patch.object(turbo_benchmark, "make_contact_sheet", side_effect=lambda rows, *rest, **kw: calls.append((rows, kw["column_labels"]))):
                turbo_benchmark.report(args)
            summary = json.loads((output / "evaluation_summary.json").read_text())
            self.assertEqual([row["checkpoint_step"] for row in summary["checkpoints"]], [1600, 2200])
            self.assertEqual([row[0] for row in calls[1][0]], ["second", "first"])
            self.assertEqual(calls[1][1], ("control", "candidate 1600", "candidate 2200"))
            self.assertEqual(summary["qualitative_grids"]["full_contact_sheet"], "arbitrary-run_full_contact_sheet.png")
            ranking = json.loads((output / "checkpoint_ranking.json").read_text())
            self.assertEqual(ranking["ranking"]["pck_020"][0]["checkpoint_step"], 1600)

    def test_report_keeps_numerical_baseline_when_its_sample_images_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = make_spec(root)
            baseline_root = root / "baseline"
            spec_payload = json.loads(spec_path.read_text())
            spec_payload["baseline"] = {"output_root": str(baseline_root), "checkpoint_step": 1500, "label": "baseline 1500"}
            spec_path.write_text(json.dumps(spec_payload))
            config = load_turbo_experiment_spec(spec_path)
            output = config.output_root; output.mkdir()
            stems = ("first", "second")
            spec = {"kind": "turbo_fixed_pose", "seed": 420200, "stems": list(stems), "per_stem_seeds": {}, "sample_identities": {}, "turbo": turbo_metadata(), "experiment_name": config.experiment_name}
            baseline_root.mkdir()
            (baseline_root / "turbo_spec.json").write_text(json.dumps(spec))
            (baseline_root / "pck_clip_results.json").write_text(json.dumps({"checkpoints": [score_row(1500)]}))
            (output / "turbo_spec.json").write_text(json.dumps(spec))
            (output / "generation_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0, "stems": list(stems), "generated_steps": {stem: [1550] for stem in stems}}))
            for stem in stems:
                directory = output / "fixed_pose" / stem; directory.mkdir(parents=True)
                Image.new("RGB", (1, 1)).save(directory / "control.png")
                Image.new("RGB", (1, 1)).save(directory / "step_001550.png")
                (directory / "metadata.json").write_text(json.dumps({"stem": stem, "control_scale": 1.0, **turbo_metadata()}))
            (output / "pck_clip_results.json").write_text(json.dumps({"metadata": turbo_metadata(), "control_scale": 1.0, "experiment_name": config.experiment_name, "checkpoints": [score_row(1550)]}))
            args = SimpleNamespace(spec=str(spec_path), checkpoint_root=None, hf_repo_id=None, hf_namespace=None, output_root=None, diagnostic_manifest=None, latent_root=None, text_conditioning_root=None, turbo_ckpt=None, dataset_root=None, reference_sidecar=None, clip_model_id=None)
            calls = []
            with patch.object(turbo_benchmark, "_dataset_and_spec", return_value=(object(), stems, spec)), patch.object(turbo_benchmark, "make_contact_sheet", side_effect=lambda rows, *rest, **kw: calls.append((rows, kw["column_labels"]))):
                turbo_benchmark.report(args)
            summary = json.loads((output / "evaluation_summary.json").read_text())
            self.assertEqual(summary["baseline"]["result"], score_row(1500))
            self.assertEqual(summary["deltas_vs_baseline"]["1550"], turbo_benchmark._deltas(score_row(1550), score_row(1500), config))
            self.assertFalse(summary["baseline_visual_artifacts_available"])
            self.assertEqual(summary["baseline_visual_artifacts_missing_count"], 2)
            self.assertEqual(calls[1][1], ("control", "candidate 1550"))
            (output / "fixed_pose" / "second" / "step_001550.png").unlink()
            with patch.object(turbo_benchmark, "_dataset_and_spec", return_value=(object(), stems, spec)):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    turbo_benchmark.report(args)

    def test_metrics_are_shared_and_no_train_backward_optimizer_or_resume_path_exists(self):
        self.assertIs(turbo_benchmark.score_authoritative_pck, score_authoritative_pck)
        source = Path(turbo_benchmark.__file__).read_text().lower()
        self.assertIn("confidence_threshold=.5", source); self.assertIn("turbo_scoring_geometry", source)
        for forbidden in ("torch.optim", "optimizer.", "backward(", ".backward(", "model.train(", "resume"):
            self.assertNotIn(forbidden, source)
        self.assertEqual(assert_exact_diagnostic_stems(("a", "b"), ("b", "a"), expected_count=2), ("a", "b"))


if __name__ == "__main__": unittest.main()
