import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pose_controlnet.config import TrainConfig
from pose_controlnet.evaluation import CHECKPOINT_STEPS
from pose_controlnet.post1500_evaluation import (fixed_timestep_loss_and_sensitivity, merge_checkpoint_results,
    score_authoritative_pck, telemetry_audit, timestep_distribution_audit)


def _flow_payload(steps):
    spec = {"seed": 420100, "stems": ["a"]}
    return {"kind": "fixed_flow", "spec": spec, "checkpoints": [{"checkpoint_step": step, "loss": step} for step in steps]}


class _Dataset:
    records = [("x", 0, (4, 4), "a")]
    def __getitem__(self, _):
        return {"stem": "a", "latent": torch.ones(16, 4, 4), "control": torch.ones(16, 4, 4),
                "context": torch.ones(2, 1, 1, dtype=torch.bfloat16), "mask": torch.ones(2, dtype=torch.bool)}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(patch=2)


class Post1500EvaluationTest(unittest.TestCase):
    def test_exact_series_merge_rejects_incomplete_or_conflicting_history(self):
        old = _flow_payload(CHECKPOINT_STEPS[:11]); new = _flow_payload(CHECKPOINT_STEPS[11:])
        merged = merge_checkpoint_results(old, new)
        self.assertEqual(tuple(row["checkpoint_step"] for row in merged["checkpoints"]), CHECKPOINT_STEPS)
        bad = _flow_payload(CHECKPOINT_STEPS[11:]); bad["checkpoints"][0]["loss"] = -1
        with self.assertRaises(ValueError): merge_checkpoint_results(merged, bad)

    def test_timestep_audit_is_deterministic_and_bucket_weighted(self):
        cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards")
        buckets = {(1024, 1024): 3, (768, 1280): 1}
        first = timestep_distribution_audit(buckets, cfg, seed=99, samples_per_bucket=64)
        second = timestep_distribution_audit(buckets, cfg, seed=99, samples_per_bucket=64)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["bucket_weight"] for row in first["per_bucket"]), 4)
        self.assertEqual(first["overall_bucket_weighted"]["sample_count"], 128)
        self.assertEqual(first["overall_bucket_weighted"]["effective_bucket_weight"], 256.0)

    def test_pck_pool_excludes_danbooru_and_breaks_down_single_multi(self):
        source = [[float(index), 0., 2. if index in (5, 6, 7, 9, 11, 13, 15) else 0.] for index in range(17)]
        sidecar = {"records": [
            {"stem": "real_human_humanart_1", "source": "humanart", "status": "available", "people": [{"annotation_id": 1, "keypoints": source}]},
            {"stem": "coco_1_1", "source": "coco", "status": "available", "mode": "single", "people": [{"annotation_id": 2, "keypoints": source}, {"annotation_id": 3, "keypoints": source}]},
            {"stem": "danbooru_anime_1", "source": "danbooru", "status": "unavailable", "people": []},
        ]}
        geometry = {stem: {"source_size": [20, 20], "resized_size": [20, 20], "crop_box": [0, 0, 20, 20]} for stem in ("real_human_humanart_1", "coco_1_1", "danbooru_anime_1")}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for stem in geometry:
                (root / f"{stem}.png").touch()
            def detector(path):
                if path.stem.startswith("coco"):
                    return [{"keypoints": source}]
                return [{"keypoints": source}]
            result = score_authoritative_pck(sidecar=sidecar, geometry_by_stem=geometry, image_for=lambda stem: root / f"{stem}.png", detector=detector)
        self.assertEqual(result["unavailable_excluded_sample_count"], 1)
        self.assertIsNone(result["per_source"]["Danbooru unavailable"]["pck_020"])
        self.assertEqual(result["single_person"]["pck_020"], 1.0)
        self.assertLess(result["multi_person"]["pck_020"], 1.0)

    def test_control_sensitivity_is_forward_only_and_repeatable(self):
        dataset, model = _Dataset(), _Model(); cfg = TrainConfig(raw_ckpt="raw", shard_dir="shards")
        def forward(_model, image, control, *_args, **_kwargs):
            return image + control
        with patch("pose_controlnet.post1500_evaluation.forward_pose_control", side_effect=forward):
            first = fixed_timestep_loss_and_sensitivity(model, dataset, ["a"], cfg, torch.device("cpu"), timesteps=(.1, .9))
            second = fixed_timestep_loss_and_sensitivity(model, dataset, ["a"], cfg, torch.device("cpu"), timesteps=(.1, .9))
        self.assertEqual(first, second)
        self.assertGreater(first["timesteps"][0]["control_sensitivity_rms"]["mean"], 0)

    def test_telemetry_parser_reports_clipping_only_from_logged_norm(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (
                {"global_step": 499, "train/loss": 3.},
                {"global_step": 500, "train/loss": 2., "train/global_grad_norm": .5, "diagnostics/control_input_grad_norm/control_half": .1},
                {"global_step": 501, "train/global_grad_norm": 1.2, "diagnostics/lora_grad_norm/a": .2},
            )) + "\n")
            report = telemetry_audit(path, end_step=501)
        self.assertEqual(report["max_grad_norm_1_observability"]["above_1_count"], 1)
        self.assertEqual(report["control_input_grad_norms"]["control_half"]["sample_count"], 1)

    def test_audit_entrypoint_contains_no_training_or_optimizer_invocation(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "post1500_audit.py").read_text().lower()
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)


if __name__ == "__main__":
    unittest.main()
