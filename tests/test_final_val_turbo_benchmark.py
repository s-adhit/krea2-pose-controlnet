import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from pose_controlnet.evaluation import make_evaluation_spec
from pose_controlnet.turbo_runtime import turbo_metadata
from scripts import final_val_turbo_benchmark as evaluator


class _Dataset:
    def __init__(self, stems):
        self.records = [("shard.pt", index, (4, 4), stem) for index, stem in enumerate(stems)]
        self.samples = {
            stem: {"stem": stem, "latent": torch.full((16, 4, 4), index + 1.0),
                   "control": torch.ones(16, 4, 4), "context": torch.ones(2, 1, 1),
                   "mask": torch.ones(2, dtype=torch.bool), "prompt": f"prompt {stem}"}
            for index, stem in enumerate(stems)
        }

    def __getitem__(self, index):
        return self.samples[self.records[index][3]]


class FinalValTurboBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.stems = [*(f"coco_{index:02d}" for index in range(16)),
                      *(f"painting_humanart_{index:02d}" for index in range(12)),
                      *(f"real_human_humanart_{index:02d}" for index in range(12)),
                      *(f"sculpture_humanart_{index:02d}" for index in range(8))]
        self.dataset = _Dataset(self.stems)
        self.spec = make_evaluation_spec(self.dataset, split="val", count=48, seed=420300,
                                         kind="final_val_turbo_fixed_pose", stems=self.stems)
        self.spec["turbo"] = {**turbo_metadata(), "control_scale": 1.0}
        self.spec["benchmark"] = {"name": "final_val_benchmark_48",
            "source_counts": {"coco": 16, "painting": 12, "real_human": 12, "sculpture": 8},
            "orientation_counts": {"landscape": 16, "near_square": 17, "portrait": 15}}

    def test_frozen_spec_validation_and_cached_identity_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.json"; path.write_text(json.dumps(self.spec))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded, observed = evaluator.load_final_spec(path, expected_sha256=digest)
            self.assertEqual(loaded, self.spec); self.assertEqual(observed, digest)
            evaluator.validate_cached_contract(self.dataset, loaded)
            changed = copy.deepcopy(loaded); changed["per_stem_seeds"][self.stems[0]]["sampling"] += 1
            with self.assertRaisesRegex(ValueError, "cache identity/seed"):
                evaluator.validate_cached_contract(self.dataset, changed)

    def test_rejects_nonfinal_contract_and_unselected_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.json"; bad = copy.deepcopy(self.spec); bad["turbo"]["steps"] = 7
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "locked 8-step"):
                evaluator.load_final_spec(path, expected_sha256=None)
        with self.assertRaisesRegex(ValueError, "supports only"):
            evaluator.candidate_checkpoint("turbo-base")

    def test_candidate_checkpoint_keeps_gate_e_validation_when_metadata_is_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step_000123.pt"; checkpoint.write_bytes(b"metadata-present")
            candidates = {"candidate": {
                "checkpoint_root": temporary, "step": 123, "label": "candidate",
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "production_provenance": {"format": 1, "run_name": "unused", "max_steps": 123},
            }}
            state = {"global_step": 123, "gate_e": {"present": True}, "config": {}}
            with patch.object(evaluator, "CANDIDATES", candidates), \
                 patch.object(evaluator, "load_training_state", return_value=state), \
                 patch.object(evaluator, "controlled_branch_metadata", return_value={"validation": "gate_e"}) as controlled:
                _, resolved, metadata = evaluator.candidate_checkpoint("candidate")
            self.assertEqual(resolved, checkpoint)
            self.assertEqual(metadata, {"validation": "gate_e"})
            controlled.assert_called_once_with([(123, checkpoint)])

    def test_candidate_checkpoint_accepts_only_pinned_legacy_production_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step_000123.pt"; checkpoint.write_bytes(b"legacy-metadata-absent")
            provenance = {"format": 1, "run_name": "trusted-run", "max_steps": 456}
            candidates = {"candidate": {
                "checkpoint_root": temporary, "step": 123, "label": "candidate",
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "production_provenance": provenance,
            }}
            state = {"global_step": 123, "config": {}, "production_pose_control": {
                **provenance, "current_step": 123,
            }}
            with patch.object(evaluator, "CANDIDATES", candidates), \
                 patch.object(evaluator, "load_training_state", return_value=state), \
                 patch.object(evaluator, "controlled_branch_metadata", side_effect=AssertionError("legacy checkpoint must not require gate_e")):
                _, _, metadata = evaluator.candidate_checkpoint("candidate")
            self.assertEqual(metadata["validation"], "pinned_final_val_legacy_production_metadata")
            self.assertEqual(metadata["checkpoint_sha256"], candidates["candidate"]["sha256"])

            state["production_pose_control"]["run_name"] = "wrong-run"
            with patch.object(evaluator, "CANDIDATES", candidates), \
                 patch.object(evaluator, "load_training_state", return_value=state):
                with self.assertRaisesRegex(ValueError, "compatible production provenance"):
                    evaluator.candidate_checkpoint("candidate")

    def test_final_sidecar_requires_exact_frozen_order_and_non_diagnostic_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            records = [{"stem": stem, "source": "coco" if stem.startswith("coco_") else "humanart", "status": "unavailable"}
                       for stem in self.stems]
            path.write_text(json.dumps({"records": records}))
            loaded, _ = evaluator._load_final_sidecar(path, self.stems)
            self.assertEqual(loaded["records"], records)
            records.reverse(); path.write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(ValueError, "in order"):
                evaluator._load_final_sidecar(path, self.stems)

    def test_canonical_final_v3_sidecar_has_frozen_provenance_and_adapts_for_pck(self):
        root = Path("docs/evaluation/final-val-benchmark-selection")
        spec, _ = evaluator.load_final_spec(root / "final_val_benchmark_spec.json")
        sidecar, digest = evaluator._load_final_sidecar(root / "final_val_benchmark_48_pose_targets_v3", spec["stems"])
        self.assertEqual(digest, "3cc4defc282cb11e956ec06517eff4e8369622d4c0b3b567ab2247efb4a499a7")
        self.assertEqual([record["stem"] for record in sidecar["records"]], spec["stems"])
        self.assertEqual({record["source"] for record in sidecar["records"]}, {"coco", "humanart"})

    def test_generation_artifacts_require_complete_matching_final_contract(self):
        candidate = {"label": "parent-4000", "step": 4000}
        stems = ["first", "second"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); digest = "frozen-digest"
            for stem in stems:
                directory = root / "fixed_pose" / stem; directory.mkdir(parents=True)
                Image.new("RGB", (2, 2), "white").save(directory / "step_004000.png")
                metadata = {"stem": stem, "candidate": candidate["label"], "checkpoint_step": 4000,
                            "final_spec_sha256": digest, "control_scale": 1.0, **turbo_metadata()}
                (directory / "metadata.json").write_text(json.dumps(metadata))
            (root / "generation_results.json").write_text(json.dumps({
                "stems": stems, "candidate": candidate["label"], "final_spec_sha256": digest,
                "turbo": {**turbo_metadata(), "control_scale": 1.0},
                "generated_steps": {stem: [4000] for stem in stems},
            }))
            self.assertEqual(evaluator._generation_status(root, stems, candidate, digest), "complete")
            bad = json.loads((root / "fixed_pose" / "second" / "metadata.json").read_text())
            bad["checkpoint_step"] = 4300
            (root / "fixed_pose" / "second" / "metadata.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "contract-inconsistent"):
                evaluator._generation_status(root, stems, candidate, digest)

    def test_qualitative_paths_select_candidate_generated_image_not_dataset_rgb(self):
        candidate = {"label": "parent-4000", "step": 4000}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "fixed_pose" / "sample"; directory.mkdir(parents=True)
            control = directory / "control.png"; control.write_bytes(b"control")
            generated = directory / "step_004000.png"; generated.write_bytes(b"generated")
            dataset_rgb = root / "dataset" / "sample.jpg"; dataset_rgb.parent.mkdir(); dataset_rgb.write_bytes(b"reference")

            self.assertEqual(
                evaluator._qualitative_image_paths(root, "sample", candidate),
                [control, generated],
            )
            self.assertNotEqual(evaluator._qualitative_image_paths(root, "sample", candidate)[1], dataset_rgb)

    def test_qualitative_paths_fail_closed_when_generated_image_is_missing(self):
        candidate = {"label": "parent-4000", "step": 4000}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "fixed_pose" / "sample"; directory.mkdir(parents=True)
            (directory / "control.png").write_bytes(b"control")

            with self.assertRaisesRegex(FileNotFoundError, "missing generated output"):
                evaluator._qualitative_image_paths(root, "sample", candidate)


if __name__ == "__main__":
    unittest.main()
