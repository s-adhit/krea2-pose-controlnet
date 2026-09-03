import importlib.util
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from pose_controlnet.style_lora import (
    STYLE_LORA_SPECS,
    StyleLoRAAdapter,
    StyleLoRAAudit,
    _Delta,
    applied_style_lora,
    audit_style_lora,
    map_style_target,
    sha256,
)


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "style_lora_composition.py"
SPEC = importlib.util.spec_from_file_location("style_lora_composition", MODULE_PATH)
assert SPEC and SPEC.loader
composition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(composition)


class StyleLoRACompositionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This hashes and structurally inspects the four pinned, local files;
        # no model weights, CUDA, or network access are involved.
        cls.audits = {style: audit_style_lora(style) for style in STYLE_LORA_SPECS}

    def test_frozen_style_hashes_pairs_rank_shapes_and_complete_resolution(self):
        self.assertEqual(sha256(composition.SPEC_FILE), composition.SPEC_SHA256)
        for style, audit in self.audits.items():
            self.assertTrue(audit.supported, (style, audit.errors))
            self.assertEqual(audit.sha256, STYLE_LORA_SPECS[style]["sha256"])
            self.assertEqual((audit.tensor_count, audit.target_count, audit.rank, audit.dtype), (528, 264, 32, "F32"))
            self.assertEqual(len(audit.mapping), 264)

    def test_realism_direct_namespace_mapping_is_explicit_and_complete(self):
        audit = self.audits["realism"]
        self.assertEqual(audit.mapping["base_model.model.blocks.0.attn.wq"], "blocks.0.attn.wq")
        self.assertEqual(audit.mapping["base_model.model.first"], "first")
        self.assertEqual(audit.scaling_rule["effective_multiplier"], 1.0)
        with self.assertRaisesRegex(ValueError, "Unsupported realism"):
            map_style_target("transformer.blocks.0.attn.wq", "base_model_model")

    def test_unpaired_shape_invalid_and_unsupported_realism_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            unpaired = Path(directory) / "unpaired.safetensors"
            save_file({"transformer.img_in.lora_A.weight": torch.zeros(31, 64)}, unpaired)
            result = audit_style_lora("darkbrush", expected_path=unpaired, expected_sha256=sha256(unpaired))
            self.assertFalse(result.supported)
            self.assertTrue(any("unpaired_A_B" in error for error in result.errors))

            bad_realism = Path(directory) / "bad_realism.safetensors"
            save_file({"base_model.model.unknown.lora_A.weight": torch.zeros(32, 3),
                       "base_model.model.unknown.lora_B.weight": torch.zeros(4, 32)}, bad_realism)
            result = audit_style_lora("realism", expected_path=bad_realism, expected_sha256=sha256(bad_realism))
            self.assertFalse(result.supported)
            self.assertTrue(any("incomplete_target_resolution" in error for error in result.errors))

    def test_style_strength_and_hook_scope_do_not_mutate_or_leak(self):
        model = torch.nn.Module()
        model.target = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.target.weight.copy_(torch.tensor([[2.0, -1.0]]))
        audit = StyleLoRAAudit(style_id="toy", path="unused", sha256="0" * 64, namespace="toy", supported=True,
                               tensor_count=2, target_count=1, rank=32, dtype="F32", metadata=None,
                               scaling_rule={"effective_multiplier": 1.0}, mapping={"toy": "target"}, errors=())
        a = torch.zeros(32, 2); a[0, 0] = 1.0
        b = torch.zeros(1, 32); b[0, 0] = 3.0
        adapter = StyleLoRAAdapter(audit, [_Delta("target", a, b)])
        x = torch.tensor([[4.0, 5.0]])
        baseline = model.target(x).detach().clone()
        with applied_style_lora(model, adapter, 2.0):
            styled = model.target(x).detach().clone()
        self.assertTrue(torch.equal(model.target(x), baseline))
        self.assertTrue(torch.equal(styled, baseline + torch.tensor([[24.0]])))
        with applied_style_lora(model, adapter, 0.0):
            self.assertTrue(torch.equal(model.target(x), baseline))
        self.assertTrue(torch.equal(model.target.weight, torch.tensor([[2.0, -1.0]])))

        class ExpandedInput(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            def forward(self, value):
                return torch.nn.functional.linear(value, self.weight)

        image_control = torch.nn.Module(); image_control.first = ExpandedInput()
        first_adapter = StyleLoRAAdapter(audit, [_Delta("first", a, b)])
        first_x = torch.tensor([[4.0, 5.0, 9.0, 10.0]]); first_baseline = image_control.first(first_x).detach().clone()
        with applied_style_lora(image_control, first_adapter, 1.0):
            self.assertTrue(torch.equal(image_control.first(first_x), first_baseline + torch.tensor([[12.0]])))
        self.assertTrue(torch.equal(image_control.first(first_x), first_baseline))

    def test_frozen_matrix_keeps_seed_control_and_semantics_identical_across_variants(self):
        rows = composition.load_rows()
        self.assertEqual([row["condition_id"] for row in rows], list(composition.EXPECTED_CONDITIONS))
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.png"; control.write_bytes(b"authoritative-control")
            contract = {"style_strength": 1.0, "sampling_seeds": {row["stem"]: 100 + index for index, row in enumerate(rows)},
                        "control_sha256": {row["stem"]: sha256(control) for row in rows},
                        "buckets": {row["stem"]: [768, 512] for row in rows}, "style_loras": composition._immutable_style_loras({style: self.audits[style].json() for style in STYLE_LORA_SPECS}),
                        "turbo": composition.guide.TURBO, "candidate_kind": "trainable_tensor_interpolation", "checkpoint_step": None,
                        "checkpoint_interpolation": {"alpha": .25}}
            for row in rows:
                metadata = [composition._metadata(row, style, contract, control) for style in composition.STYLE_ORDER]
                self.assertEqual({entry["seed"] for entry in metadata}, {contract["sampling_seeds"][row["stem"]]})
                self.assertEqual({entry["control_sha256"] for entry in metadata}, {sha256(control)})
                self.assertEqual({entry["prompt"] for entry in metadata}, {row["prompt"]})
                self.assertTrue(all(entry["geometry"] == composition.NATIVE_GEOMETRY and entry["source_rgb_fallback_used"] is False for entry in metadata))
                required = {"condition_id", "stem", "prompt", "style_id", "style_strength", "seed", "control_path", "control_sha256", "bucket", "geometry", "candidate", "steps", "cfg", "mu", "mu_resolution_dependent"}
                self.assertTrue(all(required <= set(entry) for entry in metadata))
                self.assertIn("style_lora", metadata[1])

    def test_v2_trigger_prompts_are_exact_deterministic_and_preserve_semantics(self):
        v1_rows = composition.load_rows("v1")
        v2_rows = composition.load_rows("v2-triggers")
        v2 = composition.EXPERIMENTS["v2-triggers"]
        self.assertEqual(sha256(v2.path), v2.sha256)
        self.assertEqual(len(v1_rows), len(v2_rows))
        self.assertEqual([row["condition_id"] for row in v1_rows], [row["condition_id"] for row in v2_rows])
        self.assertEqual([row["stem"] for row in v1_rows], [row["stem"] for row in v2_rows])
        prompt_provenance = composition._prompt_provenance(v2_rows, composition.STYLE_ORDER, v2)
        self.assertEqual([entry["stem"] for entry in prompt_provenance], [row["stem"] for row in v2_rows])
        for v1_row, row in zip(v1_rows, v2_rows):
            with self.subTest(condition=row["condition_id"]):
                self.assertEqual(row["semantic_base_prompt"], v1_row["prompt"])
                parts = {style: composition._prompt_parts(row, style, v2) for style in composition.STYLE_ORDER}
                self.assertEqual({part["semantic_base_prompt"] for part in parts.values()}, {row["semantic_base_prompt"]})
                provenance_parts = {entry["style_id"]: entry for entry in next(
                    entry for entry in prompt_provenance if entry["stem"] == row["stem"]
                )["variants"]}
                self.assertEqual(provenance_parts, {style: {"style_id": style, **part} for style, part in parts.items()})
                self.assertEqual(parts["pose-only"]["trigger_phrase"], "")
                self.assertEqual(parts["realism"]["trigger_phrase"], "")
                self.assertEqual(parts["pose-only"]["effective_prompt"], row["semantic_base_prompt"])
                self.assertEqual(parts["realism"]["effective_prompt"], row["semantic_base_prompt"])
                for style, phrase in (("darkbrush", "monochrome ink wash style"),
                                      ("rainywindow", "rainy window style"),
                                      ("retroanime", "Purple retro anime style")):
                    self.assertEqual(parts[style]["trigger_phrase"], phrase)
                    self.assertEqual(parts[style]["effective_prompt"], f"{row['semantic_base_prompt']}, {phrase}")
                    self.assertEqual(parts[style], composition._prompt_parts(row, style, v2))

    def test_v1_and_v2_cannot_share_immutable_provenance_or_output_identity(self):
        v1, v2 = composition.EXPERIMENTS["v1"], composition.EXPERIMENTS["v2-triggers"]
        self.assertNotEqual((v1.kind, v1.sha256), (v2.kind, v2.sha256))
        v1_identity = composition._canonical_json({"kind": v1.kind, "frozen_spec": {"sha256": v1.sha256}, "experiment_id": v1.experiment_id})
        v2_identity = composition._canonical_json({"kind": v2.kind, "frozen_spec": {"sha256": v2.sha256}, "experiment_id": v2.experiment_id,
                                                    "prompt_construction": composition._prompt_provenance(composition.load_rows("v2-triggers"), composition.STYLE_ORDER, v2)})
        self.assertNotEqual(v1_identity, v2_identity)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            composition._validate_or_write_provenance(output, v1_identity)
            with self.assertRaisesRegex(ValueError, "conflicting immutable Style-LoRA provenance"):
                composition._validate_or_write_provenance(output, v2_identity)

    def test_staged_immutable_provenance_lifecycle_and_drift_fail_closed(self):
        """Audit/preflight/generate share identity; stage output never alters it."""
        provenance = composition._canonical_json({
            "kind": composition.KIND,
            "candidate": composition.POSE_CANDIDATE,
            "frozen_spec": {"sha256": composition.SPEC_SHA256, "record_count": 4},
            "style_strength": 1.0,
            "style_loras": {"darkbrush": {"sha256": "a" * 64, "mapping": {"source": "target"},
                                           "scaling_rule": {"effective_multiplier": 1.0}}},
            "checkpoint_interpolation": {"candidate_id": "mix-025", "alpha": 0.25,
                                           "endpoints": [{"candidate": "parent-4000", "sha256": "b" * 64},
                                                         {"candidate": "finish-control-a4300", "sha256": "c" * 64}]},
            "conditions": [{"stem": "one", "prompt": "locked"}], "sampling_seeds": {"one": 42},
            "control_sha256": {"one": "d" * 64}, "buckets": {"one": [768, 512]},
            "turbo": {"steps": 8, "cfg": 0.0, "mu": 1.15}, "geometry": composition.NATIVE_GEOMETRY,
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for stage, state in (
                ("audit", {"audit_output": "audits/darkbrush.json", "timestamp": "first"}),
                ("preflight", {"checkpoint_preflight": "checkpoint_preflight.json", "stage_complete": True}),
                ("generate", {"generated_artifacts": {"one": {"darkbrush": "generations/one/darkbrush.png"}}}),
                ("score", {"score_artifact": "pck_clip_results.json", "reference_sidecar": "/other/path/sidecar"}),
                ("report", {"report_path": "evaluation_summary.json"}),
                ("summary", {"summary_timestamp": "last"}),
            ):
                self.assertEqual(composition._validate_or_write_provenance(output, provenance), provenance)
                payload = composition._stage_payload(provenance, **state)
                composition._write(output / f"{stage}.json", payload)
                self.assertEqual(composition._read_json(output / f"{stage}.json")["immutable_provenance"], provenance)

            # The original failure mode: live tuples become JSON lists on disk.
            tuple_provenance = {"audit_result": {"errors": ()}}
            self.assertEqual(composition._validate_or_write_provenance(output / "tuple", tuple_provenance), {"audit_result": {"errors": []}})
            self.assertEqual(composition._validate_or_write_provenance(output / "tuple", tuple_provenance), {"audit_result": {"errors": []}})

            for key, mutate in (
                ("frozen spec", lambda value: value["frozen_spec"].update(sha256="e" * 64)),
                ("style strength", lambda value: value.update(style_strength=0.5)),
                ("Style-LoRA hash", lambda value: value["style_loras"]["darkbrush"].update(sha256="f" * 64)),
                ("candidate", lambda value: value.update(candidate="other-candidate")),
            ):
                drifted = copy.deepcopy(provenance)
                mutate(drifted)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "conflicting immutable Style-LoRA provenance"):
                    composition._validate_or_write_provenance(output, drifted)


if __name__ == "__main__":
    unittest.main()
