import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

import inference
from pose_controlnet.resolution_policy import RESOLUTION_768_BUCKETS


class _Conditioner:
    def __call__(self, prompts):
        self.prompts = prompts
        return torch.ones((1, 3, 12, 4)), torch.ones((1, 3), dtype=torch.bool)


class InferenceTest(unittest.TestCase):
    def _cli(self, root: Path, *extra: str) -> list[str]:
        return [
            "--turbo-ckpt", str(root / "turbo.safetensors"),
            "--pose-lora-ckpt", str(root / "pose.pt"),
            "--prompt", "a dancer", "--pose-image", str(root / "pose.png"),
            "--output", str(root / "output.png"), *extra,
        ]

    def test_cli_defaults_are_the_locked_turbo_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = inference.parser().parse_args(self._cli(Path(temporary)))
        request = inference.request_from_args(args)
        self.assertEqual((request.steps, request.cfg, request.mu), (8, 0.0, 1.15))
        self.assertEqual((request.width, request.height), (None, None))
        self.assertEqual(request.geometry_mode, inference.NATIVE_GEOMETRY)
        self.assertEqual(request.candidate, "mix-025")
        self.assertEqual(request.control_scale, 1.0)

    def test_cli_defaults_to_frozen_candidate_without_an_explicit_pose_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self._cli(root)
            index = command.index("--pose-lora-ckpt")
            del command[index:index + 2]
            request = inference.request_from_args(inference.parser().parse_args(command))
            self.assertIsNone(request.pose_lora_checkpoint)
            self.assertEqual(request.candidate, "mix-025")

    def test_dynamic_768_geometry_uses_the_shared_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose = root / "pose.png"; Image.new("RGB", (400, 800), "white").save(pose)
            request = inference.PoseInferenceRequest(
                root / "turbo.safetensors", root / "pose.pt", "pose", pose, root / "out.png",
                width=None, height=None, dynamic_768_bucket=True,
            )
            prepared = inference.prepare_pose_control(request)
        self.assertEqual(prepared.mode, inference.DYNAMIC_768_GEOMETRY)
        self.assertIn(tuple(prepared.geometry["bucket"]), RESOLUTION_768_BUCKETS)
        self.assertEqual(prepared.image.size, tuple(prepared.geometry["bucket"]))

    def test_explicit_geometry_and_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pose = root / "pose.png"; Image.new("RGB", (120, 80)).save(pose)
            request = inference.PoseInferenceRequest(root / "t", root / "p", "pose", pose, root / "out.png", width=640, height=960)
            prepared = inference.prepare_pose_control(request)
        self.assertEqual(prepared.geometry["bucket"], [640, 960])
        with self.assertRaisesRegex(inference.InferenceError, "divisible by 16"):
            inference._validate_dimensions(777, 768)
        with self.assertRaisesRegex(inference.InferenceError, "positive"):
            inference._validate_dimensions(0, 768)

    def test_missing_checkpoint_and_malformed_pose_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); bad_pose = root / "pose.bin"; bad_pose.write_bytes(b"not-an-image")
            request = inference.PoseInferenceRequest(root / "missing-turbo", root / "missing-pose", "pose", bad_pose, root / "out.png")
            with self.assertRaisesRegex(FileNotFoundError, "Turbo checkpoint"):
                inference.generate_pose(request)
            turbo = root / "turbo.safetensors"; turbo.write_bytes(b"turbo")
            with self.assertRaisesRegex(FileNotFoundError, "pose control checkpoint"):
                inference.generate_pose(inference.PoseInferenceRequest(turbo, root / "missing-pose", "pose", bad_pose, root / "out.png"))
            with self.assertRaisesRegex(inference.InferenceError, "Malformed pose image"):
                inference.load_pose_image(bad_pose)

    def test_malformed_checkpoint_metadata_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            turbo = root / "turbo.safetensors"; turbo.write_bytes(b"turbo")
            checkpoint = root / "pose.pt"; checkpoint.write_bytes(b"not-a-torch-checkpoint")
            pose = root / "pose.png"; Image.new("RGB", (32, 32)).save(pose)
            request = inference.PoseInferenceRequest(turbo, checkpoint, "pose", pose, root / "out.png", device="cpu")
            with self.assertRaisesRegex(inference.InferenceError, "checkpoint metadata is incompatible"):
                inference.load_inference_runtime(request)

    def test_seed_generator_is_deterministic(self):
        first = torch.rand(4, generator=inference.seed_generator(123, "cpu"))
        second = torch.rand(4, generator=inference.seed_generator(123, "cpu"))
        self.assertTrue(torch.equal(first, second))

    def test_callable_api_generates_and_emits_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            turbo = root / "turbo.safetensors"; turbo.write_bytes(b"turbo")
            checkpoint = root / "pose.pt"; checkpoint.write_bytes(b"pose")
            pose = root / "pose.png"; Image.new("RGB", (60, 120), "white").save(pose)
            request = inference.PoseInferenceRequest(turbo, checkpoint, "a dancer", pose, root / "output.png", seed=77,
                                                      width=768, height=768)
            conditioner = _Conditioner()
            runtime = inference.InferenceRuntime(object(), object(), conditioner, torch.device("cpu"), 4300)
            pixels = np.zeros((768, 768, 3), dtype=np.uint8)
            with patch("inference.encode_preprocessed_image", return_value=torch.ones((16, 96, 96))) as encode, \
                 patch("inference.sample_turbo_pose_image", return_value=pixels) as sample:
                result = inference.generate_pose(request, runtime=runtime)
            metadata = json.loads(result.metadata.read_text())
            self.assertTrue(result.output.is_file())
            self.assertEqual(conditioner.prompts, ["a dancer"])
            self.assertEqual(encode.call_args.kwargs["generator"].initial_seed(), 77)
            self.assertEqual(sample.call_args.args[4], 77)
            self.assertEqual(metadata["checkpoint_step"], 4300)
            self.assertEqual(metadata["geometry_mode"], "explicit")
            self.assertEqual(metadata["output_path"], str(request.output.resolve()))
            self.assertEqual(metadata["turbo"]["mu_resolution_dependent"], False)
            self.assertIn("candidate", metadata)
            self.assertIn("style_lora", metadata)

    def test_frozen_mix_025_validates_hashes_and_blends_only_model_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.pt"; parent.write_bytes(b"parent")
            finish = root / "finish.pt"; finish.write_bytes(b"finish")
            request = inference.PoseInferenceRequest(root / "turbo", None, "pose", root / "pose.png", root / "out.png",
                                                      parent_checkpoint=parent, finish_checkpoint=finish)
            endpoints = {
                "parent-4000": {"path": str(parent), "sha256": inference.sha256_file(parent), "step": 4000},
                "finish-control-a4300": {"path": str(finish), "sha256": inference.sha256_file(finish), "step": 4300},
            }
            states = [
                {"global_step": 4000, "config": {"raw_ckpt": "raw"}, "model": {"first.weight": torch.tensor([1., 2.], dtype=torch.bfloat16)}, "optimizer": {}},
                {"global_step": 4300, "config": {"raw_ckpt": "raw"}, "model": {"first.weight": torch.tensor([5., 6.], dtype=torch.bfloat16)}, "scheduler": {}},
            ]
            with patch.object(inference, "CANONICAL_ENDPOINTS", endpoints), \
                 patch("inference.load_release_contract", return_value={"release_id": "test"}), \
                 patch("inference.load_training_state", side_effect=states):
                candidate = inference.resolve_pose_candidate(request)
            self.assertEqual(candidate.candidate_id, "mix-025")
            self.assertEqual(candidate.provenance["interpolation"]["alpha"], .25)
            self.assertTrue(torch.equal(candidate.trainable_state["first.weight"], torch.tensor([2., 3.], dtype=torch.bfloat16)))

            endpoints["parent-4000"]["sha256"] = "0" * 64
            with patch.object(inference, "CANONICAL_ENDPOINTS", endpoints), \
                 patch("inference.load_release_contract", return_value={"release_id": "test"}):
                with self.assertRaisesRegex(inference.InferenceError, "SHA-256 mismatch"):
                    inference.resolve_pose_candidate(request)

    def test_native_geometry_is_default_and_never_falls_back_to_dynamic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pose = root / "pose.png"; Image.new("RGB", (64, 128), "white").save(pose)
            native = inference.PoseInferenceRequest(root / "t", None, "pose", pose, root / "out.png")
            prepared = inference.prepare_pose_control(native)
            self.assertEqual(prepared.mode, inference.NATIVE_GEOMETRY)
            self.assertEqual(prepared.geometry["bucket"], [64, 128])
            dynamic = inference.PoseInferenceRequest(root / "t", None, "pose", pose, root / "out.png",
                                                      width=None, height=None, dynamic_768_bucket=True,
                                                      geometry_mode=inference.DYNAMIC_768_GEOMETRY)
            self.assertEqual(inference.prepare_pose_control(dynamic).mode, inference.DYNAMIC_768_GEOMETRY)

    def test_style_lora_scope_strength_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); style = root / "style.safetensors"; style.write_bytes(b"style")
            request = inference.PoseInferenceRequest(root / "t", None, "pose", root / "pose.png", root / "out.png",
                                                      style_lora_paths=(style,), style_name="darkbrush", style_strength=0.0)
            audit = inference.StyleLoRAAudit("darkbrush", str(style), inference.sha256_file(style), "official_transformer",
                                             True, 528, 264, 32, "F32", None, {"effective_multiplier": 1.0}, {}, ())
            with patch("inference.audit_style_lora", return_value=audit):
                resolved = inference.resolve_style_lora(request)
            self.assertEqual(resolved.strength, 0.0)
            configured = inference.PoseInferenceRequest(root / "t", None, "pose", root / "pose.png", root / "out.png",
                                                         style_lora_paths=(style,), style_name="darkbrush", style_strength=0.5)
            with patch("inference.audit_style_lora", return_value=audit):
                self.assertEqual(inference.resolve_style_lora(configured).strength, 0.5)
            with self.assertRaisesRegex(inference.InferenceError, "multiple Style-LoRAs"):
                inference._validate_request(inference.PoseInferenceRequest(
                    root / "t", None, "pose", root / "pose.png", root / "out.png", style_lora_paths=(style, style)))
            self.assertEqual(inference.STYLE_DEFAULT_STRENGTHS["darkbrush"], .75)
            control = inference.PreparedPoseControl(Image.new("RGB", (64, 64)), inference.NATIVE_GEOMETRY,
                                                     {"source_size": [64, 64], "resized_size": [64, 64], "crop_box": [0, 0, 64, 64], "bucket": [64, 64]})
            pose = root / "pose.png"; pose.write_bytes(b"pose")
            metadata = inference.build_metadata(request, control, None, style=resolved)
            self.assertEqual(metadata["style_lora"]["strength"], 0.0)
            self.assertEqual(metadata["style_lora"]["path"], str(style.resolve()))

    def test_zero_strength_skips_style_adapter_loading_and_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); turbo = root / "turbo.safetensors"; turbo.write_bytes(b"turbo")
            request = inference.PoseInferenceRequest(turbo, None, "pose", root / "pose.png", root / "out.png", device="cpu")
            audit = inference.StyleLoRAAudit("darkbrush", "unused", "0" * 64, "official_transformer", True,
                                             528, 264, 32, "F32", None, {"effective_multiplier": 1.0}, {}, ())
            style = inference.ResolvedStyleLoRA(audit, 0.0)
            candidate = inference.ResolvedPoseCandidate("mix-025", {}, {"config": {"raw_ckpt": "raw"}, "model": {}},
                                                        None, {"candidate_id": "mix-025"})
            model = MagicMock(); model.eval.return_value = model
            with patch("inference.build_turbo_pose_model", return_value=model), \
                 patch("inference.raw_to_turbo_control_compatibility"), \
                 patch("inference.load_trainable_state_dict"), \
                 patch("inference.load_krea_vae", return_value=object()), \
                 patch("inference.PoseTextConditioner", return_value=object()), \
                 patch("inference.StyleLoRAAdapter.load", side_effect=AssertionError("zero strength must not load an adapter")):
                runtime = inference.load_inference_runtime(request, candidate=candidate, style=style)
            self.assertIsNone(runtime.style_adapter)

    def test_no_duplicate_dynamic_bucket_policy(self):
        self.assertIs(inference.RESOLUTION_768_BUCKETS, RESOLUTION_768_BUCKETS)
        source = Path(inference.__file__).read_text(encoding="utf-8")
        self.assertIn("choose_bucket(source.size, RESOLUTION_768_BUCKETS)", source)
        self.assertNotIn("(704, 896)", source)


if __name__ == "__main__":
    unittest.main()
