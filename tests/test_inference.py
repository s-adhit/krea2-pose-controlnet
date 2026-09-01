import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual((request.width, request.height), (768, 768))
        self.assertEqual(request.control_scale, 1.0)

    def test_cli_requires_an_explicit_pose_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self._cli(root)
            index = command.index("--pose-lora-ckpt")
            del command[index:index + 2]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                inference.parser().parse_args(command)

    def test_dynamic_768_geometry_uses_the_shared_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose = root / "pose.png"; Image.new("RGB", (400, 800), "white").save(pose)
            request = inference.PoseInferenceRequest(
                root / "turbo.safetensors", root / "pose.pt", "pose", pose, root / "out.png",
                width=None, height=None, dynamic_768_bucket=True,
            )
            prepared = inference.prepare_pose_control(request)
        self.assertEqual(prepared.mode, "production-dynamic-768")
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
            request = inference.PoseInferenceRequest(turbo, checkpoint, "a dancer", pose, root / "output.png", seed=77)
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

    def test_no_duplicate_dynamic_bucket_policy(self):
        self.assertIs(inference.RESOLUTION_768_BUCKETS, RESOLUTION_768_BUCKETS)
        source = Path(inference.__file__).read_text(encoding="utf-8")
        self.assertIn("choose_bucket(source.size, RESOLUTION_768_BUCKETS)", source)
        self.assertNotIn("(704, 896)", source)


if __name__ == "__main__":
    unittest.main()
