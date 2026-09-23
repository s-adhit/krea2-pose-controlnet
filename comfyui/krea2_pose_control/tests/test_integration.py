from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

import krea2_pose_control
from krea2_pose_control import nodes
from krea2_pose_control import controlnet_aux
from krea2_pose_control import generation_smoke
from krea2_pose_control.dwpose import (
    COCO17_NAMES as DWPOSE_COCO17_NAMES, DWPOSE_BODY18_FOR_COCO17,
    DWPOSE_BODY18_NAMES, dwpose_person_confidence, normalize_dwpose_body18,
)
from krea2_pose_control.nodes import (
    Krea2PoseExtractor, _image_to_pil, _pil_to_image, extract_coco17_condition, extract_coco17_people,
)
from krea2_pose_control.renderer import (
    BODY_COLORS, BODY_LIMBS, COCO_TO_BODY18, ENDPOINT_RADIUS, LINE_WIDTH,
    _body18, render_coco17_people,
)
from krea2_pose_control.krea2_pose_runtime.runtime import RELEASE_SHA256, ReleaseValidationError, validate_release_artifact
from krea2_pose_control.krea2_pose_runtime import diffusion as runtime_diffusion
from krea2_pose_control.krea2_pose_runtime import turbo_runtime, vae as runtime_vae


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
FROZEN_COCO_TO_BODY18 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
FROZEN_BODY_LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
)
FROZEN_BODY_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
)


def _frozen_posebridge_render(size, coco17):
    """Independent literal implementation of the frozen historic contract."""
    canvas = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    body = [[0.0, 0.0, 0.0] for _ in range(18)]
    for coco_index, body_index in enumerate(FROZEN_COCO_TO_BODY18):
        body[body_index] = list(coco17[coco_index])
    left_shoulder, right_shoulder = body[5], body[2]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0:
        body[1] = [
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
            min(left_shoulder[2], right_shoulder[2]),
        ]
    for index, (first, second) in enumerate(FROZEN_BODY_LIMBS):
        if body[first][2] > 0 and body[second][2] > 0:
            start = tuple(int(round(value)) for value in body[first][:2])
            end = tuple(int(round(value)) for value in body[second][:2])
            draw.line((start, end), fill=FROZEN_BODY_COLORS[index], width=3)
    for x, y, confidence in body:
        if confidence > 0:
            cx, cy = int(round(x)), int(round(y))
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 255))
    return canvas


class ComfyIntegrationTests(unittest.TestCase):
    def test_qwen_vae_boundary_removes_only_singleton_temporal_axis(self):
        class FakeVAE:
            config = SimpleNamespace(z_dim=2, latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0])

            def __init__(self):
                self.encode_inputs = []
                self.decode_inputs = []

            def encode(self, pixels):
                self.encode_inputs.append(pixels.clone())
                raw = torch.arange(12, dtype=pixels.dtype).reshape(1, 2, 1, 3, 2).add_(1)
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: raw))

            def decode(self, raw):
                self.decode_inputs.append(raw.clone())
                return SimpleNamespace(sample=torch.ones((raw.shape[0], 3, raw.shape[2], raw.shape[3], raw.shape[4]),
                                                         dtype=raw.dtype))

        fake = FakeVAE()
        encoded = runtime_vae.encode_preprocessed_image(
            fake, Image.new("RGB", (16, 24)), device="cpu", generator=torch.Generator().manual_seed(42)
        )
        self.assertEqual(tuple(fake.encode_inputs[0].shape), (1, 3, 1, 24, 16))
        self.assertEqual(tuple(encoded.shape), (1, 2, 3, 2))
        decoded = runtime_vae.decode_normalized_latents(fake, encoded)
        self.assertEqual(tuple(fake.decode_inputs[0].shape), (1, 2, 1, 3, 2))
        self.assertEqual(tuple(decoded.shape), (1, 3, 3, 2))

    def test_qwen_vae_boundary_rejects_non_singleton_temporal_axes(self):
        class BadTemporalVAE:
            config = SimpleNamespace(z_dim=2, latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0])

            def encode(self, pixels):
                raw = torch.ones((1, 2, 2, 3, 2), dtype=pixels.dtype)
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: raw))

            def decode(self, raw):
                return SimpleNamespace(sample=torch.ones((1, 3, 2, 3, 2), dtype=raw.dtype))

        fake = BadTemporalVAE()
        with self.assertRaisesRegex(ValueError, "temporal dimension must be exactly 1"):
            runtime_vae.encode_preprocessed_image(
                fake, Image.new("RGB", (16, 24)), device="cpu", generator=torch.Generator().manual_seed(42)
            )
        with self.assertRaisesRegex(ValueError, "temporal dimension must be exactly 1"):
            runtime_vae.decode_normalized_latents(fake, torch.ones((1, 2, 3, 2)))

    def test_pose_control_and_noise_reach_patchify_as_matching_4d_latents(self):
        class FakeModel:
            config = SimpleNamespace(patch=2)

        captured_decode = []

        def decode(latent):
            captured_decode.append(latent)
            return torch.zeros((1, 3, 4, 4), dtype=latent.dtype)

        sample = {
            "latent": torch.zeros((1, 2, 4, 4)),
            "control": torch.ones((1, 2, 4, 4)),
            "context": torch.zeros((3, 5)),
            "mask": torch.ones(3, dtype=torch.bool),
        }
        with patch.object(turbo_runtime, "patchify_and_position", wraps=runtime_diffusion.patchify_and_position) as patchify, \
             patch.object(turbo_runtime, "forward_pose_control", side_effect=lambda model, image, *args, **kwargs: torch.zeros_like(image)):
            pixels = turbo_runtime.sample_turbo_pose_image(FakeModel(), decode, sample, torch.device("cpu"), 42)
        self.assertEqual(pixels.shape, (4, 4, 3))
        self.assertEqual([tuple(call.args[0].shape) for call in patchify.call_args_list], [(1, 2, 4, 4), (1, 2, 4, 4)])
        self.assertEqual(tuple(captured_decode[0].shape), (1, 2, 4, 4))

    def test_patchify_rejects_qwen_5d_latents_and_accepts_diffusion_4d_latents(self):
        text_mask = torch.ones((1, 3), dtype=torch.bool)
        tokens, _, _ = runtime_diffusion.patchify_and_position(torch.zeros((1, 2, 4, 4)), 3, 2, text_mask)
        self.assertEqual(tuple(tokens.shape), (1, 4, 8))
        with self.assertRaisesRegex(ValueError, "B×C×H×W"):
            runtime_diffusion.patchify_and_position(torch.zeros((1, 2, 1, 4, 4)), 3, 2, text_mask)

    def test_controlnet_aux_discovers_sibling_custom_nodes_src(self):
        with tempfile.TemporaryDirectory() as directory:
            custom_nodes = Path(directory) / "ComfyUI" / "custom_nodes"
            package_dir = custom_nodes / "krea2_pose_control"
            source = custom_nodes / "comfyui_controlnet_aux" / "src"
            package_dir.mkdir(parents=True)
            (source / "custom_controlnet_aux").mkdir(parents=True)
            (source / "custom_controlnet_aux" / "__init__.py").write_text("")
            (source / "custom_controlnet_aux" / "dwpose.py").write_text("MARKER = 'sibling'\n")
            self.assertEqual(controlnet_aux.discover_controlnet_aux_src(package_dir=package_dir), source)
            original_modules = {name: sys.modules.pop(name) for name in list(sys.modules)
                                if name == "custom_controlnet_aux" or name.startswith("custom_controlnet_aux.")}
            try:
                with patch.object(controlnet_aux.sys, "path", [str(custom_nodes)]):
                    module = controlnet_aux.load_dwpose_module(package_dir=package_dir)
                    self.assertEqual(module.MARKER, "sibling")
                    self.assertEqual(controlnet_aux.sys.path[0], str(source))
            finally:
                for name in list(sys.modules):
                    if name == "custom_controlnet_aux" or name.startswith("custom_controlnet_aux."):
                        del sys.modules[name]
                sys.modules.update(original_modules)

    def test_controlnet_aux_already_importable_needs_no_discovery(self):
        module = ModuleType("custom_controlnet_aux.dwpose")
        with patch.object(controlnet_aux.importlib, "import_module", return_value=module) as imported, \
             patch.object(controlnet_aux, "discover_controlnet_aux_src") as discovered:
            self.assertIs(controlnet_aux.load_dwpose_module(), module)
        imported.assert_called_once_with("custom_controlnet_aux.dwpose")
        discovered.assert_not_called()

    def test_controlnet_aux_missing_dependency_has_actionable_error(self):
        with patch.object(controlnet_aux.importlib, "import_module", side_effect=ModuleNotFoundError(
                "No module named 'custom_controlnet_aux'", name="custom_controlnet_aux")), \
             patch.object(controlnet_aux, "discover_controlnet_aux_src", return_value=None):
            with self.assertRaisesRegex(controlnet_aux.ControlNetAuxDependencyError,
                                        "Install it as a sibling custom node"):
                controlnet_aux.load_dwpose_module()

    def test_generation_smoke_reuses_pinned_runtime_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = root / "condition.png"
            Image.new("RGB", (32, 48)).save(condition)
            turbo, raw, release = (root / name for name in ("turbo.safetensors", "raw.safetensors", "release.safetensors"))
            for path in (turbo, raw, release):
                path.write_bytes(b"placeholder")
            output = root / "output.png"
            with patch.object(generation_smoke, "load_runtime", return_value=object()) as load, \
                 patch.object(generation_smoke, "generate", return_value=Image.new("RGB", (32, 48))) as run:
                self.assertEqual(generation_smoke.main([
                    "--pose-condition", str(condition), "--release-artifact", str(release),
                    "--turbo-path", str(turbo), "--raw-path", str(raw), "--output-image", str(output),
                ]), 0)
            load.assert_called_once_with(turbo_path=turbo, release_artifact=str(release), base_raw_path=raw)
            self.assertEqual(run.call_args.kwargs["seed"], generation_smoke.DEFAULT_SEED)
            self.assertEqual(run.call_args.kwargs["control_scale"], 1.0)
            self.assertEqual(run.call_args.kwargs["steps"], 8)
            self.assertEqual(run.call_args.kwargs["mu"], 1.15)
            self.assertTrue(output.is_file())

    def test_exact_coco17_topology(self):
        self.assertEqual(len(COCO_TO_BODY18), 17)
        self.assertEqual(len(BODY_LIMBS), 17)
        self.assertEqual(len(BODY_COLORS), 17)
        self.assertEqual(COCO_TO_BODY18, (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10))
        self.assertEqual(BODY_LIMBS[0], (1, 2))
        self.assertEqual(BODY_LIMBS[-1], (15, 17))

    def test_torchvision_coco_v1_order_and_all_joint_assignments(self):
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

        self.assertEqual(tuple(KeypointRCNN_ResNet50_FPN_Weights.COCO_V1.meta["keypoint_names"]), COCO17_NAMES)
        # Every source joint has a coordinate which uniquely identifies its
        # semantic source. A left/right or distal-joint swap cannot pass.
        points = [[10.0 + index * 11.0, 20.0 + index * 13.0, 0.9] for index in range(17)]
        body = _body18(points, confidence_threshold=0.5)
        expected = dict(zip(COCO17_NAMES, FROZEN_COCO_TO_BODY18))
        for coco_index, name in enumerate(COCO17_NAMES):
            body_index = expected[name]
            self.assertEqual(body[body_index], points[coco_index], name)
        self.assertEqual(body[1], [70.5, 91.5, 0.9])

    def test_neck_uses_only_the_two_shoulder_joints(self):
        points = [[0.0, 0.0, 0.0] for _ in range(17)]
        points[5] = [20.0, 40.0, 0.9]
        points[6] = [80.0, 60.0, 0.8]
        points[7] = [999.0, 999.0, 0.99]
        self.assertEqual(_body18(points, 0.5)[1], [50.0, 50.0, 0.8])
        points[6][2] = 0.49
        self.assertEqual(_body18(points, 0.5)[1], [0.0, 0.0, 0.0])

    def test_renderer_matches_frozen_posebridge_limb_and_color_contract(self):
        points = [[15.0 + index * 15.0, 25.0 + index * 11.0, 0.9] for index in range(17)]
        expected = _frozen_posebridge_render((300, 240), points)
        actual = render_coco17_people((300, 240), [points], keypoint_confidence_threshold=0.5)
        self.assertEqual((LINE_WIDTH, ENDPOINT_RADIUS), (3, 4))
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_extractor_preserves_order_and_uses_torchvision_joint_scores(self):
        points = [[float(index * 10), float(index * 10 + 1), 1.0] for index in range(17)]
        joint_scores = [0.1 + index / 20.0 for index in range(17)]

        class FakeWeights:
            def transforms(self):
                return lambda image: torch.zeros((3, image.height, image.width))

        class FakeDetector:
            def __call__(self, images):
                return [{
                    "scores": torch.tensor([0.95]),
                    "keypoints": torch.tensor([points]),
                    "keypoints_scores": torch.tensor([joint_scores]),
                }]

        captured = []
        def render(size, people, **kwargs):
            captured.extend(people)
            return Image.new("RGB", size)

        with patch.object(nodes, "_keypoint_rcnn_detector", return_value=(FakeDetector(), FakeWeights())), \
             patch.object(nodes, "render_coco17_people", side_effect=render):
            extract_coco17_condition(Image.new("RGB", (200, 100)), backend="keypoint_rcnn", device="cpu")
        self.assertEqual(len(captured), 1)
        expected_points = [joint[:2] if score >= 0.5 else [0.0, 0.0]
                           for joint, score in zip(points, joint_scores)]
        expected_scores = [score if score >= 0.5 else 0.0 for score in joint_scores]
        self.assertEqual([joint[:2] for joint in captured[0]], expected_points)
        self.assertTrue(torch.allclose(torch.tensor([joint[2] for joint in captured[0]]),
                                       torch.tensor(expected_scores)))

    def test_dwpose_joint_order_left_right_and_source_coordinates(self):
        body = [[100.0 + index * 7.0, 200.0 + index * 11.0, 0.9] for index in range(18)]
        # The synthetic neck must not be read as a physical COCO joint.
        body[1] = [999.0, 999.0, 1.0]
        coco = normalize_dwpose_body18(body, keypoint_confidence=0.5)
        self.assertEqual(DWPOSE_COCO17_NAMES, COCO17_NAMES)
        self.assertEqual(len(DWPOSE_BODY18_FOR_COCO17), 17)
        self.assertEqual(DWPOSE_BODY18_NAMES[1], "neck")
        for coco_index, body_index in enumerate(DWPOSE_BODY18_FOR_COCO17):
            self.assertEqual(coco[coco_index], body[body_index], COCO17_NAMES[coco_index])
        self.assertEqual(coco[5], body[5])   # left shoulder
        self.assertEqual(coco[6], body[2])   # right shoulder
        self.assertEqual(coco[15], body[13]) # left ankle
        self.assertEqual(coco[16], body[10]) # right ankle
        self.assertNotIn([999.0, 999.0, 1.0], coco)

    def test_dwpose_confidence_filtering_and_missing_joints(self):
        body = [[float(index), float(index + 100), 0.9] for index in range(18)]
        body[5][2] = 0.49
        body[2] = None
        coco = normalize_dwpose_body18(body, keypoint_confidence=0.5)
        self.assertEqual(coco[5], [0.0, 0.0, 0.0])
        self.assertEqual(coco[6], [0.0, 0.0, 0.0])
        self.assertEqual(coco[7], body[6])
        self.assertAlmostEqual(dwpose_person_confidence(coco), 0.9)

    def test_dwpose_multiple_people_remain_separate_and_ranked(self):
        first = np.array([[10.0 + index, 20.0 + index, 0.60] for index in range(134)], dtype=np.float32)
        second = np.array([[1000.0 + index, 2000.0 + index, 0.95] for index in range(134)], dtype=np.float32)

        class FakeDWPose:
            def dw_pose_estimation(self, image):
                self.shape = image.shape
                return np.stack((first, second))

        detector = FakeDWPose()
        with patch.object(nodes, "_dwpose_detector", return_value=detector):
            people = extract_coco17_people(Image.new("RGB", (320, 180)), backend="dwpose", device="cpu",
                                            person_confidence=0.5, keypoint_confidence=0.5, max_people=2)
        self.assertEqual(detector.shape, (180, 320, 3))
        self.assertEqual(len(people), 2)
        self.assertEqual(people[0][0], second[0].tolist())
        self.assertEqual(people[1][0], first[0].tolist())
        self.assertTrue(all(point[0] >= 1000 for point in people[0]))
        self.assertTrue(all(point[0] < 100 for point in people[1]))

    def test_dwpose_conversion_has_exact_frozen_renderer_parity_and_ignores_extras(self):
        body = [[30.0 + index * 9.0, 40.0 + index * 6.0, 0.9] for index in range(134)]
        # Deliberately hostile values in hands/face cannot affect the body-only result.
        body[24:] = [[9999.0, 8888.0, 1.0] for _ in body[24:]]

        class FakeDWPose:
            def dw_pose_estimation(self, image):
                return np.array([body], dtype=np.float32)

        with patch.object(nodes, "_dwpose_detector", return_value=FakeDWPose()):
            coco = extract_coco17_people(Image.new("RGB", (300, 240)), backend="dwpose", device="cpu")[0]
        expected = _frozen_posebridge_render((300, 240), coco)
        actual = render_coco17_people((300, 240), [coco], keypoint_confidence_threshold=0.5)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertNotIn((9999.0, 8888.0, 1.0), [tuple(point) for point in coco])

    def test_backend_selector_defaults_to_dwpose_and_keypoint_rcnn_fallback_works(self):
        self.assertEqual(Krea2PoseExtractor.INPUT_TYPES()["required"]["backend"][0], ["dwpose", "keypoint_rcnn"])

        class FakeDWPose:
            def dw_pose_estimation(self, image):
                return np.array([[[float(index), float(index), 0.9] for index in range(134)]], dtype=np.float32)

        with patch.object(nodes, "_dwpose_detector", return_value=FakeDWPose()):
            self.assertEqual(len(extract_coco17_people(Image.new("RGB", (20, 10)), device="cpu")), 1)

        class FakeWeights:
            def transforms(self):
                return lambda image: torch.zeros((3, image.height, image.width))

        class FakeRCNN:
            def __call__(self, images):
                return [{"scores": torch.tensor([0.9]),
                         "keypoints": torch.tensor([[[float(i), float(i + 1), 1.0] for i in range(17)]]),
                         "keypoints_scores": torch.full((1, 17), 0.9)}]

        with patch.object(nodes, "_keypoint_rcnn_detector", return_value=(FakeRCNN(), FakeWeights())):
            fallback = extract_coco17_people(Image.new("RGB", (20, 10)), backend="keypoint_rcnn", device="cpu")
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0][5][:2], [5.0, 6.0])
        self.assertAlmostEqual(fallback[0][5][2], 0.9)

    def test_conditioning_renderer_synthesizes_only_neck(self):
        points = [[0.0, 0.0, 0.0] for _ in range(17)]
        points[5] = [20.0, 20.0, 0.9]
        points[6] = [40.0, 20.0, 0.9]
        points[8] = [50.0, 35.0, 0.9]
        rendered = render_coco17_people((64, 64), [points], keypoint_confidence_threshold=0.5)
        self.assertEqual(rendered.size, (64, 64))
        self.assertEqual(rendered.getpixel((45, 27)), BODY_COLORS[2])
        self.assertEqual(rendered.getpixel((35, 20)), BODY_COLORS[0])
        no_shoulders = render_coco17_people((64, 64), [[[0.0, 0.0, 0.0] for _ in range(17)]])
        self.assertIsNone(no_shoulders.getbbox())

    def test_tensor_range_conversion(self):
        original = torch.tensor([[[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]]], dtype=torch.float32)
        image = _image_to_pil(original)
        restored = _pil_to_image(image)
        self.assertEqual(tuple(restored.shape), (1, 1, 2, 3))
        self.assertTrue(torch.all(restored >= 0) and torch.all(restored <= 1))
        self.assertTrue(torch.allclose(restored[0], original, atol=1 / 255))

    def test_release_sha_validation_rejects_wrong_bytes(self):
        self.assertEqual(RELEASE_SHA256, "6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "krea2-pose-control-mix025.safetensors"
            artifact.write_bytes(b"not the frozen release")
            with self.assertRaises(ReleaseValidationError):
                validate_release_artifact(artifact, inspect_tensors=False)

    def test_workflow_json_validity(self):
        root = Path(__file__).resolve().parents[2] / "workflows"
        for name, required_node in (("krea2_pose_from_reference.json", "Krea2PoseExtractor"),
                                    ("krea2_pose_from_condition.json", "Krea2PoseCondition")):
            workflow = json.loads((root / name).read_text())
            self.assertEqual(workflow["version"], 0.4)
            self.assertIn("Krea2PoseGenerate", [node["type"] for node in workflow["nodes"]])
            self.assertIn(required_node, [node["type"] for node in workflow["nodes"]])
            self.assertTrue(workflow["links"])

    def test_custom_node_import(self):
        self.assertEqual(set(krea2_pose_control.NODE_CLASS_MAPPINGS),
                         {"Krea2PoseExtractor", "Krea2PoseCondition", "Krea2PoseGenerate"})


if __name__ == "__main__":
    unittest.main()
