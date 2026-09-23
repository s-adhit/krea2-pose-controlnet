"""ComfyUI nodes for the frozen Krea-2 Pose Control-LoRA v1 release."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import numpy as np
import torch
from PIL import Image

from .krea2_pose_runtime.runtime import Runtime, generate, load_runtime
from .controlnet_aux import ControlNetAuxDependencyError, load_dwpose_module
from .dwpose import dwpose_person_confidence, normalize_dwpose_body18
from .renderer import render_coco17_people

_DETECTORS: dict[str, Any] = {}
_DWPOSE_DETECTORS: dict[tuple[str, str | None], Any] = {}
_RUNTIMES: dict[tuple[str, str, str], Runtime] = {}


def _image_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("ComfyUI IMAGE must have H×W×3 or H×W×4 layout")
    if not torch.isfinite(image).all():
        raise ValueError("IMAGE contains NaN or Inf")
    pixels = image[..., :3].detach().float().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(pixels, "RGB")


def _pil_to_image(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _keypoint_rcnn_detector(device: str):
    if device not in _DETECTORS:
        try:
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
        except ImportError as error:
            raise RuntimeError("torchvision with Keypoint R-CNN is required for the keypoint_rcnn backend") from error
        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        _DETECTORS[device] = (keypointrcnn_resnet50_fpn(weights=weights).to(device).eval(), weights)
    return _DETECTORS[device]


def _dwpose_detector(device: str, *, model_dir: str | Path | None = None):
    """Load ComfyUI-ControlNet-Aux DWPose without importing its preview node."""
    resolved_dir = None if model_dir is None else str(Path(model_dir).expanduser().resolve())
    if resolved_dir is not None and not Path(resolved_dir).is_dir():
        raise RuntimeError(f"DWPose model directory does not exist: {resolved_dir}")
    key = (device, resolved_dir)
    if key in _DWPOSE_DETECTORS:
        return _DWPOSE_DETECTORS[key]
    if resolved_dir is not None:
        existing = os.environ.get("AUX_ANNOTATOR_CKPTS_PATH")
        if existing is not None and Path(existing).expanduser().resolve() != Path(resolved_dir):
            raise RuntimeError("AUX_ANNOTATOR_CKPTS_PATH is already set to a different DWPose model directory")
        os.environ["AUX_ANNOTATOR_CKPTS_PATH"] = resolved_dir
    try:
        detector_type = load_dwpose_module().DwposeDetector
    except (ControlNetAuxDependencyError, AttributeError) as error:
        raise RuntimeError(
            "The dwpose backend requires ComfyUI-ControlNet-Aux (including its DWPose/ONNX dependencies). "
            f"{error} "
            "This node intentionally will not accept a native OpenPose/DWPose raster."
        ) from error
    try:
        detector = detector_type.from_pretrained(
            "yzd-v/DWPose", det_filename="yolox_l.onnx", pose_filename="dw-ll_ucoco_384.onnx",
            torchscript_device=device,
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to load DWPose. Ensure yzd-v/DWPose/yolox_l.onnx and "
            "yzd-v/DWPose/dw-ll_ucoco_384.onnx are available in the configured ControlNet-Aux cache."
        ) from error
    _DWPOSE_DETECTORS[key] = detector
    return detector


def _keypoint_rcnn_people(image: Image.Image, *, person_confidence: float,
                           keypoint_confidence: float, max_people: int | None, device: str) -> list[list[list[float]]]:
    model, weights = _keypoint_rcnn_detector(device)
    tensor = weights.transforms()(image.convert("RGB")).to(device)
    with torch.inference_mode():
        output = model([tensor])[0]
    keypoint_scores = output.get("keypoints_scores")
    if keypoint_scores is None:
        raise RuntimeError("COCO_V1 Keypoint R-CNN did not return per-joint keypoints_scores")
    people: list[list[list[float]]] = []
    for score, keypoints, joint_scores in zip(
        output["scores"].detach().cpu().tolist(), output["keypoints"].detach().cpu().tolist(),
        keypoint_scores.detach().cpu().tolist(),
    ):
        if float(score) < person_confidence:
            continue
        if len(keypoints) != 17 or len(joint_scores) != 17:
            raise RuntimeError("COCO_V1 Keypoint R-CNN returned a non-COCO-17 result")
        # torchvision's keypoints[..., 2] is an inference visibility marker,
        # not a per-joint confidence.  Omit weak physical joints before the
        # frozen renderer; it will synthesize only its permitted neck.
        people.append([[float(x), float(y), float(joint_score)] if float(joint_score) >= keypoint_confidence
                       else [0.0, 0.0, 0.0]
                       for (x, y, _visibility), joint_score in zip(keypoints, joint_scores)])
        if max_people is not None and len(people) >= max_people:
            break
    return people


def _dwpose_people(image: Image.Image, *, person_confidence: float,
                   keypoint_confidence: float, max_people: int | None, device: str,
                   dwpose_model_dir: str | Path | None) -> list[list[list[float]]]:
    detector = _dwpose_detector(device, model_dir=dwpose_model_dir)
    # The ControlNet-Aux runtime returns [people, whole-body joints, x/y/score]
    # here. Its first 18 joints are Body-18; reading that structured output
    # retains DWPose confidences and bypasses its preview raster entirely.
    source_rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    try:
        keypoints_info = detector.dw_pose_estimation(source_rgb)
    except AttributeError as error:
        raise RuntimeError("Installed DWPose runtime does not expose structured whole-body keypoints") from error
    if keypoints_info is None:
        return []
    keypoints_info = np.asarray(keypoints_info)
    if keypoints_info.ndim != 3 or keypoints_info.shape[1] < 18 or keypoints_info.shape[2] < 3:
        raise RuntimeError("DWPose returned malformed structured whole-body keypoints")
    ranked: list[tuple[float, int, list[list[float]]]] = []
    for index, body_keypoints in enumerate(keypoints_info[:, :18, :3]):
        coco17 = normalize_dwpose_body18(body_keypoints.tolist(), keypoint_confidence=keypoint_confidence)
        confidence = dwpose_person_confidence(coco17)
        if confidence >= person_confidence:
            ranked.append((confidence, index, coco17))
    # DWPose does not expose a comparable detector-box score here. Ranking the
    # mean retained body confidence is deterministic and never mixes people.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked if max_people is None else ranked[:max_people]
    return [person for _confidence, _index, person in selected]


def extract_coco17_people(image: Image.Image, *, backend: str = "dwpose", person_confidence: float = 0.5,
                           keypoint_confidence: float = 0.5, max_people: int | None = None,
                           device: str | None = None, dwpose_model_dir: str | Path | None = None) -> list[list[list[float]]]:
    """Detect people and return only frozen-contract COCO-17 source pixels."""
    if backend not in {"dwpose", "keypoint_rcnn"}:
        raise ValueError("backend must be 'dwpose' or 'keypoint_rcnn'")
    if not 0 <= person_confidence <= 1 or not 0 <= keypoint_confidence <= 1:
        raise ValueError("person and keypoint confidence thresholds must be in [0, 1]")
    if max_people is not None and max_people < 1:
        raise ValueError("max_people must be positive when supplied")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "dwpose":
        return _dwpose_people(image, person_confidence=person_confidence, keypoint_confidence=keypoint_confidence,
                               max_people=max_people, device=device, dwpose_model_dir=dwpose_model_dir)
    return _keypoint_rcnn_people(image, person_confidence=person_confidence, keypoint_confidence=keypoint_confidence,
                                 max_people=max_people, device=device)


def extract_coco17_condition(image: Image.Image, *, backend: str = "dwpose", person_confidence: float = 0.5,
                              keypoint_confidence: float = 0.5, max_people: int | None = None,
                              device: str | None = None, dwpose_model_dir: str | Path | None = None) -> Image.Image:
    """Reference RGB -> exact frozen PoseBridge conditioning raster."""
    people = extract_coco17_people(
        image, backend=backend, person_confidence=person_confidence, keypoint_confidence=keypoint_confidence,
        max_people=max_people, device=device, dwpose_model_dir=dwpose_model_dir,
    )
    return render_coco17_people(image.size, people, keypoint_confidence_threshold=keypoint_confidence)


class Krea2PoseExtractor:
    """Reference IMAGE -> exact project COCO-17 conditioning raster."""
    CATEGORY = "Krea2 Pose Control"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_condition",)
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference_image": ("IMAGE",),
            "backend": (["dwpose", "keypoint_rcnn"], {"default": "dwpose"}),
            "person_confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "keypoint_confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "max_people": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
        }}

    def extract(self, reference_image, backend="dwpose", person_confidence_threshold=0.5,
                keypoint_confidence_threshold=0.5, max_people=0):
        rendered = [
            _pil_to_image(extract_coco17_condition(
                _image_to_pil(item), backend=str(backend), person_confidence=float(person_confidence_threshold),
                keypoint_confidence=float(keypoint_confidence_threshold),
                max_people=None if int(max_people) == 0 else int(max_people),
            ))
            for item in reference_image
        ]
        return (torch.cat(rendered, dim=0),)


class Krea2PoseCondition:
    """Pass through an existing project-format condition after safe IMAGE validation."""
    CATEGORY = "Krea2 Pose Control"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_condition",)
    FUNCTION = "validate"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose_control_image": ("IMAGE",)}}

    def validate(self, pose_control_image):
        if pose_control_image.ndim != 4 or pose_control_image.shape[-1] not in (3, 4):
            raise ValueError("pose-control IMAGE must be B×H×W×3 or B×H×W×4")
        if pose_control_image.shape[1] < 1 or pose_control_image.shape[2] < 1 or not torch.isfinite(pose_control_image).all():
            raise ValueError("pose-control IMAGE has invalid geometry or non-finite values")
        # Valid controls remain bit-for-bit unchanged; this node does not convert generic OpenPose maps.
        return (pose_control_image,)


class Krea2PoseGenerate:
    """Frozen Krea-2 Turbo v1 generation from a project-format pose raster."""
    CATEGORY = "Krea2 Pose Control"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "pose_condition": ("IMAGE",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
            "control_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.05}),
            "steps": ("INT", {"default": 8, "min": 8, "max": 8}),
            "mu": ("FLOAT", {"default": 1.15, "min": 1.15, "max": 1.15, "step": 0.01}),
            "geometry_mode": (["native_aspect", "explicit_size"], {"default": "native_aspect"}),
            "width": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
            "release_artifact_path": ("STRING", {"default": ""}),
            "krea2_turbo_path": ("STRING", {"default": ""}),
            "krea2_raw_path": ("STRING", {"default": ""}),
        }}

    def generate(self, prompt, pose_condition, seed=42, control_scale=1.0, steps=8, mu=1.15,
                 geometry_mode="native_aspect", width=1024, height=1024,
                 release_artifact_path="", krea2_turbo_path="", krea2_raw_path=""):
        key = (str(Path(krea2_turbo_path).expanduser()), str(release_artifact_path), str(Path(krea2_raw_path).expanduser()))
        if key not in _RUNTIMES:
            _RUNTIMES[key] = load_runtime(turbo_path=krea2_turbo_path, release_artifact=release_artifact_path,
                                          base_raw_path=krea2_raw_path)
        dimensions = (None, None) if geometry_mode == "native_aspect" else (int(width), int(height))
        images = []
        for offset, condition in enumerate(pose_condition):
            output = generate(_RUNTIMES[key], prompt=prompt, pose_image=_image_to_pil(condition),
                              seed=int(seed) + offset, control_scale=float(control_scale), steps=int(steps), mu=float(mu),
                              width=dimensions[0], height=dimensions[1])
            images.append(_pil_to_image(output))
        return (torch.cat(images, dim=0),)
