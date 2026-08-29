# DWPose suitability and architecture audit

Date: 2026-08-29.  Scope: research only.  No model was downloaded, no GPU
inference was run, no dependency was installed, and no training code or loss
was changed.

## Project baseline

The repository contains no DWPose implementation or dependency.  Its current
environment is Python 3.10.12 on aarch64 with the host-owned PyTorch 2.7.0
(CUDA runtime 12.8); `mmcv`, `mmengine`, `mmdet`, `mmpose`, and `onnxruntime`
are all absent.  The Phase-1 sidecar remains authoritative: every available
person target is `coco17`, in persisted training coordinates, with its
per-joint `reward_joint_valid` mask.  This audit does not alter that contract.

## Official DWPose surfaces

The official [IDEA-Research/DWPose repository](https://github.com/IDEA-Research/DWPose)
has two materially different surfaces:

1. Its `onnx` / `opencv_onnx` ControlNet-facing path uses
   `yolox_l.onnx` for detection and `dw-ll_ucoco_384.onnx` for pose.  The
   former repository path creates ONNX Runtime sessions for both models and
   runs detector output into `inference_pose`.
2. Its native training/research tree is a vendored MMPose fork.  Official
   training commands use `DWPoseDistiller`, then `pth_transfer.py` to turn the
   distilled checkpoint into a regular pose checkpoint.  The author's
   [yzd-v/DWPose model repository](https://huggingface.co/yzd-v/DWPose/tree/main)
   publishes both `dw-ll_ucoco_384.onnx` and `dw-ll_ucoco_384.pth`; the latter
   is a 407 MB PyTorch/MMEngine checkpoint, not a standalone pure-PyTorch
   module.

The native 384 model configuration is
[`rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py`](https://github.com/IDEA-Research/DWPose/blob/onnx/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py):

- `TopdownPoseEstimator` with a `CSPNeXt` P5 large backbone, `arch='P5'`,
  `in_channels=1024`, and an `RTMCCHead`.
- The head has 133 output joints and uses SimCC with `input_size=(288, 384)`
  (width, height) and split ratio 2.  Before decoding, the expected raw heads
  are therefore `pred_x: [B, 133, 576]` and `pred_y: [B, 133, 768]`.
- The model normalizes RGB with mean `[123.675, 116.28, 103.53]` and standard
  deviation `[58.395, 57.12, 57.375]`.  The published evaluator's BGR-to-RGB
  conversion must be accounted for exactly when constructing a future crop.

This establishes that `dw-ll_ucoco_384` is a whole-body DWPose checkpoint
implemented as a top-down RTMPose-family estimator; it is not a distinct,
dependency-free DWPose network definition.

## Raw outputs, autograd, and supplied boxes

The official native head's `forward(feats)` returns `pred_x, pred_y` directly;
it uses only PyTorch operations through its SimCC layers.  Decoding occurs in
the separate `predict` path.  Therefore, *if the native stack can be loaded*,
calling `model.extract_feat(crops)` followed by `model.head.forward(feats)`
would retain a valid autograd path from raw SimCC tensors to `crops`.  The
pose model parameters can be frozen without `torch.no_grad()` so gradients
still reach decoded RGB.

The official `TopdownPoseEstimator` accepts an `(N, C, H, W)` tensor and its
`predict` implementation first extracts features, then calls the head.  It
does not run a detector.  The ControlNet wrapper is merely a composition:
`inference_detector(...)` then `inference_pose(session, out_bbox, oriImg)`.
Thus YOLOX, NMS, and detector thresholds can be excluded from a top-down
critic by using constant Phase-1 sidecar boxes.

A compliant future graph would use a differentiable fixed-box crop/affine
sample (for example, PyTorch grid sampling) from decoded RGB, then normalize
and feed it to the frozen pose estimator, selecting raw SimCC bins for the
first 17 joints.  The box is data, not a learned/differentiated detector
result.  Argmax, confidence thresholding, skeleton rendering, PCK, and
Hungarian matching remain outside this graph.  This is a feasibility finding,
not an implementation recommendation.

The ONNX wrapper exposes its two pre-decode SimCC outputs internally, but its
path uses ONNX Runtime and NumPy/OpenCV preprocessing and explicitly decodes
with `np.argmax`.  Those tensors cannot be connected to the repository's
PyTorch RGB tensor for backpropagation.  Exposing raw ONNX outputs does not
change that boundary.

## COCO-17 mapping

The native model uses the standard COCO-WholeBody order.  Its first 17 output
channels are exactly the Phase-1 COCO body joints, in the same order:

| DWPose raw channel | Phase-1 joint |
| ---: | --- |
| 0--4 | nose, left_eye, right_eye, left_ear, right_ear |
| 5--10 | left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist |
| 11--16 | left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle |

Channels 17--22 are foot keypoints, followed by 68 face keypoints and 21
keypoints for each hand.  The OpenPose rendering wrapper inserts a synthetic
neck for display only; it is neither one of these first 17 channels nor an
authoritative target.  No target remapping or Phase-1 semantic change is
required for the body channels.

## Candidate decision table

| Candidate | Backend | Dependencies | Raw outputs | Input differentiability | Detector separable | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| A. Official ControlNet path | ONNX Runtime (`dw-ll_ucoco_384.onnx`) + YOLOX ONNX | ONNX Runtime; NumPy/OpenCV; detector model for normal pipeline | Yes, internal SimCC arrays before NumPy decode | No: runtime/NumPy/OpenCV break PyTorch autograd | Yes: pose routine takes boxes | Not recommended for a critic; suitable only for offline preprocessing/evaluation |
| B. Official native PyTorch path | DWPose fork of MMPose, TopdownPoseEstimator / CSPNeXt / RTMCC | MMPose fork, MMEngine, MMCV, MMDetection; plus its runtime packages | Yes: `head.forward` returns SimCC tensors | Yes in principle, when directly using tensor forward path | Yes | Not recommended: the official config imports `CSPNeXt` from MMDetection and the fork pins `mmcv>=2.0.0,<2.1.0`, `mmdet>=3.0.0,<3.2.0`, `mmengine>=0.4.0,<1.0.0` |
| C. Third-party TorchScript conversion | Example: `hr16/DWPose-TorchScript-BatchSize5` distributed through ComfyUI auxiliary tooling | Converter-specific, unpinned; model artifact and wrapper | Unverified against official raw SimCC contract | Nominally possible for a clean TorchScript graph, but unproven | Likely | Do not adopt: it is not an official PyTorch checkpoint/runtime and lacks required checkpoint/architecture/output-parity evidence |

## Compatibility finding and decision

The native path is technically differentiable and detector-separable, but it
does **not** avoid the abandoned dependency problem.  The official configuration
directly scopes the backbone to `mmdet`, and the DWPose fork explicitly pins
MMCV/MMDetection/MMEngine versions.  No compatibility evidence exists for
that old extension-bearing stack on this repository's ARM64 PyTorch 2.7/CUDA
12.8 GH200 environment; adding it would violate this audit's dependency
boundary and risk replacing or compiling against the host-owned stack.

**Recommendation: abandon DWPose as the differentiable critic.**  Retain the
official ONNX pair as an optional future *non-gradient* preprocessing or
evaluation backend only.  Do not install dependencies, download checkpoints,
or implement a reward on the basis of this audit.

## Exact next action

Record the decision, keep the Phase-1 sidecar and flow-only training objective
unchanged, and do not start a DWPose critic implementation.  If offline pose
evaluation is later requested, separately audit the ONNX model's licensing,
artifact hashes, and preprocessing parity; it must remain outside the
training graph.
