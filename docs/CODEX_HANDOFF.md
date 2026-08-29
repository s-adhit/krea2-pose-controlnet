# Project handoff

## Current objective and decisions

The DWPose-only suitability audit is complete.  No critic exists and none was
implemented.  The decision is **not to use DWPose as a differentiable critic**:
the only official native PyTorch route recreates the MMEngine/MMCV/MMDetection/
MMPose dependency family that led to the abandoned RTMPose path.  The official
ONNX pair may be considered only for a separately approved non-gradient
preprocessing/evaluation use case.

The production training objective remains flow-matching MSE only.  Phase-1
provenance remains unchanged: `data/pose_targets_authoritative_v1.jsonl` and
sidecar v3 are authoritative; `reward_joint_valid` masks (including the seven
visible Human-Art source-OOB masks) remain intact.  Danbooru remains flow-only.
No target semantics, control reconstruction, paired preprocessing, or external
Keypoint R-CNN PCK evaluation changed.

## Verified findings

- Environment: Python 3.10.12, aarch64, host-owned PyTorch 2.7.0/CUDA 12.8;
  `mmcv`, `mmengine`, `mmdet`, `mmpose`, and `onnxruntime` are absent.
- Official ControlNet DWPose is `yolox_l.onnx` plus
  `dw-ll_ucoco_384.onnx`.  It obtains raw SimCC values but uses ONNX Runtime,
  NumPy/OpenCV, and `np.argmax`; it cannot backpropagate to this training
  graph.  Detector and pose routines are separable, so it is usable top-down
  with supplied boxes only outside a gradient path.
- The official native checkpoint `dw-ll_ucoco_384.pth` exists in the author's
  model repository, but loads through the DWPose MMPose fork, not a standalone
  PyTorch module.  Its 384 model is a top-down CSPNeXt-L / RTMCCHead / SimCC
  whole-body estimator with 133 joints.  The raw head forward exposes
  `pred_x` and `pred_y`; expected 384-model shapes are `[B,133,576]` and
  `[B,133,768]`, so input gradients are technically possible if the native
  stack loads.
- This is not a clean path: its config imports CSPNeXt from MMDetection, and
  its exact pins are `mmcv>=2.0.0,<2.1.0`, `mmdet>=3.0.0,<3.2.0`, and
  `mmengine>=0.4.0,<1.0.0`.  Compatibility with ARM64 PyTorch 2.7/CUDA 12.8
  is unverified and requires the prohibited dependency experiment.
- DWPose channels 0--16 match Phase-1 COCO-17 exactly (nose through right
  ankle); the rendered OpenPose neck is synthetic and not a target.

See `docs/DWPOSE_AUDIT.md` for sources, candidate table, architecture, and
reward-graph boundary analysis.

## Files changed this session

- Added: `docs/DWPOSE_AUDIT.md`.
- Modified: `docs/CODEX_HANDOFF.md`.
- No Python, config, data, manifest, dependency, model, or training files
  changed; no model files were downloaded and no GPU inference was run.

## Checks and blockers

- PASS: targeted repository/dependency inventory and Python environment probe.
- PASS: `git diff --check` after the documentation update.
- Blocker/decision: a differentiable DWPose critic is rejected because the
  official native route has the same dependency risk as RTMPose.  Do not
  install MM* packages, ONNX Runtime, or download DWPose checkpoints for this
  critic.

## Exact next recommended action

Keep DWPose out of the training graph and retain flow-only training.  If an
offline pose-evaluation utility is later requested, perform a new, isolated
ONNX-only artifact/hash/preprocessing-parity audit; do not connect it to loss
or backpropagation.
