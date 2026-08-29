# Project handoff

## Current objective and decisions

The current bounded milestone is the audit-only fixed-box differentiable
torchvision Keypoint R-CNN pose critic. It is implemented but **the real-image
GH200 domain audit has not run**, so the critic is HOLD for any future
experiment. Production training remains flow-matching MSE only: `train.py`,
the training objective, VAE decoding, x0/timestep logic, dependency stack,
Phase-1 provenance, and the existing external Keypoint R-CNN PCK evaluator
were not changed.

Phase-1 remains authoritative. The immutable v3 sidecar's COCO17 training
coordinates, fixed boxes, and `reward_joint_valid` mask are the sole target
inputs. The seven reviewed Human-Art source-OOB joints stay masked; Danbooru
stays `pose_reward_available=false` and is excluded. RTMPose and DWPose remain
rejected as differentiable critics because they require the unsupported MM*
dependency family; no MM*/ONNX dependencies were installed.

## Verified critic design

- `pose_controlnet/keypoint_critic.py` loads the exact model used by the
  external evaluator: `keypointrcnn_resnet50_fpn` with official COCO_V1
  weights (`DEFAULT` resolves to COCO_V1), freezes every parameter, and forces
  evaluation mode.
- From torchvision 0.22 source, its differentiable tensor path is
  `GeneralizedRCNNTransform -> backbone -> keypoint_roi_pool -> keypoint_head
  -> keypoint_predictor`. The default keypoint ROI pool is 14×14 and the
  predictor is 14→28 transpose-convolution then 28→56 bilinear upsample, so
  raw logits are `[total_fixed_people, 17, 56, 56]` in Phase-1 COCO17 order.
- Fixed authoritative `xyxy` boxes are passed through only torchvision's
  transform resize and then directly to the keypoint ROI branch. The wrapper
  bypasses `model.rpn`, all ROI box classification/regression operations,
  detector scores/thresholds, NMS, torchvision keypoint decoding, and every
  other detector decision.
- The module provides spatial-softmax ROI coordinates, masked coordinate Huber,
  normalized Gaussian heatmap targets / masked KL, and detached-only soft/
  argmax PCK, error, entropy, and peak-probability diagnostics. It does not
  attach any loss to training.

## Files changed this session

- Added `pose_controlnet/keypoint_critic.py`.
- Added `scripts/audit_keypoint_critic.py`.
- Added `tests/test_keypoint_critic.py`.
- Added `docs/KEYPOINT_RCNN_CRITIC_AUDIT.md`.
- Rewrote this handoff. No existing source-code file was modified.

## Checks and blockers

- PASS: `PYTHONPATH=. python -m unittest tests.test_keypoint_critic` — 10 CPU
  tests passed: spatial and ROI mapping, masks including invalid-zero KL,
  Gaussian normalization/KL finiteness, synthetic heatmap autograd, detached
  diagnostics, and mocked frozen-model contract.
- PASS: `python -m py_compile pose_controlnet/keypoint_critic.py
  scripts/audit_keypoint_critic.py tests/test_keypoint_critic.py`.
- PASS: `PYTHONPATH=. python scripts/audit_keypoint_critic.py --help`.
- PASS: `git diff --check` after tracked-file changes; no unrelated worktree
  changes were present at session start.
- HOLD: no official COCO_V1 checkpoint load or real-image/GH200 forward was
  run in this Codex sandbox. Therefore no domain-quality assertion, PCK claim,
  or training integration is justified.

## Exact next action

On the actual GH200 shell, run the real-image audit before making any decision
about future pose-loss work:

```bash
PYTHONPATH=. python scripts/audit_keypoint_critic.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_audit.json
```

Use the actual immutable v3 sidecar location if it differs. Confirm finite
metrics, nonzero finite RGB gradient for each source's first sample, and no
critic parameter gradient. Do not modify `train.py`, decode VAE outputs, or
introduce a pose loss unless separately authorized after that audit.
