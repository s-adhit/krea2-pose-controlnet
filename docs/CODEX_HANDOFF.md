# Project handoff

## Current objective

The standalone ComfyUI integration remains scoped to frozen Krea-2 Pose Control-LoRA v1 and its exact COCO-17 PoseBridge conditioning format. The latest completed milestone audited the poor reference-image extraction smoke results before any detector replacement.

## Frozen facts in force

- Release `krea2-pose-control-mix025.safetensors`: SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`; 450 tensors / 215,488,512 trainable parameters.
- Release defaults: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0, native/aspect geometry, and no Style-LoRA.
- Frozen renderer contract: torchvision COCO-17 order maps to historic Body-18 by `(0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10)`; Body-18 index 1 is a neck synthesized only from COCO left/right shoulders (5/6). It draws the 17 ordered rainbow limbs with 3px strokes, then white radius-4 joints.
- Normal host remains ARM64 GH200 / 96 GB, Python 3.10.12, PyTorch 2.7.0 CUDA 12.8, cuDNN 9.8, Triton 3.3.0. Sandbox CUDA absence is not host evidence.

## Extraction audit result

- `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1` metadata confirms exact COCO order: nose; left/right eyes; left/right ears; left/right shoulders, elbows, wrists, hips, knees, ankles.
- The ComfyUI mapping, historic `PoseBridge` reconstruction, and `reference_pose.py` agree limb-for-limb and color/order-for-color/order. The mapping is correct: left/right shoulders, elbows, wrists, hips, knees, ankles, eyes, and ears all reach their intended Body-18 indices.
- A separate renderer-integration defect was fixed in `Krea2PoseExtractor`: torchvision's `keypoints[..., 2]` is its inference visibility marker (the installed 0.22 implementation emits `1`), not a per-joint confidence. The extractor now passes `keypoints_scores` to the frozen renderer, so its keypoint threshold can omit weak joints instead of drawing all 17 joints on each accepted box.
- Classification is **C, both**: the unconditional per-joint rendering was a confirmed integration cause of implausible long/collapsed limbs; the observed extra jester people can only originate from separate accepted detector boxes, since the renderer creates neither people nor cross-person limbs. Actual smoke input/output files were not present in this workspace, so detector quality for each stylized image is not independently quantified.

## ComfyUI delivery and completed gates

- Package: `comfyui/krea2_pose_control/`; workflows and instructions remain under `comfyui/`.
- Focused synthetic tests now assert torchvision order, all uniquely-labelled COCO-to-Body-18 assignments, shoulder-only neck synthesis, exact frozen raster parity, and extractor pass-through of `keypoints_scores` rather than the visibility column.

PASS:

```bash
uv run python -m unittest comfyui/krea2_pose_control/tests/test_integration.py
uv run python -m unittest tests/test_reference_pose.py tests/test_keypoint_critic.py
uv run python -m compileall -q comfyui/krea2_pose_control
git diff --check
```

## Files changed this session

- `comfyui/krea2_pose_control/nodes.py`
- `comfyui/krea2_pose_control/tests/test_integration.py`
- `docs/CODEX_HANDOFF.md`

## Exact next recommended action

Re-run the four original smoke inputs with this corrected per-joint score handoff and retain the raw person boxes plus `keypoints_scores` for review. If stylized-image failures remain, test a ComfyUI-compatible DWPose/OpenPose-style detector backend (DWPose first) behind the same COCO-17 normalization and this unchanged frozen PoseBridge renderer; do not feed its native OpenPose raster to the model and do not alter release or evaluation artifacts.
