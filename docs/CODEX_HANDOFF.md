# Project handoff

## Current objective

The standalone ComfyUI package now has a DWPose reference-image extractor that preserves the frozen Krea-2 PoseBridge COCO-17 conditioning contract. The next bounded action is a visual comparison of DWPose conditions for the four original reference images; do not run generation yet.

## Frozen facts in force

- Release `krea2-pose-control-mix025.safetensors`: SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`; 450 tensors / 215,488,512 trainable parameters.
- Release defaults: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0, native/aspect geometry, no Style-LoRA.
- Frozen renderer contract remains unchanged: COCO-17 maps to historic Body-18 by `(0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10)`; only shoulders synthesize its neck; 17 ordered rainbow limbs use 3px strokes; all retained endpoints are white radius-4 circles.
- The normal host is ARM64 GH200 / 96 GB, Python 3.10.12, PyTorch 2.7.0 CUDA 12.8, cuDNN 9.8, Triton 3.3.0. Sandbox CUDA absence is not host evidence.

## DWPose implementation

- `Krea2PoseExtractor` now defaults to `dwpose`; `keypoint_rcnn` remains selectable fallback. Person confidence, per-keypoint confidence, and `max_people` are configurable.
- `dwpose.py` maps DWPose structured Body-18 entries to COCO-17 in the exact required order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. It reads neither DWPose's neck nor hand, finger, foot-detail, or dense face entries.
- The node calls the ComfyUI-ControlNet-Aux structured whole-body API at source RGB geometry, retains DWPose confidences, filters missing/weak joints before rendering, ranks people from retained body confidence, and sends each separate COCO-17 person only to the frozen renderer. No native DWPose/OpenPose raster is consumed by Krea.
- Preflight accepts `--backend`, `--person-confidence`, `--keypoint-confidence`, `--max-people`, `--device`, and `--dwpose-model-dir`; it writes the frozen condition and reports person/joint counts without generation.
- Dependency: install ComfyUI-ControlNet-Aux through ComfyUI Manager and restart ComfyUI. Its local checkpoint/cache root needs `yzd-v/DWPose/yolox_l.onnx` and `yzd-v/DWPose/dw-ll_ucoco_384.onnx`. `--dwpose-model-dir` is an optional local checkpoint root override; no original DWPose checkout or NFS path is required.

## Completed / green checks

- Synthetic integration coverage proves DWPose joint-order and left/right mapping, source-coordinate orientation, confidence/missing-joint omission, per-person separation/ranking, no hand/face leakage, exact frozen-renderer parity after conversion, default selector behavior, and Keypoint R-CNN fallback behavior.
- PASS: `python -m unittest comfyui/krea2_pose_control/tests/test_integration.py` (15 tests).
- PASS: `python -m compileall -q comfyui/krea2_pose_control`.
- PASS: `python -m krea2_pose_control.preflight --help` from `comfyui/`.
- PASS: `git diff --check`.

## Exact GH200 extraction-smoke invocation

The original four reference files are absent from this workspace. Copy them to `docs/comfyui/dwpose-smoke/` using the four documented `*-reference.png` names, then from `ComfyUI/custom_nodes` run the commands in `comfyui/README.md` under **GH200 extraction-smoke commands (no generation)**. They are four explicit DWPose preflights for `elegant-wandering-knight`, `psychedelic-swordswoman`, `comic-fashion`, and `dark-fantasy-jester`; all output only `docs/comfyui/dwpose-smoke/*-condition.png`.

## Files changed this session

- `comfyui/krea2_pose_control/dwpose.py` (new body-only normalizer)
- `comfyui/krea2_pose_control/nodes.py`
- `comfyui/krea2_pose_control/preflight.py`
- `comfyui/krea2_pose_control/tests/test_integration.py`
- `comfyui/workflows/krea2_pose_from_reference.json`
- `comfyui/README.md`
- `docs/CODEX_HANDOFF.md`

## Unresolved / exact next action

ComfyUI-ControlNet-Aux, its DWPose models, and the four original RGB references are not installed/present in this audit workspace, so no real DWPose extraction has been performed. On the GH200, run the four documented no-generation DWPose preflights, inspect their frozen PoseBridge rasters against the prior Keypoint R-CNN conditions, retain the JSON person/joint counts, and visually compare them. Do not change the renderer, release identity, training/evaluation artifacts, benchmark values, or checkpoints.
