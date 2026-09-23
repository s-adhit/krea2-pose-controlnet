# Project handoff

## Current objective

Standalone ComfyUI integration for frozen Krea-2 Pose Control-LoRA v1 is implemented. It is scoped to the release's exact COCO-17 skeleton condition and canonical Krea-2 Turbo path; it adds no new training capability.

## Frozen facts in force

- Release `krea2-pose-control-mix025.safetensors`: SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`; 450 tensors / 215,488,512 trainable parameters.
- Release defaults: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, fixed mu (no resolution-dependent shift), control scale 1.0, native/aspect geometry, no Style-LoRA.
- Renderer contract is historic PoseBridge topology: COCO-17 mapped to unified Body-18, neck only when both shoulders exist, 17 ordered rainbow limbs with 3px strokes, white radius-4 endpoints. No hands/fingers/dense face landmarks.
- Normal host remains ARM64 GH200 / 96 GB, Python 3.10.12, PyTorch 2.7.0 CUDA 12.8, cuDNN 9.8, Triton 3.3.0. Codex sandbox CUDA absence is not host evidence.

## ComfyUI delivery

- Custom-node package: `comfyui/krea2_pose_control/`; copy directly to `ComfyUI/custom_nodes/krea2_pose_control/`.
- Workflows: `comfyui/workflows/krea2_pose_from_reference.json` and `comfyui/workflows/krea2_pose_from_condition.json`.
- Installation, model layout, defaults, and limitations: `comfyui/README.md`.
- The package vendors package-relative MMDiT model surgery and Turbo sampling. It has no runtime `sys.path` mutation, no dependency on this repository path, and no NFS/checkpoint fallback.
- `Krea2PoseExtractor` uses torchvision `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`, selects all score-qualified people (optional cap), uses only 17 body joints, preserves source canvas geometry, and never creates missing joints.
- `Krea2PoseCondition` leaves a finite valid ComfyUI IMAGE untouched. Raster provenance cannot be inferred from pixels, so docs explicitly prohibit generic OpenPose inputs.
- `Krea2PoseGenerate` requires local Turbo and materialized release files, supports `hf://owner/repo/filename` through the HF cache, checks SHA/tensor/parameter identity, and has no Style-LoRA or historical NFS path. Raw input is optional provenance-path validation; Turbo is the actual inference base.
- No-generation preflight `krea2_pose_control.preflight` validates model paths/release identity, imports the nodes, then extracts/renders a condition.

## Verification this session

PASS:

```bash
uv run python -m unittest comfyui/krea2_pose_control/tests/test_integration.py
uv run python -m compileall -q comfyui/krea2_pose_control
PYTHONPATH=comfyui uv run python -m krea2_pose_control.preflight --help
git diff --check
```

Focused tests cover exact COCO-17 topology, renderer neck/limb behavior, Comfy tensor/PIL range conversion, release SHA rejection, both workflow JSONs, and custom-node import. No full model generation was attempted.

## Files changed this session

- `comfyui/krea2_pose_control/` (nodes, renderer, geometry, vendored runtime, preflight, focused tests)
- `comfyui/workflows/krea2_pose_from_reference.json`
- `comfyui/workflows/krea2_pose_from_condition.json`
- `comfyui/README.md`
- `docs/CODEX_HANDOFF.md`

## Unresolved / next action

No live release artifact, Turbo checkpoint, or reference-image path was available in this sandbox, so release-positive validation, detector download, and actual generation are unrun. On the GH200, after copying the package into ComfyUI, run:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH="$PWD/comfyui" uv run python -m krea2_pose_control.preflight \
  --reference-image /ABS/PATH/reference.jpg \
  --release-artifact /ABS/PATH/krea2-pose-control-mix025.safetensors \
  --turbo-path /ABS/PATH/krea2_turbo.safetensors \
  --raw-path /ABS/PATH/krea2_raw.safetensors \
  --output-condition /tmp/krea2_pose_condition.png
```

Next step: end-to-end reference-image -> condition -> generation smoke test on the GH200 with verified release and Turbo files. Do not modify release/evaluation artifacts or checkpoints; do not commit or push without explicit authorization.
