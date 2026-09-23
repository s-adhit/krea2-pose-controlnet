# Project handoff

## Current objective

The standalone ComfyUI package supports frozen Krea-2 Pose Control-LoRA v1. DWPose is the default reference extractor; Keypoint R-CNN remains its selectable fallback. The current bounded work added package-local standalone dependency discovery and a one-shot frozen generation smoke entry point. No generation was run from this audit shell because CUDA is not visible there.

## Frozen facts in force

- Release: `krea2-pose-control-mix025.safetensors`, SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`, 450 tensors / 215,488,512 trainable parameters.
- Release runtime: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0, native/aspect geometry, no Style-LoRA; rank-64 control/LoRA state trained with Raw provenance.
- Frozen PoseBridge renderer is unchanged: DWPose structured Body-18 maps to COCO-17 in `(0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10)` order; renderer alone synthesizes neck from shoulders; no native DWPose raster, hands, feet, or dense-face points reach Krea.
- The validated smoke condition is `docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png` (832×1216).

## Standalone DWPose behavior

- `controlnet_aux.py` first accepts an already-importable `custom_controlnet_aux.dwpose`; otherwise it checks only the custom node's sibling `comfyui_controlnet_aux/src` and explicit `custom_nodes` roots already on `sys.path`.
- It adds just that discovered `src` path to the current process when needed. It has no hard-coded home path, no original-training-repository runtime dependency, and emits an actionable error if ControlNet-Aux is absent or incomplete.
- No manual `comfyui_controlnet_aux/src` `PYTHONPATH` entry is required. Real import verification passed with ComfyUI’s venv and only `PYTHONPATH=/home/ubuntu/ComfyUI/custom_nodes:/home/ubuntu/krea2-pose-controlnet/comfyui`.
- Cached DWPose ONNX files are present at `/home/ubuntu/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/`.

## Completed / green checks

- PASS: `PYTHONPATH=./comfyui python -m unittest comfyui/krea2_pose_control/tests/test_integration.py` — 19 tests. Includes sibling-source discovery, already-importable dependency, missing-dependency error, existing DWPose normalization/renderer behavior, and pinned headless smoke invocation.
- PASS: `python -m compileall -q comfyui/krea2_pose_control`.
- PASS: `python -m krea2_pose_control.preflight --help` under the documented two-root `PYTHONPATH`.
- PASS: `python -m krea2_pose_control.generation_smoke --help` under the same environment.
- PASS: real ComfyUI-vendored DWPose import via automatic sibling discovery, using `/home/ubuntu/ComfyUI/.venv/bin/python`.
- PASS: release artifact SHA-256 matches the frozen release identity.
- PASS: `git diff --check`.

## Exact supported smoke command

From `/home/ubuntu/krea2-pose-controlnet/comfyui`, run the `generation_smoke` command in `comfyui/README.md`. It uses only the frozen release, Turbo, and Raw provenance artifact paths; condition `docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png`; default prompt; seed 42; and writes exactly one image to `docs/comfyui/generation-smoke/elegant-wandering-knight-seed42.png`.

## Current blocker / next action

This Codex audit shell reports `torch.cuda.is_available() == False` (including the visible ComfyUI venv), so it must not execute the actual GPU generation. On the real GH200 production shell, run the exact README extraction verification command first, then the exact one-shot generation command. The command changes only the specified output image; it does not alter model/release artifacts. Inspect the result and record its path/checksum. Do not change renderer, release identity, training/evaluation artifacts, or benchmark values.

## Files changed this session

- `comfyui/krea2_pose_control/controlnet_aux.py` (new discovery helper)
- `comfyui/krea2_pose_control/nodes.py`
- `comfyui/krea2_pose_control/generation_smoke.py` (new headless smoke entry point)
- `comfyui/krea2_pose_control/tests/test_integration.py`
- `comfyui/README.md`
- `docs/CODEX_HANDOFF.md`
