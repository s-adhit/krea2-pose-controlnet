# Project handoff

## Current objective

The standalone ComfyUI package ships frozen Krea-2 Pose Control-LoRA v1. The package-local Qwen VAE boundary was repaired after the real GH200 headless generation smoke reached sampling and raised `ValueError: too many values to unpack (expected 4)` in `patchify_and_position`. Do not run a production training job from this project without explicit approval.

## Frozen facts in force

- Release: `krea2-pose-control-mix025.safetensors`, SHA-256 `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`, 450 tensors / 215,488,512 trainable parameters.
- Release runtime: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0, native/aspect geometry, no Style-LoRA; rank-64 control/LoRA state trained with Raw provenance.
- Frozen PoseBridge renderer is unchanged: DWPose structured Body-18 maps to COCO-17 in `(0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10)` order; renderer alone synthesizes neck from shoulders; no native DWPose raster, hands, feet, or dense-face points reach Krea.
- The validated smoke condition is `docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png` (832×1216).

## Qwen VAE / diffusion shape contract

`AutoencoderKLQwenImage` receives and returns one-frame video tensors. The package-local VAE wrapper is the only code that handles this time axis:

- rendered RGB condition: `1×3×1×H×W` (for the frozen 832×1216 condition, `1×3×1×1216×832`);
- raw and normalized Qwen latent: `B×16×1×H/8×W/8` (`1×16×1×152×104` for that condition);
- after the VAE wrapper removes exactly the singleton temporal axis: normalized Krea diffusion latent `B×16×H/8×W/8` (`1×16×152×104`);
- zero image template and seeded noise use the identical 4-D convention; with patch 2 both patchify to `1×3952×64` tokens;
- sampled latent is reconstructed as `1×16×152×104`; the wrapper restores its temporal axis before `vae.decode` (`1×16×1×152×104`), validates the decoded `1×3×1×1216×832`, then returns `1×3×1216×832` pixels.

The canonical `inference.py` and project VAE preprocessing establish the same Qwen input/decode semantics; cached single-image records there are `C×H×W` and its sampler adds batch once. The package runtime instead retains batch while removing only time, so its sampler must not add a second batch axis.

## Root cause and repair

- Root cause: `comfyui/.../vae.py` correctly received Qwen `B×C×1×H×W` but only executed `squeeze(2)`, producing `B×C×H×W`. `comfyui/.../turbo_runtime.py` then indexed both `sample["latent"][None]` and `sample["control"][None]`, making `1×B×C×H×W`. `patchify_and_position` correctly requires 4-D Krea latents and failed.
- Repair: VAE encode/decode now explicitly validate and convert only Qwen's singleton temporal axis; decode restores it. The package-local Turbo sampler consumes 4-D B×C×H×W latents directly, asserts matching image/control shape and its one-image batch contract, and creates matching 4-D noise. `patchify_and_position` rejects non-4-D input with an actionable VAE-boundary error; it does not accept 5-D tensors. MMDiT geometry, trained/release tensors, schedule, and renderer/detector behavior are unchanged.

## Completed / green checks

- PASS: `PYTHONPATH=./comfyui python -m unittest comfyui/krea2_pose_control/tests/test_integration.py` — 23 tests, including mocked Qwen encode/decode conversion, non-singleton-time rejection, 4-D patchify assertion, and matching control/noise conventions. No Qwen download.
- PASS: `python -m compileall -q comfyui/krea2_pose_control`.
- PASS: `git diff --check`.
- PASS (previous): package dependency loading, Qwen VAE/text loading, and Turbo/Raw release paths on the real GH200 smoke shell; sampling alone exposed the repaired shape defect.
- Pending after this repair: rerun the exact GH200 one-shot generation smoke below. Do not run it from the Codex audit shell unless CUDA is visible there.

## Files changed this session

- `comfyui/krea2_pose_control/krea2_pose_runtime/vae.py`
- `comfyui/krea2_pose_control/krea2_pose_runtime/turbo_runtime.py`
- `comfyui/krea2_pose_control/krea2_pose_runtime/diffusion.py`
- `comfyui/krea2_pose_control/tests/test_integration.py`
- `docs/CODEX_HANDOFF.md`

## Exact GH200 generation-smoke retry command

```bash
cd /home/ubuntu/krea2-pose-controlnet/comfyui
PYTHONPATH=/home/ubuntu/ComfyUI/custom_nodes:/home/ubuntu/krea2-pose-controlnet/comfyui \
/home/ubuntu/ComfyUI/.venv/bin/python -m krea2_pose_control.generation_smoke \
  --pose-condition /home/ubuntu/krea2-pose-controlnet/docs/comfyui/dwpose-smoke/elegant-wandering-knight-condition.png \
  --release-artifact /lambda/nfs/adhit/krea2-pose/release/krea2-pose-control-lora-v1/krea2-pose-control-mix025.safetensors \
  --turbo-path /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors \
  --raw-path /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --output-image /home/ubuntu/krea2-pose-controlnet/docs/comfyui/generation-smoke/elegant-wandering-knight-seed42.png
```
