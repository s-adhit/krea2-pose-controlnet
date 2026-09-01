# Project handoff

## Current objective and state

The canonical local user-facing inference entrypoint is now `inference.py`.
It is a thin composition of the existing Krea-2 Turbo model loader, strict
project ControlInputLayer/LoRA loader, Qwen VAE, online Qwen text conditioner,
and locked Turbo sampler. It does not reimplement model layers,
ControlInputLayer, or sampling math.

The retained inference-evaluation candidates are `parent-4000`,
`finish-control-a4300`, and `finish-anneal-b4200`, exposed as names in
`POSE_CHECKPOINT_CANDIDATES`. No release checkpoint is locked: every CLI run
requires an explicit `--pose-lora-ckpt` (or `--control-ckpt`) file path.

The finishing pose-anneal contract remains unchanged: global update 4001 uses
`lambda_pose=.04`; update 4500 uses literal `lambda_pose=0.0` and writes
`step_004500.pt`. Do not run recovery or training from Codex without explicit
authorization.

## Inference interface

The standard command is:

```bash
PYTHONPATH=. python inference.py \
  --turbo-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors \
  --pose-lora-ckpt /path/to/selected/step_004300.pt \
  --prompt "editorial photograph of a dancer" \
  --pose-image /path/to/skeleton.png \
  --output /path/to/result.png \
  --seed 42 --width 768 --height 768 \
  --steps 8 --cfg 0 --mu 1.15 --control-scale 1.0
```

Required inputs are `--turbo-ckpt`, `--pose-lora-ckpt` (`--control-ckpt`
alias), `--prompt`, `--pose-image`, and `--output`. Additional options are
`--seed`, `--width`, `--height`, `--steps`, `--cfg`, `--mu`, and
`--control-scale`. Defaults are Turbo, 8 steps, CFG 0, mu 1.15, control scale
1.0, seed 42, and 768x768. The established sampler explicitly has no
resolution-dependent shift; noncanonical steps/CFG/mu values fail clearly.

`--dynamic-768-bucket` replaces `--width/--height`; it selects the shared
production `RESOLUTION_768_BUCKETS` policy from pose-image aspect ratio and
uses the exact shared resize-to-cover / center-crop helpers. Explicit
dimensions must be positive multiples of 16. The `<output-stem>.json` sidecar
contains prompt, seed, output dimensions, steps, CFG, mu, control scale, both
checkpoint paths, checkpoint step when available, pose input, geometry mode
and full geometry, locked Turbo metadata, and absolute output path.

Python callers use `PoseInferenceRequest(...)` and `generate_pose(request)`,
which returns `PoseInferenceResult`. Optional `InferenceRuntime` injection
keeps the wrapper suitable for a future ComfyUI or HF demo adapter without
separate model implementations. Style-LoRA composition is intentionally not
implemented.

## This session: files and verification

Changed: `inference.py`, `pose_controlnet/paired_preprocessing.py`,
`pose_controlnet/vae_preprocessing.py`, `tests/test_inference.py`, and this
handoff. The VAE now exposes `encode_preprocessed_image`, which paired
encoding also uses. Paired preprocessing publicly exposes the exact geometry
application helper used by inference.

PASS (CPU, no network):

```bash
PYTHONPATH=. python -m unittest tests.test_inference \
  tests.test_paired_preprocessing tests.test_vae_preprocessing \
  tests.test_turbo_evaluation -v
```

36 tests passed. Coverage includes CLI/defaults, explicit checkpoint
requirement, shared explicit/dynamic geometry, metadata sidecar, deterministic
seed propagation, bad dimensions, missing/malformed files and checkpoint
metadata, callable API generation with mocked heavy execution, and no copied
dynamic bucket list.

PASS:

```bash
python -m py_compile inference.py pose_controlnet/paired_preprocessing.py \
  pose_controlnet/vae_preprocessing.py tests/test_inference.py
```

No training, real evaluation, network activity, commit, or push occurred.
Known limitations: no real generation ran in this session; it requires the
GH200 shell, local Turbo/pose checkpoint files, and locally available VAE/text
weights. No candidate has been selected or released, and style-LoRA
composition remains intentionally unsupported.

Next action: choose one explicit retained pose checkpoint for a separately
authorized local inference smoke, then inspect its image and JSON sidecar.
