# Phase 1 handoff

## Current objective

Gate C2 — real Qwen/Krea VAE integration and paired RGB/control latent smoke
verification. The project-owned VAE encoding path and focused unit tests are
complete. The required three-sample real-data/GH200 smoke check remains
blocked by unavailable artifacts in this audit workspace.

## Decisions in force

- Krea-2 Raw uses `diffusers.AutoencoderKLQwenImage` from
  `Qwen/Qwen-Image`, subfolder `vae`; no transformer/base-model artifact is
  downloaded by the project-owned loader.
- Installed project environment verification: `diffusers 0.40.0` exposes
  `AutoencoderKLQwenImage`.
- Input is RGB normalized from `[0, 255]` to `[-1, 1]` in BF16 on the VAE
  device, with Qwen's one-frame video layout `B×3×1×H×W`.
- The encoder uses `latent_dist.sample()`, matching Diffusers' Qwen-Image
  ControlNet path. Normalize the raw `B×16×1×H/8×W/8` latent exactly as
  `(z - vae.config.latents_mean) / vae.config.latents_std` per channel, then
  remove batch/time axes to the downstream `16×H/8×W/8` layout. Shard
  serialization, when separately implemented, must explicitly store float32;
  this encoding path retains its BF16 compute dtype.
- RGB/control resolution and geometry remain exclusively in `DatasetIndex` and
  `preprocess_pair`; no path-resolution or crop logic was duplicated.

## Verified environment facts

- Host verification remains: Linux aarch64, GH200, Python 3.10.12, torch
  2.7.0+cu128, CUDA runtime 12.8, cuDNN 9.8, Triton 3.3.0, and `uv 0.12.5`.
- This Codex audit shell is CPU-only. It has the project `.venv` and
  `diffusers 0.40.0`, but not `data/full/` and cannot resolve the Hugging Face
  Hub DNS endpoint. No CUDA conclusion was drawn from this shell.

## Completed/green checks

- Gate B physical dataset resolution: PASS (previously verified).
- Gate C paired geometry: PASS (previously verified).
- VAE helper unit tests: PASS — PIL range/layout, per-channel normalization,
  invalid/nonfinite rejection, paired encode layout/shape/nonzero signal, and
  diagnostic reporting.
- Regression paired-preprocessing tests: PASS.

## Gate C2 real-data smoke

- Status: BLOCKED in the audit workspace, not passed.
- Required dataset root `data/full/` is absent (no `images/` or
  `conditioning_images/` directories), so `DatasetIndex.discover` correctly
  rejects the attempted smoke command before it can select a square, portrait,
  and landscape record.
- Exact VAE-only Hub download was attempted through `hf` but could not begin
  because this sandbox has a temporary DNS failure. No model weights or
  unrelated components were downloaded.
- Therefore there are no real sample latent statistics yet. Do not treat unit
  test output or review images as the Gate C2 smoke result.

## Exact checks this session

- `.venv/bin/python -m unittest -v tests/test_vae_preprocessing.py tests/test_paired_preprocessing.py` — PASS (12 tests).
- `.venv/bin/python -m py_compile pose_controlnet/vae_preprocessing.py` — PASS.
- `.venv/bin/python` Diffusers class check — PASS: version `0.40.0`, required
  class available.
- `hf models info Qwen/Qwen-Image ...` and VAE-only dry run — blocked by DNS.
- `.venv/bin/python -m pose_controlnet.vae_preprocessing --dataset-root data/full --device cpu --scan-limit 16` — expected BLOCKED: missing dataset root.

## Files changed this session

- `pose_controlnet/vae_preprocessing.py`
- `tests/test_vae_preprocessing.py`
- `docs/CODEX_HANDOFF.md`

## Current blockers

- Run the final smoke on the GH200 production shell/service environment where
  the immutable PoseBridge snapshot is mounted and Hugging Face access is
  available. This audit workspace cannot access those artifacts.

## Exact next action

On the GH200 host, after confirming the mounted snapshot root, run:

```bash
.venv/bin/python -m pose_controlnet.vae_preprocessing \
  --dataset-root <mounted-posebridge-snapshot> --device cuda --scan-limit 256
```

Confirm one square, portrait, and landscape report; matching finite RGB/control
latent shapes; BF16 encoding; and nonzero control latent statistics. Record
the reported values in this handoff. Do not create full shards or begin Gate D.
