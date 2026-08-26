# Phase 1 handoff

## Current objective

Gate E real Krea-2 Raw control-path verification is implemented and locally tested, but the real GH200 checkpoint forward/backward remains **PENDING**. Do not begin Gate F, a production training loop, 10/100-step training, or the 6000-step run.

## Decisions in force

- Base is gated `krea/Krea-2-Raw/raw.safetensors`; rank is exactly 64; targets are the eight main-block linear paths `attn.{wq,wk,wv,wo,gate}` and `mlp.{gate,up,down}` in every one of 28 blocks (224 modules).
- Control is clean, spatially aligned, patchified with the same geometry as the image, and channel-concatenated with noisy image tokens: width 64 + 64 = 128 without changing token count.
- Flow diagnostic uses seed 42, logistic-normal timestep sampling plus the configured resolution shift, `x_t=t*noise+(1-t)*image`, target `noise-image`, and flow-matching MSE only.
- Standard zero-impact LoRA initialization makes A's first gradient exactly zero because B starts at zero. The diagnostic records first-backward B/control gradients, takes exactly one AdamW step, proves real/zero-control output divergence, then performs one additional backward with no second optimizer step to prove both A and B gradients are finite and nonzero.

## Completed/green gates and checks

- User host-verified persistent latent dataset: train 16,503; val 889; diagnostic_val 24; total 17,416; full hard verification PASS.
- Official architecture contract encoded and locally checked: hidden 6144; 28 blocks; 48 query heads; 12 KV heads; head dimension 128; MLP width 16,384; 12 input text layers; two layerwise plus two refiner text blocks with 20 heads/20 KV heads.
- Strict checkpoint preflight compares every safetensors key and shape against a meta Krea model; build then uses `load_state_dict(..., strict=True, assign=True)` and rejects any missing/unexpected key.
- ControlInput assertions cover `(6144,128)` expanded weight, exact raw image-half copy, zero control half, unchanged tokens, and output hidden width 6144.
- Trainability audit permits only `first.{weight,bias}` and LoRA `A/B`; expected static counts are **215,488,512 trainable** and **12,819,673,676 frozen**.
- One real persistent sample loaded locally from `train-00000.pt`, sample 0 (`coco_100098_193288`, bucket 1216x832): image/control shapes `(16,104,152)`, image RMS `0.5127766728`, std `0.5112274885`, control RMS `0.7648329139`, std `0.7523605824`; finite, nonzero, caption present.
- Project venv import blocker fixed with project-owned `networkx>=3.1` via uv; no PyTorch/CUDA/cuDNN/Triton/NVIDIA package changed.
- Targeted unit suite PASS (5 tests): `.venv/bin/python -m unittest -v tests/test_gate_e.py`.
- Syntax PASS: `.venv/bin/python -m py_compile base_model/k2_lora.py pose_controlnet/data.py pose_controlnet/diffusion.py pose_controlnet/model.py scripts/gate_e_real_diagnostic.py`.
- Patch hygiene PASS: `git diff --check`.

## Gate E status / blocker

- **Gate E is not yet PASS.** This audit shell reports `torch.cuda.is_available() == False`; the real diagnostic correctly exits before model loading.
- No local `raw.safetensors` exists under `/lambda/nfs/adhit/krea2-pose`; sandbox Hugging Face CLI access failed with DNS resolution unavailable. The official artifact is about 26.3 GB and gated.
- Therefore real loss, ControlInput weight/gradient norms, LoRA gradient norms, real/zero output deltas, runtime trainable/frozen counts, and peak VRAM are not yet available. Do not invent or infer these values; `scripts/gate_e_real_diagnostic.py` emits all of them as JSON after a successful host run.

## Files changed this session

- `base_model/k2_lora.py`
- `pose_controlnet/data.py`
- `pose_controlnet/diffusion.py`
- `pose_controlnet/model.py`
- `scripts/gate_e_real_diagnostic.py`
- `tests/test_gate_e.py`
- `pyproject.toml`
- `uv.lock`
- `docs/CODEX_HANDOFF.md`

## Exact next action

Run from the normal GH200 shell. If the artifact is not already present, first download the one official monolithic raw checkpoint (requires accepted Krea license/HF access):

```bash
mkdir -p /lambda/nfs/adhit/krea2-pose/models/krea-2-raw
hf download krea/Krea-2-Raw raw.safetensors \
  --local-dir /lambda/nfs/adhit/krea2-pose/models/krea-2-raw
```

Then execute exactly one bounded Gate E diagnostic:

```bash
.venv/bin/python scripts/gate_e_real_diagnostic.py \
  --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --device cuda --seed 42 \
  --output-json /tmp/gate-e-real.json
```

PASS requires JSON `status=PASS`, strict zero incompatible keys, control-half first-backward gradient finite and >0, first-backward representative LoRA B gradient >0, post-step representative LoRA A/B gradients both >0, no frozen gradients, and finite nonzero post-step real-vs-zero-control output difference. Copy the exact JSON values into this handoff, mark Gate E PASS, and stop; Gate F remains a separate milestone.
