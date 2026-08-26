# Phase 1 handoff

## Current objective

Gate E real Krea-2 Raw control-path verification is implemented and locally tested, but the corrected real GH200 diagnostic rerun is **PENDING**. Do not begin Gate F, a production training loop, 10/100-step training, or the 6000-step run.

## Decisions in force

- Base is gated `krea/Krea-2-Raw/raw.safetensors`; rank is exactly 64; targets are the eight main-block linear paths `attn.{wq,wk,wv,wo,gate}` and `mlp.{gate,up,down}` in every one of 28 blocks (224 modules).
- Control is clean, spatially aligned, patchified with the same geometry as the image, and channel-concatenated with noisy image tokens: width 64 + 64 = 128 without changing token count.
- Flow diagnostic uses seed 42, logistic-normal timestep sampling plus the configured resolution shift, `x_t=t*noise+(1-t)*image`, target `noise-image`, and flow-matching MSE only.
- Standard zero-impact LoRA initialization makes A's first gradient exactly zero because B starts at zero. The diagnostic records first-backward B/control gradients, takes exactly one AdamW step, proves real/zero-control output divergence, then performs one additional backward with no second optimizer step to prove both A and B gradients are finite and nonzero.

## Completed/green checks

- User host-verified persistent latent dataset: train 16,503; val 889; diagnostic_val 24; total 17,416; full hard verification PASS.
- Official architecture contract and strict checkpoint preflight are encoded; control input is `(6144,128)` with an exact pretrained image-half copy and exactly zero control-half weights.
- Trainability audit permits only `first.{weight,bias}` and rank-64 LoRA `A/B`; expected static counts are 215,488,512 trainable and 12,819,673,676 frozen.
- One real persistent sample loaded locally from `train-00000.pt`, sample 0 (`coco_100098_193288`, bucket 1216x832); image/control latents are finite, nonzero, aligned, and captioned.
- Targeted unit suite PASS (5 tests): `.venv/bin/python -m unittest -v tests/test_gate_e.py`.
- Syntax PASS: `.venv/bin/python -m py_compile scripts/gate_e_real_diagnostic.py tests/test_gate_e.py`.

## Diagnostic bug found and fixed

- The first real GH200 attempt reported initial real-vs-zero control `max_abs=0.0625`, `rms=0.0084880879`, despite `count_nonzero(control_half)==0`.
- This was a diagnostic methodology bug: it compared a train-mode, autograd, `grad_ckpt=True` real-control prediction retained through `loss.backward()` against an eval-style no-grad, `grad_ckpt=False` zero-control prediction.
- Initial functional comparison now runs before any backward with both forwards in `model.eval()`, one `torch.no_grad()` context, and `grad_ckpt=False`, using identical precomputed image/context/timestep/position/mask inputs. It requires the exact result `{max_abs: 0.0, rms: 0.0}` with no tolerance.
- An isolated `model.first` real-control versus zero-control comparison now records max-absolute and RMS differences and also requires exact zero.
- The gradient-bearing first forward/backward runs only after restoring `model.train()`.
- After the optimizer step, both functional forwards again share eval/no-grad/`grad_ckpt=False` conditions and must have a finite nonzero difference. The script then restores train mode and creates a separate gradient-bearing real-control forward for the LoRA A/B proof.

## Gate E status / blocker

- **Gate E is not yet PASS.** The corrected script must be rerun from the normal GH200 shell with CUDA and the real checkpoint.
- This Codex audit shell does not expose CUDA. Do not infer real loss, gradient norms, output deltas, or peak VRAM locally.

## Files changed this session

- `scripts/gate_e_real_diagnostic.py`
- `tests/test_gate_e.py`
- `docs/CODEX_HANDOFF.md`

## Exact next action

Run exactly one corrected Gate E diagnostic from the normal GH200 shell:

```bash
.venv/bin/python scripts/gate_e_real_diagnostic.py \
  --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --device cuda --seed 42 \
  --output-json /tmp/gate-e-real.json
```

PASS requires JSON `status=PASS`; strict zero incompatible checkpoint keys; isolated `model.first` and full-model initial differences exactly zero; control-half first-backward gradient finite and >0; first-backward representative LoRA B gradient >0; post-step functional difference finite and nonzero; post-step representative LoRA A/B gradients both >0; and no frozen gradients. Copy the exact JSON values into this handoff, mark Gate E PASS, and stop. Gate F remains a separate milestone.
