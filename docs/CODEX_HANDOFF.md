# Phase 1 handoff

## Current objective

Repair cached-vs-online Qwen text-conditioning equivalence, then regenerate and hard-verify the host text cache. No 10-step, 100-step, or production run has been started.

## Decisions in force

- Base: Krea-2 Raw; skeleton control is clean latent channel concatenation.
- LoRA rank/alpha 64, BF16, flow-matching MSE only, seed 42, frozen backbone.
- AdamW: lr `1e-4`, betas `(0.9, 0.99)`, weight decay `0.0`; warmup 200 optimizer steps; effective batch remains 32.
- Gate-F remains `compile=False`, `gradient_checkpointing=False` by default.
- Canonical latent root: `/lambda/nfs/adhit/krea2-pose/posebridge_latents`.
- Canonical cached text root: `/lambda/nfs/adhit/krea2-pose/text_conditioning`.

## Completed gates/findings

- Gates A–E remain PASS (environment, manifests/resolution, paired VAE preprocessing, latent archives, real control-path proof).
- Root cause confirmed: the old mixed-length `PoseTextConditioner` appended suffix tokens after the batch-padded prompt tensor. A short caption had `prompt-valid, internal-padding, suffix-valid`; v1 used `mask.sum()` plus a prefix slice, persisting padding and dropping suffix tokens.
- `PoseTextConditioner` now encodes captions independently, preserving tokenizer/template, `PREFIX_IDX=34`, selected layers `(2,5,...,35)`, BF16 hidden states, and Krea txtfusion semantics. It restores only trailing batch padding.
- `compact_valid_conditioning` is the canonical boolean extraction for normal and unconditional entries. Cached masks are contiguous all-true sequences; training collate restores trailing padding only.
- Text cache `FORMAT_VERSION=2`. Metadata, shard payloads, and `unconditional.pt` must all declare v2. v1 contents cannot be reused. Preparation remains atomic and metadata is incomplete until validation passes.
- The verifier accepts `--stem coco_100098_193288` and derives `dataset_root` from latent metadata when omitted. It requires exact BF16 tensor, mask, dtype, and shape equality with max absolute difference `0.0`.

## Files changed this session

- `pose_controlnet/text_encoder.py`
- `pose_controlnet/text_conditioning.py`
- `scripts/verify_text_conditioning.py`
- `tests/test_text_conditioning.py`

## Tests run

PASS:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_text_conditioning
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile \
  pose_controlnet/text_encoder.py pose_controlnet/text_conditioning.py \
  scripts/verify_text_conditioning.py tests/test_text_conditioning.py
git diff --check
```

Seven focused regressions cover mixed short/long prompts, internal padding before suffixes, suffix preservation, independent-online equality, unconditional extraction, dynamic right-padding, and v1 metadata rejection. The Codex shell did not run Qwen or mutate host cache artifacts.

## Exact next action

From the GH200 host, regenerate the rejected v1 cache in place; it replaces archives atomically and does not delete the root:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python prepare_text_conditioning.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --output-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --device cuda --shard-samples 64
```

Then hard-verify representative captions and the original failing stem:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python scripts/verify_text_conditioning.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --output-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --online-equivalence --device cuda --samples-per-split 3 \
  --stem coco_100098_193288
```
