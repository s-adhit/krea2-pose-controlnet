# Phase 1 handoff

## Current objective

The Gate-F telemetry timing bug is fixed: sparse control/LoRA gradient diagnostics are captured after accumulated backward work and clipping, immediately before the optimizer update clears gradients. No training was run this session.

## Decisions in force

- Base: Krea-2 Raw; skeleton control is clean latent channel concatenation.
- LoRA rank/alpha 64, BF16, flow-matching MSE only, seed 42, frozen backbone.
- AdamW: lr `1e-4`, betas `(0.9, 0.99)`, weight decay `0.0`; warmup 200 optimizer steps; effective batch remains 32.
- Gate-F remains `compile=False`, with zero checkpointed main blocks by default.
- Global gradient clipping, AdamW settings, scheduler behavior, accumulation, and `zero_grad(set_to_none=True)` semantics remain unchanged.
- Canonical latent root: `/lambda/nfs/adhit/krea2-pose/posebridge_latents`.
- Canonical cached text root: `/lambda/nfs/adhit/krea2-pose/text_conditioning`.

## Completed gates/findings

- Gates A–E remain PASS (environment, manifests/resolution, paired VAE preprocessing, latent archives, real control-path proof).
- Root cause confirmed: the old mixed-length `PoseTextConditioner` appended suffix tokens after the batch-padded prompt tensor. A short caption had `prompt-valid, internal-padding, suffix-valid`; v1 used `mask.sum()` plus a prefix slice, persisting padding and dropping suffix tokens.
- `PoseTextConditioner` now encodes captions independently, preserving tokenizer/template, `PREFIX_IDX=34`, selected layers `(2,5,...,35)`, BF16 hidden states, and Krea txtfusion semantics. It restores only trailing batch padding.
- `compact_valid_conditioning` is the canonical boolean extraction for normal and unconditional entries. Cached masks are contiguous all-true sequences; training collate restores trailing padding only.
- Text cache `FORMAT_VERSION=2`. Metadata, shard payloads, and `unconditional.pt` must all declare v2. v1 contents cannot be reused. Preparation remains atomic and metadata is incomplete until validation passes.
- Normal cached entries now persist `stem`, `context`, and `mask`. The writer adds `record.stem` before `_validate_entry(..., expected_stem=record.stem)` and its atomic shard save, eliminating the immediate `Invalid stem` failure.
- The verifier accepts `--stem coco_100098_193288` and derives `dataset_root` from latent metadata when omitted. It requires exact BF16 tensor, mask, dtype, and shape equality with max absolute difference `0.0`.
- Main-block checkpointing is now selective: `--gradient-checkpointing-blocks N` accepts 0–28 and checkpoints the first N entries of `model.blocks` in execution order. N=0 checkpoints none. Legacy `--gradient-checkpointing` remains an all-28 shorthand; the count option overrides it.
- Only expensive Krea `model.blocks` are checkpointed. Cached text conditioning, frozen Qwen/text fusion, VAE, control construction, and helpers remain outside checkpointing.
- The prior real 10-step smoke showed `diagnostics/control_input_grad_norm/{full,control_half}=0` because it sampled after `optimizer_update()` had called `zero_grad`. Diagnostics now use an optional pre-step callback inside that same optimizer boundary, after clipping and before `optimizer.step()`.
- The regression test injects synthetic ControlInput and LoRA gradients, captures nonzero diagnostics via the pre-step callback, then proves the post-update gradients are cleared.

## Files changed this session

- `train.py`
- `tests/test_train_mechanics.py`
- `docs/CODEX_HANDOFF.md`

## Tests run

PASS:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile \
  train.py tests/test_train_mechanics.py
git diff --check
```

Focused regressions cover the pre-step gradient capture plus existing configuration, checkpointing, and flow-loss mechanics. No real training was run.

## Exact next action

From the GH200 host, repeat the real 10-step Gate-F smoke only when authorized, and confirm sparse W&B metrics now show nonzero control-half/full and representative LoRA gradient norms on the diagnostic cadence. Then continue the existing MB2/accum16 profiling plan, beginning with four checkpointed main blocks:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --run-name mb2-gc4 --max-steps 1 --microbatch-size 2 \
  --gradient-accumulation-steps 16 --no-compile \
  --gradient-checkpointing-blocks 4
```

Repeat with `--gradient-checkpointing-blocks 8`, `12`, and `16` (using a distinct `--run-name`) and record peak allocated/reserved memory plus seconds per step. Do not begin a longer run.
