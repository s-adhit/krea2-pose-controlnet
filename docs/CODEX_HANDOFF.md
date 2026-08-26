# Phase 1 handoff

## Current objective

Cached Qwen text conditioning is implemented and unit-tested. Generate and hard-verify the persistent host cache before rerunning Gate-F memory probes. No 10-step, 100-step, or production run has been started.

## Decisions in force

- Base: Krea-2 Raw; skeleton control is clean latent channel concatenation.
- LoRA rank/alpha 64, BF16, flow-matching MSE only, seed 42, frozen backbone.
- AdamW: lr `1e-4`, betas `(0.9, 0.99)`, weight decay `0.0`; warmup 200 optimizer steps; effective batch remains 32.
- Gate-F remains `compile=False`, `gradient_checkpointing=False` by default.
- Canonical latent root: `/lambda/nfs/adhit/krea2-pose/posebridge_latents`.
- Canonical cached text root: `/lambda/nfs/adhit/krea2-pose/text_conditioning`.

## Completed gates/findings

- Gates A–E remain PASS (environment, manifests/resolution, paired VAE preprocessing, latent archives, real control-path proof).
- Exact online Qwen contract: `PoseTextConditioner` combines selected Qwen hidden states `(2,5,...,35)` into `[batch, tokens, 12, hidden]`, with its matching post-prefix boolean attention mask. This unchanged pair enters Krea `txtfusion`; no pooled/reduced representation is used.
- New text cache archives use BF16 contexts and bool masks, atomic resumable 64-sample shards, immutable stem ordering, and a separate unconditional empty-caption entry. Preparation validates latent `stem` and caption identity against immutable manifests first.
- Cached training is the default. It does not instantiate Qwen and logs `text_conditioning=cached text_encoder_loaded=False`. Seeded 10% caption dropout selects cached unconditional conditioning at training time. Use `--online-text-conditioning` only for diagnostics.

## Files changed this session

- `pose_controlnet/text_conditioning.py` (new archive preparation, validation, cache lookup, online-vs-cached equivalence smoke)
- `prepare_text_conditioning.py` (new host preparation command)
- `scripts/verify_text_conditioning.py` (new hard verification command)
- `pose_controlnet/data.py` (stem-aware cached context collation)
- `train.py` (cached hot path and seeded cached dropout)
- `tests/test_text_conditioning.py` (new)

## Tests run

PASS:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile \
  train.py pose_controlnet/data.py pose_controlnet/text_conditioning.py \
  prepare_text_conditioning.py scripts/verify_text_conditioning.py \
  tests/test_text_conditioning.py
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest \
  tests.test_train_mechanics tests.test_text_conditioning
git diff --check
```

The 15 focused tests passed. No host Qwen generation, VAE/latent changes, training, dependency changes, commit, or push occurred.

## Exact next action

From the GH200 host, generate the cache. `prepare_text_conditioning.py` derives the immutable dataset root from verified latent-shard metadata:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python prepare_text_conditioning.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --output-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --device cuda --shard-samples 64
```

Then run `scripts/verify_text_conditioning.py` with `--online-equivalence` and the dataset root recorded in latent `shards.json`, before cached MB1/MB2 one-step benchmarks using `--no-compile --no-gradient-checkpointing`.
