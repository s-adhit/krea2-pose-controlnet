# Phase 1 handoff

## Current objective

Gate E is host-verified **PASS**. The first real Gate-F GH200 smoke failed
before optimizer step 1 with `torch._dynamo.exc.FailOnRecompileLimitHit`.
The bounded runtime-controls fix is complete and unit-tested. Run the 1-step
no-compile/no-gradient-checkpointing correctness smoke next; do not begin a
10-step, 100-step, or 6000-step run first.

## Decisions in force

- Base: Krea-2 Raw, canonical checkpoint:
  `/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors`.
- Data: verified persistent latent archives only:
  `/lambda/nfs/adhit/krea2-pose/posebridge_latents`.
- Control: clean skeleton latent, spatial channel concatenation; no RGB
  reference control.
- LoRA: rank/alpha 64 on the exact 224 established target modules.
- BF16, flow-matching MSE only, seed 42, caption dropout 0.10, frozen backbone.
- AdamW only: lr `1e-4`, betas `(0.9, 0.99)`, weight decay `0.0`; warmup is
  200 **optimizer** steps; target effective batch is 32.
- GH200 host: ARM64, torch 2.7.0/CUDA 12.8, BF16/SDPA/compile verified.
  Do not replace the host accelerator stack.
- Gate-F defaults: `compile=False`, `gradient_checkpointing=False`. Both have
  explicit positive/negative CLI switches and are checkpointed as run config.

## Gates complete

- Gate A environment: PASS.
- Gate B dataset resolution/manifests: PASS (16,503 train / 889 val / 24 diagnostic).
- Gate C paired preprocessing + Qwen VAE: PASS.
- Gate D persistent latent shards: PASS.
- Gate E real Krea control-path diagnostic: PASS. It strictly loaded the
  checkpoint, verified rank-64/224 targets/frozen backbone, nonzero control
  gradient, and post-step real-vs-zero-control divergence.

## Gate F implementation

- The Dynamo failure came specifically from the former unconditional
  `@torch.compile(fullgraph=True)` on `RMSNorm.forward` at
  `base_model/mmdit.py:154`. `RMSNorm` is shared by rank-3 text/MLP
  activations and rank-4 attention Q/K activations, so the full-graph compiled
  primitive had an invalid shared specialization contract. No Dynamo
  recompile/cache limit was raised.
- Unconditional decorators were removed from RMSNorm, positional encoding,
  and the final layer, making correctness independent of compilation.
- `--compile` is opt-in and only compiles `model.txtmlp.forward` with
  `dynamic=True`; this module boundary has a stable rank-3
  `[batch, text_length, text_features]` contract and does not use fullgraph.
- `--gradient-checkpointing` is opt-in. The training loop now passes its value
  to `forward_pose_control`; it no longer hard-codes `grad_ckpt=True`.
- Startup logs `runtime: compile=... gradient_checkpointing=...` before work.

- `train.py` is now the bounded Gate-F entry point. It requires explicit
  `--max-steps`, rejects values outside 1..100, and therefore cannot inherit
  the config's 6000-step default.
- It reads only the verified `.pt` latent archives. `PreparedLatentShardDataset`
  builds a compact read-only shard/sample/bucket index and lazily caches one
  archive; it never invokes VAE or image preprocessing.
- Bucket-homogeneous deterministic batches are reconstructed from `(seed,
  epoch, batch_position)`. This position and all RNG state, including the
  dedicated flow-noise generator, are checkpointed/restored.
- Effective batch is explicitly `microbatch * accumulation * world_size`.
  AdamW contains exactly the model's intended trainable tensors; runtime audit
  rejects frozen or unexpected optimizer tensors.
- Loss construction uses intended shifted logistic-normal timesteps, noises
  image latents only, retains clean controls, and uses flow MSE only.
- Accumulation scales loss, updates/clips/schedules only at optimizer
  boundaries, and logs the resulting pre-clip global norm. Warmup advances
  only on optimizer updates.
- Validation is inference-only, bounded by `--validation-batches`, restores
  training mode, and cannot update the optimizer.
- One existing `TrainingTelemetry` instance provides nonfatal W&B plus
  independent JSONL metrics. Training logs loss, LR, global norm,
  timing/throughput, CUDA memory, validation loss, and sparse control/LoRA
  diagnostics.
- Full `.pt` checkpoints are written atomically, deserialize-validated, and
  include trainable model, optimizer, scheduler, global step, data position,
  config, Python/NumPy/Torch/CUDA RNG, and flow-generator state. `--resume`
  restores that state. SIGINT/SIGTERM requests controlled stop/checkpoint at
  the current optimizer boundary.

## Files changed this session

- `train.py`
- `pose_controlnet/config.py`
- `base_model/mmdit.py`
- `tests/test_train_mechanics.py`
- `docs/CODEX_HANDOFF.md`

## Tests run

PASS:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile \
  train.py pose_controlnet/config.py pose_controlnet/diffusion.py \
  base_model/mmdit.py tests/test_train_mechanics.py
git diff --check
```

The focused 13-test suite proves: bounded-smoke defaults keep both switches
off; positive/negative CLI flags propagate; no-compile runtime does not invoke
`torch.compile`; and `_flow_loss` passes disabled gradient checkpointing to
`forward_pose_control`.

No real training, VAE preprocessing, shard regeneration, production launch,
commit, or push occurred.

## Exact next action: user-run 1-step Gate-F correctness smoke

Run from the normal GH200 host shell:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py \
  --run-name gate-f-correctness-1 \
  --max-steps 1 \
  --microbatch-size 1 \
  --gradient-accumulation-steps 32 \
  --no-compile \
  --no-gradient-checkpointing
```

Expected effective batch: `1 × 32 × 1 = 32`.

Confirm startup prints `runtime: compile=False gradient_checkpointing=False`,
then inspect finite loss/gradients, metrics, and its checkpoint. Use the same
two runtime flags for the one-step effective-batch-32 memory probes:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py --run-name gate-f-mb2-1 \
  --max-steps 1 --microbatch-size 2 --gradient-accumulation-steps 16 \
  --no-compile --no-gradient-checkpointing
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py --run-name gate-f-mb4-1 \
  --max-steps 1 --microbatch-size 4 --gradient-accumulation-steps 8 \
  --no-compile --no-gradient-checkpointing
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py --run-name gate-f-mb8-1 \
  --max-steps 1 --microbatch-size 8 --gradient-accumulation-steps 4 \
  --no-compile --no-gradient-checkpointing
```
