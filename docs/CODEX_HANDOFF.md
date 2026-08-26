# Phase 1 handoff

## Current objective

Gate E is host-verified **PASS**. Gate F production training mechanics are
implemented and unit-tested, but the real GH200 10-optimizer-step smoke has
**not** been run. The user must run that bounded smoke next; do not begin a
100-step or 6000-step run in this session.

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

## Gates complete

- Gate A environment: PASS.
- Gate B dataset resolution/manifests: PASS (16,503 train / 889 val / 24 diagnostic).
- Gate C paired preprocessing + Qwen VAE: PASS.
- Gate D persistent latent shards: PASS.
- Gate E real Krea control-path diagnostic: PASS. It strictly loaded the
  checkpoint, verified rank-64/224 targets/frozen backbone, nonzero control
  gradient, and post-step real-vs-zero-control divergence.

## Gate F implementation

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
- `pose_controlnet/data.py`
- `pose_controlnet/checkpointing.py`
- `tests/test_train_mechanics.py`

## Tests run

PASS:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest \
  tests.test_train_mechanics tests.test_wandb_logging tests.test_gate_e
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile \
  train.py pose_controlnet/config.py pose_controlnet/data.py \
  pose_controlnet/checkpointing.py
git diff --check
```

No real training, VAE preprocessing, shard regeneration, production launch,
commit, or push occurred.

## Exact next action: user-run Gate-F 10-step smoke

Run from the normal GH200 host shell:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python train.py \
  --run-name gate-f-smoke-10 \
  --max-steps 10 \
  --microbatch-size 1 \
  --gradient-accumulation-steps 32 \
  --max-grad-norm 1.0 \
  --validation-batches 1 \
  --val-every 10 \
  --save-every 10 \
  --diagnostics-every 10
```

Expected effective batch: `1 × 32 × 1 = 32`.

Inspect: all ten finite train losses; finite nonzero global gradients and
control/LoRA diagnostics; LR at step 10 equals `1e-4 * 10 / 200 = 5e-6`;
finite validation flow loss; no OOM/NaN; JSONL and W&B telemetry; and a valid
full checkpoint at
`/lambda/nfs/adhit/krea2-pose/checkpoints/gate-f-smoke-10/step_000010.pt`.
