# Phase 1 handoff

## Current objective

Production-safe W&B telemetry and an independent local JSONL metrics fallback
are implemented and verified. This session deliberately did not touch VAE
preprocessing, latent shards, model downloads, training mechanics,
checkpointing, HF uploads, or production training.

## Decisions in force

- Krea-2 Raw, rendered skeleton control, spatial channel concatenation,
  rank-64 LoRA, BF16, and flow-matching MSE only remain unchanged.
- `TrainingTelemetry` in `pose_controlnet.wandb_logging` is project-owned and
  failure-isolated: local JSONL is attempted independently; W&B import/init,
  network, log, image-log, and finish failures are recorded in memory but never
  raised to training.
- Default W&B target is entity `adhit-projects`, project
  `Krea-2-PoseControl-Lora`. `TrainConfig` can override entity/project/mode,
  and `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE`, and `WANDB_DISABLED`
  override at runtime. Default local log path is `runs/metrics.jsonl`.
- Configuration fields whose names imply credentials (`api_key`, token,
  password, secret) are excluded from W&B config serialization. No API key is
  read, stored, or written by project telemetry.

## Verified environment facts

- Host verification remains: Linux aarch64, GH200, Python 3.10.12, torch
  2.7.0+cu128, CUDA runtime 12.8, cuDNN 9.8, Triton 3.3.0, and `uv 0.12.5`.
- User-confirmed W&B host verification is green: a real remote test run synced
  metrics to `adhit-projects/Krea-2-PoseControl-Lora`.
- This audit shell remains CPU-only; no CUDA conclusion was drawn here.

## Completed gates

- Gate B physical dataset resolution: PASS. Read-only dataset indexing resolves
  17,416 immutable manifest records with expected split counts and rejects
  ambiguity, missing counterparts, split overlap, and empty captions.
- Gate C paired geometry: PASS. RGB/control pair geometry is shared exactly
  across the fixed Krea buckets and is covered by focused tests.
- W&B/local telemetry implementation: PASS. Named future-training interfaces
  cover train loss/LR/grad norm/throughput, validation flow loss, control and
  LoRA diagnostics, CUDA memory, checkpoint metadata, HF upload status and
  remote-checkpoint age, and sparse diagnostic images.

## Exact checks

- `python -m unittest -v tests/test_wandb_logging.py` — PASS (7 tests): normal
  init, init failure, log failure, JSONL output and named interfaces,
  disabled/offline modes, environment overrides, and credential exclusion.
- `python -m unittest -v tests/test_dataset_index.py tests/test_paired_preprocessing.py`
  — PASS (13 tests; regression check run before the final W&B-only test pass).
- `git diff --check` — PASS before this handoff rewrite; rerun after it.
- `git status --short` before this handoff rewrite: user-existing `M .gitignore`;
  session changes `M pose_controlnet/config.py`, `M pose_controlnet/wandb_logging.py`,
  and `?? tests/test_wandb_logging.py`.

## Files changed this session

- `pose_controlnet/wandb_logging.py`
- `pose_controlnet/config.py`
- `tests/test_wandb_logging.py`
- `docs/CODEX_HANDOFF.md`

## Current blockers

None for the telemetry milestone. W&B real-remote verification is confirmed by
the user; this isolated test suite uses fakes and does not require credentials.

## Exact next recommended action

Await a separately bounded implementation assignment. Do not start production
training. When a training loop exists, instantiate `TrainingTelemetry` once,
call its named interfaces at the configured cadences, and close it during
controlled shutdown; do not add a second logging implementation.
