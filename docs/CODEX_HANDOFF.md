# Phase 1 handoff

## Current bounded objective and status

The interrupted timestep-exposure continuation is ready for a **validated recovery only**. No training/resume, checkpoint write, HF upload, W&B action, commit, or push occurred in this session. Do not use the original `--timestep-lowmid-1500-to1800` selector: it is intentionally locked to the immutable step-1500 source and would restart the experiment from 1500.

The new narrow selector is:

```bash
uv run python train.py --recover-timestep-lowmid-1500-to1800
```

It selects only the newest fully validated local continuation checkpoint from the fixed namespace below; it does not accept `--resume`, an arbitrary path, or mutable timestep flags.

## Recovery checkpoint audit

Immutable source remains:

`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`

Audited local run:

`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-1500-timestep-lowmid20-to1800`

Both checkpoints fully deserialized through `load_training_state` and passed the recovery semantic validator, including schema, model structure/finiteness, finite AdamW moments, optimizer identities, scheduler, config compatibility, CPU/CUDA RNG, and flow-generator state:

| checkpoint | global step | epoch | batch position | scheduler / warmup |
|---|---:|---:|---:|---:|
| `step_001525.pt` | 1525 | 2 | 7902 | 1525 / 200 |
| `step_001526.pt` | 1526 | 2 | 7918 | 1526 / 200 |

Both have one AdamW group, 450 complete parameter moment entries, LR/base LR `5e-5`, betas `(0.9, 0.99)`, weight decay `0.0`, one CUDA RNG state, and a 16-byte CUDA flow-generator state. Both exactly match the fixed continuation configuration: run/max steps `pose-learning-1500-timestep-lowmid20-to1800` / `1800`, seed 42, BF16 path, original buckets/model/data/dropout settings, and auxiliary pre-shift sampler `prob=0.20`, support `[0.04359494981207863, 0.3773562340267345)`.

Selected checkpoint:

`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-1500-timestep-lowmid20-to1800/step_001526.pt`

The selector keeps the same local directory, W&B run name/config semantics, and HF target:

`adhit-420/Krea-2-PoseControl-LoRA-checkpoints/pose-learning-1500-timestep-lowmid20-to1800/full/`

The preserved `save_every=25` and `hf_mirror_every_steps=100` produce and mirror checkpoints 1600, 1700, and 1800 normally.

## Files changed and validation

- `train.py`: added the fixed `--recover-timestep-lowmid-1500-to1800` selector and a semantic validator for only this interrupted continuation.
- `tests/test_train_mechanics.py`: focused acceptance, rejection, and newest-valid-local-selection coverage.
- `docs/CODEX_HANDOFF.md`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile train.py tests/test_train_mechanics.py`

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics` — 42 tests.

PASS: direct semantic validation of both real checkpoints; selected step 1526.

PASS: `git diff --check`.

## Exact next action

Only with explicit authorization to resume training on the GH200 host:

```bash
cd /home/ubuntu/Krea-2-Pose-ControlNet
tmux new-session -d -s pose-learning-1500-timestep-lowmid20-to1800 \
  'cd /home/ubuntu/Krea-2-Pose-ControlNet && export UV_CACHE_DIR=/tmp/krea_uv_cache && exec uv run python train.py --recover-timestep-lowmid-1500-to1800'
```

Immediately verify:

```bash
tmux capture-pane -pt pose-learning-1500-timestep-lowmid20-to1800 -S -200 \
  | rg '\[timestep-recovery\]|effective_batch|runtime'
```

Expected recovery fields: selected `step_001526.pt`, global/scheduler step 1526, warmup 200, LR `5e-5`, preserved AdamW/RNG/data state, auxiliary probability/support, `(1600, 1700, 1800)` checkpoint cadence, the existing HF namespace, and the existing W&B run name. This continues the same experiment; it does not restart from 1500.
