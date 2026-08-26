# Phase 1 handoff

## Current objective

An explicit, checkpointed extended-training opt-in is implemented for the authorized step-100 to step-500 continuation. The normal Gate-F entry point still rejects values above 100. No training was started in this session.

## Decisions and verified state

- Krea-2 Raw; clean skeleton-control latent channel concat; rank/alpha 64 LoRA; BF16 flow-matching MSE; AdamW `1e-4`, betas `(0.9, 0.99)`, no weight decay; warmup 200; MB2/accum16/effective batch32; GC blocks 6; compile off; cached Qwen conditioning; seed 42.
- `train.py --max-steps` accepts `1..100` by default. Values above 100 require `--allow-extended-training`; the resulting `allow_extended_training` value is included in `TrainConfig`, checkpointed through the existing config payload, and printed at startup. Resume, optimizer/scheduler, data/RNG, telemetry, and checkpoint/mirror behavior are otherwise unchanged.
- Gates A–E and Gate-F mechanics are reported green; real checkpoints 20/40/60/80/100 are at `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100`. Their stochastic validation losses are not a checkpoint-comparison metric.
- No project-owned pose estimator/PCK interface exists. No heavyweight metric dependency was added.

## Implemented evaluation contract

- `evaluate.py fixed-flow` creates/reuses `fixed_flow_spec.json`: default 32 deterministic `val` stems, seed `420100`, per-stem timestep/noise/sampling seeds, and SHA-256 identities of image latent, control latent, cached context, and mask. It calls existing `sample_flow_timestep`, `make_flow_pair`, `forward_pose_control`, and MSE. Changed latent/text inputs fail rather than silently changing the fixed set.
- Baseline step 0 is fresh verified Krea + initial zero-impact control/LoRA state. Steps 20–100 are full-schema validated and load through exact trainable-state validation. JSON results include mean/median/std/per-sample loss.
- `evaluate.py fixed-pose` uses default eight deterministic `diagnostic_val` stems, seed `420200`, cached conditional/unconditional text, existing Euler sampler, 8 steps, CFG 3.5, and Qwen VAE inverse normalization/decode. It writes `fixed_pose/<stem>/control.png`, metadata, steps 0–100, and `comparison_grid.png`.
- `comparison_grid.png` now has labeled `control | step0 | step20 | step40 | step60 | step80 | step100` columns and fixed 320×320 cells. Each image is centered and thumbnail-scaled without stretching; `--comparison-grid-thumbnail-width` and `--comparison-grid-thumbnail-height` can override those display-only defaults. This does not alter stems, inputs, sampling, checkpoints, files, metadata, or inference behavior.

## Files changed this session

- `train.py`, `pose_controlnet/config.py`, `tests/test_train_mechanics.py`
- `docs/CODEX_HANDOFF.md`

## Tests run

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_train_mechanics` (25 tests); `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile train.py pose_controlnet/config.py tests/test_train_mechanics.py`; `git diff --check`.

Coverage: default 100-step acceptance, 101-step rejection without opt-in, authorized 500-step acceptance, accepted resume request from step 100 to 500, plus existing warmup/optimizer, resume state, deterministic data/RNG, checkpoint mirroring, cached-text, and GC mechanics coverage.

## Exact GH200 commands / outputs

From `/home/ubuntu/Krea-2-Pose-ControlNet`:

```bash
uv run python evaluate.py fixed-flow --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-100
```

Writes `fixed_flow_spec.json` and `fixed_flow_results.json` under `/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-100`.

```bash
uv run python evaluate.py fixed-pose --samples 1 --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-100-smoke
```

Writes the six-checkpoint one-sample smoke beneath `.../pose-learning-100-smoke/fixed_pose/<stem>/`.

```bash
uv run python evaluate.py fixed-pose --samples 8 --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-100
```

Writes full results and `fixed_pose/comparison_grid.png` beneath `.../evaluation/pose-learning-100`. `--dataset-root` is optional because the verified source root is read from `posebridge_latents/shards.json`.

## Next action

Run the already authorized continuation from the GH200 host shell, preserving the private HF target used for the step-100 run:

```bash
uv run python train.py --run-name pose-learning-100 --max-steps 500 --allow-extended-training --microbatch-size 2 --gradient-accumulation-steps 16 --gradient-checkpointing-blocks 6 --resume auto --hf-repo-id "${HF_REPO_ID:?set to the existing private checkpoint mirror repo}"
```

This resumes from the newest valid local checkpoint (step 100) first, with the existing HF fallback. Evaluate the resulting checkpoints using the fixed-flow/fixed-pose commands above before any further extension.
