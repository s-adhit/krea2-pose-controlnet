# Phase 1 handoff

## Current objective

Post-100-step deterministic evaluation / checkpoint-comparison gate is implemented. No training was started in this session.

## Decisions and verified state

- Krea-2 Raw; clean skeleton-control latent channel concat; rank/alpha 64 LoRA; BF16 flow-matching MSE; AdamW `1e-4`, betas `(0.9, 0.99)`, no weight decay; warmup 200; MB2/accum16/effective batch32; GC blocks 6; compile off; cached Qwen conditioning; seed 42.
- Gates A–E and Gate-F mechanics are reported green; real checkpoints 20/40/60/80/100 are at `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100`. Their stochastic validation losses are not a checkpoint-comparison metric.
- No project-owned pose estimator/PCK interface exists. No heavyweight metric dependency was added.

## Implemented evaluation contract

- `evaluate.py fixed-flow` creates/reuses `fixed_flow_spec.json`: default 32 deterministic `val` stems, seed `420100`, per-stem timestep/noise/sampling seeds, and SHA-256 identities of image latent, control latent, cached context, and mask. It calls existing `sample_flow_timestep`, `make_flow_pair`, `forward_pose_control`, and MSE. Changed latent/text inputs fail rather than silently changing the fixed set.
- Baseline step 0 is fresh verified Krea + initial zero-impact control/LoRA state. Steps 20–100 are full-schema validated and load through exact trainable-state validation. JSON results include mean/median/std/per-sample loss.
- `evaluate.py fixed-pose` uses default eight deterministic `diagnostic_val` stems, seed `420200`, cached conditional/unconditional text, existing Euler sampler, 8 steps, CFG 3.5, and Qwen VAE inverse normalization/decode. It writes `fixed_pose/<stem>/control.png`, metadata, steps 0–100, and `comparison_grid.png`.

## Files changed this session

- `evaluate.py`, `pose_controlnet/evaluation.py`, `tests/test_evaluation.py`
- `pose_controlnet/diffusion.py`, `pose_controlnet/vae_preprocessing.py`, `base_model/k2_lora.py`
- `docs/CODEX_HANDOFF.md`

## Tests run

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_evaluation tests.test_train_mechanics tests.test_vae_preprocessing` (32 tests); `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile evaluate.py pose_controlnet/evaluation.py pose_controlnet/diffusion.py pose_controlnet/vae_preprocessing.py base_model/k2_lora.py tests/test_evaluation.py`; `git diff --check`.

Coverage: deterministic repeated fixed flow, checkpoint-independent timestep/noise, required checkpoint order, trainable-state interface, deterministic pose filenames/metadata, no gradients/optimizer effects, and eval-mode restoration.

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

Run fixed flow twice and require identical results/spec identity; then run the one-sample smoke and full grid. Interpret deterministic loss and fixed-control evidence before authorizing further training.
