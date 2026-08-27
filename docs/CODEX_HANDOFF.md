# Phase 1 handoff

## Current bounded objective and status

Turbo benchmark support is extended for incremental evaluation of exact
checkpoints 900 and 1200. No training, optimizer/backward work, model/sampler
change, checkpoint mutation, or image generation was performed in this session.

## Turbo contract and exact checkpoint routing

- Turbo remains Krea-2 Turbo, 8 steps, CFG 0.0, mu 1.15, with
  `mu_resolution_dependent=false`; schedule source remains
  `https://github.com/krea-ai/krea-2/blob/main/sampling.py`.
- Output root remains
  `/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0`.
- `--steps` accepts an explicit unique subset of `800 900 1200 1500`; the
  no-argument default remains the legacy `(800, 1500)` pair.
- Exact 900/1200 recovery uses HF repo
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`, run namespace
  `pose-learning-1500`, and therefore only
  `pose-learning-1500/full/step_000900.pt` and
  `pose-learning-1500/full/step_001200.pt` (plus matching completion markers).
  Existing shared recovery performs exact filename, marker, SHA-256, full
  deserialization/schema, and embedded `global_step` validation; no latest,
  nearest, or timed substitution is permitted.

## Incremental evaluation behavior

- Generation skips existing per-stem step images, so existing 800/1500 images
  remain untouched and `--steps 900 1200` generates only missing 900/1200
  images.
- Generation metadata merges prior generated-step records with new work.
- Scoring merges requested rows with existing score rows in canonical numeric
  order. Report requires and emits `800, 900, 1200, 1500`, including the
  five-column control/grid order and `evaluation_summary.json`
  `machine_readable_table` order.
- Diagnostic stems, prompts, controls, seeds, buckets, paired geometry, VAE,
  sampler, PCK, and CLIP semantics are unchanged.

## Files changed this session

- `pose_controlnet/turbo_evaluation.py`
- `scripts/turbo_benchmark.py`
- `tests/test_turbo_evaluation.py`
- `docs/CODEX_HANDOFF.md`

## Verified tests

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_turbo_evaluation tests.test_evaluation tests.test_post1500_evaluation` (30 tests).
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile scripts/turbo_benchmark.py pose_controlnet/turbo_evaluation.py tests/test_turbo_evaluation.py`.
- Regression coverage verifies CLI 900/1200 selection, exact HF namespace/path,
  legacy image reuse/only-missing generation, incremental result merging/canonical order, Turbo
  schedule pinning, and absence of optimizer/backward paths.
- PASS: `git diff --check`.

## Exact GH200 commands (do not train)

```bash
export UV_CACHE_DIR=/tmp/krea_uv_cache
cd /home/ubuntu/Krea-2-Pose-ControlNet

uv run python scripts/turbo_benchmark.py preflight --steps 900 1200
uv run python scripts/turbo_benchmark.py generate --steps 900 1200
uv run python scripts/turbo_benchmark.py score --steps 800 900 1200 1500
uv run python scripts/turbo_benchmark.py report --steps 800 900 1200 1500
```

Stop before expensive generation unless explicitly directed to run it. Before
ending an implementation session, run `git diff --check` and `git status --short`.
