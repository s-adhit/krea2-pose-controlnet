# Project handoff

## Current objective and enforced decisions

`train.py` remains unchanged and production training remains flow-matching MSE
only. Gate E is the separately invoked bounded Gaussian-heatmap-KL smoke
continuation: `lambda_pose=2e-5`, inclusive pose timestep window `[0.10,
0.20]`, microbatch `1`, gradient accumulation `32`, and target effective batch
`32`. Do not launch production training from the Codex sandbox.

Gate-E training is complete at global step `1700`. Its checkpoints are local
under `/lambda/nfs/adhit/krea2-pose/checkpoints/gate-e-parent1500-kl-l2e5-t010-020-mb1-ga32/`:
steps `1550`, `1600`, `1650`, and `1700`. The supplied SHA-256 for
`step_001700.pt` is
`b454cfff01e6c2608415abc54d910682be9705d1ea337b342511fe1586828415`.
Measured resumed exposure was `20 / 2504 = 0.7987%` active/eligible samples;
`18 / 90` optimizer steps had at least one active sample.

## Verified gates and decisions

- Gates A, A.5, B, and C: PASS as previously documented. Gate D remains
  IMPLEMENTED / GH200 RUN REQUIRED.
- Gate E training continuation is complete; its Turbo evaluation has not yet
  been run. Gate E is not PASS until that evaluation/inspection is complete.
- Gate-E checkpoints store top-level `gate_e` metadata: pose loss/window,
  critical model/training config, and trainable state names. Resume validation
  remains fail-closed.
- The generic Turbo evaluator now supports a `direct_local` exact-checkpoint
  selector for bounded local branches without an HF checkpoint mirror. It
  checks direct filenames, embedded global steps, and every configured SHA;
  this changes no sampler, VAE, diagnostic, CLIP, or PCK semantics.
- `configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json` fixes Krea-2 Turbo,
  8 steps, CFG 0, fixed non-resolution-dependent `mu=1.15`, official schedule,
  control scale 1.0, the established 24 diagnostics, and authoritative
  21-sample numerical PCK (Danbooru remains excluded). It reuses the established
  LR-only step-1500 result rather than regenerating it.
- Gate-E output is isolated at `docs/evaluation/gate-e-kl-l2e5-t010-020/`.
  It does not overwrite any historical evaluation output.

## Files changed this session

- `pose_controlnet/turbo_evaluation.py`
- `scripts/turbo_benchmark.py`
- `configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json`
- `tests/test_turbo_evaluation.py`
- `docs/CODEX_HANDOFF.md`

Existing untracked Gate-B/C/D/E audit files remain user-owned and were not
overwritten. No training or historical evaluation artifacts were changed.

## Tests and checks

- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest
  tests.test_turbo_evaluation tests.test_turbo_lr5e5_evaluation
  tests.test_turbo_timestep_evaluation tests.test_post1500_evaluation` — 30
  CPU tests. Covers Gate-E spec/metadata, direct exact-local selection and SHA
  mismatch failure, unchanged shared Turbo/PCK/CLIP contracts, and post-1500
  PCK behavior.
- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile
  pose_controlnet/turbo_evaluation.py scripts/turbo_benchmark.py
  tests/test_turbo_evaluation.py`.
- PASS: `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python
  scripts/turbo_benchmark.py preflight --help`.
- PASS: Gate-E experiment-spec load check.
- PASS: `git diff --check`.

## Exact next GH200 action (do not run from Codex)

Run the complete staged Gate-E Turbo evaluation from the repository root:

```bash
PYTHONPATH=. python scripts/turbo_benchmark.py preflight --spec configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json && \
PYTHONPATH=. python scripts/turbo_benchmark.py generate --spec configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json && \
PYTHONPATH=. python scripts/turbo_benchmark.py score --spec configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json && \
PYTHONPATH=. python scripts/turbo_benchmark.py report --spec configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json
```

Inspect `docs/evaluation/gate-e-kl-l2e5-t010-020/evaluation_summary.json` and
the contact sheets. It contains CLIP, overall/subset PCK, coverage and people
counts, plus deltas for all reported PCK variants and aggregate metrics versus
step 1500.
