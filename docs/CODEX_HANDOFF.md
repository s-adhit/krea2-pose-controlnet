# Phase 1 handoff

## Current objective and status

The Turbo evaluator is now spec-driven and evaluation-only. The current
ControlInput-LR2x experiment is configured in
`configs/evaluation/controlinput_lr2x_turbo.json`; no checkpoint subset is a
permanent source/config invariant. No preflight, generation, scoring,
training, resume, checkpoint mutation, upload, commit, or push was run in
this session.

## Generic Turbo evaluator

Use `scripts/turbo_benchmark.py` with `--spec PATH` and normally exactly one
of `--steps ...` or `--all-checkpoints` for `preflight`, `generate`, and
`score`. A spec may instead provide an optional generic `steps` default;
the current spec intentionally does not.
`report` discovers completed scored results under the configured output root;
it needs no step arguments. Explicit CLI overrides are available for the
checkpoint root, HF repo/namespace, output root, and normal runtime paths.

`pose_controlnet/turbo_evaluation.py` owns spec validation, direct-root
checkpoint discovery, and strict exact-local checkpoint validation. Every
selected checkpoint must be the direct `step_XXXXXX.pt` file in the configured
root and pass its matching configured HF completion-marker, SHA-256, schema,
and embedded-step checks. There is no nearest/latest, alternate-run, or remote
payload fallback.

Generation and scoring discover prior completed outputs dynamically. Partial,
inconsistent, invalid-image, or contract-mismatched outputs fail closed and
are not overwritten. Reports dynamically sort all scored checkpoints, preserve
canonical diagnostic-manifest order, and render control, optional reference,
configured baseline, and every completed checkpoint column.

The current spec resolves the canonical 24-record diagnostic manifest and
pins Krea-2 Turbo: 8 steps, CFG 0.0, mu 1.15, non-resolution-dependent shift,
official schedule, and control scale 1.0. PCK (confidence .5, authoritative
Hungarian/bbox-normalized implementation and Danbooru behavior) and CLIP stay
in the shared implementations. The baseline is a configured external LR-only
step-1500 artifact and is never regenerated.

## Current user-run commands (GH200 only, when authorized)

```bash
cd /home/ubuntu/Krea-2-Pose-ControlNet
export UV_CACHE_DIR=/tmp/krea_uv_cache
uv run python scripts/turbo_benchmark.py preflight --spec configs/evaluation/controlinput_lr2x_turbo.json --steps 1600 1700 1900 2000 2100 2300 2400 2500 2700
uv run python scripts/turbo_benchmark.py generate --spec configs/evaluation/controlinput_lr2x_turbo.json --steps 1600 1700 1900 2000 2100 2300 2400 2500 2700
uv run python scripts/turbo_benchmark.py score --spec configs/evaluation/controlinput_lr2x_turbo.json --steps 1600 1700 1900 2000 2100 2300 2400 2500 2700
uv run python scripts/turbo_benchmark.py report --spec configs/evaluation/controlinput_lr2x_turbo.json
```

Discovery example (do not run without authorization):

```bash
uv run python scripts/turbo_benchmark.py preflight --spec configs/evaluation/controlinput_lr2x_turbo.json --all-checkpoints
```

## Files and validation in this session

- Changed: `scripts/turbo_benchmark.py`, `pose_controlnet/turbo_evaluation.py`,
  `tests/test_turbo_evaluation.py`, this handoff.
- Added: `configs/evaluation/controlinput_lr2x_turbo.json`.
- Removed obsolete branch-pinned ControlInput-LR2x Turbo entrypoint and its
  fixed-step test.
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile
  pose_controlnet/turbo_evaluation.py scripts/turbo_benchmark.py
  tests/test_turbo_evaluation.py`
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest
  tests.test_turbo_evaluation` (7 tests).

## Exact next action

Review the generic evaluator/spec diff. Do not run evaluation stages or
production training without explicit user authorization.
