# Project handoff

## Current objective

The frozen 48-image final validation benchmark now has an opt-in Turbo
evaluation entry point. This session added no training, network operation,
image generation, commit, or push. The historical 24-item diagnostic Turbo
benchmark contract was not changed.

## Locked final-val contract

- Immutable selection:
  `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl`.
- Immutable spec:
  `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json`,
  SHA-256 `93a5254e57fa208263f6188573e0760ffedd954bf3b3b3425109ea0178957cd0`.
- Exactly 48 held-out `val` stems: COCO 16, painting 12, real_human 12,
  sculpture 8; orientation counts 16/17/15.
- Locked Turbo settings: Krea-2 Turbo, 8 steps, CFG 0, mu 1.15 with no
  resolution shift, and control scale 1.0.
- Only real controlled checkpoint candidates are accepted:
  - `parent-4000`: `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt`
  - `finish-control-a4300`: `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-finish-control-4000-to4500/step_004300.pt`
- Turbo base/zero adapter and checkpoint interpolation are intentionally
  unsupported.

## Completed implementation

- Added `scripts/final_val_turbo_benchmark.py` with staged `preflight`,
  `generate`, `score`, and `report` actions.
  - Pins the frozen spec SHA-256 and validates its 48 stems, quotas, seed,
    identities, and locked Turbo settings.
  - Recomputes all selected val cached latent/control/text SHA-256 identities
    and per-stem seeds via shared `make_evaluation_spec`.
  - Resolves controls exclusively through `validate_posebridge_snapshot` /
    `DatasetIndex`, including the full sharded physical snapshot validation.
  - Validates exact checkpoint filename, embedded step, SHA in output
    provenance, and controlled-branch metadata before work begins.
  - Reuses the Turbo runtime, raw-to-Turbo compatibility check, PCK scoring,
    CLIP scoring implementation, PNG/artifact validation pattern, and contact
    sheets. Outputs fail closed on a conflicting, partial, or corrupt set.
  - PCK needs an explicit immutable 48-stem authoritative sidecar in frozen
    order; the historical diagnostic sidecar is rejected and there is no
    fallback.
- Added `tests/test_final_val_turbo_benchmark.py`.
- Added runnable command documentation at
  `docs/evaluation/final-val-benchmark-selection/README.md`.

## Focused verification

PASS:

```bash
PYTHONPATH=. python -m py_compile scripts/final_val_turbo_benchmark.py
PYTHONPATH=. python -m unittest tests.test_final_val_benchmark_spec tests.test_final_val_turbo_benchmark tests.test_turbo_evaluation -v
# 20 tests passed
PYTHONPATH=. python scripts/final_val_turbo_benchmark.py --help
git diff --check
```

## Exact run commands

Run preflight then generation from the GH200 host shell; no generation was run
in this session:

```bash
uv run python scripts/final_val_turbo_benchmark.py preflight --candidate parent-4000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000
uv run python scripts/final_val_turbo_benchmark.py generate --candidate parent-4000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000
uv run python scripts/final_val_turbo_benchmark.py preflight --candidate finish-control-a4300 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
uv run python scripts/final_val_turbo_benchmark.py generate --candidate finish-control-a4300 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
```

After the separate immutable final-val pose sidecar exists, use its exact path
in the documented `score` commands, then run the corresponding `report`
commands in the README.

## Files changed this session

- `scripts/final_val_turbo_benchmark.py`
- `tests/test_final_val_turbo_benchmark.py`
- `docs/evaluation/final-val-benchmark-selection/README.md`
- `docs/CODEX_HANDOFF.md`

The pre-existing/untracked frozen selection, immutable spec, spec builder, and
its focused test remain untouched.

## Exact next task

Create and validate an immutable authoritative pose-reference sidecar for the
same frozen 48 final-val stems (including its provenance and frozen order), so
the implemented PCK scoring commands can run. Do not generate images, train,
or add Turbo base/zero-adapter or interpolation evaluation in that task.
