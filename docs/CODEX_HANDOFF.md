# Project handoff

## Current objective

The final-val Turbo evaluator is ready to preflight the two frozen selected
controlled checkpoints. No checkpoint was modified. No generation, training,
network operation, commit, or push occurred.

## Final-val checkpoint compatibility contract

- Only `parent-4000` and `finish-control-a4300` are accepted.
- Each candidate is pinned to its exact absolute root, `step_XXXXXX.pt`
  filename, embedded `global_step`, and SHA-256:
  - parent-4000 / step 4000:
    `0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd`.
  - finish-control-a4300 / step 4300:
    `17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff`.
- `load_training_state` remains the full-training checkpoint schema gate.
- If `gate_e` is present, the evaluator still uses the unchanged strict
  `controlled_branch_metadata()` validation used by historical diagnostics.
- These two legitimate older checkpoints lack `gate_e`; for them only, the
  evaluator instead verifies their pinned project-owned
  `production_pose_control` format, run name, maximum step, and current step.
  This validated provenance is recorded in final-val outputs.

## Completed / green gates

PASS:

```bash
python -m py_compile scripts/final_val_turbo_benchmark.py tests/test_final_val_turbo_benchmark.py
python -m unittest tests.test_final_val_turbo_benchmark -v
# 7 tests passed
git diff --check
```

Focused coverage proves metadata-present checkpoints retain strict historical
validation, while metadata-absent final-val checkpoints require the pinned
SHA and production provenance and reject mismatched provenance.

## Files changed this session

- `scripts/final_val_turbo_benchmark.py`
- `tests/test_final_val_turbo_benchmark.py`
- `docs/CODEX_HANDOFF.md`

Pre-existing untracked final-val sidecar/build-script work remains untouched.

## Exact next preflight commands

Run from the GH200 host shell only:

```bash
uv run python scripts/final_val_turbo_benchmark.py preflight --candidate parent-4000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000
uv run python scripts/final_val_turbo_benchmark.py preflight --candidate finish-control-a4300 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
```

After both preflights pass, the next bounded task is the already-authorized
final-val generation/scoring workflow for only those two candidates. Do not
evaluate Turbo base/zero-adapter, interpolate checkpoints, or alter the
frozen benchmark contract.
