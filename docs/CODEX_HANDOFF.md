# Project handoff

## Current objective

The frozen 48-image final validation benchmark is now bound to deterministic
held-out val-cache identities. This stage created only the immutable benchmark
spec and its focused tests; it did not run training, inference, networking,
Turbo baselines, checkpoint interpolation, generation, commit, or push.

## Locked final benchmark contract

- Frozen selection:
  `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl`.
- Immutable spec:
  `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json`.
- Stem order is the frozen source-then-stem ascending order, not a new
  seed-ranked selection order.
- Fixed quotas: COCO 16, painting 12, real_human 12, sculpture 8.
- Orientation counts: landscape 16, near_square 17, portrait 15.
- The spec uses the complete held-out `val` cache at
  `/lambda/nfs/adhit/krea2-pose/posebridge_latents` plus the matching
  `/lambda/nfs/adhit/krea2-pose/text_conditioning` cache. The `_768` cache is
  not appropriate: it does not have matching val text-conditioning identities.
- Dataset identity is solely the shared `make_evaluation_spec` per-stem latent,
  control, context, and mask SHA-256 values. Absolute shard/cache paths are
  absent from the generated spec.
- `benchmark.provenance` records repository-relative paths, SHA-256 digests,
  and record counts for the frozen selection (48), `val.jsonl` (889),
  `diagnostic_val.jsonl` (24), and candidate pool (96).
- Locked Turbo metadata is Krea-2 Turbo, 8 steps, CFG 0.0, mu 1.15,
  resolution-independent mu, and control scale 1.0.
- `write_immutable_spec` writes atomically on first creation and later rejects
  any non-identical replacement. Historical diagnostic benchmark behavior is
  unchanged.

## Completed implementation

- Added `scripts/create_final_val_benchmark_spec.py`.
  - Validates frozen count, unique membership, documented ordering, fixed
    source quotas, val membership, diagnostic exclusion, candidate-pool
    membership, and frozen candidate-pool digest.
  - Calls `pose_controlnet.evaluation.make_evaluation_spec` directly for all
    deterministic seeds and cached sample identities.
  - Adds final-benchmark counts/provenance and the centralized Turbo contract.
- Added `tests/test_final_val_benchmark_spec.py`.
  - Covers deterministic output, stable frozen order, quotas, Turbo contract,
    all required provenance keys, absence of absolute shard paths, immutable
    write behavior, and diagnostic-overlap rejection.
- Generated `final_val_benchmark_spec.json` successfully against the mounted
  production val cache. Its source provenance is recorded inside the spec.

## Verification

PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_final_val_benchmark_spec tests.test_freeze_final_val_benchmark tests.test_evaluation -v
# 14 tests passed

PYTHONPATH=. python -m py_compile scripts/create_final_val_benchmark_spec.py
PYTHONPATH=. python scripts/create_final_val_benchmark_spec.py
# wrote 48-record final validation benchmark spec

git diff --check -- scripts/create_final_val_benchmark_spec.py tests/test_final_val_benchmark_spec.py docs/CODEX_HANDOFF.md
```

The full-tree `git diff --check` remains non-clean solely because the
user-modified `candidate_review.csv` contains pre-existing trailing whitespace;
that input was not changed in this stage.

## Files changed this session

- `scripts/create_final_val_benchmark_spec.py`
- `tests/test_final_val_benchmark_spec.py`
- `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json`
- `docs/CODEX_HANDOFF.md`

The pre-existing modified `candidate_review.csv` and untracked frozen
`final_val_benchmark_48.jsonl` were preserved untouched.

## Exact next task

In a separate bounded stage, add the final-val Turbo evaluation entry point
that consumes `final_val_benchmark_spec.json`; do not alter this frozen spec or
the historical diagnostic benchmark contract.
