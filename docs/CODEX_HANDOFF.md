# Project handoff

## Current objective

The benchmark-freezing stage is implemented. It converts the manually reviewed
96-candidate validation pool into a deterministic, provenance-bearing
48-record JSONL selection once the review CSV has exactly 48 `keep=yes` rows.
No training, inference, interpolation, network access, commit, or push was
performed.

## Decisions and verified inputs

- Candidate pool: `docs/evaluation/final-val-benchmark-selection/candidate_pool_96.jsonl`.
- Required immutable pool SHA256:
  `a72607f65d104ed09a083588bb210b8fb4e7ab22db3f2224ba939b838d906056`.
- The pre-existing read-only audit established 96 unique candidate validation
  stems, no current `keep=yes` review selections, and zero val/diagnostic
  overlap.
- Freeze quotas are fixed: COCO 16, painting 12, real_human 12, sculpture 8.
- The candidate pool, review CSV, source manifests, and `scripts/turbo_benchmark.py`
  remain unmodified in this session.

## Completed implementation

- Added `scripts/freeze_final_val_benchmark.py`.
  - Uses Python's `csv.DictReader` for the review CSV.
  - Verifies the exact candidate-pool SHA256 before any selection is accepted.
  - Validates unique candidate/review/manifest stems; validates candidate
    membership in val; rejects diagnostic overlap, non-candidates, duplicate
    reviews, non-val stems, caption/source mismatches, invalid `keep` values,
    wrong selection count, and wrong quotas.
  - Atomically writes the stable source/stem-sorted artifact at the default
    path `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl`.
    Each artifact record retains candidate metadata, carries the verified pool
    digest, and preserves review `difficulty`, `pose_type`, `multi_person`, and
    `notes` fields for later `make_evaluation_spec` integration.
- Added focused unit tests in `tests/test_freeze_final_val_benchmark.py` using
  only temporary fixtures. They cover deterministic successful output and hash,
  count, quota, duplicate, non-candidate, non-val, diagnostic-overlap, and
  caption-mismatch failures.

## Verification

PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_freeze_final_val_benchmark -v
# 2 tests passed

PYTHONPATH=. python -m py_compile scripts/freeze_final_val_benchmark.py
git diff --check
```

Current working tree also includes the user-provided, untracked
`docs/evaluation/final-val-benchmark-selection/` candidate materials, plus the
new untracked script and focused test. No frozen artifact exists yet because
the live review sheet has zero keep selections, as expected.

## Exact next task

Complete `candidate_review.csv` by marking exactly 48 reviewed candidates
`keep=yes` while meeting the fixed source quotas, then run:

```bash
PYTHONPATH=. python scripts/freeze_final_val_benchmark.py
```

Review the generated `final_val_benchmark_48.jsonl`; only after it is accepted
should a separate bounded task integrate its frozen stems with
`make_evaluation_spec`.
