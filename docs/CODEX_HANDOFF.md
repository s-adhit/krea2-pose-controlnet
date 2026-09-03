# Project handoff

## Current objective

The reproducible prompting-guide experiment is implemented as separate,
opt-in tooling. It has not been run on GH200. No training, network access,
commit, or push occurred in this session.

## Frozen prompting-study contract

- Entry point: `scripts/prompting_guide_study.py` with staged actions
  `preflight`, `generate`, `score`, `report`, and read-only `summary`.
- Frozen source: `docs/evaluation/prompting-guide/prompting_study.jsonl`.
  Byte SHA-256 is
  `4fae6d39ac7354d451ca13556d3a2a89e303691ceb30727b8561b13b494450df`.
- The loader requires exactly 64 rows: exactly 8 unique pose conditions and
  exactly these eight ordered modes per stem: `P0_minimal`, `P1_style`,
  `P2_environment`, `P3_neutral`, `P4_supportive`, `P5_conflicting`,
  `P6_semantic_prior`, and `P7_framing_count_conflict`. It rejects SHA drift,
  schema drift, missing modes, unexpected names, duplicate stem/mode pairs,
  conflicting classes, or an incomplete matrix.
- Candidate is hard-locked to `mix-025`. It validates the existing pinned
  interpolation endpoints and provenance. Geometry is native/aspect-preserving
  cached-latent bucket geometry. Turbo is locked to 8 steps, CFG 0, mu 1.15,
  and control scale 1.0.
- All eight study stems are verified members of the frozen final-val set and
  must have a pinned final-val sampling seed. That same seed is used for every
  prompt mode of its stem; no substitute seed path exists.
- Controls are resolved only through the authoritative final-val DatasetIndex
  helper. Their SHA-256 values, frozen prompt mapping, seeds, native buckets,
  candidate interpolation provenance, and exact prompt text are retained in
  immutable output provenance and per-generation metadata.
- The tool requires the canonical final-val v3 pose sidecar for scoring. It
  scores PCK separately for every one of the 64 images and CLIP against the
  row's exact experimental prompt, then verifies and aggregates those records
  by prompt mode and pose class.
- It fails closed on candidate/provenance conflicts, missing/incomplete or
  duplicate generation manifests, corrupted images, orphan controls/metadata,
  wrong control hashes, wrong native bucket/geometry, non-locked Turbo fields,
  and incomplete or reordered scored records. Source RGB is never sampled,
  copied, or used as a qualitative fallback.

## Expected output artifacts

The separate output root contains immutable `prompting_study_provenance.json`,
one copied authoritative control per stem under `controls/`, 64 generation
directories with exact-prompt metadata, `pck_clip_results.json`, and:

- `comparison_grids/<stem>.png` for every skeleton, each with pose control
  plus P0 through P7;
- `prompting_study_contact_sheet.png` covering all eight skeletons;
- `metrics_by_prompt_mode.json` and `metrics_by_pose_class.json` compact
  aggregate tables;
- `evaluation_summary.json` binding the above to provenance.

The frozen 48 source-caption benchmark, frozen prompt-injection benchmark,
canonical `inference.py`, and all training code remain untouched.

## Files changed this session

- `scripts/prompting_guide_study.py` (new isolated study tool)
- `tests/test_prompting_guide_study.py` (new focused contract tests)
- `docs/CODEX_HANDOFF.md`

The frozen `docs/evaluation/prompting-guide/prompting_study.jsonl` is present
as an untracked supplied input and was not modified.

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_prompting_guide_study tests.test_frozen_prompt_turbo tests.test_final_val_turbo_benchmark -v
# 27 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/prompting_guide_study.py tests/test_prompting_guide_study.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/prompting_guide_study.py --help
git diff --check
```

Focused coverage includes pinned SHA/matrix validation, hash drift rejection,
unexpected/duplicate modes, locked Turbo settings, reuse of frozen final-val
seeds, incomplete or orphaned generation artifact rejection, and complete
ordered score-record requirements. Existing prompt-injection and final-val
evaluator tests remain green.

## Exact GH200 commands

Run from the repository root in the GH200 host shell. Use a new output root;
do not write under any frozen final-val or prompt-injection root.

```bash
uv run python scripts/prompting_guide_study.py preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/mix-025
uv run python scripts/prompting_guide_study.py generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/mix-025
uv run python scripts/prompting_guide_study.py score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/mix-025 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/prompting_guide_study.py report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/mix-025
```

Exact compact result-summary command:

```bash
uv run python scripts/prompting_guide_study.py summary --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/mix-025
```

## Exact next task

Inspect prompting results and write root-level `prompting.md`.
