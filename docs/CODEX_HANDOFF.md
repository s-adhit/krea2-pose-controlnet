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

## Gate-E Turbo runtime state

Gate-E preflight, generation, and scoring completed successfully in the
isolated `docs/evaluation/gate-e-kl-l2e5-t010-020/` output. It has generated
and scored all four configured current-branch checkpoints: `1550`, `1600`,
`1650`, and `1700`. Do not regenerate step 1500, alter historical artifacts,
or rerun expensive stages.

The report stage failed only because the generic visual contact-sheet code
required historical per-sample baseline PNGs at
`docs/evaluation/turbo-8step-cfg0-lr5e5/fixed_pose/*/step_001500.png`. Those
PNGs are absent, while the required historical numerical artifacts remain:
`turbo_spec.json`, `pck_clip_results.json` (including exact step 1500), and
`evaluation_summary.json`.

The generic report now keeps the established step-1500 numerical baseline and
all existing delta math untouched. It scans only the optional historical PNG
column: if every configured sample PNG exists, contact sheets retain that
column; otherwise the whole baseline visual column is omitted so every grid row
has equal columns. Current-branch control/1550/1600/1650/1700 files remain
required and fail closed. The resulting `evaluation_summary.json` records
`baseline_visual_artifacts_available` and
`baseline_visual_artifacts_missing_count`; it does not substitute any image or
checkpoint for the numerical baseline.

## Verified gates and decisions

- Gates A, A.5, B, and C: PASS as previously documented. Gate D remains
  IMPLEMENTED / GH200 RUN REQUIRED.
- Gate E is not PASS until the report-only command below completes and its
  generated summary/contact sheets are inspected.
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

- `scripts/turbo_benchmark.py`
- `tests/test_turbo_evaluation.py`
- `docs/CODEX_HANDOFF.md`

Existing untracked Gate-B/C/D/E audit files remain user-owned and were not
overwritten. Gate-E evaluation artifacts were not overwritten.

## Tests and checks

- PASS: `PYTHONPATH=. python -m unittest tests.test_turbo_evaluation
  tests.test_turbo_lr5e5_evaluation tests.test_turbo_timestep_evaluation
  tests.test_post1500_evaluation` — 31 CPU tests. Includes the regression:
  a valid step-1500 numerical baseline plus missing baseline PNGs reports
  successfully with unchanged deltas and explicit availability metadata;
  deleting a current-branch image still fails closed.
- PASS: `PYTHONPATH=. python -m py_compile scripts/turbo_benchmark.py
  tests/test_turbo_evaluation.py`.
- PASS: `git diff --check`.

## Exact next GH200 action (report only)

```bash
PYTHONPATH=. python scripts/turbo_benchmark.py report --spec configs/evaluation/gate_e_kl_l2e5_t010_020_turbo.json
```

Inspect `docs/evaluation/gate-e-kl-l2e5-t010-020/evaluation_summary.json` and
the contact sheets. Confirm the baseline numerical identity/deltas and that
`baseline_visual_artifacts_available` is `false` with the expected missing
count before declaring Gate E evaluated.
