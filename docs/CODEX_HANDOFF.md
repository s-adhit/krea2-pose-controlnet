# Project handoff

## Current objective

The final-val Turbo qualitative-report selection bug is fixed in the workspace.
The report-only rebuild is ready but cannot be written from this Codex sandbox:
`/lambda/nfs` is mounted read-only here. No model generation, PCK/CLIP scoring,
checkpoint, frozen spec, sidecar, training, network operation, commit, or push
occurred.

## Final-val qualitative report fix

- Bug: `full_contact_sheet.png` and `checkpoint_selection_grid.png` could
  visually present reference RGB in the generated-output position.
- `report` now uses `_qualitative_image_paths()` as the sole report image
  resolver. For each stem it returns only:
  1. `<candidate output-root>/fixed_pose/<stem>/control.png` (pose control);
  2. `<candidate output-root>/fixed_pose/<stem>/step_XXXXXX.png` (that
     candidate's actual generated output).
- A missing generated PNG fails closed with `FileNotFoundError`; no dataset RGB
  fallback exists. Column labels are now `pose control` and
  `generated output (<candidate>)`.
- The same rows and labels feed both qualitative sheets. Historical diagnostic
  behavior and frozen final-val benchmark/spec/sidecar are unchanged.

## Completed / green checks

PASS:

```bash
python -m unittest tests.test_final_val_turbo_benchmark -v
# 9 tests passed
git diff --check
```

Focused coverage proves the qualitative resolver selects the exact
candidate-specific `step_004000.png` rather than a dataset RGB path, and that
it fails closed when the expected generated image is absent.

The script was run through the edited workspace module and reached report-image
writing, then failed at the expected output path with `OSError: [Errno 30]
Read-only file system`. The existing report artifacts were not changed.

## Exact report rerun commands

Run these from the writable GH200 host shell. They perform only the `report`
action and reuse the complete existing generations and score artifacts:

```bash
uv run python scripts/final_val_turbo_benchmark.py report --candidate parent-4000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000
uv run python scripts/final_val_turbo_benchmark.py report --candidate finish-control-a4300 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
```

Each command rewrites only its report artifacts:
`checkpoint_selection_grid.png`, `full_contact_sheet.png`, and
`evaluation_summary.json`.

## Files changed this session

- `scripts/final_val_turbo_benchmark.py`
- `tests/test_final_val_turbo_benchmark.py`
- `docs/CODEX_HANDOFF.md`

Pre-existing untracked final-val sidecar/build-script work remains untouched.

## Exact next recommended action

Run the two report-only commands from the writable GH200 host shell, then
review the rebuilt sheets. Do not regenerate or rescore these completed
final-val outputs unless an independently authorized benchmark change is
required.
