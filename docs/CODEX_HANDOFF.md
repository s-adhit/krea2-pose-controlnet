# Project handoff

## Current bounded objective

The overfit-capacity evaluator has a fixed native-evaluation contract. This
session changed only CPU-side evaluation, report, comparison, and paired
geometry code. No training, generation, GPU evaluation, checkpoint/output
mutation, deletion, commit, or push was performed.

## Resolution policy in force

- Low resolution is a training-only capacity axis. Training may use `native`
  or `768`; the training CLI still owns `--resolution`.
- Every capacity evaluation is `native`, independently of training resolution.
  The evaluation CLI has no resolution or resolution-cache option.
- Native evaluation reads the persisted original source size, resized size,
  crop box, and bucket from the native latent shard for every stem. It rebuilds
  RGB target and pose control with that exact shared geometry; it never reads
  an alternate-resolution cache.
- Evaluation records both fields explicitly: `training_resolution` is read
  from checkpoint provenance, while `evaluation_resolution` is always
  `native`. Evaluation resolution is never inferred from the training value.

## Completed and non-authoritative state

- Completed 768-trained checkpoint root (read-only):
  `/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-mixed-r64-mse-res768`
  with exact checkpoints `0, 50, 100, 200, 300, 400, 500`.
- Its training provenance remains `training_resolution = 768`; checkpoint
  metadata and weights were not modified.
- The earlier 768-geometry evaluation for that experiment was interrupted and
  is non-authoritative. It must not be resumed or compared.
- Canonical native-evaluation root:
  `/lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation/overfit32-mixed-r64-mse-res768`.
  The evaluator refuses incompatible/malformed/incomplete content there before
  writing. Its error gives an archive command; do not delete artifacts.

Read-only inspection confirmed that canonical directory currently contains a
`training_set/` tree but no `generation_results.json`; it is incomplete and
must be archived before native generation.

If that refusal occurs, first confirm the archive destination does not exist,
then an operator may preserve the old partial directory with:

```bash
mv -- /lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation/overfit32-mixed-r64-mse-res768 /lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation/overfit32-mixed-r64-mse-res768.partial-768-eval-archive
```

After that archive, foreground native generation (not from Codex) is:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse-res768 --stage generate
```

Then the foreground native qualitative report command is:

```bash
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse-res768 --stage report
```

The report remains: pose control, target training RGB, then steps `0, 50,
100, 200, 300, 400, 500`, all at native geometry. Future 768 pose-loss
experiments follow the same native-evaluation rule. Comparison labels are
`Native train / Native eval`, `768 train / Native eval`, and
`768+pose train / Native eval`; entries without explicit native evaluation
provenance are excluded.

## Files changed this session

- `pose_controlnet/paired_preprocessing.py`
- `pose_controlnet/capacity_resolution.py`
- `scripts/evaluate_overfit_capacity.py`
- `scripts/run_overfit_capacity.py`
- `scripts/summarize_overfit_capacity.py`
- `tests/test_overfit_capacity.py`
- `tests/test_overfit_evaluation_resolution.py`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
python -m py_compile pose_controlnet/paired_preprocessing.py pose_controlnet/capacity_resolution.py scripts/evaluate_overfit_capacity.py scripts/summarize_overfit_capacity.py scripts/run_overfit_capacity.py tests/test_overfit_capacity.py tests/test_overfit_evaluation_resolution.py
python -m unittest tests.test_capacity_experiment_axes tests.test_overfit_capacity tests.test_overfit_evaluation_resolution -v
git diff --check
```

The 26 CPU tests cover native and 768 training provenance, fixed native
evaluation, exact Mixed-32 order and checkpoint steps, paired persisted
geometry, no evaluator 768/cache mode, stale-output refusal/archive guidance,
safe valid-native reuse, report provenance, comparison exclusion, no
optimizer/backward path, and unchanged 768 training ownership.

## Exact next action

An operator should archive the incompatible old canonical evaluation directory,
then run the foreground native generation command above. Do not train or modify
the completed checkpoints.
