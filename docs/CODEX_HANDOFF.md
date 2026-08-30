# Project handoff

## Current bounded objective

Backward-compatible score-only provenance handling is complete for the
already-generated legacy native Mixed-32 experiment
`overfit32-mixed-r64-mse`. No training, generation, GPU evaluation,
checkpoint/output mutation, commit, or push occurred in this session.

## Exact legacy compatibility rule

Only `overfit32-mixed-r64-mse` may map an absent/`none` checkpoint resolution
to `training_resolution = native`. It must be located exactly at
`/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-mixed-r64-mse`;
its requested stems must exactly equal the immutable Mixed-32 manifest order;
the generation metadata must identify that experiment and exactly the steps
`0,50,100,200,300,400,500`; evaluation must be native; and its persisted
native geometry must validate and match verified latent geometry. Any present
legacy checkpoint schedule must also match exactly.

All recorded checkpoint resolution fields must be absent/`none`; an explicit
native/768/other value or any conflict fails closed. Resolution-manifest/cache
metadata, a checkpoint resolution manifest, or an alternate-resolution cache
also fails closed. Arbitrary experiments with `resolution=none` remain invalid.

Successful compatibility resolution records:

```text
training_resolution = native
evaluation_resolution = native
training_resolution_source = legacy_native_compatibility
```

The source marker is emitted inside `evaluation_provenance` in new score
metrics/summary output; it does not claim the legacy checkpoint recorded
`native`. The native-evaluation behavior of
`overfit32-mixed-r64-mse-res768` remains `training_resolution=768` and
`evaluation_resolution=native`. PCK, CLIP, detector, matching, coverage,
sidecar, and native geometry scoring logic were not changed.

## Files changed this session

- `scripts/evaluate_overfit_capacity.py`
- `tests/test_overfit_evaluation_resolution.py`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS:

```bash
PYTHONPATH=. python -m py_compile scripts/evaluate_overfit_capacity.py pose_controlnet/capacity_resolution.py pose_controlnet/reference_pose.py tests/test_capacity_reference_pose.py tests/test_overfit_evaluation_resolution.py tests/test_overfit_mixed_reference_pose.py
PYTHONPATH=. python -m unittest tests.test_capacity_reference_pose tests.test_overfit_evaluation_resolution tests.test_overfit_mixed_reference_pose -v
```

The 27 CPU/no-network tests cover the exact legacy allowlist and explicit
provenance, arbitrary-`none`, contradictory resolution, wrong root, wrong
Mixed-32 order, invalid supplied checkpoint schedule, unchanged 768/native
provenance, no checkpoint-metadata write, and unchanged scoring-call
definitions.

## Exact next action

Foreground score-only retry (not run in this session):

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse --stage score-only --reference-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl
```

Afterward, inspect:

```bash
cat /lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation/overfit32-mixed-r64-mse/training_set_overfit_metrics.json
```
