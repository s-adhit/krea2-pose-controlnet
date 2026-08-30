# Project handoff

## Current bounded objective

The circular legacy-native provenance dependency is fixed for the archived
Mixed-32 legacy experiment `overfit32-mixed-r64-mse`. The change only enables
its foreground native regeneration; no training, GPU generation, checkpoint or
evaluation-output mutation, deletion, commit, or push occurred in this session.

## Legacy-native compatibility rule

The allowlist is only `overfit32-mixed-r64-mse`, at exactly
`/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-mixed-r64-mse`.
For `--stage generate`, it can resolve absent/`none` checkpoint resolution to
`training_resolution = native` only when all checkpoint resolution fields are
absent/`none`, the checkpoint files are exactly steps
`0,50,100,200,300,400,500`, the immutable Mixed-32 order matches exactly,
there is no alternate-resolution cache/manifest, and every persisted native
RGB/control latent geometry is present, aligned, and valid. Generation also
preflights that each indexed RGB/control source pair recovers the exact
persisted native geometry before writing an artifact.

`--stage report` and `--stage score-only` do not infer this legacy resolution
from checkpoints. They require regenerated `generation_results.json` metadata
with `training_resolution = native`, `evaluation_resolution = native`, and
`evaluation_provenance.training_resolution_source = legacy_native_compatibility`.
Any conflict fails closed. The completed 768 experiment remains
`training_resolution = 768`, `evaluation_resolution = native`.

PCK, CLIP, detector, matching, sidecar handling, deterministic seeds, Turbo
generation behavior, and native geometry behavior were not changed.

## Files changed this session

- `scripts/evaluate_overfit_capacity.py`
- `pose_controlnet/capacity_resolution.py`
- `tests/test_overfit_evaluation_resolution.py`
- `docs/CODEX_HANDOFF.md`

## Verification

PASS (CPU/no-network only):

```bash
PYTHONPATH=. python -m unittest tests.test_capacity_reference_pose tests.test_overfit_evaluation_resolution tests.test_overfit_mixed_reference_pose -v
PYTHONPATH=. python -m py_compile scripts/evaluate_overfit_capacity.py pose_controlnet/capacity_resolution.py pose_controlnet/reference_pose.py tests/test_capacity_reference_pose.py tests/test_overfit_evaluation_resolution.py tests/test_overfit_mixed_reference_pose.py
git diff --check
```

The 34 tests cover no-metadata generation inference, exact allowlist/root/stem
order/schedule checks, contradictory resolution, alternate cache/manifest,
missing latent geometry, exact paired source recovery, emitted compatibility
marker, report/score metadata requirements, no checkpoint metadata write, and
unchanged 768/native behavior.

## Exact next actions (not run in this session)

Foreground native regeneration:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse --stage generate
```

Foreground report:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse --stage report
```

Foreground score-only:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse --stage score-only --reference-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl
```
