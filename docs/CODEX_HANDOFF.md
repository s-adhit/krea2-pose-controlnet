# Project handoff

## Current bounded objective

The authoritative exact-manifest Mixed-32 reference-pose sidecar required for
quantitative scoring is built and CPU-verified. This session changed only
sidecar construction/loading and score-only geometry selection. No training,
generation, GPU evaluation, checkpoint/output mutation (other than the new
requested immutable sidecar), deletion, commit, or push occurred.

## Exact sidecar

- Records: `data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl`
- Metadata: `data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl.metadata.json`
- Records SHA-256: `95ef6d7d6aa69bc7784f38340abf9e19285d097a4ebf306f8c06f0e3b9cfb3d4`
- Input manifest SHA-256: `18ed9279a1bd05eece600ff950c6d42a4fb84efee7cbf24d82b28882920cc17d`
- Authoritative numerical source: `/lambda/nfs/adhit/krea2-pose/pose_targets_v3/records.jsonl`
  (SHA-256 `dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f`).
- Coverage: 32 records; 26 eligible authoritative targets (6 COCO, 7 HumanArt
  painting, 7 HumanArt real human, 6 HumanArt sculpture); 6 Danbooru records
  explicitly unavailable for numerical pose scoring.
- Compatibility is explicitly limited to `overfit32-mixed-r64-mse` and
  `overfit32-mixed-r64-mse-res768`, which share the exact manifest stems.

The sidecar retains authoritative source-space person keypoints and source
visibility data plus per-record provenance. It deliberately contains no old
bucket/crop/resized coordinates. During score-only evaluation, source-space
references are transformed from the exact native geometry persisted in
`generation_results.json`; the 768 training cache is not used for scoring.
Danbooru is excluded from numerical PCK rather than treated as a failure.

## Implementation and verification

- Added `scripts/build_overfit_capacity_reference_pose.py`.
- Extended `pose_controlnet/reference_pose.py` with immutable generic
  exact-manifest construction/loading while preserving existing COCO-sidecar
  behavior.
- `scripts/evaluate_overfit_capacity.py --stage score-only` now gets scoring
  geometry from persisted native generation metadata. Existing Keypoint R-CNN
  COCO_V1, confidence >= 0.5, deterministic Hungarian, bbox-diagonal PCK,
  unmatched-reference, CLIP, and coverage semantics are unchanged.
- Added `tests/test_overfit_mixed_reference_pose.py` for exact order, eligible
  joins, Danbooru unavailability, fail-closed cases, source SHA, native
  transform, no 768 leak, dual-experiment reuse, no detector builder path,
  existing COCO compatibility, and evaluation-only scoring.

PASS:

```bash
python -m py_compile pose_controlnet/reference_pose.py scripts/build_overfit_capacity_reference_pose.py scripts/evaluate_overfit_capacity.py tests/test_capacity_reference_pose.py tests/test_overfit_evaluation_resolution.py tests/test_overfit_mixed_reference_pose.py
python -m unittest tests.test_capacity_reference_pose tests.test_overfit_evaluation_resolution tests.test_overfit_mixed_reference_pose -v
```

A read-only CPU validation loaded the new sidecar against the already-generated
native geometry for `overfit32-mixed-r64-mse-res768`: 32 geometry records, 26
available targets, 6 unavailable targets, and the authoritative source SHA
matched exactly. It did not instantiate a detector or score any images.

## Operator commands

Build (the target is immutable and will refuse overwrite):

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/build_overfit_capacity_reference_pose.py --manifest configs/overfit_capacity/manifests/overfit32-mixed-r64-mse.jsonl --authoritative-source /lambda/nfs/adhit/krea2-pose/pose_targets_v3 --output data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl
```

Foreground native score-only evaluation (not run in this session):

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse-res768 --stage score-only --reference-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl
```

Print metrics after that score-only command completes:

```bash
cat /lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation/overfit32-mixed-r64-mse-res768/training_set_overfit_metrics.json
```

## Exact next action

An operator may run the foreground score-only command above. Do not train,
regenerate, alter checkpoints, or run the 6000-step production job.
