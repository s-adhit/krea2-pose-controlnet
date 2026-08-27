# Phase 1 handoff

## Current bounded objective and status

Post-1500 evaluation/audit tooling is implemented and unit-tested. This session
fixed its exact checkpoint preflight recovery routing. It did **not** train,
construct an optimizer, start optimizer steps, alter model/LR/LoRA/checkpoint/
data/sampler state, regenerate any images, or commit/push. The expensive GH200
evaluation has deliberately not been run.

## Training/archive facts in force

- Training is complete through step 1500.
- Canonical trajectory is exactly:
  `0,20,40,60,80,100,200,225,350,475,500,600,700,800,900,1000,1100,1200,1300,1400,1500`.
  Never add 300 or 400.
- Roots: `pose-learning-100` for <=100, `pose-learning-500` for 200..500,
  `pose-learning-1500` for 600..1500.
- Root cause of the host preflight failure: shared evaluation resolution only
  attempted HF recovery for steps >=600, so a legitimately pruned local
  `pose-learning-500/step_000200.pt` failed before it could recover.
- Exact recovery routing from `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`:
  `200,225,350,475,500 -> pose-learning-500/full/step_XXXXXX.pt`; and
  `600,700,800,900,1000,1100,1200,1300,1400,1500 ->
  pose-learning-1500/full/step_XXXXXX.pt`. Recovery copies are segregated by
  run namespace. Local valid checkpoints remain preferred (including step
  500). Resolution requires the matching completion marker, SHA-256, full
  checkpoint deserialization/schema, and embedded `global_step`; it never
  substitutes timed, nearest, latest, or other-namespace checkpoints.
- PCK references: 24 diagnostic records, 21 authoritative Human-Art/COCO,
  3 Danbooru unavailable/excluded. Eligibility remains
  `source_visible AND rendered_in_control`; detector is torchvision Keypoint
  R-CNN COCO_V1 at confidence >=0.5 with deterministic Hungarian matching,
  reference bbox-diagonal normalization, and <= thresholds.

## What this session changed

Three-archive canonical resolution with exact per-run HF recovery,
incremental/repeated fixed-flow,
full-diagnostic and smoke specs, and fixed-pose reuse are in `evaluate.py` /
`pose_controlnet.evaluation`. `post1500_evaluation` and `post1500_audit.py`
provide read-only merging, timestep/data/telemetry audits, pooled authoritative
PCK, CLIP, control sensitivity, plots, grids, and terminal summary. Unmatched
reference people now remain in the PCK denominator.

## Verified gates/tests

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile evaluate.py pose_controlnet/evaluation.py pose_controlnet/post500_evaluation.py pose_controlnet/post1500_evaluation.py scripts/post1500_audit.py tests/test_evaluation.py tests/test_post500_evaluation.py tests/test_post1500_evaluation.py`
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_evaluation tests.test_post1500_evaluation tests.test_train_mechanics` (48 tests).
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile pose_controlnet/evaluation.py pose_controlnet/checkpointing.py scripts/post1500_audit.py tests/test_evaluation.py tests/test_post1500_evaluation.py`.

Coverage: canonical 0..1500 order; local early and valid-local mid resolution;
exact mid/final HF namespaces with separate recovery copies; completion-marker,
checksum, schema/deserialization, and embedded-step validation; rejection of
wrong namespace and nearest-step replacement; fixed-pose reuse; deterministic
fixed-flow/timestep/control calculations, pooled PCK with single/multi and
Danbooru exclusion, telemetry parsing, and no optimizer/backward in audit.

## Important audit finding before host execution

`sample_flow_timestep` currently samples `sigmoid(N(0,1))`, then applies the
resolution shift; it is not a uniform-u sampler. The audit reports this actual
implementation without changing it. Any mismatch with prior verbal
descriptions is a decision gate, not a reason to alter training here.

## Exact GH200 execution plan (do not train)

Set once:

```bash
export UV_CACHE_DIR=/tmp/krea_uv_cache
export MPLCONFIGDIR=/tmp/krea_mpl
cd /home/ubuntu/Krea-2-Pose-ControlNet
```

### A. Checkpoint recovery/status preflight

```bash
uv run python scripts/post1500_audit.py preflight
```

### B. Cheap timestep, telemetry, and source/data-balance audits

```bash
uv run python scripts/post1500_audit.py cheap
```

### C. Deterministic fixed-flow extension and exact merge

```bash
uv run python evaluate.py fixed-flow --steps 600 700 800 900 1000 1100 1200 1300 1400 1500 --verify-repeat
uv run python scripts/post1500_audit.py merge-flow
```

### D. One-sample new fixed-pose smoke at step 1500

```bash
uv run python evaluate.py fixed-pose --steps 1500 --stems real_human_humanart_17000000000288 --spec-name fixed_pose_step1500_smoke_spec.json
```

### E. Authoritative PCK smoke at step 1500

```bash
uv run python scripts/reference_pose_gate.py smoke --step 1500 --device cuda
```

### F. Full fixed-pose generation, new checkpoints only

```bash
uv run python evaluate.py fixed-pose --full-diagnostic --steps 600 700 800 900 1000 1100 1200 1300 1400 1500
```

### G. Full authoritative PCK + unchanged CLIP scoring

```bash
uv run python scripts/post1500_audit.py pck-clip --allow-missing-images
```

`--allow-missing-images` permits the historical compact 0..500 image set; it
reports each checkpoint's actual reference/evaluable sample count rather than
treating absent historical full-set images or unavailable Danbooru as zero.
Remove it only after every canonical step has all 24 images.

### H. Fixed-timestep loss/control audit, grids, plots, report, and export

```bash
uv run python scripts/post1500_audit.py loss-control
uv run python scripts/post1500_audit.py report
```

Products: `evaluation_summary.json`, fixed-flow/CLIP/PCK/coverage/timestep/
control/telemetry plots, `500_vs_800_vs_1100_vs_1500.png`, and terminal table.

## Decision gates after H

Do not resume training automatically. Review independent winners (fixed-flow,
CLIP, pooled PCK .05/.10/.20, single, multi), per-source coverage, loss and
control sensitivity by timestep, actual timestep mass, gradient/throughput/
memory telemetry, and compact versus full-set sample counts. Only then choose
whether to stop, continue to 2000, alter LR/timestep/control exposure, or
optimize runtime; all such changes are outside this milestone and require
explicit authorization.
