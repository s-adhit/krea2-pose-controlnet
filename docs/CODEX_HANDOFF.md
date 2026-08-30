# Project handoff

## Current objective

The deterministic 32-sample fresh-R64 capacity/overfit harness is implemented
and ready for an operator-run GH200 experiment. Codex did **not** train,
generate, evaluate, mutate checkpoints, commit, or push.

## Scientific/architecture contract

This is TRAINING-SET OVERFIT, not generalization: each run trains and evaluates
the same 32 immutable train samples. It tests the existing frozen Krea-2 Raw
base, fresh rank-64 Pose-Control LoRA, channel-concatenated ControlInputLayer,
and flow-MSE only. No resume/trained LoRA state, ControlNet, pose reward,
critic, KL, coordinate loss, warmup, new regularization, rank/target change,
or architecture change is allowed.

`scripts/train_overfit_capacity.py` is a thin harness over `train.py`'s fresh
builder, `_flow_loss`, AdamW, scheduler, cached-text dropout, telemetry, and
atomic checkpoint primitive; production `train.py` is unchanged.

## Manifests

`configs/overfit_capacity/manifests/` contains six exact 32-unique immutable
train subsets:

- COCO, Painting, Real Human, Sculpture: 16 one-person, 8 two-person, 4
  three/four-person, 4 five-plus-person records from authoritative metadata.
- Danbooru: 32 deterministic immutable train records; PCK is explicitly
  unavailable because there are no authoritative targets.
- Mixed: 6 COCO, 7 Painting, 7 Real Human, 6 Sculpture, 6 Danbooru. Every
  mixed stem is reused from its homogeneous manifest.

`scripts/build_overfit_capacity_manifests.py` reproduces selection; persisted
manifests define the experiment rather than filesystem order.

## Exact configuration

- Krea-2 Raw fresh base; rank/alpha 64/64; existing 28 × 8 targets
  `attn.wq/wk/wv/wo/gate`, `mlp.gate/up/down`; `first.weight/bias` only.
- Model: 13,035,162,188 total params; 215,488,512 trainable (1.6531326%).
  LoRA 214,695,936; ControlInputLayer 792,576; 224 LoRA modules.
- AdamW betas `(0.9,0.99)`, WD 0, LR `1e-4` constant from step 1. This is the
  authoritative fresh `TrainConfig` R64 LR; `5e-5` is continuation-only.
- Warmup 0; microbatch 1 × accumulation 8 = effective batch 8; steps
  0/50/100/200/300/400/500. At 50…500: 400…4000 presentations or
  12.5/25/50/75/100/125 dataset-equivalent passes.
- Existing deterministic 10% cached-text caption dropout remains; no new
  spatial augmentation; control dropout 0.

Checkpoints go to
`/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/<experiment>/`.
Evaluation uses exact same 32 records, Turbo 8 steps/CFG 0/mu 1.15, fixed
per-stem seeds, existing VAE/PCK/CLIP, and target RGB + pose control contact
sheets. It has no optimizer/backward path.

## GH200 commands — do not run from Codex

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/run_overfit_capacity.py overfit32-coco-r64-mse
PYTHONPATH=. python scripts/run_overfit_capacity.py overfit32-coco-r64-mse overfit32-danbooru-r64-mse overfit32-mixed-r64-mse
PYTHONPATH=. python scripts/run_overfit_capacity.py $(PYTHONPATH=. python -c 'from pose_controlnet.overfit_capacity import OVERFIT_EXPERIMENTS; print(*OVERFIT_EXPERIMENTS)')
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-coco-r64-mse --stage all
```

Outputs: per-stem `control.png`, `target.png`, seven checkpoint PNGs,
`training_set_overfit_metrics.json`, selection/full contact sheets,
`overfit_summary.json`, and suite-level `capacity_comparison_summary.json`.

Eight forward/backward microbatches occur per optimizer step, nominally ~4×
fewer than accumulation 32. Until a 10–20-step measurement, treat the old
30 sec/update only as a conservative ceiling: <=4.2 h train per 500-step run,
plus 224 Turbo generations and PCK/CLIP scoring. Measure actual throughput:

```bash
python - <<'PY'
import json, statistics
p='/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints/overfit32-coco-r64-mse/metrics.jsonl'
r=[json.loads(x) for x in open(p) if 'sec_per_step' in x][-20:]
s=statistics.mean(x['sec_per_step'] for x in r); print({'mean_sec_per_step':s,'remaining_seconds':s*(500-r[-1]['step'])})
PY
```

## Checks

- PASS: py_compile all new capacity files.
- PASS: `tests.test_overfit_capacity`, `test_train_mechanics`,
  `test_turbo_evaluation`, and `test_post1500_evaluation` (68 tests).
- PASS: CPU-only COCO `--preflight` (no model construction/output writes).
- PASS: `git diff --check` before handoff update.

Changed: manifests/helpers, selector, train/evaluate/runner/summary scripts,
focused tests, and this handoff. Existing unrelated untracked critic/evaluation
work remains untouched.
