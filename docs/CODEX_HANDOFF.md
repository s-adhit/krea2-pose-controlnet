# Project handoff

## Current bounded objective

The overfit-capacity harness now exposes only two experiment axes: paired
resolution policy and optional audited auxiliary pose loss. No training,
generation, GPU evaluation, checkpoint/output mutation, commit, or push was
performed in this session.

## Completed mixed native result and comparison contract

- The completed native flow-MSE experiment remains
  `overfit32-mixed-r64-mse`, with the immutable 32-stem order: 6 COCO, 7
  Human-Art Painting, 7 Human-Art Real Human, 6 Sculpture, 6 Danbooru.
- Dynamic Mixed-32 namespace identity is derived from the scientific config:
  `overfit32-mixed-r64-mse-resnative`,
  `overfit32-mixed-r64-mse-res768`,
  `overfit32-mixed-r64-coord-l1e5-res768`, or `...-kl-...`.
  Checkpoint, evaluation, metrics, cache, and W&B run names use that identity.
- Native means the verified persisted latent geometry is read verbatim. It is
  never recomputed or silently resampled.
- `768` rebuilds the paired RGB/control pixels and VAE latents in an isolated,
  deterministic per-experiment cache. It cannot use native latents. The exact
  buckets, all 64-pixel aligned, are: 768x768, 704x896, 896x704, 640x960,
  960x640, 576x1024, 1024x576, 512x1152, 1152x512.
- The cache validates its full scientific config, exact ordered Mixed-32 stems,
  per-stem crop/bucket/latent geometry, finite equal RGB/control latents, and
  VAE factor-eight shapes. `resolution_manifest.json` records source dimensions,
  native bucket, requested bucket, resize/crop geometry, and latent dimensions.

## Pose-loss contract

- Default is exactly flow-MSE only: `--pose-loss none --lambda-pose 0`, zero
  forced exposure, and no auxiliary timestep window.
- Selectable modes are `gaussian_heatmap_kl` and
  `normalized_coordinate_huber`. Enabled pose loss requires positive lambda,
  an explicit `[min,max]` window inside `(0,1)`, optional forced exposure in
  `[0,1]`, and an immutable authoritative sidecar.
- The capacity trainer reuses the audited x0-hat -> autograd VAE decode ->
  frozen fixed-box Keypoint R-CNN -> raw logits path. It remaps authoritative
  source labels through the requested geometry, masks invalid/OOB joints, and
  rejects missing eligible targets. Danbooru is explicitly unavailable and
  never receives numerical pose reward.
- Rank/alpha 64/64, 224 LoRA targets, frozen Raw base, ControlInputLayer,
  AdamW LR 1e-4, warmup 0, microbatch 1, accumulation 8/effective batch 8,
  500 steps, and checkpoints 0/50/100/200/300/400/500 remain fixed.

## Files changed this session

- `pose_controlnet/overfit_capacity.py`
- `pose_controlnet/capacity_resolution.py`
- `pose_controlnet/paired_preprocessing.py`
- `scripts/train_overfit_capacity.py`
- `scripts/run_overfit_capacity.py`
- `scripts/evaluate_overfit_capacity.py`
- `tests/test_capacity_experiment_axes.py`
- `docs/CODEX_HANDOFF.md`

## Commands and verification

PASS before final follow-up validation:

```bash
python -m py_compile pose_controlnet/overfit_capacity.py pose_controlnet/capacity_resolution.py pose_controlnet/paired_preprocessing.py scripts/train_overfit_capacity.py scripts/run_overfit_capacity.py scripts/evaluate_overfit_capacity.py
python -m unittest tests.test_overfit_capacity -v
```

Run on GH200 (not from Codex) for 768 MSE-only:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/run_overfit_capacity.py --base-experiment mixed32 --resolution 768 --pose-loss none --stage train
```

Pose-loss template (supply the authoritative sidecar):

```bash
PYTHONPATH=. python scripts/run_overfit_capacity.py --base-experiment mixed32 --resolution 768 --pose-loss normalized_coordinate_huber --lambda-pose 1e-5 --forced-pose-exposure-probability 0.1 --pose-timestep-min 0.1 --pose-timestep-max 0.2 --pose-target-sidecar <immutable-sidecar> --stage train
```

Foreground generation/report (not from Codex):

```bash
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse-res768 --resolution 768 --stage generate
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment overfit32-mixed-r64-mse-res768 --resolution 768 --stage report
PYTHONPATH=. python scripts/summarize_overfit_capacity.py --output-root /lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation --checkpoint-root /lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints overfit32-mixed-r64-mse overfit32-mixed-r64-mse-res768 overfit32-mixed-r64-coord-l1e5-res768
```

The 768 buckets contain roughly 57--63% of the native pixels, so a rough
throughput expectation is materially faster than the observed 7.2--7.4 sec
native optimizer step but not linearly proportional. After 20 steps, compute
measured ETA from metrics JSONL with:

```bash
python -c "import json,sys; r=[json.loads(x) for x in open(sys.argv[1])][-20:]; s=sum(x['sec_per_step'] for x in r)/len(r); print({'mean_sec_step':s,'eta_remaining_hours':s*(500-r[-1]['global_step'])/3600})" <checkpoint-root>/<experiment>/metrics.jsonl
```

## Exact next recommended action

Run the final CPU test suite and static checks, review the uncommitted diff,
then have an operator run the 768 MSE-only GH200 command above. Do not start
the configurable pose-loss experiment until its immutable authoritative target
sidecar is specified.
