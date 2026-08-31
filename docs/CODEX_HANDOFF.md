# Project handoff

## Current objective and status

`pose-control-production-3000` now has an explicit provenance-safe cooldown
continuation path. It is a new scientific run, never an exact-resume disguise.
No cooldown training, real GPU evaluation, real image generation, network
action, checkpoint upload, commit, or push occurred in this session.

Required parent checkpoint:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000/step_003000.pt
```

The new run is `pose-control-production-cooldown-3000-to5000`, in its own
directory and its own W&B run. Every new checkpoint and `run_metadata.json`
persist the parent absolute path/SHA-256/run name/global step plus the
continuation schedule.

## Locked cooldown science

All parent science remains locked: dynamic 768 policy; frozen Raw backbone;
ControlInputLayer; R64/alpha64 LoRA; 224 target topology; flow MSE +
`normalized_coordinate_huber`; `lambda_pose=0.04`; natural pose window
`[0.10,0.20]`; forced exposure 0; AdamW `(0.9,0.99)`, zero decay, max norm 1;
microbatch 1 / accumulation 32; BF16; four persistent pinned workers with
prefetch 4; no checkpointing, compile, or fused AdamW; seed/RNG/data position.

Only the scheduler changes. It runs global steps 3001..5000 (2,000 updates),
with no warmup and cosine `1e-4` to `1e-5`. For update `s`, `i=s-3001`:

```text
lr = 1e-5 + (1e-4 - 1e-5) * 0.5 * (1 + cos(pi * i / 1999))
```

Thus step 3001 uses `1e-4` and step 5000 uses `1e-5` exactly. Global checkpoint
names are `3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000`.

The launcher requires `--continue-from`, `--lr-schedule cosine`, and
`--lr-final 1e-5` together. It rejects any parent other than step 3000,
immutable science/artifact identity mismatch, absent Adam state, or malformed
deterministic position. It restores weights, AdamW moments, Python/NumPy/Torch
RNG, timestep-generator state, and data position, replacing only the scheduler.
Ordinary exact `--resume` retains its prior fail-closed behavior. Restart this
continuation with the same flags plus `--resume auto`; it can resume the new
continuation W&B run but never the parent W&B run.

W&B: project `Krea-2-PoseControl-Lora`, entity `adhit-projects`, name
`pose-control-production-cooldown-3000-to5000`. HF is optional; the command
below uses private `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`, mirror
cadence 500 global steps: `3500, 4000, 4500, 5000`.

## Exact foreground cooldown command (do not run from Codex)

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py \
  --run-name pose-control-production-cooldown-3000-to5000 \
  --max-steps 5000 --save-every 250 --diagnostics-every 50 \
  --continue-from /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000/step_003000.pt \
  --continue-from-step 3000 --lr-schedule cosine --lr-final 1e-5 \
  --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-projects \
  --wandb-name pose-control-production-cooldown-3000-to5000 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 500
```

## Completed 3k milestone results

Native (`PCK@.05/.10/.20`, coverage, CLIP):

```text
500:  .057039 / .183252 / .419903, .903846, .337271
1000: .055825 / .157767 / .405340, .903846, .337827
1500: .057039 / .184466 / .459951, .903846, .339513
2000: .047330 / .160194 / .422330, .923077, .339663
2500: .063107 / .188107 / .468447, .923077, .337012
3000: .425971 / .605583 / .713592, .884615, .333933
```

Dynamic-768 at 3000: PCK `.356796 / .557039 / .691748`, coverage `.846154`,
CLIP `.334231`. The major pose transition appears at step 3000 in both modes;
the cooldown curve should verify and explain it. Hypothesis: lower-LR
continuation may preserve/improve pose with stable/better image quality, trade
image quality for pose, or degrade both.

## Evaluation and dynamic contact sheets (do not run from Codex)

The harness accepts explicit positive checkpoint lists; historical six-step
defaults are unchanged. Evaluate cooldown checkpoints in both modes:

```bash
cd /home/ubuntu/krea2-pose-controlnet
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python scripts/evaluate_production_milestones.py evaluate \
  --checkpoint-root /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000 \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/pose-control-production-cooldown-3000-to5000 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --turbo-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors \
  --reference-sidecar /home/ubuntu/krea2-pose-controlnet/data/manifests/diagnostic_reference_pose.json \
  --diagnostic-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/diagnostic_val.jsonl \
  --canonical-reference-spec /home/ubuntu/krea2-pose-controlnet/docs/evaluation/turbo-8step-cfg0/turbo_spec.json \
  --steps 3500 4000 4500 5000 --modes native dynamic-768
```

Render source-preserving continuous sheets (no metric/model work):

```bash
PYTHONPATH=. python scripts/evaluate_production_milestones.py contact-sheet \
  --evaluation-root /lambda/nfs/adhit/krea2-pose/evaluation/pose-control-production-cooldown-3000-to5000 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --steps 3500 4000 4500 5000 --modes native dynamic-768 \
  --output-dir docs/evaluation/pose-control-production-cooldown-3000-to5000
```

The command reuses the existing renderer and produces
`native_full_contact_sheet.png` and `dynamic768_full_contact_sheet.png`. Stem
order comes from `generation_results.json`, must agree at every requested
step/mode, and missing source/generated images fail loudly.

## Verification and next decision

CPU/no-network PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_production_training tests.test_production_milestone_evaluation -v
PYTHONPATH=. python -m py_compile pose_controlnet/production_training.py pose_controlnet/production_milestone_evaluation.py scripts/evaluate_production_milestones.py tests/test_production_training.py tests/test_production_milestone_evaluation.py
```

Tests cover Adam-moment restoration; no second warmup; cosine endpoints and
monotonicity; global checkpoint/HF scheduling; parent provenance/identity
failure; checkpoint reload; new W&B identity; exact-resume regressions;
arbitrary contact-sheet steps/modes; stem order; source pair resolution;
isolated output names; and missing image failure.

After evaluating 3000 baseline and 3500/4000/4500/5000 in native and
dynamic-768, choose among: A stable/better pose with stable CLIP; B preserved
pose with better image/CLIP; C pose/image tradeoff; D both degrade (3000 was
optimal). Do not add pose-loss annealing or flow-only finishing first.

Files changed this session: `pose_controlnet/production_training.py`,
`pose_controlnet/production_milestone_evaluation.py`,
`scripts/evaluate_production_milestones.py`, their two test files, and this
handoff.
