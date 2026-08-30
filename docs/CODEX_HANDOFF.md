# Project handoff

## Current bounded objective

The definitive Mixed-32 768 training experiment is prepared, but has **not**
been calibrated or trained. It is a fresh coordinate-Huber branch, distinct
from the completed MSE-only baseline. No 500-step training, generation, GPU
evaluation, checkpoint/evaluation mutation, commit, or push occurred.

## Baseline hurdle

`overfit32-mixed-r64-mse-res768` trained at 768 and was evaluated natively.
At step 500 it recorded PCK@.05 `.172269`, PCK@.10 `.358543`, PCK@.20
`.553221`, detection coverage `.866667`, 39 matched people, 6 unmatched
reference people, and CLIP `.333764`.

## Prepared coordinate-Huber contract

- Exact immutable Mixed-32 stem order and sidecar:
  `data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl`.
  The six Danbooru records remain in identity but are explicitly excluded from
  numerical pose reward.
- Training only permits `normalized_coordinate_huber` for a pose-enabled
  capacity run. The audited path is `x0_hat -> autograd VAE decode -> frozen
  fixed-box Keypoint R-CNN -> soft expected normalized coordinates -> SmoothL1`.
- The JSONL source coordinates are reprojected through the exact paired 768
  crop before entering the existing fixed-box critic; it is never rewritten.
- R64/alpha64, 224 LoRA targets, trainable `ControlInputLayer`, frozen Raw
  base, AdamW `1e-4`, warmup 0, microbatch 1, accumulation/effective batch 8,
  500 steps, and checkpoints `0,50,100,200,300,400,500` are unchanged.
- Native evaluation remains the only evaluation geometry. The compact
  comparison table pairs every required checkpoint and never declares a winner.
- Pose exposure remains configurable. Start with the conservative selectable
  window `[.10,.20]`; the command below uses zero forced exposure.

Run identities round-trip the selected lambda, for example
`overfit32-mixed-r64-coord-l2.5e-5-res768`; no lambda has been selected or
hardcoded for the future run.

## Gradient calibration rationale and command

Choose lambda from trainable-gradient norms, not loss scalars. The audit uses
deterministic representative eligible Mixed-32 microbatches at 768, a fresh
model, the actual trainable parameter set, and `torch.autograd.grad`; it does
not construct an optimizer, update parameters, write a cache/checkpoint, or
run generation/evaluation.

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/audit_pose_gradient_balance.py \
  --sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --pose-timestep-min 0.10 --pose-timestep-max 0.20 --timesteps 0.10 0.15 0.20 \
  --samples-per-source 2 --candidate-lambda 1e-6 3e-6 1e-5 3e-5 1e-4 \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/overfit_capacity/audits/mixed32-coordinate-res768-gradient-balance.json
```

Read `aggregate.raw_flow_grad_norm`, `aggregate.raw_pose_grad_norm`, and
`aggregate.implied_lambda.lambda_5pct` / `lambda_10pct`. The candidate panel
is exactly `||lambda * grad L_pose|| / ||grad L_flow||`; select a finite
candidate near `.05`–`.10`, record it, then substitute it below. The utility
does not select a winner or lambda itself.

## Future operator commands (not run)

```bash
cd /home/ubuntu/krea2-pose-controlnet
LAMBDA_POSE='<selected calibration value>'
EXPERIMENT='overfit32-mixed-r64-coord-l<same selected value token>-res768'
PYTHONPATH=. python scripts/run_overfit_capacity.py --stage train \
  --base-experiment mixed32 --resolution 768 \
  --pose-loss normalized_coordinate_huber --lambda-pose "$LAMBDA_POSE" \
  --forced-pose-exposure-probability 0.0 --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
  --pose-target-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl
```

`EXPERIMENT` must use Python's compact float token with leading exponent zero
removed (`2.5e-05` becomes `2.5e-5`); the runner verifies this identity.

Foreground native generation, report, and score-only commands:

```bash
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment "$EXPERIMENT" --stage generate
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment "$EXPERIMENT" --stage report
PYTHONPATH=. python scripts/evaluate_overfit_capacity.py --experiment "$EXPERIMENT" --stage score-only --reference-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl
```

Compact native-only comparison (steps 0/50/100/200/300/400/500):

```bash
PYTHONPATH=. python scripts/summarize_overfit_capacity.py \
  --output-root /lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation \
  --checkpoint-root /lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints \
  --compare overfit32-mixed-r64-mse-res768 "$EXPERIMENT"
```

## Files changed and verification

Changed: `pose_controlnet/capacity_pose.py`, `pose_controlnet/overfit_capacity.py`,
`pose_controlnet/reference_pose.py`, `scripts/audit_pose_gradient_balance.py`,
`scripts/train_overfit_capacity.py`, `scripts/run_overfit_capacity.py`,
`scripts/summarize_overfit_capacity.py`, focused tests, and this handoff.

PASS (CPU/no-network):

```bash
PYTHONPATH=. python -m unittest tests.test_mixed_coordinate_capacity tests.test_capacity_experiment_axes tests.test_overfit_evaluation_resolution tests.test_overfit_mixed_reference_pose tests.test_overfit_capacity -v
PYTHONPATH=. python -m py_compile pose_controlnet/capacity_pose.py pose_controlnet/overfit_capacity.py pose_controlnet/reference_pose.py scripts/audit_pose_gradient_balance.py scripts/train_overfit_capacity.py scripts/run_overfit_capacity.py scripts/summarize_overfit_capacity.py tests/test_mixed_coordinate_capacity.py
PYTHONPATH=. python scripts/train_overfit_capacity.py --preflight --base-experiment mixed32 --resolution 768 --pose-loss normalized_coordinate_huber --lambda-pose 2.5e-5 --forced-pose-exposure-probability 0.0 --pose-timestep-min 0.10 --pose-timestep-max 0.20 --pose-target-sidecar data/manifests/overfit_capacity_reference_pose/overfit32-mixed-r64-mse.jsonl --no-wandb
git diff --check
```

Next action: run only the gradient calibration command on the GH200, inspect
the 5–10% gradient-ratio panel, and explicitly choose the lambda before any
training command.
