# Project handoff

## Current objective

The ready next experiment is a matched 500-update finishing A/B from:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt
```

Step 4000 is the preferred balanced branch point: it retains the observed pose
gain without entering the less-settled 4250+ tail. No training, real
evaluation, image generation, network activity, upload, commit, or push
occurred while implementing this experiment.

Both branches preserve dynamic-768; frozen Raw; ControlInputLayer; R64/alpha64
LoRA; 224 targets; flow MSE plus unchanged normalized-coordinate Huber pose
term; natural pose window `[.10,.20]`; forced exposure 0; AdamW
`(.9,.99)`/zero decay/max norm 1; microbatch 1/accumulation 32; BF16; four
persistent pinned workers/prefetch 4; and checkpointing/compile/fused AdamW
off. Parent Adam moments, RNG, timestep generator, and deterministic data
position are restored; there is no new warmup.

## Exact A/B contract

Both run global updates `4001..4500` and save/may mirror at `4100 4200 4300
4400 4500`. Let `i=s-4001` and `f=i/499` for optimizer update `s`.

```text
lr(s) = 5e-6 + (2e-5 - 5e-6) * 0.5 * (1 + cos(pi * f))
```

This is exactly `2e-5` at 4001 and `5e-6` at 4500; it replaces only the parent
scheduler. Branch A (`finish-control`) uses `lambda_pose(s)=.04`. Branch B
(`finish-pose-anneal`) uses `lambda_pose(s)=.04*(1-f)` in the existing
coordinate-loss call: `.04` at 4001, `0` at 4500, and
`.032064/.024048/.016032/.008016` at 4100/4200/4300/4400.

The LR and lambda states are checkpointed, so `--resume auto` restores the
next update exactly. Every finishing checkpoint and `run_metadata.json`
persists parent absolute path/SHA-256/run/step, branch type, schedules and
endpoints, continuation length, global numbering, and immutable
science/artifact identities. Finishing accepts only the named step-4000
cooldown artifact with matching 3k→5k provenance. It starts a new W&B run;
only exact resume of that branch may reuse its own id. Existing ordinary and
cooldown exact-resume checks remain fail-closed.

## Foreground launches (do not run from Codex)

Branch A:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py \
  --run-name pose-control-finish-control-4000-to4500 \
  --max-steps 4500 --save-every 100 --diagnostics-every 50 \
  --continue-from /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt \
  --continue-from-step 4000 --lr-schedule cosine --lr-start 2e-5 --lr-final 5e-6 \
  --pose-lambda-schedule constant \
  --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-projects \
  --wandb-name pose-control-finish-control-4000-to4500 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 100
```

Branch B:

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_production.py \
  --run-name pose-control-finish-anneal-4000-to4500 \
  --max-steps 4500 --save-every 100 --diagnostics-every 50 \
  --continue-from /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt \
  --continue-from-step 4000 --lr-schedule cosine --lr-start 2e-5 --lr-final 5e-6 \
  --pose-lambda-schedule linear --pose-lambda-final 0 \
  --wandb --wandb-project Krea-2-PoseControl-Lora --wandb-entity adhit-projects \
  --wandb-name pose-control-finish-anneal-4000-to4500 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-mirror-every-steps 100
```

For a restart add `--resume auto` to the identical branch command; do not use
it on a first launch. Remove both `--hf-*` options to keep HF disabled.

## Evaluation plan only (do not run from Codex)

Run this for each `<branch>` (`pose-control-finish-control-4000-to4500` and
`pose-control-finish-anneal-4000-to4500`):

```bash
cd /home/ubuntu/krea2-pose-controlnet
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python scripts/evaluate_production_milestones.py evaluate \
  --checkpoint-root /lambda/nfs/adhit/krea2-pose/checkpoints/<branch> \
  --output-root /lambda/nfs/adhit/krea2-pose/evaluation/<branch> \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --turbo-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors \
  --reference-sidecar /home/ubuntu/krea2-pose-controlnet/data/manifests/diagnostic_reference_pose.json \
  --diagnostic-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/diagnostic_val.jsonl \
  --canonical-reference-spec /home/ubuntu/krea2-pose-controlnet/docs/evaluation/turbo-8step-cfg0/turbo_spec.json \
  --steps 4100 4200 4300 4400 4500 --modes native dynamic-768
```

Dynamic-768 contact sheet for each branch:

```bash
PYTHONPATH=. python scripts/evaluate_production_milestones.py contact-sheet \
  --evaluation-root /lambda/nfs/adhit/krea2-pose/evaluation/<branch> \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --steps 4100 4200 4300 4400 4500 --modes dynamic-768 \
  --output-dir docs/evaluation/<branch>
```

Use `--modes native dynamic-768` for full sheets. Compare against parent 4000:
native PCK `.548544/.656553/.739078`, coverage `.884615`, CLIP `.327484`;
dynamic-768 PCK `.495146/.649272/.734223`, coverage `.865385`, CLIP `.331195`.
Anneal improving CLIP/visual quality with retained pose supports anneal;
substantial pose loss favors control or parent; both degrading leaves parent
4000 preferred; unexpected control improvement warrants studying the lower-LR
effect. Do not add a flow-only branch.

## Verification this session

CPU/no-network production tests pass: `PYTHONPATH=. python -m unittest
tests.test_production_training -v`. Coverage includes both branches, accepted
and rejected parent step, parent/provenance/artifact failure, preserved Adam
moments, no warmup, exact monotonic LR/lambda endpoints and intermediate
anneal values, 4100..4500 checkpoint/HF cadence, scheduler resume position,
new W&B identities, and ordinary/cooldown resume regressions.

Files changed: `pose_controlnet/production_training.py`,
`tests/test_production_training.py`, and this handoff. Next action: review,
then only with authorization launch the two foreground branches.
