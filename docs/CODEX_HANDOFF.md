# Project handoff

## Current objective and enforced decisions

The completed bounded milestone implements tooling only: Gate D is a
read-only actual-trainable gradient calibration, and Gate E is a separately
invoked, bounded pose-reward smoke continuation. `train.py` remains unchanged
and normal production training remains flow-matching MSE only. Gate E's
selected invocation policy is parent step 1500, Gaussian heatmap KL,
`lambda_pose=2e-5`, inclusive pose-timestep window `[0.10, 0.20]`,
microbatch size 2, accumulation 16, 200 steps, and saves every 50 steps.
Do not launch either tool from the Codex sandbox.

Phase-1 provenance remains authoritative and unchanged. Canonical sidecar SHA:
`dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f`.
The exact required parent is
`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`
with SHA256 `6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0`.

## Verified gates and decisions

- Gate A and A.5: PASS. Temperature 1.0; primary critic loss is Gaussian
  heatmap KL; normalized-coordinate Huber is diagnostic/fallback only; raw
  pixel-coordinate Huber is rejected.
- Gate B: PASS. Exact VAE decode, geometry, gradients, and frozen-boundary
  audit completed on GH200.
- Gate C: PASS. Exact convention is `x_t=t*noise+(1-t)*x0`,
  `v_target=noise-x0`, `x0_hat=x_t-t*v_hat`. Primary pose timestep `.20`,
  secondary `.10`, stress/upper-bound diagnostic `.30`; `.40` is held for
  strong gradient escalation/outliers. `.02`/`.05` are not primary candidates
  because `d x0_hat / d v_hat = -t` attenuates exposure.
- Gate D: IMPLEMENTED, GH200 RUN REQUIRED. It remains audit-only: no optimizer
  construction or update. It validates exact rank-64/alpha-64 LoRA plus
  ControlInput trainables, independent equal-state flow/Gaussian-KL graphs,
  norms/dot/cosine, candidate lambda panel, combined gradients, and
  source/timestep statistics.
- Gate E: TOOLING IMPLEMENTED, RUNTIME BUG FIXED, BUT NOT PASS. The smoke
  loader now uses the same `PreparedLatentShardDataset[index]` semantics as
  `train.py`, then collates that exact item list and preserves aligned stems.
  Epoch rollover, batch-position advancement, microbatch planning, and
  accumulation scaling match the production continuation loop. It retains
  production flow sampling/MSE and only decodes/runs the critic for
  `pose_reward_available=true` active samples. A real GH200 run remains
  required; do not mark Gate E PASS.

## Files changed this session

- `pose_controlnet/pose_reward_tools.py`
- `scripts/audit_pose_gradient_balance.py`
- `scripts/train_pose_reward_smoke.py`
- `tests/test_pose_reward_tools.py`
- `docs/KEYPOINT_RCNN_CRITIC_AUDIT.md`
- `docs/CODEX_HANDOFF.md`

This runtime-fix session modified only `scripts/train_pose_reward_smoke.py`,
`tests/test_pose_reward_tools.py`, and this handoff.

Existing untracked Gate B/C audit files were preserved; do not overwrite them.

## Tests/checks

- PASS: `PYTHONPATH=. python -m unittest tests.test_keypoint_critic
  tests.test_keypoint_critic_audit tests.test_pose_reward_tools` — 32 CPU
  tests. Coverage includes incremental norm/dot/cosine, zero safety, lambda
  formula, combined gradients, trainable/frozen selection, deterministic
  state, active/availability masks, inactive pose-graph skipping, invalid-joint
  masking, inactive/active total loss, required lambda/output isolation, no
  Gate-D optimizer step, existing-loader checkpoint schema, and the Gate-E
  index-only dataset microbatch fetch/collation/stem path.
- PASS: `python -m py_compile pose_controlnet/pose_reward_tools.py
  scripts/audit_pose_gradient_balance.py scripts/train_pose_reward_smoke.py
  tests/test_pose_reward_tools.py`.
- PASS: `PYTHONPATH=. python scripts/audit_pose_gradient_balance.py --help`;
  `PYTHONPATH=. python scripts/train_pose_reward_smoke.py --help`.
- PASS: `git diff --check`.

## Blockers and exact GH200 run order

The only blocker is the required Gate-D real GH200 output and its human review.
Do not mark Gate D PASS or launch Gate E automatically.

1. Run Gate D:

   ```bash
   PYTHONPATH=. python scripts/audit_pose_gradient_balance.py \
     --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
     --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
     --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
     --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
     --split train --samples-per-source 4 --timesteps .10 .20 .30 --device cuda \
     --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_d_gradient_balance.json
   ```

2. Inspect Gate D, especially per-source/timestep ratio/cosine and sculpture
   outliers.
3. Gate-E policy is selected as above. Only after the required Gate-D review,
   launch Gate E with this corrected command (the isolated run directory must
   not already exist):

   ```bash
   PYTHONPATH=. python scripts/train_pose_reward_smoke.py \
     --parent-checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt \
     --expected-parent-sha256 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0 \
     --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
     --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
     --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
     --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
     --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints \
     --run-name gate-e-parent1500-heatmap-kl-lambda2e-5-t010-020 \
     --lambda-pose 2e-5 --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
     --max-steps 200 --save-every 50 \
     --microbatch-size 2 --gradient-accumulation-steps 16 \
     --device cuda
   ```

4. Inspect the first 50/100/200-step checkpoints and local metrics before any
   longer branch.
