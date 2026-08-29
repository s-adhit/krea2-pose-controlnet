# Project handoff

## Current objective and decisions

The bounded milestone is complete: Gate B and Gate C Keypoint R-CNN audit
tooling is implemented, but neither gate has run on GH200. This remains strictly
audit-only. `train.py` is unchanged; production remains flow-matching MSE only.
No `lambda_pose`, dependency, Phase-1 provenance, VAE/training implementation,
or external Keypoint R-CNN PCK evaluator was altered. Do not train or implement
Gate D in this milestone.

Phase-1 remains authoritative. Canonical sidecar SHA:
`dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f`.
The deterministic rebuild remains byte-identical (17,416 records, 15,161
reward available, 2,255 unavailable, 444,235 valid reward joints, seven
reviewed source-OOB masks, 21/21 diagnostic coverage).

Gate A and Gate A.5 are complete. A.5 retained temperature 1.0, selected
Gaussian heatmap KL as the primary audit candidate, retains normalized
coordinate Huber as fallback/diagnostic, and rejects raw pixel-coordinate
Huber for training consideration. Mean RGB gradient norms: Gaussian KL — COCO
2.3343, painting 9.6581, real 7.2826, sculpture 7.1236; normalized Huber —
COCO .002822, painting .101544, real .054005, sculpture .020164.

## Completed implementation

- Added `pose_controlnet/keypoint_critic_audit.py`: dependency-free helpers
  for deterministic noise/seed policy, timestep validation, exact
  `x0_hat = x_t - t*v_hat` reconstruction, metric deltas/aggregation, and
  frozen/Phase-1-geometry contracts.
- Added `scripts/audit_keypoint_critic_vae.py` (Gate B). It uses the exact
  project Qwen/Krea VAE helpers: BF16 `AutoencoderKLQwenImage`, posterior
  sampling, per-channel latent normalization/inverse normalization, and
  in-graph decoded `[-1,1]` to critic `[0,1]` RGB conversion. It audits eight
  deterministic samples/source by default, outputs original/round-trip
  metrics/deltas plus detached RGB L1/MSE, and independently measures finite
  nonzero latent gradients for Gaussian KL and normalized Huber.
- Added `scripts/audit_keypoint_critic_timestep.py` (Gate C). It uses existing
  prepared latent shards and cached text conditioning, project
  channel-concat/model loading, exact `make_flow_pair`, and the exact pinned
  step-1500 parent checkpoint. It verifies SHA
  `6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0`,
  sweeps `.02 .05 .10 .20 .30 .40` by default, reports original/VAE/x0_hat
  metrics and VAE-baseline deltas, and independently measures `dL/dv_hat` and
  `dL/dx0_hat` for both retained candidates. It never accumulates parameter
  gradients or steps an optimizer.
- Added `tests/test_keypoint_critic_audit.py`; updated
  `docs/KEYPOINT_RCNN_CRITIC_AUDIT.md` with the final A.5 result, Gate B/C
  status, contracts, and GH200 commands.

## Tests/checks

- PASS: `PYTHONPATH=. python -m unittest tests.test_keypoint_critic
  tests.test_keypoint_critic_audit` — 22 CPU tests.
- PASS: `python -m py_compile pose_controlnet/keypoint_critic_audit.py
  scripts/audit_keypoint_critic_vae.py scripts/audit_keypoint_critic_timestep.py
  tests/test_keypoint_critic.py tests/test_keypoint_critic_audit.py`.
- PASS: `PYTHONPATH=. python scripts/audit_keypoint_critic_vae.py --help` and
  `PYTHONPATH=. python scripts/audit_keypoint_critic_timestep.py --help`.
- PASS: `git diff --check`.

## Current blockers and exact next action

The only blocker is the required real GH200 execution; the Codex sandbox does
not establish real VAE/critic/checkpoint behavior. Run in this exact order:

1. Gate B:

   ```bash
   PYTHONPATH=. python scripts/audit_keypoint_critic_vae.py \
     --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
     --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
     --samples-per-source 8 --device cuda \
     --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_b_vae.json
   ```

2. Inspect Gate B: finite metrics, finite nonzero `z0` gradients for both
   candidates, stable geometry, and frozen VAE/critic checks.
3. Only if Gate B is acceptable, Gate C:

   ```bash
   PYTHONPATH=. python scripts/audit_keypoint_critic_timestep.py \
     --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
     --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
     --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
     --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
     --split train --samples-per-source 4 --device cuda \
     --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_c_timestep.json
   ```

4. Inspect Gate C quality and gradient exposure together; low `t` naturally
   attenuates `dL/dv_hat` because `d x0_hat / d v_hat = -t`. Gate D is the
   subsequent milestone, not part of this task.
