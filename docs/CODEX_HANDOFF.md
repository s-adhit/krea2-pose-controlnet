# Phase 1 handoff

## Current objective and status

Completed a read-only feasibility audit for a future differentiable
ControlNet++-style pose-consistency reward. No training, generation, resume,
checkpoint/data mutation, network download, commit, or push occurred. The
intended parent remains exactly the clean LR-only checkpoint
`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`
(2.5 GB); the ControlInput-LR2x continuation is explicitly out of scope.

## Verified current mechanics

- `make_flow_pair` implements `x_t = t*noise + (1-t)*x0` and target
  `v = noise-x0`; `train.py:_flow_loss` MSE-trains tokenized model velocity.
  Therefore the correct reconstructed normalized clean latent is
  `x0_hat = x_t - t*v_hat`, after unpatchifying model output with the same
  `einops` layout used by `sample_eval_image`.
- The Qwen Image VAE is `diffusers 0.40.0` `AutoencoderKLQwenImage`. Existing
  `decode_normalized_latents` is deliberately inference-only, but the wrapped
  VAE `decode` has no `no_grad`; a separate grad-enabled decode helper can
  freeze VAE parameters yet pass gradients from decoded RGB to `x0_hat`.
- The complete 17,416-sample shard audit found one schema only:
  `stem,file_name,text,split,bucket,source_size,resized_size,crop_box,
  image_latent,control_latent`. Geometry is present for all records, but no
  raw keypoints, confidence/visibility, boxes, person grouping, DWPose,
  RTMPose, OpenPose JSON, or other pose metadata exists.
- The local HF snapshot contains exactly 17,495 JPG + 17,495 PNG and only its
  four JSONL metadata/manifest files. Diagnostic-only
  `data/manifests/diagnostic_reference_pose.json` has 21 annotated references
  (4 COCO/17 Human-Art) and 3 unavailable Danbooru entries; it is not a
  training target source.
- No DWPose, RTMPose, MMPose, MMCV, or SimCC package/config/checkpoint is
  installed. The cached torchvision 0.22 COCO Keypoint R-CNN checkpoint is
  available and its source exposes pre-argmax `keypoint_logits`, but its
  current evaluator wrapper is inference-only and it is a heatmap—not
  SimCC—fallback. CUDA is unavailable in the Codex audit shell, so its actual
  host gradient path remains to be probed later.

## Feasibility / blockers

The differentiable VAE and `x0_hat` portions are feasible. The preferred
fixed-box raw-SimCC reward is blocked before implementation by two missing
artifacts: (1) an every-sample, geometry-aligned per-person target sidecar and
(2) a specifically approved, locally installed top-down RTMPose/DWPose
implementation/checkpoint exposing raw SimCC logits. Raster controls alone
cannot safely reconstruct people, visibility, or joint coordinates.

Do not repurpose authoritative Keypoint-RCNN/PCK evaluation into training.
The original LR-only run's measured step-1500 telemetry was 28.49 sec/update,
92.93e9 peak allocated bytes, and 100.47e9 reserved bytes, leaving little
headroom for a decoded-VAE plus frozen reward graph; profile microbatch 1
before any reward experiment.

## Required implementation-phase scope

Add a versioned read-only pose-target sidecar/cache (do not rewrite immutable
manifests or existing latent shards), a target builder once an authoritative
source/model is selected, a grad-enabled VAE decode helper, a new reward
module, and a step-1500-only training branch/audits. Likely files:
`train.py`, `pose_controlnet/config.py`, `pose_controlnet/data.py`,
`pose_controlnet/vae_preprocessing.py`, new `pose_controlnet/pose_reward.py`,
new target-builder/audit scripts, and focused tests. `checkpointing.py` need
not change if the reward/cache identity is stored in `TrainConfig`, which is
already checkpointed.

## Commands run this session

- PASS (read-only): full shard field audit: 16,503 train + 889 val + 24
  diagnostic; all exactly the schema above.
- PASS (read-only): local package/asset inspection; no RTMPose/DWPose/SimCC,
  cached Keypoint R-CNN weights found.
- BLOCKED AS EXPECTED: synthetic CUDA-only fixed-box Keypoint-RCNN autograd
  probe; Codex sandbox has no CUDA. This does not invalidate host GH200.
- PASS (read-only): flow, VAE decode, preprocessing, dataset, evaluator,
  checkpoint/resume, metadata, and telemetry inspection.

## Exact next action

Obtain/approve an authoritative all-sample pose-target provenance and a
specific top-down raw-SimCC pose checkpoint, then implement only the target
cache plus a host-only `x0_hat`/VAE/reward gradient audit from the exact
LR-only step-1500 parent. Do not train until that audit is green and lambda is
calibrated from measured separate flow/reward gradients.
