# Fixed-box torchvision Keypoint R-CNN critic audit

Status: **Gate A (real-RGB critic feasibility) PASS; Gate A.5 loss-gradient
comparison PASS; Gate B PASS; Gate C PASS. Gate D gradient-calibration tooling
is implemented but requires a real GH200 run. Gate E smoke tooling is
implemented but blocked on Gate-D review and an explicit operator policy.**
This is an audit-only module. It does not modify `train.py`, the Phase-1
flow-matching objective, provenance, or the external Keypoint R-CNN PCK
evaluator.

## Pinned estimator and torchvision 0.22 path

The external evaluator in `pose_controlnet/post500_evaluation.py` uses
`torchvision.models.detection.keypointrcnn_resnet50_fpn` with
`KeypointRCNN_ResNet50_FPN_Weights.DEFAULT`. In torchvision 0.22.0 that is
`KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`; the audit critic uses that exact
weight selection too. Its 17 COCO keypoint channels are, in order: nose,
left/right eye, left/right ear, left/right shoulder, left/right elbow,
left/right wrist, left/right hip, left/right knee, and left/right ankle. This
is the Phase-1 `coco17` order.

The exact source-level fixed-box tensor path is:

```text
KeypointRCNN.transform (GeneralizedRCNNTransform.forward)
  -> model.backbone(image_list.tensors)
  -> model.roi_heads.keypoint_roi_pool(features, fixed_model_boxes, image_list.image_sizes)
  -> model.roi_heads.keypoint_head(...)
  -> model.roi_heads.keypoint_predictor(...)
```

`KeypointRCNN.__init__` creates a `MultiScaleRoIAlign` with FPN feature maps
`["0", "1", "2", "3"]`, `output_size=14`, and `sampling_ratio=2` when no
custom keypoint pool is supplied. `KeypointRCNNHeads` preserves the 14×14
spatial size. `KeypointRCNNPredictor` applies a stride-2, kernel-4 transpose
convolution (14→28) then 2× bilinear interpolation (28→56). Therefore the
raw, pre-decode output is exactly **`[total_fixed_people, 17, 56, 56]`**.

The transform expects floating RGB in `[0, 1]`, normalizes it with the
torchvision detection default ImageNet mean/std `[0.485, 0.456, 0.406]` /
`[0.229, 0.224, 0.225]`, then in evaluation resizes the shorter side to 800
subject to `max_size=1333`, and zero-pads the batch to divisibility by 32.
Fixed training-frame boxes are passed as transform targets solely to use the
same torchvision `resize_boxes` geometry. ROIAlign receives those transformed
boxes; soft heatmap coordinates are mapped through the original authoritative
training-frame ROI boxes for direct comparison to Phase-1 coordinates.

## Explicit detector bypass and reward boundary

`FixedBoxKeypointRCNNCritic` calls the path above directly. It never calls
`GeneralizedRCNN.forward`, `model.rpn`, `roi_heads.box_roi_pool`,
`roi_heads.box_head`, `roi_heads.box_predictor`, or
`RoIHeads.postprocess_detections`. Thus the graph contains no RPN proposals,
detector filtering/scores/thresholds, box regression, NMS, or detector boxes.
The only boxes are authoritative Phase-1 data.

All critic parameters are `requires_grad_(False)` and the wrapper forces eval
mode, but it intentionally does not use `torch.no_grad()` around the tensor
path: gradients must reach the RGB input. The raw heatmaps are never decoded
with torchvision's `heatmaps_to_keypoints` in the reward path. Argmax and PCK
exist only in the detached diagnostic helper.

## Differentiable coordinate geometry and candidate losses

For each `[56, 56]` joint heatmap, `spatial_softmax(logits / temperature)` is
formed, followed by expected heatmap `(x, y)`. A cell is mapped by its center:

```text
x_training = x0 + (x_heatmap + 0.5) * (x1 - x0) / heatmap_width
y_training = y0 + (y_heatmap + 0.5) * (y1 - y0) / heatmap_height
```

where `(x0, y0, x1, y1)` is the supplied authoritative training-frame box.
The default temperature is 1.0. No checkpoint-specific calibration was found
in the pretrained head, so this default is deliberately not tuned by
assumption.

Three independent audit candidates are exposed:

- `masked_coordinate_huber` / `coordinate_huber`: raw training-pixel
  per-coordinate Huber, averaged only over valid person/joint pairs.
- `normalized_coordinate_huber`: applies `(x - x0) / max(x1 - x0, eps)` and
  `(y - y0) / max(y1 - y0, eps)` independently to both the soft prediction
  and the authoritative target, then applies the same validity-masked Huber
  reduction. The authoritative target coordinates are not modified.
- `gaussian_heatmap_target` plus `masked_gaussian_heatmap_kl` /
  `gaussian_heatmap_kl`: a normalized Gaussian in raw heatmap coordinates and
  target-KL-predicted distribution loss, also averaged only over valid pairs.

Both consume only the authoritative Phase-1 `joint_provenance` coordinates and
`reward_joint_valid`. Invalid joints—including the seven reviewed Human-Art
source-OOB joints—have exactly zero contribution. Danbooru is excluded because
its sidecar records have `pose_reward_available=false`.

`detached_pose_diagnostics` reports normalized soft-coordinate error, soft
PCK@0.05/@0.10, optional argmax PCK@0.05/@0.10, entropy, and peak probability
under `torch.no_grad()`. None is a backward loss.

## Phase-1 provenance

The current canonical immutable sidecar record SHA is
`dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f`.
A deterministic current-code rebuild using this sidecar reproduced it
byte-for-byte with 17,416 records, 15,161 reward available, 2,255 unavailable,
444,235 valid reward joints, seven reviewed source-OOB masked joints, and
21/21 diagnostic coverage. The older `c98f...` value is historical only and
must not be treated as the current canonical sidecar fingerprint.

## Gate A: real-image GH200 feasibility audit

The completed Gate-A script selected usable records in deterministic stem order
from each available source: `coco`, `humanart_painting`,
`humanart_real_human`, and `humanart_sculpture`. It rebuilds the final RGB
through `preprocess_pair` and fails if its source resize/crop/bucket geometry
does not equal the immutable sidecar record. It uses all authoritative people
in each selected image as fixed boxes. Gate A backpropagated raw pixel
coordinate Huber for the first sample of each source and confirmed finite,
nonzero RGB gradients with no critic parameter gradients.

| source | soft PCK .05/.10 | argmax PCK .05/.10 | normalized soft error | coordinate Huber | Gaussian KL | RGB Huber gradient norm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| COCO | .9070 / .9728 | .9048 / .9728 | .02293 | 5.876 | 2.399 | 4.95 |
| Human-Art painting | .3842 / .5979 | .3931 / .5471 | .10508 | 46.478 | 4.366 | 631.18 |
| Human-Art real | .6023 / .7000 | .6591 / .7341 | .08409 | 32.096 | 4.157 | 225.50 |
| Human-Art sculpture | .6963 / .8494 | .7185 / .8395 | .05058 | 20.967 | 3.471 | 58.15 |

Gate A establishes critic feasibility. It also shows that raw pixel-coordinate
Huber produces strongly domain-dependent RGB gradient scales; it does not
choose a loss or authorize VAE/x0/training integration.

## Gate A.5: bounded loss-gradient comparison — PASS

The audit now evaluates all three losses for **eight deterministic images per
source by default**. Each candidate/image pair uses a separate critic forward
and `torch.autograd.grad` graph from identical RGB values, so RGB gradient
norms cannot be accumulated or reused. Every source JSON result includes the
valid-joint-weighted loss aggregate, every sample's three losses and
`rgb_grad_norm_coordinate_pixels`, `rgb_grad_norm_coordinate_normalized`, and
`rgb_grad_norm_heatmap_kl`, plus mean, median, population std, min, and max of
each gradient-norm series.

All candidates are means over valid person/joint observations. Source
aggregation weights each image mean by its valid-joint count, so crowded images
do not gain a larger per-observation weight. PCK, argmax, and other diagnostics
remain detached. `--temperature-sweep` is optional and diagnostic-only: it
reports just soft PCK and normalized coordinate error at 0.5, 1.0, and 2.0;
temperature remains fixed and unlearned.

The GH200 run retained temperature **1.0** and measured these mean RGB gradient
norms over the deterministic eight-sample/source panel:

| candidate | COCO | Human-Art painting | Human-Art real | Human-Art sculpture |
| --- | ---: | ---: | ---: | ---: |
| Gaussian heatmap KL | 2.3343 | 9.6581 | 7.2826 | 7.1236 |
| Normalized-coordinate Huber | .002822 | .101544 | .054005 | .020164 |

Gaussian heatmap KL is the primary audit candidate for the next gates.
Normalized-coordinate Huber remains the fallback/diagnostic candidate. Raw
pixel-coordinate Huber is rejected for training consideration because its
gradient scale is strongly domain-dependent. This is not an authorization to
add any pose objective to production training: production remains
flow-matching MSE plus the canonical normalized-coordinate pose-consistency
Huber auxiliary loss.

The reproducible A.5 command was:

```bash
PYTHONPATH=. python scripts/audit_keypoint_critic.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --samples-per-source 8 \
  --temperature-sweep \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_a5.json
```

Use the actual immutable sidecar path if it differs from the illustrative
`pose_targets_v3` path above.

## Gate B: exact VAE round-trip audit — PASS

`scripts/audit_keypoint_critic_vae.py` selects eight deterministic usable
records/source from `coco`, `humanart_painting`, `humanart_real_human`, and
`humanart_sculpture`; Danbooru remains excluded. It validates the sidecar
geometry through the existing paired preprocessing path, then uses the exact
project VAE helpers:

- `Qwen/Qwen-Image`, subfolder `vae`, class `AutoencoderKLQwenImage`;
- BF16 VAE execution;
- posterior sampling with a deterministic stem-derived generator;
- normalized latent `(raw - latents_mean) / latents_std`;
- inverse normalization before decode; and
- VAE `[-1, 1]` output converted in-graph to critic `[0, 1]` RGB.

For original RGB and decoded `z0`, it reports Gaussian KL,
normalized-coordinate Huber, normalized soft-coordinate error, soft PCK@.05,
soft PCK@.10, entropy, peak probability, and signed round-trip deltas. It also
reports detached RGB L1/MSE reconstruction error. Separate graphs measure
finite nonzero `z0` gradient norms for Gaussian KL and normalized Huber; the
JSON contains per-sample values and mean/median/std/min/max per source/loss.
It asserts that fixed boxes, COCO17 targets, and `reward_joint_valid` do not
change, and that frozen VAE/critic parameters receive no gradients.

Run Gate B first on the GH200 host shell:

```bash
PYTHONPATH=. python scripts/audit_keypoint_critic_vae.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --samples-per-source 8 \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_b_vae.json
```

Gate B passed on GH200: the exact VAE path retained valid geometry, finite
metrics, finite nonzero latent gradients for the retained candidates, and a
clean frozen VAE/critic boundary.

## Gate C: step-1500 x0_hat timestep audit — PASS

`scripts/audit_keypoint_critic_timestep.py` is a separate read-only audit. It
uses existing prepared latent shards plus cached text conditioning (rather than
an alternative inference path), the existing control channel-concatenation
forward, and the exact project flow helper:

```text
x_t = t * noise + (1 - t) * x0
v   = noise - x0
x0_hat = x_t - t * v_hat
```

The default timestep sweep is `.02 .05 .10 .20 .30 .40`; deterministic noise
is SHA256-derived from seed 42 and stem. It validates the exact default parent
checkpoint path and SHA256
`6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0`,
requires embedded step 1500, and loads it with the project model and
trainable-state loader. For each source/timestep it reports primary Gaussian
KL, fallback normalized Huber, all detached pose diagnostics, and deltas from
the VAE-round-trip baseline. Independent graphs report both `dL/dv_hat` and
`dL/dx0_hat` norms for both losses, without parameter gradients or optimizer
updates.

The quality and gradient reports must be read together: since
`d x0_hat / d v_hat = -t`, low timesteps naturally attenuate the gradient to
`v_hat` even when decoded pose quality is high. Gate C must not select a
timestep based on PCK alone.

Only after Gate B is acceptable, run Gate C on the GH200 host shell:

```bash
PYTHONPATH=. python scripts/audit_keypoint_critic_timestep.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --split train \
  --samples-per-source 4 \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_c_timestep.json
```

Gate C passed on GH200. The resulting interpretation is deliberately bounded:
primary pose timestep `.20`; secondary `.10`; stress/upper-bound diagnostic
`.30`; and `.40` is held because of strong gradient escalation/outliers.
Timesteps `.02`/`.05` are not primary pose candidates because the exact
`d x0_hat / d v_hat = -t` path heavily attenuates exposure. This does not
change production flow sampling or the flow-only production objective.

## Gate D: actual LoRA/control gradient calibration — IMPLEMENTED, GH200 RUN REQUIRED

`scripts/audit_pose_gradient_balance.py` is read-only. It loads the exact
rank-64/alpha-64 step-1500 trainable state, confirms the production trainable
boundary, and uses deterministic source samples/noise with independent flow and
Gaussian-KL autograd graphs. It reports per sample/source/timestep flow and
pose norms, ratio, dot product, cosine, candidate 1/5/10/20% lambdas,
combined-gradient diagnostics, and meaningful LoRA/ControlInput groups. It
does not build an optimizer or update a parameter.

Run it on the GH200 host shell:

```bash
PYTHONPATH=. python scripts/audit_pose_gradient_balance.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --split train --samples-per-source 4 --timesteps .10 .20 .30 --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_d_gradient_balance.json
```

Gate D is not PASS until this output, including sculpture outliers, has been
reviewed and an operator explicitly chooses `lambda_pose` and a pose-timestep
window.

## Gate E: isolated pose-reward smoke tooling — IMPLEMENTED, BLOCKED ON GATE-D REVIEW

`scripts/train_pose_reward_smoke.py` is a separate bounded continuation tool;
it does not alter `train.py`. It requires an isolated output run name,
`lambda_pose`, and an explicit inclusive pose-timestep window. It restores the
exact step-1500 parent, preserves production flow-timestep sampling and flow
MSE, and applies Gaussian heatmap KL only to Phase-1
`pose_reward_available=true` samples inside that supplied window. Danbooru
therefore remains flow-only. VAE, critic, and base weights stay frozen while
their differentiable path remains intact for active samples. Checkpoints use
the existing full trainable-state schema and default to frequent configurable
saves.

Do not launch this until Gate D has been reviewed. Template GH200 command:

```bash
PYTHONPATH=. python scripts/train_pose_reward_smoke.py \
  --parent-checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt \
  --expected-parent-sha256 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0 \
  --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints \
  --run-name <isolated_gate_e_run_name> --lambda-pose <reviewed_lambda_pose> \
  --pose-timestep-min <reviewed_min> --pose-timestep-max <reviewed_max> \
  --max-steps 200 --save-every 50 \
  --microbatch-size <profiled_microbatch> --gradient-accumulation-steps <accumulation> \
  --device cuda
```

## PASS/HOLD criteria

The current CPU gate passes when synthetic spatial/ROI mapping, validity
masking, Gaussian normalization/KL finiteness, synthetic heatmap gradients,
detached diagnostics, and mocked frozen-boundary checks pass. It does not
establish real-image suitability.

Gate A.5 completed with finite independent losses and nonzero RGB gradients
across the four sources. Gates B/C have no arbitrary quality threshold, but
must fail loudly on a non-finite value, absent/zero required latent/output
gradient, broken frozen boundary, changed Phase-1 geometry, checkpoint identity
mismatch, or unavailable required artifact. No training-loss integration is
authorized by this document.
