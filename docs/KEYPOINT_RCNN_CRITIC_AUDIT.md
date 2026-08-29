# Fixed-box torchvision Keypoint R-CNN critic audit

Status: **Gate A (real-RGB critic feasibility) PASS; Gate A.5 loss-gradient
comparison is implemented but not yet run on GH200.** This is an audit-only
module. It does not modify `train.py`, the Phase-1 flow-matching objective, VAE
decoding, timestep logic, provenance, or the external Keypoint R-CNN PCK
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

## Gate A.5: bounded loss-gradient comparison

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

Run on the actual GH200 shell (not the Codex sandbox):

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
`pose_targets_v3` path above. Do not use its result to select a training loss
without separate authorization.

## PASS/HOLD criteria

The current CPU gate passes when synthetic spatial/ROI mapping, validity
masking, Gaussian normalization/KL finiteness, synthetic heatmap gradients,
detached diagnostics, and mocked frozen-boundary checks pass. It does not
establish real-image suitability.

Gate A.5 may be marked complete only after it reports finite independent losses
and finite nonzero RGB gradients for all three candidates across all four
sources, with no critic parameter gradients. It is an empirical comparison:
there is no arbitrary PCK, gradient, or loss threshold and it does not choose a
training objective. Hold it if official COCO_V1 loading fails, fixed
boxes/sidecar geometry fail validation, a value is non-finite, an RGB gradient
is absent/zero, or a critic parameter receives a gradient. No training-loss or
VAE/timestep integration is authorized by this document.
