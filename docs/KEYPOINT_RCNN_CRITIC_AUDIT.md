# Fixed-box torchvision Keypoint R-CNN critic audit

Status: **implementation and CPU contract checks complete; real-image GH200
domain audit is HOLD/not run.** This is an audit-only module. It does not
modify `train.py`, the Phase-1 flow-matching objective, VAE decoding, timestep
logic, provenance, or the external Keypoint R-CNN PCK evaluator.

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

Two independent audit candidates are exposed:

- `masked_coordinate_huber` / `coordinate_huber`: per-coordinate Huber,
  averaged only over valid person/joint pairs.
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

## Real-image GH200 audit

The script selects the first 16 usable records in deterministic stem order
from each available source: `coco`, `humanart_painting`,
`humanart_real_human`, and `humanart_sculpture`. It rebuilds the final RGB
through `preprocess_pair` and fails if its source resize/crop/bucket geometry
does not equal the immutable sidecar record. It uses all authoritative people
in each selected image as fixed boxes. For the first sample of each source it
backpropagates coordinate Huber to RGB and asserts finite nonzero RGB gradient
and absent critic parameter gradients.

Run on the actual GH200 shell (not the Codex sandbox):

```bash
PYTHONPATH=. python scripts/audit_keypoint_critic.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_audit.json
```

Use the actual immutable sidecar path if it differs from the illustrative
`pose_targets_v3` path above. The script reports joint count, both candidate
losses, normalized soft error, soft and detached-argmax PCK, entropy, peak
probability, and first-sample RGB input-gradient norm for every source.

## PASS/HOLD criteria

The current CPU gate passes when synthetic spatial/ROI mapping, validity
masking, Gaussian normalization/KL finiteness, synthetic heatmap gradients,
detached diagnostics, and mocked frozen-boundary checks pass. It does not
establish real-image suitability.

The GH200 audit may be marked PASS only after it completes for all four sources
with finite reported values, finite nonzero first-sample RGB gradients, and no
critic parameter gradients. It is an empirical domain audit: there is no
near-zero error or arbitrary PCK threshold. Hold the candidate if loading the
official COCO_V1 weights fails, fixed boxes/sidecar geometry fail validation,
values are non-finite, RGB gradients are absent/zero, or a critic parameter
receives a gradient. Even a PASS remains audit evidence only; no training-loss
or VAE/timestep integration is authorized by this document.
