# Unified pose-target sidecar v3

`data/pose_targets_authoritative_v1.jsonl` is the sole numerical source for
active COCO and Human-Art pose targets. The builder joins it by exact final
PoseBridge stem with the persisted geometry in the latent shards. It does not
read raw COCO JSON, a separate Human-Art adapter, a source specification, a
pose detector, or a control raster to obtain numerical targets. Active
Danbooru records remain explicit flow-only records with
`pose_reward_available: false`.

## Provenance and two explicit consumers

Every available person's 17 COCO-order joints retain the raw
`keypoints_source` triple without clipping or visibility changes. Its
`joint_provenance` entry explicitly contains:

- `source_coordinate` and `source_visibility_confidence`;
- `source_in_bounds` against the declared source canvas;
- the unclipped affine `training_coordinate`;
- `final_in_frame` in the persisted crop/bucket geometry;
- `reward_joint_valid` and `reward_invalid_reason`.

The `consumer_semantics` field separates two uses of the same provenance:

- Historical reconstruction reads raw `keypoints_source` in the source frame.
  This faithfully represents the input received by the original renderer,
  including off-canvas geometry that rasterization naturally clips.
- Any future pose-reward consumer must use `joint_provenance` and only joints
  whose `reward_joint_valid` is true. Pose reward is not implemented yet.

The v3 compatibility `keypoints_training` field is clipped to the final pixel
canvas for raster diagnostics; it never replaces the preserved raw source
coordinate.

## Reviewed source-coordinate anomaly policy

`humanart_original_source_oob_v1` is an explicit, versioned contract for the
current authoritative export. It allows exactly seven visible original
Human-Art source-coordinate defects, all masked from reward use with
`reward_invalid_reason: source_coordinate_out_of_bounds`:

- `painting_humanart_2000000000804`, annotation `2000000007651`:
  `left_wrist`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.
- `sculpture_humanart_14000000001208`, annotation `14000000088574`:
  `left_ankle`, `right_ankle`.

The policy pins their stem, annotation ID, joint index, raw coordinates, and
source dimensions. The source audit and full builder enumerate joint names,
raw coordinates, dimensions, and directional overshoots. Any other visible
source-out-of-bounds joint, or an altered/missing reviewed one, fails the
active-dataset audit/build pending review. Neither affected sample is made
unavailable: its other valid joints remain usable future reward supervision.

## Build and source audit

```bash
python scripts/audit_pose_target_sources.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --authoritative-jsonl data/pose_targets_authoritative_v1.jsonl

python scripts/build_pose_target_sidecar.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --authoritative-jsonl data/pose_targets_authoritative_v1.jsonl \
  --output /new/nonexistent/pose_targets_v3
```

The builder writes a new immutable directory with deterministic
`records.jsonl` and `metadata.json`, refusing to overwrite an existing
destination. Loading verifies the records SHA-256.

## Reconstruction audit

Historical source controls are rerendered in their raw source frame and then
passed through exactly the persisted PIL resize/crop preprocessing before
comparison. This is the primary geometry comparison: a direct fixed-3px
final-frame vector render is retained only as a diagnostic, because it is not
raster-equivalent after a source control has been scaled.

The deterministic primary gate reports foreground IoU at thresholds 1, 10,
and 32, symmetric foreground distance-transform mean, and p95 distance. The
default acceptance criteria are IoU at threshold 10 >= 0.63, symmetric mean
distance <= 2.75 final-frame pixels, and p95 <= 3 pixels. They are calibrated
from the deterministic 16-per-available-source v3 baseline: IoU
0.6380–0.8321, symmetric mean distance 0.0917–2.6909 px, and p95 1–3 px.
The six-item forensic subset (IoU about 0.674–0.769, mean distance about
0.177–0.303 px, p95 1–2 px) remains useful supporting evidence, but is not a
replacement for the full audit. A fixed-stroke IoU >= 0.995 is inappropriate
after source-raster scaling.

```bash
python scripts/audit_control_reconstruction.py \
  --sidecar /new/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-dir /new/pose_target_audit_v3 \
  --per-source 16
```
