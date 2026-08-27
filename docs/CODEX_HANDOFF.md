# Phase 1 handoff

## Current bounded objective and status

Authoritative-reference PCK is **blocked before scoring**. No training process
was stopped or modified; no training, image regeneration, optimizer, LR, LoRA,
or checkpoint change was made in this session.

The committed sidecar `data/manifests/diagnostic_reference_pose.json` has 24
diagnostic records: 21 `available` (17 Human-Art, 4 COCO) and 3 unavailable
Danbooru records. Danbooru must remain `pose_metric_status: unavailable` with
reason `authoritative_reference_pose_unavailable`, null pose metrics, and no
aggregate-PCK denominator contribution.

## Authoritative provenance

- Human-Art records originate from `training_humanart.json` or
  `validation_humanart.json`; their source COCO-17 coordinates remain the only
  eligible PCK reference. Do not use the synthesized unified-18 neck for PCK.
- COCO normal stems are `coco_<image_id>_<annotation_id>`; crowd stems are
  `coco_<image_id>_crowd` and contain the qualifying source persons used for
  the crowd control.
- The sidecar currently does not persist per-person `iscrowd`, explicit
  renderer-inclusion, `has_core_visibility`, or visible-limb values. Any PCK
  implementation must not invent these facts; it must either receive that
  provenance or prove source-control inclusion without deriving keypoints from
  rasters.

## Exact geometry established from project code

`pose_controlnet.paired_preprocessing.resize_center_crop_geometry` is the
authoritative paired path used before VAE encoding in `prepare_shards.py`:

```text
bucket = nearest log-aspect bucket
scale = max(bucket_w / source_w, bucket_h / source_h)
resized_w = round(source_w * scale)
resized_h = round(source_h * scale)
crop_left = (resized_w - bucket_w) // 2
crop_top  = (resized_h - bucket_h) // 2
x_output = x_source * resized_w / source_w - crop_left
y_output = y_source * resized_h / source_h - crop_top
```

There is no pad. RGB and control share this exact LANCZOS resize and crop.
Latents are 1/8 of the bucket dimensions; fixed-pose decoding returns the
bucket dimensions. Fixed-pose `control.png` is a copied source raster, while
the matching generated image uses the persisted bucket recorded in its
`metadata.json`.

Verified required examples:

- `real_human_humanart_17000000000288`: source/control 665x1000 -> bucket
  832x1216; resized 832x1251; crop `(0,17,832,1233)`;
  `x'=x*832/665`, `y'=y*1251/1000-17`.
- `coco_156320_crowd`: 640x400 -> 1344x768; resized 1344x840; crop
  `(0,36,1344,804)`; `x'=x*1344/640`, `y'=y*840/400-36`.
- `coco_299468_426600`: 640x640 -> 1024x1024; resized 1024x1024; no crop.
- `painting_humanart_10000000000838`: 300x607 -> 704x1472; resized 728x1472;
  crop `(12,0,716,1472)`; `x'=x*728/300-12`, `y'=y*1472/607`.

## Geometry gate evidence and blocker

Validation used existing source control PNGs only as evidence, never as a
keypoint source: the distance from each sidecar visible COCO-17 joint to the
nearest non-black control pixel was measured. Raster dimensions exactly matched
authoritative source dimensions in all cases.

- `real_human_humanart_17000000000288`: 34/34 exact pixels.
- `coco_156320_crowd`: 39/39 exact pixels.
- `coco_299468_426600`: 13/13 exact pixels.
- `painting_humanart_10000000000838`: 45/46 exact pixels. The remaining
  visible source joint is Human-Art annotation `10000000065251`, COCO-17 index
  15, `(224.7366, 217.9995)`: nearest control pixel is 24.331 px away.

This fails the requested exact geometry/control validation gate. Treat this as
an unexplained renderer/provenance discrepancy, not an approximation to waive.
Do not compute, publish, or plot PCK until it is resolved and per-person
renderer inclusion can be reproduced exactly.

## Existing evaluation facts

The existing post-500 fixed-pose sequence is exactly
`0,20,40,60,80,100,200,225,350,475,500`; do not add 300 or 400. Existing
fixed-flow, CLIP, checkpoint ordering, and generated images are untouched.
When the gates are repaired, PCK must use COCO-17 identity, visibility `> 0`,
generated confidence `>= 0.5`, Hungarian one-to-one association over shared
valid joints, reference valid-joint bbox diagonal normalization, and `<=` at
0.05/0.10/0.20. Unmatched references reduce coverage; unavailable Danbooru is
null/excluded, never zero.

## Files changed this session

- `docs/CODEX_HANDOFF.md`

## Commands run

- PASS (inspection): sidecar count/status inspection: 24 total, 21 available,
  3 unavailable.
- PASS (inspection): direct source-control dimension and nearest-rendered-pixel
  checks for the four required examples; the first three are exact.
- FAIL (required Gate 1): `painting_humanart_10000000000838` discrepancy above.

## Exact next action

Obtain authoritative source renderer metadata for every sidecar person
(`iscrowd`, renderer inclusion, core-visibility and visible-limb decision),
and resolve the missing visible joint for annotation `10000000065251` before
implementing/scoring PCK. Then add the requested focused tests and run the
geometry-only command before any one-sample PCK smoke.
