# Unified pose-target sidecar

## Authoritative v1 export (current path)

`data/pose_targets_authoritative_v1.jsonl` is the sole numerical target source
for active COCO and Human-Art records. It is joined by exact final PoseBridge
stem; the large original COCO/Human-Art datasets are not reread. Active
Danbooru records receive explicit v3 `pose_reward_available: false` records.

The v3 builder transforms source COCO-17 coordinates using each active
latent shard's persisted `source_size`, `resized_size`, and `crop_box`.
Visibility is preserved; the reward mask also excludes points cropped outside
the final canvas. A visible source joint outside its declared source image
(except an exact far-edge coordinate) is a provenance failure that blocks the
build. Final points and xywh boxes are clipped to the inclusive pixel canvas.

The reconstruction renderer is body-only: COCO-17 maps to unified-18, makes
the neck only while drawing, renders the standard OpenPose rainbow limbs at
thickness 3, and then draws visible white radius-4 endpoints. That synthetic
neck is never a reward target.

```bash
python scripts/build_pose_target_sidecar.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --authoritative-jsonl data/pose_targets_authoritative_v1.jsonl \
  --output /new/nonexistent/pose_targets_v3
```

`pose_controlnet.pose_targets` creates a new, versioned, read-only directory
containing deterministic `records.jsonl` and `metadata.json`. It never edits
manifests, images, controls, or latent shards; it refuses to overwrite an
existing output and verifies the JSONL SHA-256 on load.

Each training stem has exactly one explicit branch:

```text
pose_reward_available: true   target_provenance: original_annotation
pose_reward_available: false  target_provenance: unavailable
```

Available records contain the authoritative, source-grouped `people` list,
source/resize/crop/bucket geometry, source and training keypoints, joint
visibility/confidence, in-frame/reward masks, boxes when supplied, the COCO-17
body mapping, and renderer provenance. Unavailable records have
`people: null`; they are an intentional flow-only training decision, never an
empty annotation. `pose_reward_target_for_stem(...)` returns the available
record or `None` for this explicit unavailable branch.

The build fails closed only for a stem whose source claims
`pose_reward_available: true` but has a missing, malformed, mismatched, or
geometrically invalid target. It reports deterministic total, available, and
unavailable counts and percentages for every source and overall.

No path in this infrastructure derives keypoints from a raster control or runs
DWPose (or another detector) as a fallback.

## Source specification

The JSON source spec must declare every source under `sources`: `coco`,
`humanart_painting`, `humanart_real_human`, `humanart_sculpture`, and
`danbooru`.

For the first ControlNet++ experiment, use this coverage shape:

```json
{
  "sources": {
    "coco": {
      "pose_reward_available": true,
      "target_provenance": "original_annotation",
      "format": "coco_keypoints",
      "annotation_source": "COCO 2017 person keypoints",
      "joint_schema": "coco17",
      "annotation_paths": ["/absolute/path/to/annotations/person_keypoints_train2017.json"],
      "provenance_metadata": {"renderer": {"identifier": "...", "sha256": "...", "validated_historical_renderer": true, "topology": "openpose_body18"}}
    },
    "humanart_painting": {"pose_reward_available": true, "target_provenance": "original_annotation", "format": "humanart_pose_adapter_jsonl", "annotation_source": "user-supplied Human-Art export", "joint_schema": "coco17", "adapter_path": "/absolute/path/to/humanart_adapter.jsonl", "provenance_metadata": {"renderer": {"identifier": "...", "sha256": "..."}}},
    "humanart_real_human": {"pose_reward_available": true, "target_provenance": "original_annotation", "format": "humanart_pose_adapter_jsonl", "annotation_source": "user-supplied Human-Art export", "joint_schema": "coco17", "adapter_path": "/absolute/path/to/humanart_adapter.jsonl", "provenance_metadata": {"renderer": {"identifier": "...", "sha256": "..."}}},
    "humanart_sculpture": {"pose_reward_available": true, "target_provenance": "original_annotation", "format": "humanart_pose_adapter_jsonl", "annotation_source": "user-supplied Human-Art export", "joint_schema": "coco17", "adapter_path": "/absolute/path/to/humanart_adapter.jsonl", "provenance_metadata": {"renderer": {"identifier": "...", "sha256": "..."}}},
    "danbooru": {"pose_reward_available": false, "target_provenance": "unavailable", "format": "unavailable"}
  }
}
```

The renderer metadata is required only for available records because it is a
reconstruction-audit contract, not a source of targets. Exact renderer fields
must be filled from verified provenance; do not claim validation from a raster
comparison alone.

## Human-Art importer contract

The user supplies the actual Human-Art source file separately. Do **not** feed
that unknown raw schema directly to the sidecar builder. Instead, write a
small source-specific importer that preserves the original file unchanged and
emits the canonical adapter JSONL below. This keeps source-format parsing
separate from the common sidecar representation.

One non-empty JSON object per line is required:

```json
{
  "stem": "painting_humanart_10000000000838",
  "source": "humanart_painting",
  "source_image_id": "optional-original-image-id",
  "source_size": [width, height],
  "people": [
    {
      "person_id": "original-person-id",
      "annotation_id": "optional-original-annotation-id",
      "bbox_xywh": [x, y, width, height],
      "keypoints": [[x, y, visibility_or_confidence], "... 16 more COCO-17 joints"]
    }
  ]
}
```

- `stem` is required and must be the exact PoseBridge training stem. It is the
  join key; `source_image_id` is optional provenance only.
- `source` is optional but, when present, must equal the corresponding
  Human-Art source family. It prevents an accidental cross-family join.
- `people` preserves original person grouping. It may be an empty list only
  when the original annotation explicitly contains no target people.
- Each person has exactly 17 COCO-order body joints. The third value is the
  original visibility or confidence; it must be finite and non-negative.
- `bbox_xywh` is optional. If present it is source-image coordinates with a
  non-negative width and height.
- `source_size` is optional. If present, it must match the immutable shard
  geometry exactly; otherwise the build fails for that claimed sample.

The importer may translate any supplied Human-Art schema into this contract,
but it must not run a pose model or inspect a control PNG. Store the raw export
path/version/hash in `annotation_source` or provenance metadata outside the
adapter's per-record source data as appropriate.

## COCO contract

Use the original COCO person-keypoint annotation JSON, normally
`annotations/person_keypoints_train2017.json` for this training corpus. Do
not use a generated pose export. The sidecar joins each PoseBridge COCO stem
using the immutable form `coco_<image_id>_<annotation_id>.jpg` (or
`coco_<image_id>_crowd.jpg`) to COCO `images[].id` and, for the non-crowd
form, `annotations[].id`. The COCO `images[].width`/`height` must exactly
match the persisted shard source geometry. Crowd stems use every non-crowd
COCO person annotation for that image. No image download is performed.

## Commands

Once the Human-Art adapter and COCO annotation paths exist, first audit the
coverage decision and reachable authoritative artifacts:

```bash
python scripts/audit_pose_target_sources.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --source-spec /absolute/path/to/pose_reward_source_spec.json
```

Then build a new sidecar (only after the audit reports `PASS`):

```bash
python scripts/build_pose_target_sidecar.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --source-spec /absolute/path/to/pose_reward_source_spec.json \
  --output /lambda/nfs/adhit/krea2-pose/pose_targets/v2
```

The reconstruction audit selects only `pose_reward_available: true` records.
Danbooru remains in its coverage report as intentionally unavailable and is
not sampled, rendered, or reported as a reconstruction failure:

```bash
python scripts/audit_control_reconstruction.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets/v2 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-dir /lambda/nfs/adhit/krea2-pose/pose_target_audits/v2 \
  --per-source 16 --min-foreground-iou 0.995 --max-mae 0.25
```
