# Unified pose-target sidecar

`pose_controlnet.pose_targets` creates a new, versioned, read-only directory
containing `metadata.json` and deterministic `records.jsonl`. It does not edit
manifests, image/control files, or latent shards. It refuses an existing output
directory and verifies the JSONL SHA-256 when loaded.

Each record contains `schema_version`, `stem`, `source`,
`target_provenance`, `annotation_source`, source/resize/crop/bucket geometry,
the input joint schema, explicit common-body mapping, renderer provenance, and
one source-grouped `people` list. Every person retains source `xyv` or `xyc`,
source box, source confidence/visibility, clipped training coordinates,
in-frame mask, reward-visible mask, and clipped training box. The reward mask
is authoritative visibility/confidence intersected with the final crop.

The common reward schema is the 17 physical COCO body joints only. COCO-17 is
identity mapped. Historical DWPose/OpenPose body-18 uses indices
`[0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10]`; its neck (index 1) is not a
target.

## Source specification contract

Pass a JSON object with one entry per source under `sources`: `coco`,
`humanart_painting`, `humanart_real_human`, `humanart_sculpture`, and
`danbooru`. An annotated entry requires `target_provenance` exactly
`original_annotation`, `format: "coco_keypoints"`, immutable
`annotation_paths`, `annotation_source`, `joint_schema: "coco17"`, and
`provenance_metadata.renderer`. Human-Art additionally requires an explicit
`stem_image_id_regex` with named `image_id`; this prevents an assumed join.

Danbooru requires `target_provenance: "dwpose_pseudolabel"` and
`format: "historical_dwpose_jsonl"`. Its immutable export must contain `stem`,
`source_size`, and source-grouped 18-body-keypoint people. Its provenance must
contain detector, pose checkpoint and SHA-256, thresholds, body-joint mapping,
and renderer. All sources must identify the exact historical renderer. The
reconstruction gate additionally requires
`renderer.validated_historical_renderer: true`; it cannot be set based on a
raster guess.

## Commands

Read-only inventory:

```bash
python scripts/audit_pose_target_sources.py \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --source-spec /path/to/recovered_source_spec.json
```

Build only after the preceding audit is PASS:

```bash
python scripts/build_pose_target_sidecar.py \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --source-spec /path/to/recovered_source_spec.json \
  --output /lambda/nfs/adhit/krea2-pose/pose_targets/v1
```

Run the fail-closed stratified raster reconstruction gate:

```bash
python scripts/audit_control_reconstruction.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets/v1 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --output-dir /lambda/nfs/adhit/krea2-pose/pose_target_audits/v1 \
  --per-source 16 --min-foreground-iou 0.995 --max-mae 0.25
```

The audit writes per-sample metrics/mismatches and a compact stored/rebuilt/
difference contact sheet. A provenance path is accepted only if every sampled
source passes both thresholds.
