# Phase 1 handoff

## Current objective

Correct post-step-500 PCK reference-side evaluation using the authoritative pose data that produced each control raster. No training, optimizer step, model/training change, checkpoint mutation, fixed-pose regeneration, commit, or push occurred in this session.

## Verified state

- Training reached optimizer step 500; its HF checkpoint is confirmed valid and complete in `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`.
- The only valid evaluation sequence is exactly `0,20,40,60,80,100,200,225,350,475,500`; never add 300 or 400.
- Fixed-flow and fixed-pose contracts, checkpoints, generated images, CLIP implementation, confidence threshold `0.5`, CFG, sampler, and seeds remain unchanged.
- The previous GH200 scorer detected generated people but attempted to run torchvision COCO-V1 Keypoint R-CNN over control rasters. Rendered skeletons are not natural-person images, yielding `reference_people=0`; null PCK is not zero pose accuracy.

## Authoritative-reference blocker (verified)

- Production dataset snapshot: `/lambda/nfs/adhit/krea2-pose/posebridge_hf` contains RGB JPGs, control PNGs, immutable manifests, and `metadata.jsonl`.
- `metadata.jsonl` has exactly 17,495 rows and only `file_name`, `conditioning_image`, and `text`. For diagnostic stem `real_human_humanart_17000000000288` (and checked COCO/Danbooru examples), it contains no pose/keypoints, image dimensions, person grouping, visibility/confidence, renderer provenance, or source schema.
- Recursive targeted searches of the accessible `/lambda/nfs` and `/home/ubuntu` storage found no source COCO/HumanArt/Danbooru annotation/keypoint files and no pose-control generation script/notebook. DatasetIndex and preprocessing preserve only RGB/control paths and captions, so they cannot recover this lost provenance.
- Therefore no authoritative reference joints are accessible. It is unsafe to implement source mappings, coordinate transforms, or claim the required one-sample PCK smoke: doing so would infer/fabricate data. Skeleton-raster parsing is explicitly not authorized and must not be used as a fallback.
- Required external artifact to resume: the immutable source pose archive plus the exact rendering/preprocessing metadata for every diagnostic stem (source schema/joint order, per-person joints with visibility/confidence, source image dimensions, and the resize/crop transform into each generated-image pixel space), or an existing authoritative per-stem evaluation JSONL with equivalent fields.

## Existing PCK semantics needing correction once artifact is provided

- Generated image side remains `torchvision/keypointrcnn_resnet50_fpn:COCO_V1`, confidence `0.5`.
- Reference must be authoritative pose data, mapped only where semantically valid to COCO-17; invalid/invisible joints excluded.
- Association must remain deterministic one-to-one Hungarian matching over shared valid joints. Normalization is the authoritative reference person valid-joint bbox diagonal: `sqrt((max_x-min_x)^2 + (max_y-min_y)^2)`.
- Missing authoritative data must produce not-evaluable/null PCK plus a reason; never generic `0.0` coverage or null-to-zero conversion.

## Tests this session

- Pending the required authoritative pose artifact; no implementation was made because schema and provenance are unavailable.

## Exact next action

Provide/mount the authoritative pose annotation archive and renderer geometry metadata. Then implement the project-owned loader, source mappings, PCK coverage/summary/plots, focused tests, and run the requested one-sample corrected scorer. Do **not** rerun `scripts/score_post500.py` yet: its current reference side remains invalid.
