# Krea-2 Pose Control-LoRA — frozen evaluation methodology

Status: evidence-backed methodology record, audited 2026-09-19 UTC. This is the source of truth for the frozen release metrics. `Verified` denotes executable-code or retained-artifact evidence; `Reconstructed` is a mechanical conclusion from that evidence; `Unresolved` is intentionally not presented as a fact.

## PCK: pose accuracy in the generated-image coordinate system

For every renderer-qualified source reference joint \(j\), the reported metric is globally pooled PCK:

\[
\operatorname{PCK}@\tau = \frac{\sum_{j \in E}\mathbf{1}[\lVert r_j-g_j\rVert_2 \leq \tau s_{p(j)}]}{|E|},\quad \tau \in \{0.05,0.10,0.20\}.
\]

`r_j` is the authoritative source annotation transformed into the relevant output bucket; `g_j` is the matched generated-image detector keypoint. `s_p` is the **diagonal of the axis-aligned extent of that reference person’s valid reference joints**: `sqrt((max_x-min_x)^2 + (max_y-min_y)^2)`. This is neither bbox width/height/max-side nor torso/head normalization. The comparison is inclusive (`<=`). [M1] **Verified.**

The source is COCO-17 annotation data retained in the authoritative sidecar, never joints extracted from a control raster. A joint enters `E` only if it was source-visible and analytically represented by the original skeleton-renderer limb topology. A person requires at least two eligible joints and a nonzero scale. Source keypoints are resized and translated by the recorded resize-to-cover / center-crop transform before scoring. [M2] **Verified.**

This is one global joint pool: it sums correct joint counts and eligible joint counts across samples, then divides. It is not an average of per-joint, per-person, or per-image PCKs. A predicted joint below confidence `.5` or absent from a matched person remains a failed reference joint in the denominator. An unmatched eligible reference person likewise remains in the denominator with zero correct joints. [M1,M3] **Verified.**

## Person detection and matching

Generated images are evaluated with `torchvision.models.detection.keypointrcnn_resnet50_fpn` using `KeypointRCNN_ResNet50_FPN_Weights.DEFAULT` (`COCO_V1`), emitting COCO-17 keypoints. The evaluator retains detections whose person score is `>= 0.5`; a keypoint is valid when all values are finite and keypoint confidence is `>= 0.5`. [M1] **Verified.**

Reference people are the renderer-qualified, geometry-transformed authoritative people. Generated people are detector outputs. The evaluator creates a cost matrix of the mean unnormalized Euclidean pixel distance over the two people’s shared valid joints, marks no-shared-joint pairs impossible, and solves a one-to-one Hungarian assignment. There is no match-distance cutoff. `Matched` is the number of finite-cost assigned reference/generated person pairs. [M1] **Verified.**

`unmatched_reference_people` records references without an assigned prediction; their eligible joints score as zero. `unmatched_predicted_people` records extras; extras are reported but are not an additional PCK penalty. `detection_coverage = matched usable reference people / usable reference people`; `joint_evaluation_coverage` is the share of eligible joints that also had a valid detector keypoint. [M1,M3] **Verified.**

The model constructor does not override torchvision inference defaults. Audit-time torchvision `0.22.0` source gives prediction-head NMS IoU `.5`, up to 100 detections/image, RPN NMS `.7`, and inference resize bounded by short side 800 / long side 1333. The weights’ public preprocessing only converts RGB PIL to float `[0,1]`; the detector internally resizes and postprocesses outputs back to the original generated-image coordinates. These defaults are **Reconstructed**, not separately pinned in the result artifacts. [M4] **Reconstructed.**

## CLIP prompt/image similarity

CLIP is `transformers.CLIPModel` plus `CLIPProcessor` from `openai/clip-vit-base-patch32`. For every generated RGB image, the scorer tokenizes the exact prompt used for that generation (with padding, truncation, and the model text context length) and lets the named processor perform its standard image preprocessing. It obtains projected image/text features and explicitly calculates cosine similarity, without CLIP logit scaling. [M5] **Verified.**

The reported CLIP value is the arithmetic mean across all images in the completed benchmark; median and population standard deviation are retained too. It is independent of person matching and PCK eligibility. Complete frozen scoring paths raise for a missing generation rather than silently dropping it. Prompt-injection CLIP specifically uses the frozen injected prompt, not the source caption. [M5,M6] **Verified.**

## Native and dynamic-768 geometry

Native scoring consumes the persisted paired cached-latent geometry (`source_size`, `resized_size`, `crop_box`, `bucket`) and refuses stale geometry. The paired preprocessing uses resize-to-cover with `scale=max(bucket_w/source_w,bucket_h/source_h)`, `round` for resized dimensions, and a centered crop using floor integer division; it preserves aspect ratio before cropping and never pads. RGB and control received this same transform during paired preprocessing. [M7] **Verified.**

`dynamic-768` is a separate five-condition ablation, not a universal 768×768 resize. It chooses the closest log-aspect-ratio bucket from `768×768`, `704×896`, `896×704`, `640×960`, `960×640`, `576×1024`, `1024×576`, `512×1152`, and `1152×512`; resize/crop transforms the authoritative control raster, and that control is VAE-encoded using the fixed sampling seed. Source RGB is explicitly not used. PCK transforms the source annotation through that mode’s geometry, so coordinates align inside each mode. [M7,M8] **Verified.**

The comparison changes more than nominal resolution: it can change bucket dimensions, crop framing, raster resampling, and control VAE encoding. Consequently, it is an ablation of the complete geometry/control-input path, not proof of a resolution-only causal effect. **Verified as code behavior; interpretation is descriptive.**

## Blog interpretation and limitations

PCK measures whether detector-recovered body joints from the output land near renderer-qualified source joints after matching people. Higher is better. `PCK@.05` is stricter than `.10`, which is stricter than `.20`; all use the same person-diagonal normalization. CLIP measures semantic alignment of output image and prompt, rather than skeleton alignment. They can move differently because an image may satisfy text while missing the control pose, or preserve pose while changing visual/prompt detail.

Important limits: PCK depends on a COCO-trained keypoint detector and Hungarian correspondence; it does not directly judge all visual quality, hands, occluded anatomy, or prompt faithfulness. Hungarian person association uses mean raw pixel-distance cost over shared valid joints, while PCK scoring afterward uses the matched reference person's diagonal normalization; this is the defined association-and-scoring procedure, not an invalidation of the metric. Reference exclusions and detector misses are visible in retained coverage/count fields, but the benchmark remains finite. Extra generated people are counted but not directly penalized in PCK. Native/dynamic changes framing and encoding as noted above. These are descriptive limits, not claims of generalization beyond the frozen subsets.

## Evidence registry

| ID | Source / locator | Confidence |
| --- | --- | --- |
| M1 | `pose_controlnet/post500_evaluation.py`: `_valid_joints`, `_scale`, `associate_people`, `pck_for_people`, lines 33–107 | Verified |
| M2 | `pose_controlnet/reference_pose.py`: renderer qualification and source-to-bucket transform, lines 55–172 | Verified |
| M3 | `pose_controlnet/post1500_evaluation.py`: `_pool_pose`, `score_authoritative_pck`, lines 148–205 | Verified |
| M4 | audit-time installed torchvision `0.22.0`, `KeypointRCNN` constructor/default source | Reconstructed |
| M5 | `scripts/turbo_benchmark.py::_clip_score`, lines 368–374; `pose_controlnet/post500_evaluation.py::prepare_clip_scoring_inputs`, `aggregate` | Verified |
| M6 | `scripts/frozen_prompt_turbo.py::prompt_score`, lines 332–355 | Verified |
| M7 | `pose_controlnet/paired_preprocessing.py`, lines 68–105; `pose_controlnet/evaluation_geometry.py::persisted_scoring_geometry` | Verified |
| M8 | `scripts/native_vs_dynamic768.py::_inputs`, `_dynamic_sample`, lines 165–277 | Verified |

Machine-readable counterpart: [`evaluation_methods.json`](evaluation_methods.json).
