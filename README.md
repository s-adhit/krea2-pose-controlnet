# Krea-2 Pose Control-LoRA

Skeleton-conditioned image generation for Krea-2. This project adapts Krea-2 with a Control-LoRA-style control path so a rendered pose skeleton supplies body geometry while text supplies compatible appearance, style, clothing, materials, lighting, and environment.

<!-- Hero placeholder — add curated assets before enabling this image row.
![Pose condition](docs/assets/showcase/hero_pose.png) → multiple stylized generations
-->

## Showcase

### Cross-domain pose transfer

This is the report-safe technical showcase: one pose condition, interpreted across visual domains with original, non-franchise subject descriptions.

<!-- Showcase placeholders — the referenced assets do not yet exist.
| Pose condition | Fantasy mage / stained-glass subject | Street-mural interpretation | Editorial / fashion interpretation |
|---|---|---|---|
| ![Pose condition](docs/assets/showcase/hero_pose.png) | ![Stained glass](docs/assets/showcase/hero_stained_glass.png) | ![Mural](docs/assets/showcase/hero_mural.png) | ![Editorial](docs/assets/showcase/hero_editorial.png) |

| Realistic human-domain interpretation | Sculpture interpretation | Painterly interpretation |
|---|---|---|
| ![Realistic human](docs/assets/showcase/hero_realistic.png) | ![Sculpture](docs/assets/showcase/hero_sculpture.png) | ![Painterly](docs/assets/showcase/hero_painterly.png) |
-->

Planned panel: pose condition; an original fantasy mage in a stained-glass treatment; street-mural, editorial/fashion, realistic human-domain, sculpture, and painterly interpretations.

### Fan-art / social showcase

This separate, playful showcase will contain fan-art-style demonstrations: a Frieren stained-glass interpretation, Gojo action/street-mural interpretation, Jotaro fashion/comic interpretation, a meme-inspired pose, plus realistic-human and sculpture examples. These are creative, franchise-inspired examples only; they are not official affiliations, endorsements, or claims about training data.

<!-- Fan-art/social placeholders — the referenced assets do not yet exist.
| Frieren stained glass | Gojo action / street mural | Jotaro fashion / comic |
|---|---|---|
| ![Frieren-inspired stained glass](docs/assets/showcase/fanart_frieren_stained_glass.png) | ![Gojo-inspired mural](docs/assets/showcase/fanart_gojo_mural.png) | ![Jotaro-inspired fashion comic](docs/assets/showcase/fanart_jotaro_fashion_comic.png) |

| Meme-inspired pose | Realistic human | Sculpture |
|---|---|---|
| ![Meme-inspired pose](docs/assets/showcase/fanart_meme_pose.png) | ![Realistic human](docs/assets/showcase/fanart_realistic.png) | ![Sculpture](docs/assets/showcase/fanart_sculpture.png) |
-->

## What this project does

- A rendered skeleton image provides the pose condition.
- Text controls compatible appearance and scene attributes: identity, style, clothing, material, lighting, and environment.
- The pose image is VAE-encoded into a clean control latent. It is spatially aligned with the noisy image latent and injected through the expanded `ControlInputLayer`.
- Rank-64 LoRA adapters and the control input projection are the trainable adaptation; the pretrained backbone remains frozen.
- Training targets **Krea-2 Raw**. The trained control state is evaluated and deployed with **Krea-2 Turbo** after strict control/LoRA compatibility checks.

The condition is intended to carry pose geometry, not source-image semantics. Supplying a pose extracted from an image does not make this an image-reference or source-subject transfer system.

## Architecture

The control-input expansion used here was informed by [Tanmay Patil's Krea-2 ControlNet repository](https://github.com/Tanmaypatil123/Krea-2-controlnet), which demonstrates depth-conditioned control for Krea-2. This repository adapts that general control-input approach to skeleton-based pose conditioning and develops the pose data pipeline, training objective, pose-consistency supervision, evaluation, inference tooling, and checkpoint recipe used here.

At each spatial token location, the model concatenates the noisy image latent with the clean VAE-encoded pose-control latent. `ControlInputLayer` projects that widened feature vector into the existing model width; it does not add control tokens or a classical side-branch ControlNet. Its image half starts from the pretrained input projection, while its control half starts at zero. As a result, an untrained model is expected to be initially insensitive to the skeleton until optimization updates the control half.

| Component | Contract |
|---|---|
| Backbone | Krea-2 Raw, 28 transformer blocks |
| Control | Spatially aligned VAE control latent, concatenated at the input projection |
| Adaptation | LoRA rank 64, alpha 64 |
| LoRA targets | 8 modules per transformer block; 224 target modules total |
| Trainable state | `ControlInputLayer` plus LoRA tensors; approximately 215.49M parameters |

At a high level, flow matching forms a noisy image latent and predicts its velocity:

```text
x_t = t * noise + (1 - t) * x0
target velocity = noise - x0
x0_hat = x_t - t * v_hat
```

Here `x0` is the clean image latent, `x_t` is the noisy image latent, and the pose latent stays clean.

## Training objective

The canonical production objective is not flow-MSE-only. It combines flow-matching MSE with explicit pose-consistency supervision:

```text
loss = flow-matching MSE + lambda_pose * normalized-coordinate Huber
```

The canonical production/control branch uses `lambda_pose = 0.04`. Pose supervision is naturally active in an approximately `[0.10, 0.20]` timestep window; the production recipe sets forced pose-exposure probability to `0`. The pose term decodes `x0_hat`, evaluates it with a frozen fixed-box Keypoint R-CNN path, and compares normalized joint coordinates with Huber loss.

The consistency-feedback idea was inspired by [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](https://arxiv.org/abs/2404.07987), Li et al., ECCV 2024. This project’s normalized-coordinate Huber implementation is project-specific and is **not** claimed to be ControlNet++’s exact loss.

## Training recipe

| Setting | Locked production value |
|---|---:|
| Precision / seed | BF16 / 42 |
| Microbatch / accumulation / effective batch | 1 / 32 / 32 |
| Optimizer | AdamW, betas `(0.9, 0.99)`, weight decay `0` |
| Base LR / warmup / gradient clipping | `1e-4` / 200 optimizer steps / max norm `1` |
| Geometry | Dynamic 768-pixel bucket training |
| Loader | 4 workers, persistent workers, pinned memory, prefetch factor 4 |
| Runtime switches | Gradient checkpointing off, `torch.compile` off, fused AdamW off |

The work progressed through initial production training, cooldown/consolidation, and a finishing-branch comparison. The current serious checkpoint candidates are **parent-4000** (safer, more balanced) and **finish-control A4300** (more pose-specialist). The annealed/B branch is historical evidence only and is not presented as a current release candidate.

## Evaluation

Evaluation uses a locked Krea-2 Turbo contract: 8 sampling steps, CFG `0`, `mu = 1.15`, the official Turbo schedule, and no resolution-dependent shift. Where comparison applies, prompts, seeds, and geometries are fixed.

Pose scoring uses Keypoint R-CNN detections at confidence `>= 0.5`, deterministic Hungarian person matching, and bbox-diagonal-normalized PCK. Unmatched reference people fail matching rather than disappearing from the denominator. Reports also include detection coverage and CLIP image-text similarity.

Terminology is intentional:

- **Diagnostic split** is the development/selection benchmark.
- **Validation split** is held out from training, but has been inspected and used for inference benchmarking. It is not an untouched final test set.

Current qualitative status is deliberately modest: parent-4000 is the safer, more balanced candidate; A4300 is more pose-committed and suited to pose-specialist use. The B/anneal branch is historical and was rejected as the final recipe. These are development findings, not state-of-the-art claims.

## Inference

`inference.py` is the canonical local user-facing CLI. It requires a Krea-2 Turbo checkpoint, a compatible full pose-LoRA checkpoint, prompt, pose image, and output path.

```bash
PYTHONPATH=. python inference.py \
  --turbo-ckpt /path/to/turbo.safetensors \
  --pose-lora-ckpt /path/to/checkpoint.pt \
  --prompt "young woman, platinum-blonde bob, structured black fashion outfit, crimson studio background, cinematic directional lighting, high-fashion editorial photography" \
  --pose-image /path/to/control.png \
  --output /path/result.png \
  --seed 42 \
  --width 768 \
  --height 768 \
  --steps 8 \
  --cfg 0 \
  --mu 1.15 \
  --control-scale 1.0
```

The canonical sampler enforces the locked defaults: 8 steps, CFG `0`, and `mu = 1.15`. Explicit width and height are supported together and must be divisible by 16. To use the shared production dynamic-768 policy instead of explicit dimensions, pass `--dynamic-768-bucket` and omit `--width` / `--height`.

Each image is accompanied by a JSON provenance sidecar with the prompt, seed, geometry, sampling settings, checkpoint paths, and recorded checkpoint step.

For integrations, `inference.py` also exposes `PoseInferenceRequest`, `PoseInferenceResult`, `generate_pose(...)`, and `InferenceRuntime`. ComfyUI support is not available yet.

## Prompting and prompt curation

Current inference tests suggest that pose adherence is strongest when the text prompt describes appearance, style, and scene while the pose image defines body geometry. Explicit pose wording can compete with the skeleton, and composition/framing language can act like an indirect geometry constraint. Portrait-heavy prompts were the hardest current qualitative cases: full-body controls may be overwhelmed by close-up framing, subject-count mismatches, or sparse/incomplete skeletons. These are empirical observations, not yet a complete formal limitation study.

| Prompt feature | Effect | Recommendation |
|---|---|---|
| `close-up`, `portrait` | Can override a full-body condition’s framing | Avoid when preserving a full-body pose is important |
| `low angle`, `over-the-shoulder` | Can impose viewpoint and torso geometry | Use only when compatible with the condition |
| `hand on hip`, `arms raised` | Directly competes with skeleton limb placement | Let the control image specify limbs instead |
| `full body` | Usually supports a readable full-body condition | Use when it matches the skeleton’s extent |
| `multiple people` | Mismatch can break person assignment | Match prompt subject count to the control |

**Bad:**

```text
close-up portrait of a woman looking over her shoulder, one hand on her hip,
arm raised, low-angle view, ...
```

**Better:**

```text
young woman, platinum-blonde bob, structured black fashion outfit, crimson
studio background, cinematic directional lighting, high-fashion editorial photography
```

Let the pose control specify limb placement, stance, torso direction, and body geometry whenever exact pose adherence is the goal. Match subject count between prompt and control; prefer complete, readable skeletons; avoid sparse or truncated controls for showcase-quality generations; and avoid strongly contradictory portrait framing with a full-body condition. Style, clothing, lighting, environment, and material descriptors are usually safer than pose descriptors.

## Conditioning and data

Conditioning and evaluation material spans multiple visual domains, including COCO-derived human examples, Human-Art painting/real-human/sculpture domains, and anime-style/Danbooru-derived examples where applicable. These sources remain their respective owners’ material; this repository does not claim ownership or imply that third-party-derived imagery is freely redistributable. See [the archive index](docs/ARCHIVE_INDEX.md) for the committed Human-Art-derived imagery that requires redistribution review.

At inference, the skeleton is intended to convey pose geometry only. It is not a request to recover the source image’s identity, clothing, background, or visual semantics.

## Repository layout

| Path | Purpose |
|---|---|
| [`inference.py`](inference.py) | Canonical local Turbo pose-generation CLI and Python API |
| [`scripts/train_production.py`](scripts/train_production.py) | Locked production launcher |
| [`pose_controlnet/production_training.py`](pose_controlnet/production_training.py) | Production recipe, resume, checkpoint, and training mechanics |
| [`pose_controlnet/pose_consistency.py`](pose_controlnet/pose_consistency.py) | Canonical flow-MSE plus normalized-coordinate Huber objective |
| [`pose_controlnet/turbo_runtime.py`](pose_controlnet/turbo_runtime.py) | Locked Turbo sampling and Raw-to-Turbo control compatibility |
| [`pose_controlnet/resolution_policy.py`](pose_controlnet/resolution_policy.py) | Shared explicit and dynamic-768 geometry policy |
| [`data/manifests/`](data/manifests/) | Immutable train, validation, and diagnostic membership manifests |
| [`docs/inference_eval/`](docs/inference_eval/) | Current and preserved inference-evaluation evidence; consult the archive index for status |
| [`docs/ARCHIVE_INDEX.md`](docs/ARCHIVE_INDEX.md) | Canonical versus historical surfaces and redistribution-review notes |

## Training and resume

The production entry point is `scripts/train_production.py`. Its only required arguments are a run name and maximum step count; the locked recipe validates any exposed recipe switches against the production contract.

```bash
PYTHONPATH=. python scripts/train_production.py \
  --run-name pose-control-production \
  --max-steps 6000
```

Use `--resume /path/to/checkpoint.pt` for an explicit checkpoint, or `--resume auto` to select the newest valid local checkpoint in that run directory. Resume is fail-closed: checkpoint metadata must match the run identity, recipe, artifacts, scheduler, loader settings, and recorded data position before optimizer, scheduler, data-order, and RNG state are restored.

Checkpoints are written atomically and deserialize-validated. Local JSONL telemetry is durable; W&B is an optional, failure-isolated mirror. When configured with `--hf-repo-id` and a nonzero `--hf-mirror-every-steps`, completed local checkpoints are also submitted to the asynchronous Hugging Face mirror. Those remote services are not allowed to interrupt local training.

## Current status and roadmap

Production training is complete through the current candidate checkpoints. The canonical inference path exists, and prompt/pose interaction has been qualitatively investigated. Parent-4000 and A4300 remain the active candidates.

Planned work:

- Experimental checkpoint interpolation/mixing between parent-4000 and A4300 (not implemented or validated).
- Broader pose and control-scale evaluation.
- A focused prompt-conflict study.
- Style-LoRA composition experiments.
- A ComfyUI wrapper.
- A Hugging Face demo.
- Final technical and social showcase assets.

## References, prior work, and acknowledgements

- **Krea-2** — [official Krea-2 repository](https://github.com/krea-ai/krea-2) and [technical report](https://www.krea.ai/blog/krea-2-technical-report). This project follows Krea's Raw-for-training / Turbo-for-inference setup.

- **Tanmay Patil** — [Krea-2 ControlNet](https://github.com/Tanmaypatil123/Krea-2-controlnet), a public depth-conditioned Krea-2 ControlNet implementation that informed the control-input approach used in this project.

- **ControlNet** — Lvmin Zhang, Anyi Rao, and Maneesh Agrawala, [*Adding Conditional Control to Text-to-Image Diffusion Models*](https://arxiv.org/abs/2302.05543), ICCV 2023.

- **ControlNet++** — Ming Li et al., [*ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback*](https://arxiv.org/abs/2404.07987), ECCV 2024. The consistency-feedback idea influenced the explicit pose-consistency supervision used here; this repository's normalized-coordinate Huber loss is project-specific.

- **LoRA** — Edward J. Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).

- **Flow Matching** — Yaron Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747).

- **Rectified Flow** — Xingchao Liu, Chengyue Gong, and Qiang Liu, [*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*](https://arxiv.org/abs/2209.03003).

- **COCO** — Tsung-Yi Lin et al., [*Microsoft COCO: Common Objects in Context*](https://arxiv.org/abs/1405.0312), ECCV 2014.

- **Human-Art** — Xuan Ju et al., [*Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes*](https://arxiv.org/abs/2303.02760), CVPR 2023. [Official project/code](https://github.com/IDEA-Research/HumanArt).

- **Training methodology** — Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/). The project's staged overfit, debug, and scale-up workflow was influenced by this methodology.

- **CLIP** — Alec Radford et al., [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020).

- **Mask R-CNN / Keypoint R-CNN** — Kaiming He et al., [*Mask R-CNN*](https://arxiv.org/abs/1703.06870), together with the [torchvision Keypoint R-CNN implementation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html) used for pose-related scoring and supervision.
