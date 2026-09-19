---
tags:
  - krea-2
  - pose-control
  - lora
  - image-generation
---

# Krea-2 Pose Control-LoRA

Skeleton-conditioned image generation for Krea-2. The recommended public release is **`mix-025`**.

<p align="center">
  <img src="https://raw.githubusercontent.com/s-adhit/krea2-pose-controlnet/main/docs/showcase/final/hero-v1/final_showcase_collage.png" alt="Krea-2 Pose Control final showcase" width="100%">
</p>

This is the same final showcase collage, ordering, and crop used at the top of the [project README](https://github.com/s-adhit/krea2-pose-controlnet#showcase).

## Overview

Give the model a rendered pose skeleton plus a text prompt. The skeleton controls broad body geometry and placement; the prompt controls compatible subject identity, clothing, environment, lighting, and rendering style.

The public artifact is a standalone `safetensors` file for the canonical candidate:

| Release ID | File | SHA-256 | Saved trainable control/LoRA tensors |
|---|---|---|---:|
| `mix-025` | `release/krea2-pose-control-mix025.safetensors` | `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1` | 215,488,512 |

`release/` is the final public release. Existing `pose-learning/`, `pose-control-production/`, and `finish-*` folders are historical training checkpoints, not alternative public releases.

## Architecture

The rendered skeleton is VAE-encoded into a clean, spatially aligned control latent. It is channel-concatenated with the noisy image latent before an expanded input projection; the spatial token count is unchanged. The Krea-2 backbone remains frozen while the control input and rank-64 LoRA adapters supply pose control.

## Pose condition

The condition represents the 17 COCO body keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles. It contains body pose only—no face landmarks and no hand or finger keypoints.

Use skeleton rasters in this project's trained rendering convention. Generic third-party OpenPose rasters are not claimed to be compatible; use the supplied/project renderer and the examples in the [repository](https://github.com/s-adhit/krea2-pose-controlnet) as the format reference.

## Recommended inference

Canonical inference uses Krea-2 Turbo with 8 steps, CFG 0, `mu=1.15`, and control scale `1.0`. Native aspect-preserving geometry is the default; the pose image dimensions must be divisible by 16. `--dynamic-768-bucket` is an optional alternative, not the default.

```bash
PYTHONPATH=. python inference.py \
  --turbo-ckpt /path/to/krea2-turbo.safetensors \
  --release-artifact release/krea2-pose-control-mix025.safetensors \
  --prompt "fantasy mage, ornate robes, cinematic lighting" \
  --pose-image /path/to/project-format-pose.png \
  --output output.png \
  --seed 42
```

Use control scale `1.0` first. `1.25–1.50` is an optional stronger-control range, but it is not uniformly better across pose classes. See [prompting.md](https://github.com/s-adhit/krea2-pose-controlnet/blob/main/prompting.md) for prompt structure, examples, and failure modes.

One optional Style-LoRA may be applied at a time, with its own explicit strength. It is separate and reversible, and was **not** used for the canonical hero/default generation. Multi-Style-LoRA composition is outside this release scope.

## Evaluation snapshot

Frozen final-validation results for the selected candidate:

| Candidate | PCK@0.05 | PCK@0.10 | PCK@0.20 | CLIP |
|---|---:|---:|---:|---:|
| Turbo base | 0.0356 | 0.1101 | 0.3387 | 0.34402 |
| **mix-025** | **0.4521** | **0.6043** | **0.7241** | 0.33694 |

Native geometry performed better on pose PCK than dynamic-768 in the frozen comparison, while dynamic-768 had slightly higher CLIP:

| Geometry | PCK@0.05 | PCK@0.10 | PCK@0.20 | CLIP |
|---|---:|---:|---:|---:|
| **Native aspect-preserving** | **0.2920** | **0.4027** | **0.5885** | 0.30746 |
| Dynamic-768 | 0.2478 | 0.3584 | 0.5310 | **0.30968** |

See the [frozen release decision](https://github.com/s-adhit/krea2-pose-controlnet/blob/main/docs/evaluation/release/FINAL_RELEASE_DECISION.md) and its [machine-readable contract](https://github.com/s-adhit/krea2-pose-controlnet/blob/main/docs/evaluation/release/final_release_v1.json) for the benchmark context.

## Limitations

- Overlapping or interacting multi-person poses remain difficult; use clean single-person controls where possible.
- The control does not encode fingers or facial keypoints, so it cannot provide fine hand/finger or facial-pose control. Hands may still be imperfect.
- Conflicting requests about body pose, subject count, or framing can reduce control fidelity.

## Base model and attribution

This adapter depends on Krea-2: training uses **Krea-2 Raw** and canonical inference uses **Krea-2 Turbo**. Obtain and use the required base-model checkpoint under its applicable terms. See [Krea-2](https://github.com/krea-ai/krea-2) and the [Krea-2 technical report](https://www.krea.ai/blog/krea-2-technical-report).

Project code, canonical inference, pose-format examples, and prompting guidance: [s-adhit/krea2-pose-controlnet](https://github.com/s-adhit/krea2-pose-controlnet).
