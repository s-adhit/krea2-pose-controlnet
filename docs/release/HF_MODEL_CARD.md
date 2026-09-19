---
tags:
  - krea-2
  - pose-control
  - lora
  - image-generation
---

# Krea-2 Pose Control-LoRA

Pose-conditioned image generation for **Krea-2** using a 17-keypoint body skeleton.

<p align="center">
  <img src="https://raw.githubusercontent.com/s-adhit/krea2-pose-controlnet/main/docs/showcase/final/hero-v2/final/final_generation_montage.png" alt="Krea-2 Pose Control showcase" width="100%">
</p>

The pose condition controls broad body geometry and placement, while the prompt controls subject appearance, clothing, environment, lighting, and style.

## Model

Recommended release:

```text
release/krea2-pose-control-mix025.safetensors
```

* Candidate: `mix-025`
* Saved parameters: `215,488,512`
* SHA-256: `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`

The other checkpoint folders in this repository are historical training runs. `release/` contains the recommended public model.

## Pose conditioning

The model uses 17 COCO-style body keypoints:

* nose, eyes, ears
* shoulders, elbows, wrists
* hips, knees, ankles

Hands, fingers, and face landmarks are not part of the control signal.

Use pose rasters generated in this project's trained format rather than assuming arbitrary OpenPose renders are compatible.

## Inference

Recommended settings:

```text
Base: Krea-2 Turbo
Steps: 8
CFG: 0
mu: 1.15
Control scale: 1.0
Geometry: native aspect-preserving
```

```bash
PYTHONPATH=. python inference.py \
  --turbo-ckpt /path/to/krea2-turbo.safetensors \
  --release-artifact release/krea2-pose-control-mix025.safetensors \
  --prompt "fantasy mage, ornate robes, cinematic lighting" \
  --pose-image /path/to/pose.png \
  --output output.png \
  --seed 42
```

For stronger pose adherence, a control scale of `1.25–1.50` can also be useful.

See the [prompting guide](https://github.com/s-adhit/krea2-pose-controlnet/blob/main/prompting.md) for examples and prompt recommendations.

## Results

| Model       |   PCK@0.05 |   PCK@0.10 |   PCK@0.20 |    CLIP |
| ----------- | ---------: | ---------: | ---------: | ------: |
| Turbo base  |     0.0356 |     0.1101 |     0.3387 | 0.34402 |
| **mix-025** | **0.4521** | **0.6043** | **0.7241** | 0.33694 |

Native aspect-preserving inference gave better pose adherence than dynamic-768 in the frozen evaluation.

## Limitations

* Overlapping multi-person poses are harder.
* Fingers and facial pose are not directly controlled.
* Prompts that conflict with the supplied pose, framing, or subject count can reduce pose adherence.

## Links

* [GitHub](https://github.com/s-adhit/krea2-pose-controlnet)
* [Prompting guide](https://github.com/s-adhit/krea2-pose-controlnet/blob/main/prompting.md)
* [Release decision](https://github.com/s-adhit/krea2-pose-controlnet/blob/main/docs/evaluation/release/FINAL_RELEASE_DECISION.md)
* [PoseBridge dataset](https://huggingface.co/datasets/adhit-420/PoseBridge)
* [Krea-2](https://github.com/krea-ai/krea-2)
