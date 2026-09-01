
# Krea-2 Pose Control-LoRA

Skeleton-conditioned pose control for Krea-2.

The pose image controls **body geometry**. The prompt controls **identity, clothing, style, lighting, and environment**.

Training uses **Krea-2 Raw**. Inference and evaluation use **Krea-2 Turbo**.

---

## Showcase

### Fantasy Mage

| Pose condition | Generation |
|---|---|
| <img src="docs/assets/showcase/final/fantasy-mage/condition.png" width="320"> | <img src="docs/assets/showcase/final/fantasy-mage/generation.png" width="320"> |

### Gojo-inspired Mural

| Pose condition | Generation |
|---|---|
| <img src="docs/assets/showcase/final/gojo-mural/condition.png" width="320"> | <img src="docs/assets/showcase/final/gojo-mural/generation.png" width="320"> |

### Dark-fantasy Jester

| Pose condition | Generation |
|---|---|
| <img src="docs/assets/showcase/final/jester/condition.png" width="320"> | <img src="docs/assets/showcase/final/jester/generation.png" width="320"> |

<p align="center">
  <strong>Pose condition → generated interpretation</strong>
</p>

> Franchise-inspired examples are fan-art-style demonstrations only.

---

## How it works

The pose image is encoded with the VAE and spatially aligned with the noisy image latent.

The two are concatenated before entering an expanded `ControlInputLayer`:

```text
noisy image latent + pose latent
              ↓
      ControlInputLayer
              ↓
          Krea-2
````

The pretrained image-input weights are preserved while the new control portion starts from zero.

The backbone remains frozen, and the model learns pose control through:

* the expanded control input;
* rank-64 LoRA adapters across the transformer.

The control-input design was informed by [Tanmay Patil's Krea-2 ControlNet](https://github.com/Tanmaypatil123/Krea-2-controlnet), a public depth-conditioned Krea-2 ControlNet implementation. This project adapts that idea to skeleton conditioning and adds the pose training, evaluation, inference, and data pipeline used here.

---

## Training objective

Training combines the normal flow-matching objective with explicit pose consistency:

```text
loss = flow_loss + 0.04 * pose_loss
```

Flow matching uses:

```text
x_t = t * noise + (1 - t) * x0
target = noise - x0
x0_hat = x_t - t * v_hat
```

Pose consistency is calculated from the predicted clean image using a frozen Keypoint R-CNN path and a normalized-coordinate Huber loss.

The idea of explicit condition-consistency feedback was inspired by [ControlNet++](https://arxiv.org/abs/2404.07987). The exact pose loss used here is project-specific.

---

## Training setup

| Setting              | Value               |
| -------------------- | ------------------- |
| LoRA rank / alpha    | 64 / 64             |
| Trainable parameters | ~215.5M             |
| Precision            | BF16                |
| Effective batch      | 32                  |
| Optimizer            | AdamW               |
| Base LR              | `1e-4`              |
| Pose-loss weight     | `0.04`              |
| Resolution           | Dynamic 768 buckets |
| Seed                 | 42                  |

Training used a staged process:

```text
production training
      ↓
cooldown
      ↓
finishing experiments
```

The two checkpoints still under consideration are:

* **parent-4000** — more balanced
* **A4300** — stronger pose adherence

The annealed finishing branch was tested and rejected as a final candidate.

---

## Resolution buckets

Training uses nine aspect-ratio-preserving 768-class buckets:

```text
768x768
704x896   896x704
640x960   960x640
576x1024  1024x576
512x1152  1152x512
```

---

## Evaluation

The locked Turbo evaluation setup is:

```text
8 steps
CFG 0
mu 1.15
```

Pose quality is evaluated using:

* Keypoint R-CNN;
* deterministic person matching;
* bbox-normalized PCK;
* detection coverage;
* CLIP similarity.

The diagnostic split is used for development and checkpoint selection.

The validation split was held out from training, but has since been inspected during inference evaluation, so it is not treated as an untouched test set.

---

## Inference

The canonical entry point is:

```bash
PYTHONPATH=. python inference.py \
  --turbo-ckpt /path/to/turbo.safetensors \
  --pose-lora-ckpt /path/to/checkpoint.pt \
  --prompt "fantasy mage, ornate robes, cinematic lighting" \
  --pose-image /path/to/pose.png \
  --output output.png \
  --seed 42 \
  --width 768 \
  --height 768 \
  --steps 8 \
  --cfg 0 \
  --mu 1.15 \
  --control-scale 1.0
```

Dynamic production bucket selection is also available with:

```bash
--dynamic-768-bucket
```

Each output includes a JSON sidecar with the prompt, seed, resolution, checkpoint, pose image, and Turbo settings.

---

## Prompting

The model works best when the pose image controls **geometry** and the prompt controls **appearance**.

Good prompt content:

```text
character
clothing
materials
lighting
environment
art style
color palette
```

Be careful with prompt language that also tries to control pose:

```text
close-up
low angle
looking over shoulder
hand on hip
arms raised
specific stance
```

Those instructions can compete with the skeleton.

Portrait-heavy prompts are currently one of the harder cases.

For showcase and evaluation conditions, the current curation policy uses **COCO and Human-Art** pose controls.

---

## Repository

```text
inference.py
    Canonical Turbo pose inference

scripts/train_production.py
    Production training entry point

pose_controlnet/production_training.py
    Training, checkpointing, and resume

pose_controlnet/pose_consistency.py
    Pose-consistency objective

pose_controlnet/turbo_runtime.py
    Locked Turbo runtime

pose_controlnet/resolution_policy.py
    Shared resolution buckets

docs/inference_eval/
    Evaluation artifacts

docs/assets/showcase/
    Curated examples
```

---

## Status

Completed:

* pose-conditioning architecture;
* production training;
* pose-consistency supervision;
* dynamic-resolution training;
* exact checkpoint resume;
* Turbo inference;
* quantitative pose evaluation;
* parent-4000 vs A4300 comparison;
* initial showcase curation.

Still being explored:

* parent-4000 ↔ A4300 checkpoint mixing;
* style-LoRA composition;
* control-strength sweeps;
* hand/finger quality;
* difficult and multi-person poses;
* ComfyUI support;
* Hugging Face demo.

---

## References

* [Krea-2](https://github.com/krea-ai/krea-2) — [Technical Report](https://www.krea.ai/blog/krea-2-technical-report)
* [Tanmay Patil — Krea-2 ControlNet](https://github.com/Tanmaypatil123/Krea-2-controlnet)
* [ControlNet](https://arxiv.org/abs/2302.05543)
* [ControlNet++](https://arxiv.org/abs/2404.07987)
* [LoRA](https://arxiv.org/abs/2106.09685)
* [Flow Matching](https://arxiv.org/abs/2210.02747)
* [Rectified Flow](https://arxiv.org/abs/2209.03003)
* [COCO](https://arxiv.org/abs/1405.0312)
* [Human-Art](https://arxiv.org/abs/2303.02760)
* [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
* [CLIP](https://arxiv.org/abs/2103.00020)
* [Mask R-CNN](https://arxiv.org/abs/1703.06870)


