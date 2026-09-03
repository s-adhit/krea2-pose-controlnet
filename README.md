# Krea-2 Pose Control-LoRA

Skeleton-conditioned pose control for Krea-2. The pose image controls **body geometry**; the prompt controls **identity, clothing, style, lighting, and environment**.

Training uses **Krea-2 Raw**. Inference and evaluation use **Krea-2 Turbo**.

## Status

The Pose Control-LoRA has been trained for Krea-2 Raw and evaluated/deployed on Krea-2 Turbo. The current quantitative release candidate is **mix-025**, pending the remaining release experiments; it is not final.

![mix-025 held-out Turbo contact sheet](docs/evaluation/final-val-turbo/mix-025/full_contact_sheet.png)

Curated demonstrations remain available for [Fantasy Mage](docs/assets/showcase/final/fantasy-mage/generation.png), [Gojo-inspired Mural](docs/assets/showcase/final/gojo-mural/generation.png), and [Dark-fantasy Jester](docs/assets/showcase/final/jester/generation.png). Franchise-inspired examples are fan-art-style demonstrations only.

## Current evaluation candidate

`mix-025` is `75% parent-4000 + 25% A4300`, interpolated over trainable `state['model']` tensors only.

| Candidate | PCK@0.05 | PCK@0.10 | PCK@0.20 | CLIP |
|---|---:|---:|---:|---:|
| parent-4000 | 0.4301 | 0.5725 | 0.7105 | 0.33649 |
| **mix-025** | **0.4521** | **0.6043** | **0.7241** | 0.33694 |
| mix-050 | 0.4443 | 0.5991 | 0.7131 | 0.33411 |
| mix-075 | 0.4475 | 0.6036 | 0.7085 | 0.33542 |
| A4300 | 0.4404 | 0.5939 | 0.7170 | **0.33698** |

## How it works

The pose image is VAE-encoded and spatially aligned with the noisy image latent. The two are channel-concatenated before the expanded `ControlInputLayer`:

```text
noisy image latent + pose latent
              ↓
      ControlInputLayer
              ↓
          Krea-2
```

Pretrained image-input weights are preserved; the new control portion starts from zero. The backbone stays frozen while the expanded control input and rank-64 LoRA adapters learn pose control.

The control-input design was informed by [Tanmay Patil's Krea-2 ControlNet](https://github.com/Tanmaypatil123/Krea-2-controlnet), a public depth-conditioned Krea-2 ControlNet implementation. This project adapts that lineage to skeleton conditioning and provides its own pose training, evaluation, inference, and data pipeline.

## Training objective

```text
loss = flow_loss + 0.04 * pose_loss
```

Flow matching uses `x_t = t * noise + (1 - t) * x0` with target `noise - x0`. Pose consistency uses a frozen Keypoint R-CNN path and normalized-coordinate Huber loss. The idea of condition-consistency feedback was inspired by [ControlNet++](https://arxiv.org/abs/2404.07987); the exact pose loss is project-specific.

| Setting | Value |
|---|---|
| LoRA rank / alpha | 64 / 64 |
| Trainable parameters | ~215.5M |
| Precision / effective batch | BF16 / 32 |
| Optimizer / base LR | AdamW / `1e-4` |
| Pose-loss weight / seed | `0.04` / 42 |
| Resolution | Dynamic 768 buckets |

## Prompting

See the concise [prompting guide](prompting.md). The main rule is simple: **pose image = geometry; prompt = appearance, environment, and rendering**. Controlled studies show that framing/count conflicts and strong semantic priors can fight the pose condition.

![Prompting study contact sheet](docs/evaluation/prompting-guide/results/mix-025/prompting_study_contact_sheet.png)

## Style-LoRA composition

Isolated runtime composition has been implemented and tested with `darkbrush`, `rainywindow`, `retroanime`, and `realism`. Official trigger words matter for the first three; Style-LoRA deltas remain separate from the Pose-LoRA. The style-strength sweep is complete, but recommendations are still under review, so final strengths are not hard-coded here.

[Trigger-correct composition results](docs/evaluation/style-lora-composition/results/mix-025-strength-1.0-triggers) · [strength-sweep results](docs/evaluation/style-lora-composition/results/strength-sweep-v1)

![Style-LoRA strength sweep](docs/evaluation/style-lora-composition/results/strength-sweep-v1/grids/style_lora_strength_sweep_contact_sheet.png)

## Multilingual prompting

An English/Chinese fixed-pose sanity test exists. An English / Chinese / Telugu fixed-pose sanity comparison is available through the multilingual smoke harness; it is a small sanity test, not a multilingual benchmark or a claim of parity.

## Evaluation

Evaluation includes a frozen held-out 48-condition benchmark; PCK@0.05 / 0.10 / 0.20; CLIP; and an authoritative pose sidecar. Additional controlled studies cover semantic prompt-injection stress, prompting guidance, and Style-LoRA composition.

The locked Turbo contract is 8 steps, CFG 0, `mu=1.15`, and control scale 1.0. The diagnostic split is used for development and checkpoint selection; validation was held out from training but inspected for inference evaluation and is not an untouched test set.

## Inference

The canonical entry point is:

```bash
PYTHONPATH=. python inference.py \
  --turbo-ckpt /path/to/turbo.safetensors \
  --pose-lora-ckpt /path/to/checkpoint.pt \
  --prompt "fantasy mage, ornate robes, cinematic lighting" \
  --pose-image /path/to/pose.png \
  --output output.png \
  --seed 42 --width 768 --height 768 \
  --steps 8 --cfg 0 --mu 1.15 --control-scale 1.0
```

Use `--dynamic-768-bucket` for production bucket selection. Each output includes a JSON sidecar with its prompt, seed, resolution, checkpoint, pose image, and Turbo settings.

## Repository

```text
.
├── inference.py
├── prompting.md
├── pose_controlnet/
│   ├── production_training.py
│   ├── turbo_runtime.py
│   └── style_lora.py
├── scripts/
│   ├── final_val_turbo_benchmark.py
│   ├── frozen_prompt_turbo.py
│   ├── prompting_guide_study.py
│   ├── multilingual_prompt_smoke.py
│   └── style_lora_composition.py
├── docs/
│   ├── CODEX_HANDOFF.md
│   └── evaluation/
└── tests/
```

For GH200 development/restart setup, see [`scripts/bootstrap_gh200.sh`](scripts/bootstrap_gh200.sh).

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
