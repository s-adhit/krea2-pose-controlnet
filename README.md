# Krea-2 Pose Control-LoRA

Frozen pose-conditioned image generation for Krea-2 Turbo. A rendered
PoseBridge skeleton controls spatial human geometry; the prompt controls
compatible appearance and rendering.

The released control mechanism is **not** a parallel ControlNet branch. The
clean pose latent is spatially aligned and channel-concatenated with the noisy
image latent before an expanded `ControlInputLayer`. Krea-2 Raw was used for
training; the image-input backbone remains frozen while the control half of
that layer and rank-64 LoRA adapters are trained. Frozen inference uses
Krea-2 Turbo.

## Frozen release

- Artifact: `krea2-pose-control-mix025.safetensors`
- Release identity: `krea2-pose-control-lora-v1`, candidate `mix-025`
- SHA-256: `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`
- Defaults: Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0,
  native/aspect-preserving geometry, no Style-LoRA

The release decision, artifact lineage, locked runtime, and limitations are in
[docs/evaluation/release/final_release_v1.json](docs/evaluation/release/final_release_v1.json).

## Architecture

```text
rendered skeleton -> clean pose latent
noisy image latent + clean pose latent -> channel concat -> ControlInputLayer
                                                     -> frozen Krea-2 + rank-64 LoRA
```

The prompt must not compete with the condition on person count, framing, or
geometry. See [prompting.md](prompting.md) for the control/prompt contract.

## Evidence and assets

Training and evaluation are complete. The primary result record is
[docs/blog-evidence/EVALUATION_RESULTS.md](docs/blog-evidence/EVALUATION_RESULTS.md);
methodology and caveats are beside it in `docs/blog-evidence/`. Frozen
benchmark specifications and payloads live in [docs/evaluation](docs/evaluation).

Qualitative and blog-ready assets are indexed in
[docs/blog-assets/BLOG_ASSET_INDEX.md](docs/blog-assets/BLOG_ASSET_INDEX.md),
including the hero showcase, geometry comparison, hard-pose examples, and the
appendix regeneration manifest.

## Inference and ComfyUI

Use [inference.py](inference.py) for the canonical CLI contract. It validates
the frozen release before loading and writes an output provenance sidecar. The
CLI defaults to native/aspect-preserving geometry; explicit dimensions must be
positive multiples of 16.

[comfyui/README.md](comfyui/README.md) documents the completed standalone
ComfyUI integration, including the DWPose-to-PoseBridge conversion path and
frozen model paths.

## Repository status

Model development, training, evaluation, frozen release, and ComfyUI
integration are complete. The repository is in its blog/documentation and
evidence-archive phase; it does not have a future adapter roadmap.

## References

- [Krea-2](https://github.com/krea-ai/krea-2) and its [technical report](https://www.krea.ai/blog/krea-2-technical-report)
- [Tanmay Patil's Krea-2 ControlNet](https://github.com/Tanmaypatil123/Krea-2-controlnet), an architectural reference only
- [LoRA](https://arxiv.org/abs/2106.09685), [Flow Matching](https://arxiv.org/abs/2210.02747), and [ControlNet++](https://arxiv.org/abs/2404.07987)
