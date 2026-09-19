# Project handoff

## Current objective

Training-methodology and infrastructure evidence for the final Krea-2 Pose
Control-LoRA lineage is frozen for blog use. Do not alter frozen evaluation
results, NFS checkpoints, release identity, or historical logs. Hero-v2 also
remains frozen for human visual review; its authoritative winner manifest is
`docs/showcase/final/hero-v2/final_winners.json`.

## Final training evidence frozen

- Release: `mix-025` / `krea2-pose-control-lora-v1`, float32
  `0.75 * parent-4000 + 0.25 * A4300` over trainable `state['model']` only.
  Endpoint hashes were recomputed and match the release contract.
- Exact trainable state: 215,488,512 float32 parameters / 450 tensors:
  expanded ControlInputLayer plus rank-64 A/B LoRA on eight targets in each of
  28 blocks; backbone frozen.
- Recipe: AdamW `(0.9,0.99)`, eps `1e-8`, zero weight decay, clip `1.0`,
  microbatch 1, accumulation 32, effective batch 32, BF16 autocast, seed 42,
  caption dropout 0.10, control dropout 0, no compile/fused AdamW, and
  gradient checkpointing disabled (`gradient_checkpointing_blocks=0`). All
  three final-lineage metadata records have zero blocks; at `c5771ef`, the
  production recipe locks zero and `build_train_config` explicitly passes
  `gradient_checkpointing=False`/zero blocks. Generic `train.py` supports up
  to 28 blocks, but that capability was not enabled for these production runs.
- Objective: flow velocity MSE `x_t=t*eps+(1-t)*x_0`, target `eps-x_0`, plus
  `0.04` normalized-coordinate Huber (`delta=1`) only for eligible shifted
  timesteps `[0.10,0.20]`; frozen fixed-box COCO Keypoint R-CNN, without
  detector/RPN/NMS/argmax.
- Runtime conclusion: endpoint-relevant metric-step time through A4300 is
  17:51:05. Observed spans are 12:24:47 (initial), 4:10:42 (to parent-4000),
  and 1:17:25 (to A4300); 25:06:48 calendar span includes inter-run gaps.
- Host record: PyTorch normal-host device name `NVIDIA GH200 480GB` (the
  `480GB` string is a device/product name, not an HBM-capacity claim); directly
  verified PyTorch-visible total device memory `101468602368` bytes (94.5
  GiB). ARM64 host is Ubuntu 22.04.5, Python 3.10.12, PyTorch 2.7.0/CUDA 12.8,
  cuDNN 9.8, Triton 3.3.0, driver 580.105.08. `.venv` inherits system packages;
  data/checkpoints are NFS.
- Evidence outputs: `docs/blog-evidence/TRAINING_METHODS_INFRA.md`,
  `docs/blog-evidence/training_methods_infra.json`, and
  `docs/blog-evidence/RUNTIME_AUDIT.md`.
- Unresolved rather than guessed: training-time uv version and complete
  preprocessing/startup/upload/eval wall-clock totals.

## Frozen hero-v2 winners

The authoritative manifest is `docs/showcase/final/hero-v2/final_winners.json`.
Its order is also the left-to-right order of both final montages:

1. psychedelic swordswoman
2. fantasy mage
3. comic fashion
4. starry-night painterly
5. dark-fantasy jester
6. elegant male warrior / wandering knight (`canonical-v1`)
7. gothic masked noble with attendant (`canonical-v1`)
8. painterly mythic companions (`retry-a`)
9. moonlit lotus princess (`retry-b`)
10. astral empress / cosmic oracle (`retry-b`)

Explicitly excluded from the final hero are realistic female warrior, realistic
fashion/editorial portrait, original moonlit priestess/dreamy floral oracle,
and stained-glass saint/celestial figure.

## Final presentation assets

- Generation montage:
  `docs/showcase/final/hero-v2/final/final_generation_montage.png`
- Matching condition montage:
  `docs/showcase/final/hero-v2/final/final_condition_montage.png`
- Stable numbered generation/condition copies:
  `docs/showcase/final/hero-v2/final/`

`scripts/package_hero_v2_final.py` deterministically copies the selected,
already-recorded source images and controls, records prompt/seed/geometry/
candidate/control-scale provenance plus SHA-256s in the frozen manifest, and
builds the two same-order justified montages. It does not generate images.

## Documentation refreshed

- `README.md` now presents the hero-v2 generation montage and links the
  matching conditions.
- `docs/release/HF_MODEL_CARD.md` uses the same montage through the raw GitHub
  URL.
- `prompting.md` preserves the geometry-versus-appearance guidance and now has
  the final ten concise, exact-prompt examples with their condition/generation
  images, seed, native dimensions, candidate, and control scale.

## Verified release/environment facts

- Release candidate: `mix-025`; release ID: `krea2-pose-control-lora-v1`.
- Release SHA-256:
  `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
- Runtime contract: Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, native geometry,
  no Style-LoRA. Most heroes use control scale `1.0`; painterly mythic
  companions uses `1.25`.
- Normal-host verification records PyTorch device name `NVIDIA GH200 480GB`
  (a reported name, not an HBM-capacity claim) and visible device memory 94.5
  GiB. The host is Linux ARM64 with Python 3.10.12, PyTorch 2.7.0 / CUDA 12.8 /
  cuDNN 9.8 / Triton 3.3.0; BF16, SDPA, and `torch.compile` passed from the
  normal host shell. Use `uv`.

## Files changed this session

- `scripts/package_hero_v2_final.py`
- `docs/showcase/final/hero-v2/final_winners.json`
- `docs/showcase/final/hero-v2/final/` (20 selected source copies and two
  montages)
- `README.md`
- `docs/release/HF_MODEL_CARD.md`
- `prompting.md`
- `docs/CODEX_HANDOFF.md`
- `docs/blog-evidence/TRAINING_METHODS_INFRA.md`
- `docs/blog-evidence/training_methods_infra.json`

## Verification

PASS:

```bash
uv run python scripts/package_hero_v2_final.py
# visual inspection of both final montages
```

Before ending, run `git diff --check` and `git status --short`. Do not commit
or push.

## Next recommended action

Evaluation-metric verification: audit frozen final-val, hard-pose,
control-scale, and hand metrics before making blog-performance claims. Preserve
the frozen evaluation artifacts and release contract.
