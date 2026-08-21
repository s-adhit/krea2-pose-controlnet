# Krea2-Pose-ControlNet

LoRA-based pose-conditioned image generation on top of Krea-2's 13B RAW DiT.
Pose skeleton maps are channel-concatenated with the noisy latent, following
the same mechanism Krea-2's own depth-ControlNet uses — trained with an
original recipe against `pose_controlnet/`, not a fork of the reference
training code.

## Structure

- `base_model/` — `mmdit.py` / `k2_lora.py`, the frozen Krea-2 13B DiT
  architecture. Required to load the real pretrained checkpoint; not part of
  the pose-controlnet training code itself.
- `pose_controlnet/` — this project's own training stack: data loading,
  flow-matching diffusion mechanics, text conditioning, checkpointing
  (with a wall-clock HF Hub mirror), wandb logging, and seeding.
- `prepare_shards.py` — builds VAE-encoded latent shards from
  `data/full/` + `data/manifests/`.
- `train.py` — training entrypoint.
- `evaluate.py` — evaluation entrypoint (loss-based + turbo generation eval).
- `scripts/` — dataset audit/prep tooling (bucket analysis, caption
  sanity checks, VAE round-trip tests, shard verification, model prefetch).
- `data/` — `full/` and `prepared/` are gitignored (images + shards live
  only on disk/GH200, never in git). `manifests/`, `review/`, and `stats/`
  are tracked — they're the provenance record (splits, exclusions, bucket
  stats) for the PoseBridge dataset, not the images themselves.

## Reference

The base architecture (`base_model/mmdit.py`, `base_model/k2_lora.py`) was
copied from Krea-2's depth-ControlNet reference implementation:
https://github.com/Tanmaypatil123/Krea-2-controlnet

That reference repo is not vendored into this repository — only the two
architecture files needed to load the real pretrained weights are kept,
under `base_model/`. Everything else here (data pipeline, training loop,
eval, checkpointing/wandb/seed guardrails) is original to this project.

## Dataset: PoseBridge

~17,416 image/pose-skeleton pairs across 5 buckets (COCO, Danbooru, and
three Human-Art sub-buckets: painting, real_human, sculpture). Full
composition, caption methodology, and known dataset quirks are documented in
`data/stats/` and `data/review/`.

**License note**: the Human-Art sub-buckets (painting/real_human/sculpture,
~11,347 of the ~17,416 images) are under a non-commercial, no-redistribution
license. `data/full/` is never committed to this repo and should never be
uploaded to any public host.

## Status

Data prep, captioning, and validation are complete. Training infra
(checkpointing, wandb, seed) is written; GH200 provisioning and the first
real training run are in progress.