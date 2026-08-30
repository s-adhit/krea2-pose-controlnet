# Project handoff

## Current bounded objective

Prepare the immutable, full 16,503-sample 768 paired-latent cache and matching
768 pose-target sidecar required before the GH200 production-throughput matrix.
The locked candidate remains 768, flow MSE plus `normalized_coordinate_huber`,
`lambda_pose=0.04`, pose window `[0.10, 0.20]`, effective batch 32, R64/alpha64,
224 LoRA targets, trainable ControlInputLayer, frozen Krea-2 Raw base, and
native-only evaluation. No training science changed.

## Full 768 artifact contract

- Dataset snapshot: `/lambda/nfs/adhit/krea2-pose/posebridge_hf`.
- Authoritative project train manifest:
  `data/manifests/train.jsonl` SHA
  `0de42b99e40dea0726a9368a92d91ba950349999f2b2c590df85ac91df147542`.
  Its ordered parsed records are exactly identical to the snapshot manifest;
  the snapshot's raw-file SHA differs only in serialization, so both raw SHA
  and exact parsed order are deliberately checked without conflating them.
- Latent root (does **not** yet exist):
  `/lambda/nfs/adhit/krea2-pose/posebridge_latents_768`.
- Pose source: `/lambda/nfs/adhit/krea2-pose/pose_targets_v3`; its immutable
  `records.jsonl` SHA is verified as
  `dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f`.
- Pose output (does **not** yet exist):
  `/lambda/nfs/adhit/krea2-pose/pose_targets_v3_768`.
- `pose_controlnet/full_768_cache.py` is the production-only contract. It
  re-encodes RGB/control pixels under the deterministic 64-pixel-aligned 768
  bucket policy; it never reads/reuses native or Mixed-32 latents. Every shard
  preserves source/resized/crop/bucket geometry and finite aligned float32
  latents. `train_manifest_identity.json` stores all 16,503 ordered stems and
  manifest/order digests; `shards.json` becomes `complete: true` only after all
  planned shards validate.
- Valid final shards can be reused on resume; partial/corrupt shards are
  atomically rebuilt. A root with a conflicting manifest, policy, or shard plan
  is refused rather than overwritten.
- The new sidecar reprojections exact source-space keypoints/visibility from
  v3 through each cached geometry. COCO/HumanArt must resolve; Danbooru stays
  present but explicitly unavailable and has no fabricated people/keypoints.
  Sidecar metadata binds the source SHA, cache contract, manifest identity,
  ordered-stem hash, availability counts, and content SHA.
- Estimated latent storage is about 20 GiB (paired 16-channel float32 latents,
  plus small metadata); the pose sidecar should be roughly 190 MiB. Reserve at
  least 25 GiB to leave build/temporary headroom.

## Exact operator commands

The cache build VAE-encodes 16,503 RGB/control pairs and therefore requires the
GH200 CUDA environment. It is preprocessing only: no training, generation,
evaluation, checkpointing, or throughput benchmark.

```bash
cd /home/ubuntu/krea2-pose-controlnet
export DATASET=/lambda/nfs/adhit/krea2-pose/posebridge_hf
export TRAIN_MANIFEST=/home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl
export LATENT_768=/lambda/nfs/adhit/krea2-pose/posebridge_latents_768
export POSE_SOURCE=/lambda/nfs/adhit/krea2-pose/pose_targets_v3
export POSE_768=/lambda/nfs/adhit/krea2-pose/pose_targets_v3_768
PYTHONPATH=. python scripts/build_full_768_cache.py \
  --dataset-root "$DATASET" --output-root "$LATENT_768" \
  --train-manifest "$TRAIN_MANIFEST" \
  --pose-source "$POSE_SOURCE" --pose-output "$POSE_768" --device cuda
```

The same command is the safe resume command. It reuses only fully valid final
shards and refuses any scientifically conflicting root.

```bash
PYTHONPATH=. python scripts/verify_full_768_cache.py \
  --dataset-root "$DATASET" --train-manifest "$TRAIN_MANIFEST" \
  --latent-root "$LATENT_768" --pose-sidecar "$POSE_768"
```

The verifier is CPU/no-network/no-VAE. It checks exact manifest count/order,
completion marker, every planned shard/tensor/geometry, actual source-size
provenance, paired latent shape/finiteness, 768 policy only, sidecar identity,
authoritative-source SHA, and Danbooru/eligible-target availability. The
throughput runner invokes this verifier before CUDA/model work and now requires
`--dataset-root "$DATASET"`.

For the later benchmark, additionally export:

```bash
export RAW=/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors
export TEXT=/lambda/nfs/adhit/krea2-pose/text_conditioning
export DATASET=/lambda/nfs/adhit/krea2-pose/posebridge_hf
export TRAIN_MANIFEST=/home/ubuntu/krea2-pose-controlnet/data/manifests/train.jsonl
export LATENT_768=/lambda/nfs/adhit/krea2-pose/posebridge_latents_768
export POSE_768=/lambda/nfs/adhit/krea2-pose/pose_targets_v3_768
```

## This session

- Added `pose_controlnet/full_768_cache.py`, `scripts/build_full_768_cache.py`,
  `scripts/verify_full_768_cache.py`, and `tests/test_full_768_cache.py`.
- Updated `scripts/benchmark_production_trainer.py` so an unverified full 768
  cache/sidecar fails closed before GPU/model work; benchmark commands need the
  new `--dataset-root "$DATASET"` argument.
- PASS:
  `PYTHONPATH=. python -m unittest tests.test_full_768_cache tests.test_shards tests.test_pose_targets tests.test_production_throughput_benchmark -v`
- PASS:
  `PYTHONPATH=. python -m py_compile pose_controlnet/full_768_cache.py scripts/build_full_768_cache.py scripts/verify_full_768_cache.py scripts/benchmark_production_trainer.py tests/test_full_768_cache.py`
- No full cache/sidecar build was started; neither target root existed at
  inspection. No long training, generation, evaluation, A-G benchmark,
  commit, or push occurred.

## Next action

Run the exact build command on the GH200 host, then the verifier. Do not begin
the A-G throughput matrix until that verifier reports PASS.
