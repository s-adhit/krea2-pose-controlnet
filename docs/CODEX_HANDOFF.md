# Project handoff

## Current bounded objective

Production-throughput audit completed locally; no GH200 timing run has been
performed. The locked candidate is 768 training, flow MSE plus
`normalized_coordinate_huber`, `lambda_pose=0.04`, pose window `[0.10, 0.20]`,
native-only evaluation, R64/alpha64, exactly 224 LoRA targets, trainable
`ControlInputLayer`, frozen Krea-2 Raw base, AdamW `1e-4` / `(0.9,0.99)` / zero
weight decay, warmup 200, and effective batch 32.

## Throughput audit findings

- `train.py` reads cached paired image/control latents and cached text
  conditioning; it performs no per-step VAE encoding or pixel preprocessing.
  The candidate pose path necessarily retains an autograd VAE *decode* and
  frozen fixed-box Keypoint R-CNN only for pose-active samples.
- Training currently uses direct deterministic shard reads, not a DataLoader.
  The new benchmark can compare that production baseline with worker/pinning/
  prefetch candidates without changing the production default.
- Attention already calls `scaled_dot_product_attention` under
  `SDPBackend.CUDNN_ATTENTION`; no model-math rewrite is justified.
- Existing `torch.compile` only wraps the rank-stable text MLP. It remains
  opt-in and is benchmark-only until measured stable/faster.
- Existing 768 cache support is scoped to Mixed-32. The benchmark deliberately
  requires a separately prepared, immutable full-16,503-sample 768 latent root
  and exact full-768 pose sidecar, and rejects native/mismatched geometry.

## Changes this session

- Added `scripts/benchmark_production_trainer.py`: no checkpoint, generation,
  evaluation, W&B, or data mutation; 10 warmup / 20 timed optimizer steps by
  default; records forward/backward/optimizer/total/data-wait timing,
  throughput, active pose fraction, VRAM, effective batch, and projections.
- Added `pose_controlnet/throughput_benchmark.py`,
  `scripts/estimate_training_runtime.py`, and
  `scripts/summarize_production_benchmark.py`.
- Added opt-in `--fused-adamw` (default off). It preserves AdamW parameters and
  hyperparameters, rejects unsupported/non-CUDA use, and is not recommended
  until the GH200 benchmark proves it.
- Removed two unlogged control-RMS/std CUDA synchronizations from non-
  diagnostic microbatches. Finite checks and diagnostic-step values remain.
- Normal `train.py --resume` now verifies saved model/batch/optimizer/sampler/
  numerical-runtime identity before restoring weights, optimizer, scheduler,
  global step, Python/NumPy/torch/CUDA RNG, flow generator, epoch, and batch
  position. Cadences and terminal max step remain operationally adjustable.

## Benchmark matrix and exact GH200 commands

Set the four read-only inputs to the actual full 768 artifacts; do not point
them at Mixed-32 or native shards. `OUT` is a new benchmark-result directory.

```bash
cd /home/ubuntu/krea2-pose-controlnet
export RAW=/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors
export LATENT_768='<full verified 16,503-sample 768 latent root>'
export TEXT=/lambda/nfs/adhit/krea2-pose/text_conditioning
export POSE_768='<immutable full-train 768 pose sidecar directory>'
export OUT=/lambda/nfs/adhit/krea2-pose/throughput_benchmarks/2026-08-30
mkdir -p "$OUT"
bench () { PYTHONPATH=. python scripts/benchmark_production_trainer.py --raw-ckpt "$RAW" --latent-root "$LATENT_768" --text-conditioning-root "$TEXT" --pose-sidecar "$POSE_768" --output-json "$OUT/$1.json" --label "$1" "${@:2}"; }
```

Run one command at a time while a second terminal records
`nvidia-smi dmon -s pucvmt -d 1` during the timed window:

```bash
# A current production-equivalent baseline: the configured default is no checkpointing.
bench A-baseline --gradient-checkpointing-blocks 0
# B checkpointing on; compare its saved VRAM against recomputation cost.
bench B-checkpoint-all --gradient-checkpointing-blocks 28
# C one DataLoader/cache axis; retain direct-loader baseline as its own row.
bench C-loader4 --gradient-checkpointing-blocks 0 --data-loader-workers 4 --persistent-workers --pin-memory --prefetch-factor 4
# D fused AdamW axis.
bench D-fused-adamw --gradient-checkpointing-blocks 0 --fused-adamw
# E microbatch axes, always exactly effective batch 32.
bench E-micro2-acc16 --gradient-checkpointing-blocks 0 --microbatch-size 2 --gradient-accumulation-steps 16
bench E-micro4-acc8 --gradient-checkpointing-blocks 0 --microbatch-size 4 --gradient-accumulation-steps 8
# F compile candidate; setup_seconds_including_compile is reported separately.
bench F-compile --gradient-checkpointing-blocks 0 --compile
# Pose incremental cost: same best single-axis configuration, flow-only only.
bench pose-cost-flow-only --gradient-checkpointing-blocks 0 --objective flow_only
# G only after individual rows: one measured combined candidate.
bench G-combined --gradient-checkpointing-blocks 0 --data-loader-workers 4 --persistent-workers --pin-memory --prefetch-factor 4 --fused-adamw --microbatch-size 2 --gradient-accumulation-steps 16 --compile
```

Compact summary and runtime projection commands:

```bash
PYTHONPATH=. python scripts/summarize_production_benchmark.py --inputs "$OUT"/*.json
PYTHONPATH=. python scripts/estimate_training_runtime.py --seconds-per-optimizer-step '<measured optimizer_step_seconds_mean>' --effective-batch-size 32 --training-samples 16503
```

## Safety and remaining decision

Every row hard-locks effective batch 32, 768 bucket geometry, candidate loss
name/lambda/window, seed 42, R64 architecture and audited trainable parameter
set. The runner loads fresh Raw+LoRA weights for each row, emits the exact
trainable names/count, uses the same cached-caption dropout and timestep path,
and does not detach pose gradients. Fused AdamW changes only the optimizer
kernel; checkpointing/compile/data settings do not alter loss definitions.
Do not enable any option in the production command until its isolated row and
the final combined row have been compared for finite loss/gradients, matching
trainable list, pose active fraction, and acceptable numerical tolerance.

PASS (CPU/no-network):

```bash
PYTHONPATH=. python -m unittest tests.test_production_throughput_benchmark tests.test_train_mechanics tests.test_pose_reward_tools tests.test_overfit_capacity -v
PYTHONPATH=. python -m py_compile train.py pose_controlnet/config.py pose_controlnet/throughput_benchmark.py scripts/benchmark_production_trainer.py scripts/estimate_training_runtime.py scripts/summarize_production_benchmark.py scripts/train_pose_reward_smoke.py scripts/train_overfit_capacity.py tests/test_production_throughput_benchmark.py
git diff --check
```

No long production training, generation, evaluation, checkpoint write,
commit, or push occurred. Next action: provision/verify the full 768 cache and
matching immutable full pose sidecar, then run A through F individually before
the one G combined benchmark and paste the compact JSON summary back.
