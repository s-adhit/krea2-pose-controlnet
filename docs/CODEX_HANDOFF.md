# Phase 1 handoff

## Current bounded objective and status

An evaluation-only Krea-2 Turbo benchmark is implemented but deliberately has
not run the expensive GH200 generation, PCK detector, or CLIP work. It does not
train, construct an optimizer, call backward, change LoRA/training state,
mutate inputs/checkpoints, commit, or push.

## Turbo contract and upstream evidence

- Official source: `https://github.com/krea-ai/krea-2`.
- Official `inference.py` constructs both RAW and Turbo with the identical
  `SingleMMDiTConfig` (6144 features, 28 layers, patch 2, 16 channels) and
  loads Turbo from `OSS_TURBO` with strict state-dict loading.
- Official `sampling.py:timesteps` is reproduced exactly: `torch.linspace(1,
  0, steps + 1)`, then `exp(mu)/(exp(mu) + (1/t - 1)**1.0)`. Turbo is pinned to
  `steps=8`, `cfg=0.0`, `mu=1.15`; Turbo mu is explicitly **not resolution
  dependent**. CFG zero prepares only conditional text and executes one model
  forward per denoise step.
- The new builder strictly validates all Turbo base state-dict keys/shapes
  against the shared official config before control surgery. Each full training
  checkpoint must record nonempty `config.raw_ckpt`, and its trainable
  control/LoRA key set and every tensor shape must exactly match the expanded
  Turbo model before loading. This is the Raw-trained-control -> Turbo loading
  procedure; it fails rather than guessing.

## Evaluation rules in force

- Exact checkpoints only: 800 and 1500. Local valid copies are preferred;
  recovery is only `pose-learning-1500/full/step_000800.pt` and
  `pose-learning-1500/full/step_001500.pt` from
  `adhit-420/Krea-2-PoseControl-LoRA-checkpoints`. Existing completion marker,
  SHA-256, full-deserialization/schema, and embedded-step checks are reused.
- The immutable 24-record `data/manifests/diagnostic_val.jsonl` order and all
  cached latent/text identities, per-stem seeds, paired geometry, controls,
  decode behavior, and buckets are shared by both checkpoints.
- Output is isolated at `/lambda/nfs/adhit/krea2-pose/evaluation/turbo-8step-cfg0`;
  the canonical `evaluation/pose-learning-500` tree (and descendants) is
  rejected.
- Authoritative PCK and CLIP helpers are reused unchanged. The 21 available
  Human-Art/COCO samples retain source-visible-and-rendered eligibility,
  COCO-17 Keypoint R-CNN / Hungarian / bbox-diagonal / `<=` threshold
  semantics. Three Danbooru records remain unavailable and excluded from PCK
  denominators (with null PCK and the required reason).

## Files changed this session

- `base_model/k2_lora.py`, `pose_controlnet/model.py`: generic strict official
  base-checkpoint validation plus Turbo Pose-Control model builder.
- `pose_controlnet/turbo_evaluation.py`: exact Turbo schedule, CFG-disabled
  controlled sampler, isolation, exact checkpoint, diagnostic, and Raw->Turbo
  compatibility guards.
- `scripts/turbo_benchmark.py`: separate preflight/generate/score/report flow.
- `tests/test_turbo_evaluation.py`: focused schedule/CFG/control/checkpoint/
  isolation/compatibility/evaluation-only tests.

## Verified tests

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile base_model/k2_lora.py pose_controlnet/model.py pose_controlnet/turbo_evaluation.py scripts/turbo_benchmark.py tests/test_turbo_evaluation.py`
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_turbo_evaluation tests.test_evaluation tests.test_post1500_evaluation tests.test_train_mechanics` (54 tests).
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python scripts/turbo_benchmark.py --help`.

## Exact GH200 execution (do not train)

```bash
export UV_CACHE_DIR=/tmp/krea_uv_cache
export OSS_TURBO=/lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors
cd /home/ubuntu/Krea-2-Pose-ControlNet
test -f "$OSS_TURBO"
uv run python scripts/turbo_benchmark.py preflight --turbo-ckpt "$OSS_TURBO"
uv run python scripts/turbo_benchmark.py generate --turbo-ckpt "$OSS_TURBO"
uv run python scripts/turbo_benchmark.py score --turbo-ckpt "$OSS_TURBO"
uv run python scripts/turbo_benchmark.py report --turbo-ckpt "$OSS_TURBO"
```

Expected output files: `turbo_spec.json`, `checkpoint_preflight.json`,
`fixed_pose/<stem>/{control.png,metadata.json,step_000800.png,step_001500.png}`,
`generation_results.json`, `pck_clip_results.json`, `turbo_comparison_grid.png`,
and `evaluation_summary.json` with the requested machine-readable table.
