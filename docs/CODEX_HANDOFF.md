# Project handoff

## Current objective and status

The production launcher is `scripts/train_production.py`, separate from the
bounded Gate-F `train.py` path. The locked production run is not authorized to
start from Codex.

This session added the dual-mode production milestone Turbo evaluation harness.
No training, evaluation, image generation, network access, checkpoint upload,
commit, or push occurred.

`scripts/evaluate_production_milestones.py` evaluates exactly the six local
production milestones `500, 1000, 1500, 2000, 2500, 3000` in both isolated
modes:

- `native`: the historical primary benchmark. It retains the immutable native
  diagnostic latents, cached text, fixed stems/prompts/seeds, and the exact
  persisted paired geometry.
- `dynamic-768`: the deployment/generalization benchmark. It resolves the
  original diagnostic pair, selects only the shared nine-bucket 768 policy,
  applies the shared resize-to-cover/center-crop geometry, VAE-encodes the
  resulting paired control, and maps source keypoints through that exact
  geometry before PCK.

The output contract is strict and mode-separated:

```text
<output>/step_000500/native/
<output>/step_000500/dynamic-768/
...
<output>/step_003000/native/
<output>/step_003000/dynamic-768/
```

Partial artifacts fail closed rather than being overwritten. Each sample
metadata file records mode and geometry. The cross-checkpoint JSON and CSV
summaries both retain an explicit `mode` field/column, so the primary native
series cannot be conflated with dynamic-768.

## Locked production recipe

- Dataset snapshot: `/lambda/nfs/adhit/krea2-pose/posebridge_hf`
- Production train cache: `/lambda/nfs/adhit/krea2-pose/posebridge_latents_768`
- Historical diagnostic latent cache: `/lambda/nfs/adhit/krea2-pose/posebridge_latents`
- Text cache: `/lambda/nfs/adhit/krea2-pose/text_conditioning`
- Frozen Raw backbone + ControlInputLayer + rank/alpha-64 LoRA, existing 224
  target topology.
- Objective: flow MSE plus `normalized_coordinate_huber`, `lambda_pose=0.04`,
  natural pose window `[0.10, 0.20]`, forced exposure 0.
- AdamW `1e-4`, betas `(0.9, 0.99)`, weight decay 0, max-grad norm 1, 200-step
  warmup; microbatch 1, accumulation 32, BF16, seed 42.
- GH200 loader setting: workers 4, persistent workers, pinned memory,
  prefetch factor 4. Gradient checkpointing, compile, and fused AdamW remain
  disabled.

Production checkpoints save every 250 steps; HF mirrors every 500 steps yield
the exact six milestone checkpoints above.

## Completed verification

Latest CPU/no-network PASS:

```bash
PYTHONPATH=. python -m unittest tests.test_production_milestone_evaluation -v
PYTHONPATH=. python -m py_compile pose_controlnet/evaluation_geometry.py pose_controlnet/production_milestone_evaluation.py pose_controlnet/turbo_evaluation.py scripts/evaluate_production_milestones.py tests/test_production_milestone_evaluation.py
```

These tests prove the exact shared 768 bucket selector/geometry is reused,
deterministic aspect-ratio mapping, exact reference-coordinate transformation,
unchanged native persisted geometry, non-overwriting mode roots, and explicit
summary mode rows. `pose_controlnet/evaluation_geometry.py` now owns the
lightweight persisted-geometry validation shared by Turbo and the new harness;
its Turbo behavior is unchanged.

## Exact foreground dual-mode milestone evaluation command (do not run from Codex)

```bash
cd /home/ubuntu/krea2-pose-controlnet
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python scripts/evaluate_production_milestones.py evaluate --checkpoint-root /lambda/nfs/adhit/krea2-pose/checkpoints/pose-control-production-3000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/pose-control-production-3000 --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning --turbo-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-turbo/turbo.safetensors --reference-sidecar /home/ubuntu/krea2-pose-controlnet/data/manifests/diagnostic_reference_pose.json --diagnostic-manifest /home/ubuntu/krea2-pose-controlnet/data/manifests/diagnostic_val.jsonl --canonical-reference-spec /home/ubuntu/krea2-pose-controlnet/docs/evaluation/turbo-8step-cfg0/turbo_spec.json --modes native dynamic-768
```

The command uses Turbo’s pinned 8-step, CFG 0, `mu=1.15`, control-scale 1.0
contract. It uses the existing detector with threshold 0.5, unchanged Hungarian
matching/bbox-diagonal PCK semantics, unmatched-reference failures, and the
existing CLIP scoring helper. The offline variables require local model caches
and prevent a network fallback.

## Files changed this session

- `pose_controlnet/evaluation_geometry.py`
- `pose_controlnet/production_milestone_evaluation.py`
- `pose_controlnet/turbo_evaluation.py`
- `scripts/evaluate_production_milestones.py`
- `tests/test_production_milestone_evaluation.py`
- `docs/CODEX_HANDOFF.md`

## Next action

After review and only once all six local milestone checkpoints and local Turbo/
CLIP model caches exist, run the exact foreground command above from the GH200
host shell. Do not replace the native primary benchmark with dynamic-768.
