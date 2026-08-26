# Phase 1 handoff

## Current objective

Post-step-500 evaluation is implemented but has not been run on the GH200. This session corrected the comparison archive only; no training, optimizer step, commit, or push occurred.

## Verified state

- Krea-2 Raw, clean skeleton-control concat, rank/alpha-64 LoRA, BF16 flow-MSE, AdamW `1e-4` / `(0.9, 0.99)` / zero decay, warmup 200, MB2/accum16/effective32, GC6, compile off, cached Qwen, and seed42 remain in force.
- Training reached step 500. Local step 500 validates at `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt` and is being explicitly mirrored.
- The complete, real evaluation sequence is exactly `0,20,40,60,80,100,200,225,350,475,500`. Steps 20/40/60/80/100 are from `pose-learning-100`; 200/225/350/475 are proven completion-marked HF mirrors in `adhit-420/Krea-2-PoseControl-LoRA-checkpoints` under `pose-learning-500/full`; 500 is local and being mirrored. Steps 300 and 400 do not exist and must never be reconstructed, interpolated, or reported.
- `ordered_checkpoints` loads early states from `pose-learning-100` and later states from `pose-learning-500`. For a missing later checkpoint, it requests only the same numbered `pose-learning-500/full/step_XXXXXX.pt` from HF and accepts it only when the format-1 completion marker, SHA-256, full checkpoint schema, and embedded `global_step` all match. It cannot substitute a different archive step.

## Evaluation contract

- Fixed-flow remains unchanged: the stored immutable 32 stems, seed `420100`, per-stem timestep/noise/sampling seeds, and input hashes are checked before use.
- Fixed-pose remains unchanged: stored diagnostic stems, seed `420200`, fixed sampling noise, Euler 8 steps, CFG 3.5, prompts, controls, and VAE are reused.
- PCK remains torchvision COCO-V1 Keypoint R-CNN, confidence `0.5`, COCO-17 identity, Hungarian shared-joint association, and reference confident-keypoint bbox-diagonal normalization. CLIP remains `transformers.CLIPModel` / `openai/clip-vit-base-patch32` cosine similarity to immutable captions. Neither is a training loss.
- Summary, plots, terminal report, and GitHub export assert the exact real archive sequence. GitHub export allowlist is only `comparison_grid.png`, `evaluation_summary.json`, `fixed_flow_vs_step.png`, `pck_vs_step.png`, `clip_similarity_vs_step.png`, `detection_coverage_vs_step.png`, plus optional reviewed `evaluation_metrics.png`.

## Tests this session

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_evaluation tests.test_post500_evaluation tests.test_train_mechanics` (37 tests).
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile evaluate.py pose_controlnet/evaluation.py pose_controlnet/post500_evaluation.py pose_controlnet/checkpointing.py scripts/mirror_checkpoint.py scripts/score_post500.py scripts/post500_report.py tests/test_evaluation.py tests/test_post500_evaluation.py tests/test_train_mechanics.py`; `git diff --check`.

## Exact GH200 evaluation commands

From repo root with CUDA and private-HF credentials available, first confirm/mirror only local step 500:

```bash
uv run python scripts/mirror_checkpoint.py status --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-learning-500 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt
uv run python scripts/mirror_checkpoint.py mirror --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-learning-500 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt
uv run python scripts/mirror_checkpoint.py status --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-learning-500 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt
```

Then run the evaluation. Missing 200/225/350/475 are recovered exactly into `hf-recovery` only after their completion markers validate; no retraining is involved:

```bash
uv run python evaluate.py fixed-flow --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --later-checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-run-name pose-learning-500 --hf-recovery-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/hf-recovery --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python evaluate.py fixed-pose --samples 1 --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --later-checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-run-name pose-learning-500 --hf-recovery-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/hf-recovery --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500-smoke
uv run python evaluate.py fixed-pose --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --later-checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500 --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --hf-run-name pose-learning-500 --hf-recovery-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/hf-recovery --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/score_post500.py --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500 --samples 1
uv run python scripts/score_post500.py --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/post500_report.py plots --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/post500_report.py report --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/post500_report.py export --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500 --destination docs/evaluation/pose-learning-500
git status --short
git add docs/evaluation/pose-learning-500/comparison_grid.png docs/evaluation/pose-learning-500/evaluation_summary.json docs/evaluation/pose-learning-500/fixed_flow_vs_step.png docs/evaluation/pose-learning-500/pck_vs_step.png docs/evaluation/pose-learning-500/clip_similarity_vs_step.png docs/evaluation/pose-learning-500/detection_coverage_vs_step.png
```

Stage `evaluation_metrics.png` only if it exists and has been reviewed. Never stage individual generated images, controls, checkpoints, or weights.
