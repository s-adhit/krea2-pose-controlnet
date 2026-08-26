# Phase 1 handoff

## Current objective

Post-step-500 evaluation gate is implemented but not host-executed. No training, optimizer step, commit, or push occurred.

## Verified state

- Krea-2 Raw, clean skeleton-control concat, rank/alpha-64 LoRA, BF16 flow-MSE, AdamW `1e-4` / `(0.9, 0.99)` / zero decay, warmup 200, MB2/accum16/effective32, GC6, compile off, cached Qwen, seed42 remain in force.
- Training reached step 500. The 100→500 continuation OOMed after step-277 backward; it rolled back to validated step 200, restarted with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and 25-step local checkpoint cadence, then reached step 500.
- Local step 500 validates at `/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt`. Prior proven HF mirrors: 200, 225, 350, 475; 475 remains the latest proven remote recovery point.
- Sandbox HF DNS failed, so step-500 remote status was not verifiable here. The host status command below must be used.
- Required order: `0,20,40,60,80,100,200,300,400,500`. Local inspection found valid 20–100 and 500, but **200, 300, 400 are missing** from `pose-learning-500`; complete evaluation deliberately fails rather than skips them.

## Implemented evaluation contract

- `evaluate.py` preserves/reuses deterministic specs and fixed-flow hashes/seeds (420100/420200), uses 100-run directory through step 100 and 500-run directory above it, and validates all required full checkpoint schemas.
- `scripts/mirror_checkpoint.py` checks/mirrors one explicit checkpoint using the existing `pose-learning-500/full` layout, SHA-256, format-1 completion marker, local validation, background-mirror retry/credential behavior, then downloads and validates remote bytes/schema.
- `scripts/score_post500.py` is evaluation-only: torchvision COCO-V1 Keypoint R-CNN, confidence `0.5`, COCO-17 identity, Hungarian minimum shared-joint-distance association, reference confident-keypoint bounding-box diagonal PCK normalization. Missing/low-confidence joints never match; coverage/exclusions are saved. Its control-raster detection coverage must be host-smoked before treating PCK as usable.
- CLIP is `transformers.CLIPModel` / `openai/clip-vit-base-patch32`, cosine similarity to immutable captions only. Summary includes independent best metrics; no weighted score.
- `scripts/post500_report.py` generates four deterministic plots/table and only exports the strict GitHub allowlist.

## Tests

PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_evaluation tests.test_post500_evaluation tests.test_train_mechanics` (35); `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile evaluate.py pose_controlnet/evaluation.py pose_controlnet/post500_evaluation.py scripts/mirror_checkpoint.py scripts/score_post500.py scripts/post500_report.py tests/test_post500_evaluation.py`; `git diff --check`.

## GH200 host commands

From repo root, with existing private-HF credentials:

```bash
uv run python scripts/mirror_checkpoint.py status --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-learning-500 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt
uv run python scripts/mirror_checkpoint.py mirror --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-learning-500 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt
uv run python scripts/mirror_checkpoint.py status --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints --run-name pose-learning-500 --checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500/step_000500.pt
```

After recovering local steps 200/300/400, run fixed-flow, one-sample pose smoke, full pose, metric scoring, summary plots/report/export:

```bash
uv run python evaluate.py fixed-flow --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --later-checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500 --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python evaluate.py fixed-pose --samples 1 --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --later-checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500 --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500-smoke
uv run python evaluate.py fixed-pose --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-100 --later-checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-500 --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/score_post500.py --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500 --samples 1
uv run python scripts/score_post500.py --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/post500_report.py plots --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/post500_report.py report --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500
uv run python scripts/post500_report.py export --output-dir /lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500 --destination docs/evaluation/pose-learning-500
git status --short
git add docs/evaluation/pose-learning-500/comparison_grid.png docs/evaluation/pose-learning-500/evaluation_summary.json docs/evaluation/pose-learning-500/fixed_flow_vs_step.png docs/evaluation/pose-learning-500/pck_vs_step.png docs/evaluation/pose-learning-500/clip_similarity_vs_step.png docs/evaluation/pose-learning-500/detection_coverage_vs_step.png
```

Only stage optional `evaluation_metrics.png` if it exists and is reviewed. Never stage individual images, controls, checkpoints, or weights.
