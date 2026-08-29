# Project handoff

## Current objective

Prepare, but do not launch, generic controlled pose-reward continuation/evaluation. `train.py` is untouched; production remains flow-MSE only.

## Generic workflow now implemented

- Training takes parent, target, cadence, exposure, lambda, and namespace from CLI; resume remains fail-closed. `checkpoint_publication_steps(1500, 1650, 25)` yields `1525 1550 1575 1600 1625 1650`.
- `turbo_benchmark.py experiment` validates local checkpoint names/steps/optional hashes and extracts gate-E provenance/counters; inconsistent metadata fails before GPU work.
- Fixed contract retained: Turbo 8-step CFG-0/mu=1.15/control=1.0, canonical 24 samples/seed 420200, CLIP ViT-B/32, authoritative 21-sample PCK (Danbooru visual-only).
- Valid artifacts are reused; partial branch artifacts fail; absent baseline PNGs merely omit their visual column. Reports emit generic grids, summary, and separate PCK/CLIP rankings.

## Latest completed experiment result

5% + `lambda_pose=2e-5`: exposure PASS; 333/5567 active (5.98%), forced 280/5567 (5.03%), 159/200 steps pose-active. Step1600 coarse PCK@.20 `.41262 -> .41990`, Human-Art `.41227 -> .42368`, multi-person `.38439 -> .39740`; precise PCK and CLIP did not improve. Do not select `2e-5`; next hypothesis is `1e-5` with 5% exposure.

## Next controlled run — do not run from Codex

Immutable parent:

```text
/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt
sha256: 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0
```

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/train_pose_reward_smoke.py \
  --parent-checkpoint /lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt \
  --expected-parent-sha256 6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0 \
  --raw-ckpt /lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors \
  --latent-root /lambda/nfs/adhit/krea2-pose/posebridge_latents \
  --text-conditioning-root /lambda/nfs/adhit/krea2-pose/text_conditioning \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --checkpoint-dir /lambda/nfs/adhit/krea2-pose/checkpoints \
  --run-name pose-reward-kl-exposure5pct-l1e5-t010-020 \
  --lambda-pose 1e-5 --pose-timestep-min 0.10 --pose-timestep-max 0.20 \
  --forced-pose-exposure-probability 0.05 \
  --target-global-step 1650 --save-every 25 --microbatch-size 1 --gradient-accumulation-steps 32 \
  --hf-repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --hf-subdir pose-reward-kl-exposure5pct-l1e5-t010-020 --hf-mirror-every-steps 25 --device cuda \
  --wandb-entity adhit-projects --wandb-project Krea-2-PoseControl-Lora \
  --wandb-run-name pose-reward-kl-exposure5pct-l1e5-t010-020
```

Start an interactive tmux shell, then paste that command:

```bash
tmux new-session -s pose-reward-kl-exposure5pct-l1e5-t010-020 'cd /home/ubuntu/krea2-pose-controlnet && exec bash'
```

Expected local/HF steps: `1525 1550 1575 1600 1625 1650`. Check remote completion markers with:

```bash
PYTHONPATH=. python scripts/mirror_checkpoint.py list \
  --repo-id adhit-420/Krea-2-PoseControl-LoRA-checkpoints \
  --run-name pose-reward-kl-exposure5pct-l1e5-t010-020
```

Resume: set `--parent-checkpoint .../pose-reward-kl-exposure5pct-l1e5-t010-020/step_00XXXX.pt`, omit `--expected-parent-sha256`, and preserve all other experiment flags.

## Dynamic evaluation — do not run from Codex

```bash
PYTHONPATH=. python scripts/turbo_benchmark.py experiment \
  --checkpoint-root /lambda/nfs/adhit/krea2-pose/checkpoints/pose-reward-kl-exposure5pct-l1e5-t010-020 \
  --steps 1525 1550 1575 1600 1625 1650 \
  --output-root docs/evaluation/pose-reward-kl-exposure5pct-l1e5-t010-020 \
  --experiment-name pose-reward-kl-exposure5pct-l1e5-t010-020 \
  --checkpoint-label-template 'Pose KL 1e-5 {step}' \
  --baseline-output-root docs/evaluation/turbo-8step-cfg0-lr5e5 \
  --baseline-step 1500 --baseline-label 'LR-only 1500 @ 5e-5' \
  --canonical-reference-spec docs/evaluation/turbo-8step-cfg0-lr5e5/turbo_spec.json
```

Add `--expected-sha256 1650=<known-lowercase-sha256>` when available. Re-running reuses validated artifacts. Outputs: `pose-reward-kl-exposure5pct-l1e5-t010-020_full_contact_sheet.png`, `pose-reward-kl-exposure5pct-l1e5-t010-020_checkpoint_selection_grid.png`, `evaluation_summary.json`, `checkpoint_ranking.json`, and resolved `turbo_spec.json` provenance.

## Checks this session

- PASS: 74 CPU/no-network tests: pose-reward, W&B, Turbo, and exposure suites.
- PASS: `PYTHONPATH=. python -m py_compile scripts/train_pose_reward_smoke.py scripts/turbo_benchmark.py pose_controlnet/turbo_evaluation.py`.
- PASS: trainer and both Turbo CLI help commands.
- PASS: `git diff --check`.

Changed: `pose_controlnet/turbo_evaluation.py`, `scripts/turbo_benchmark.py`, `tests/test_turbo_evaluation.py`; updated pre-existing untracked trainer/tests in place. No training, evaluation, network call, commit, or push.
