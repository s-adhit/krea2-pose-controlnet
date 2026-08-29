# Project handoff

## Current objective

The normalized-coordinate-Huber pose-reward smoke experiment is complete. The
next bounded diagnostic is read-only critic alignment: score its already
generated Turbo images with the same differentiable fixed-box Keypoint R-CNN
objective used during the smoke run, then compare the internal error trend to
the unchanged authoritative external PCK. Do not train, regenerate images,
modify checkpoints, commit, or push from Codex.

## Completed experiment

Evaluation root:

```text
docs/evaluation/pose-reward-coord-exposure10pct-l1e5-t010-020
```

Branch checkpoints scored: `1525`, `1550`, `1575`, `1600`; required
retrospective baseline: step `1500` from
`docs/evaluation/turbo-8step-cfg0-lr5e5`. The immutable parent was
`/lambda/nfs/adhit/krea2-pose/checkpoints/pose-learning-900-lr5e5-to1500/step_001500.pt`
with SHA-256
`6f83449f2843414c9cd7205f6ded95bada6e8d0c17af3d612a48443a5ed75da0`.

Contract: `normalized_coordinate_huber`, temperature `1.0`, lambda `1e-5`,
forced exposure `0.10`, window `[0.10, 0.20]`, final-window-uniform-v1,
microbatch `1`, accumulation `32`.

## Critic-alignment implementation

`scripts/turbo_benchmark.py critic-alignment` is generic: it derives branch
steps, labels, Turbo provenance, and the recorded baseline from the completed
`turbo_spec.json`; optional `--steps` must exactly match the branch generated
and scored set. It first validates matching Turbo provenance, all generated
images, baseline availability, score rows, image/sidecar bucket geometry,
Phase-1 target availability, and the persisted coordinate-Huber training
configuration. It then uses only the frozen fixed-box COCO_V1 critic with raw
logits → spatial softmax at `T=1` → expected heatmap coordinates → cell-center
ROI mapping → in-ROI normalization → delta-1 Huber over `reward_joint_valid`.

The path has no sampler, generation, optimizer, parameter-gradient, or
training operation. Danbooru/other Phase-1-unavailable samples are excluded;
invalid/OOB joints are masked. `normalized_coordinate_distances` was added to
`pose_controlnet/keypoint_critic.py` so the interpretable normalized Euclidean
error shares the reward’s exact coordinate normalization.

It writes in the branch evaluation root:

```text
critic_alignment_samples.json
critic_alignment_summary.json
critic_alignment_report.md
```

The summary includes baseline plus all branch checkpoint aggregates, absolute
and percent deltas, externally precomputed PCK/CLIP/coverage, and descriptive
Pearson/Spearman correlations across five checkpoint observations. It records
that lower internal error and higher PCK should generally be negatively
correlated; it does not treat five observations as conclusive.

## Exact GH200 diagnostic command

Run only after confirming the baseline `fixed_pose/*/step_001500.png` files
are present at the baseline root; the command deliberately fails closed when
they are absent.

```bash
cd /home/ubuntu/krea2-pose-controlnet
PYTHONPATH=. python scripts/turbo_benchmark.py critic-alignment \
  --output-root docs/evaluation/pose-reward-coord-exposure10pct-l1e5-t010-020 \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --steps 1525 1550 1575 1600 \
  --experiment-name pose-reward-coord-exposure10pct-l1e5-t010-020 \
  --device cuda
```

Interpretation: A) internal critic improves and PCK improves = aligned /
promising; B) internal critic improves and PCK worsens = reward misalignment;
C) internal critic does not improve = auxiliary optimization/gradient
effectiveness problem.

## Checks completed this session

- PASS: focused CPU/no-network diagnostic, critic, and Turbo tests plus prior
  relevant suites: 130 tests.
- PASS: `PYTHONPATH=. python -m py_compile scripts/turbo_benchmark.py pose_controlnet/keypoint_critic.py tests/test_critic_alignment.py`.
- PASS: `PYTHONPATH=. python scripts/turbo_benchmark.py critic-alignment --help`.
- PASS: `git diff --check`.

Changed this session: `scripts/turbo_benchmark.py`,
`pose_controlnet/keypoint_critic.py`, `tests/test_critic_alignment.py`, and
this handoff. No GH200 critic execution, training, image generation,
checkpoint mutation, network call, commit, or push occurred.
