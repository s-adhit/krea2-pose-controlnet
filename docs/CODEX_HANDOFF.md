# Project handoff

## Current objective and decisions

The bounded milestone is Gate A.5: an audit-only loss/gradient comparison for
the fixed-box differentiable torchvision Keypoint R-CNN critic. Gate A real-RGB
feasibility completed on GH200; Gate A.5 implementation and CPU tests are
complete but the GH200 comparison has not run. Production training remains
flow-matching MSE only. `train.py`, VAE decoding, x0/timestep logic,
dependencies, Phase-1 provenance, and the external Keypoint R-CNN PCK
evaluator remain unchanged. Do not train or integrate any critic loss.

Phase-1 remains authoritative. The current canonical sidecar record SHA is
`dfc32293f1bdb76de58e34a02f95a14e515b0080b7c2f60ddd4a28c6f9fb2d8f`; a
deterministic current-code rebuild matched it byte-for-byte: 17,416 records,
15,161 reward available, 2,255 unavailable, 444,235 valid reward joints,
seven reviewed source-OOB masked joints, and 21/21 diagnostic coverage. The
older `c98f...` SHA is historical, not canonical.

## Completed and verified

- Gate A GH200 real-RGB critic feasibility PASS. Metrics are recorded in
  `docs/KEYPOINT_RCNN_CRITIC_AUDIT.md`: COCO soft PCK .05/.10 .9070/.9728 and
  RGB raw-Huber gradient 4.95; Human-Art painting .3842/.5979 and 631.18;
  real .6023/.7000 and 225.50; sculpture .6963/.8494 and 58.15. Raw
  pixel-coordinate Huber has strongly domain-dependent RGB gradient scale.
- `normalized_coordinate_huber` now normalizes both soft prediction and target
  by the same authoritative fixed person ROI, then averages Huber over valid
  person/joint observations. Raw Huber and Gaussian KL already use the same
  valid-observation mean.
- `scripts/audit_keypoint_critic.py` now defaults to eight deterministic
  samples per source and runs separate forward/autograd graphs for raw Huber,
  normalized Huber, and Gaussian KL. JSON contains each sample's three losses
  and gradient norms plus source mean/median/std/min/max gradient statistics.
  `--temperature-sweep` optionally reports detached 0.5/1.0/2.0 soft PCK and
  normalized-error diagnostics only.
- The frozen COCO_V1 fixed-box path and detector bypass remain as documented
  in `docs/KEYPOINT_RCNN_CRITIC_AUDIT.md`; critic parameters remain frozen.

## Files changed this session

- `pose_controlnet/keypoint_critic.py`
- `scripts/audit_keypoint_critic.py`
- `tests/test_keypoint_critic.py`
- `docs/KEYPOINT_RCNN_CRITIC_AUDIT.md`
- `docs/CODEX_HANDOFF.md`

## Tests/checks and blockers

- PASS: `PYTHONPATH=. python -m unittest tests.test_keypoint_critic` — 15 CPU
  tests: existing critic contracts plus normalized ROI mapping, scale
  invariance, normalized gradients, valid person/joint averaging, and isolated
  three-candidate RGB gradient graphs.
- PASS: `python -m py_compile pose_controlnet/keypoint_critic.py
  scripts/audit_keypoint_critic.py tests/test_keypoint_critic.py`.
- PASS: `PYTHONPATH=. python scripts/audit_keypoint_critic.py --help`.
- PASS: `git diff --check`.
- BLOCKER/HOLD: run Gate A.5 on the actual GH200 only. The Codex sandbox is
  not evidence for GPU checkpoint loading or real-image gradient behavior.

## Exact next recommended action

On the GH200 shell, run:

```bash
PYTHONPATH=. python scripts/audit_keypoint_critic.py \
  --sidecar /lambda/nfs/adhit/krea2-pose/pose_targets_v3 \
  --dataset-root /lambda/nfs/adhit/krea2-pose/posebridge_hf \
  --samples-per-source 8 \
  --temperature-sweep \
  --device cuda \
  --output-json /lambda/nfs/adhit/krea2-pose/keypoint_critic_gate_a5.json
```

Use the actual immutable sidecar path if it differs. Inspect all four sources
for finite losses, finite nonzero per-sample RGB gradients for all three
candidates, no critic parameter gradients, and the reported distribution of
gradient scales. Do not make a loss-selection or integration decision in this
milestone.
