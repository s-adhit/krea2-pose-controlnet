# Project handoff

## Current objective

The frozen final-val 48-item Turbo benchmark now has its canonical immutable
authoritative v3 pose-reference sidecar. No image generation, training,
network operation, commit, or push occurred. Historical 24-item diagnostic
scoring remains unchanged.

## Locked final-val pose-reference contract

- Frozen selection: `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48.jsonl`, SHA-256 `23d448d573a2ffd20adfd73fa88f34ebc08df280a051cb0931d9ecdcc1231ceb`.
- Frozen spec: `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_spec.json`, SHA-256 `93a5254e57fa208263f6188573e0760ffedd954bf3b3b3425109ea0178957cd0`.
- Canonical sidecar: `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3/`.
  - `records.jsonl` SHA-256 `3cc4defc282cb11e956ec06517eff4e8369622d4c0b3b567ab2247efb4a499a7`; exactly 48 records in frozen order.
  - `metadata.json` is read-only and binds the selection/spec hashes above and authoritative export `data/pose_targets_authoritative_v1.jsonl`, SHA-256 `6d4469ba8118f78d6bc7f99f59136ac6699d3bcec4bf89165b7ae82dabed4b4f`.
  - Coverage is 100%: COCO 16, painting 12, real_human 12, sculpture 8. All targets are original annotations.
- `scripts/build_final_val_pose_sidecar.py` fails closed on frozen-input hash/provenance drift, duplicate/missing latent stems, duplicate authoritatives, unavailable targets, source-size/geometry mismatches, and order mismatch.
- `pck_records_from_v3` is the narrow source-space representation adapter used only by `scripts/final_val_turbo_benchmark.py`; it does not detect, render, or derive pose annotations. The historical diagnostic sidecar path and historical scorer behavior were not changed.

## Completed / green gates

- Built the canonical v3 final-val sidecar against the actual persisted validation shard geometry.
- PCK loader validates canonical sidecar kind, frozen stem order, selection/spec provenance, sidecar record SHA, and authoritative-export SHA presence before adapting v3 source points for the existing scorer.

PASS:

```bash
PYTHONPATH=. python -m py_compile pose_controlnet/pose_targets.py scripts/build_final_val_pose_sidecar.py scripts/final_val_turbo_benchmark.py
PYTHONPATH=. python scripts/build_final_val_pose_sidecar.py
PYTHONPATH=. python -m unittest tests.test_pose_targets tests.test_final_val_turbo_benchmark tests.test_final_val_benchmark_spec -v
# 25 tests passed
git diff --check
```

## Files changed this session

- `pose_controlnet/pose_targets.py`
- `scripts/build_final_val_pose_sidecar.py`
- `scripts/final_val_turbo_benchmark.py`
- `tests/test_pose_targets.py`
- `tests/test_final_val_turbo_benchmark.py`
- `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3/records.jsonl`
- `docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3/metadata.json`
- `docs/CODEX_HANDOFF.md`

## Exact score/report commands

Run only after the matching 48-image generation set is complete on the GH200 host:

```bash
uv run python scripts/final_val_turbo_benchmark.py score --candidate parent-4000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/final_val_turbo_benchmark.py report --candidate parent-4000 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/parent-4000
uv run python scripts/final_val_turbo_benchmark.py score --candidate finish-control-a4300 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/final_val_turbo_benchmark.py report --candidate finish-control-a4300 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/final-val-turbo/finish-control-a4300
```

## Exact next task

Run the already-implemented final-val Turbo preflight/generation for the two allowed real controlled checkpoints, then use the score/report commands above. Do not evaluate Turbo base/zero-adapter, interpolate checkpoints, or alter the frozen benchmark contract.
