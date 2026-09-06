# Project handoff

## Current objective

Run the isolated frozen hard-pose + multi-person stress benchmark for `mix-025` on GH200. Do not train, access the network from Codex, commit, push, modify `inference.py`, modify training code, or alter historical benchmark/spec/artifact namespaces.

## Frozen stress benchmark v1

- Immutable spec: `docs/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1.json`; SHA-256 `ed2fe5021aadce7bb32a6c2cdbd13573f9631054377715498ca6c486b04572a7`.
- Runner: `scripts/hard_pose_multiperson_benchmark.py`; focused tests: `tests/test_hard_pose_multiperson_benchmark.py`.
- Exactly 12 `mix-025` generations; no checkpoint comparison. Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, control scale 1.0, `native_aspect_preserving_cached_latent_bucket`.
- Every condition uses a frozen final-val source prompt, sampling seed, cached-latent bucket, final-val v3 authoritative pose record, and pinned physical control SHA-256. Preflight refuses source/cache/prompt/seed/bucket/sidecar/candidate/historical-artifact drift.
- No RGB fallback is permitted. Scores preserve per-image PCK@0.05/0.10/0.20, CLIP, matched/predicted/unmatched people, detection coverage, and joint-evaluation coverage. Failure labels are observable-only: missed reference person, extra predicted people, low strict-PCK (<0.30), low coarse-PCK (<0.50).

## Exact frozen condition order

1. `01_seated` — `coco_268556_2203816` — single-person
2. `02_crouched` — `real_human_humanart_15000000000477` — single-person
3. `03_lying_floor` — `real_human_humanart_15000000000930` — single-person
4. `04_overhead_arms` — `coco_104715_474045` — single-person
5. `05_strong_torso_rotation` — `coco_540614_498912` — single-person
6. `06_strong_foreshortening` — `coco_125590_434183` — single-person
7. `07_inversion` — `real_human_humanart_15000000000521` — single-person
8. `08_airborne` — `coco_49731_461706` — single-person
9. `09_two_person_interaction` — `real_human_humanart_17000000001263` — multi-person
10. `10_overlapping_two_person_pose` — `real_human_humanart_15000000001893` — multi-person
11. `11_three_to_four_person_interaction` — `coco_338108_crowd` — multi-person
12. `12_crowd_dense_group` — `coco_374270_crowd` — multi-person

## Exact GH200 commands

Output root:

```bash
/lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1
```

```bash
uv run python scripts/hard_pose_multiperson_benchmark.py preflight --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1
uv run python scripts/hard_pose_multiperson_benchmark.py generate --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1
uv run python scripts/hard_pose_multiperson_benchmark.py score --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/hard_pose_multiperson_benchmark.py report --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1
uv run python scripts/hard_pose_multiperson_benchmark.py summary --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1
```

Expected files: `hard_pose_provenance.json`, `generation_results.json`, `pck_clip_results.json`, `metrics_by_pose_class.json`, `metrics_by_person_group.json`, `evaluation_summary.json`, `compact_summary.json`, `hard_pose_contact_sheet.png`, `single_person_hard_pose_grid.png`, `multi_person_stress_grid.png`, and `per_condition_grids/*.png`.

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/hard_pose_multiperson_benchmark.py tests/test_hard_pose_multiperson_benchmark.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_hard_pose_multiperson_benchmark tests.test_turbo_baseline_benchmark tests.test_final_val_benchmark_spec -v
# 16 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/hard_pose_multiperson_benchmark.py --help
git diff --check
```

No generation, scoring, network access, commit, or push was performed in Codex.

## Files changed this session

- `docs/evaluation/hard-pose-multiperson/hard-pose-multiperson-mix-025-v1.json`
- `scripts/hard_pose_multiperson_benchmark.py`
- `tests/test_hard_pose_multiperson_benchmark.py`
- `docs/CODEX_HANDOFF.md`

## Next action

Run the five GH200 commands in order. Review `compact_summary.json`, the two metric tables, observable failure taxonomy, and `pose control | generation` grids after all stages complete.
