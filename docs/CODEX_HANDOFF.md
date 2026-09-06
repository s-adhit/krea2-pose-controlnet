# Project handoff

## Current objective

Run the isolated frozen hands/fingers evaluation on GH200. It compares Pose
Control with unmodified Krea-2 Turbo; do not train, use Codex network access,
commit, push, modify `inference.py` or training, or alter historical
benchmark/spec/artifact namespaces.

## Frozen hands/fingers benchmark v1

- Spec: `docs/evaluation/hands/hands-fingers-turbo-v1.json`; SHA-256 `d61bacf17c7390b1f8e625d2c986c0098ab74638449c3cea9b7b5d67d6ff6fce`.
- Runner: `scripts/hand_finger_benchmark.py`; tests: `tests/test_hand_finger_benchmark.py`.
- Candidate order: `turbo-base`, `parent-4000`, `mix-025`, `finish-control-a4300`.
- Runtime: Krea-2 Turbo, native aspect-preserving cached latent bucket, 8 steps, CFG 0, `mu=1.15`; controlled candidates use scale 1.0. `turbo-base` is strict unmodified Turbo with no Pose-LoRA/control state.
- Primary: 6 x 3 x 4 = exactly 72 generations. Neutral-only `mix-025` scale study: 6 x scales `0.75`, `1.0`, `1.5` = 18. Total exactly 90.
- Preflight pins Turbo/checkpoint/interpolation identity, final-val cache identities, source prompts, seeds, native buckets, controls, and final-val-v3 pose sidecar. It refuses all relevant drift, partial output, and historical artifact mutation.

## Frozen conditions

1. `01_simple_standing` — `painting_humanart_10000000004967`; seed `6749320992429451996`; bucket `896x1152`
2. `02_arms_raised_overhead` — `coco_104715_474045`; seed `2395696123777290437`; bucket `832x1216`
3. `03_bent_elbow_heavy` — `real_human_humanart_15000000000477`; seed `5676061900249086193`; bucket `1472x704`
4. `04_foreshortened_arm_hand` — `coco_125590_434183`; seed `2474376788711197263`; bucket `832x1216`
5. `05_inversion_unusual_orientation` — `real_human_humanart_15000000000521`; seed `1860641454841022556`; bucket `832x1216`
6. `06_two_person_visible_hands` — `real_human_humanart_15000000001893`; seed `5745961191448260577`; bucket `832x1216`

All have authoritative v3 in-frame wrists. Full prompt/control SHA/wrist details are immutable in the spec and provenance.

## Frozen prompt modes

- `hand_neutral`: exact frozen source caption unchanged.
- `hand_compatible`: source caption plus `, natural relaxed hands and natural fingers`.
- `hand_conflicting`: source caption plus `, both arms folded tightly across the chest with hands hidden in pockets`.

The same condition/prompt-mode sampling seed is used across candidates; scale rows reuse the neutral seed.

## Output and human review

Output root:

```bash
/lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1
```

Artifacts include all requested provenance/generation/PCK+CLIP/aggregate/summary JSON files, full-image contact sheet, authoritative wrist crop metadata/sheets, per-condition grids, and scale grid. Per-image and aggregate records preserve PCK@0.05/0.10/0.20, CLIP, matched/predicted/unmatched people, and detection coverage. No finger metric or automatic anatomy judgment is used. Human-only checklist fields: missing hand, duplicated hand/fingers, malformed/anatomically implausible fingers, wrist discontinuity, arm/hand pose mismatch, and acceptable hand.

## Exact GH200 commands

```bash
uv run python scripts/hand_finger_benchmark.py preflight --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1
uv run python scripts/hand_finger_benchmark.py generate --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1
uv run python scripts/hand_finger_benchmark.py score --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/hand_finger_benchmark.py report --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1
uv run python scripts/hand_finger_benchmark.py summary --output-root /lambda/nfs/adhit/krea2-pose/evaluation/hands/hands-fingers-turbo-v1
```

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/hand_finger_benchmark.py tests/test_hand_finger_benchmark.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_hand_finger_benchmark tests.test_turbo_baseline_benchmark tests.test_hard_pose_multiperson_benchmark -v
# 20 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/hand_finger_benchmark.py --help
git diff --check
```

No generation, scoring, Codex network access, commit, or push occurred.

## Files changed this session

- `docs/evaluation/hands/hands-fingers-turbo-v1.json`
- `scripts/hand_finger_benchmark.py`
- `tests/test_hand_finger_benchmark.py`
- `docs/CODEX_HANDOFF.md`

## Next action

Run the five commands in order, then review `compact_summary.json`, the four metric tables, `hand_benchmark_contact_sheet.png`, `hand_crop_contact_sheet.png`, `per_condition_hand_grids/*.png`, and `control_scale_hand_grid.png` using the explicit human-review checklist.
