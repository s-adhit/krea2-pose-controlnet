# Phase 1 handoff

## Current bounded objective and status

The renderer-aware authoritative-reference PCK gate is implemented and green.
This session was evaluation-only: no training, optimizer, model, image
generation, checkpoint, or dataset mutation occurred. Danbooru remains
unavailable (`authoritative_reference_pose_unavailable`) and must retain null
pose metrics and no PCK denominator contribution.

## Root cause and fixed rule

The prior geometry failure was a false criterion: it required every visible
COCO-17 source joint to occur in the rendered raster. Original rendering only
draws endpoints of a limb when both unified endpoints have visibility > 0. For
Human-Art annotation `10000000065251`, COCO left ankle 15 maps to unified 13
but its left-knee neighbor (unified 12) is invisible; limb `(12,13)` is not
drawn, so the ankle is intentionally absent. It is not a geometry mismatch.

`pose_controlnet.reference_pose` now retains all raw authoritative COCO-17
joints and analytically records `source_visible`, `rendered_in_control`, and
`pck_eligible` per joint. PCK eligibility is exactly:

```text
source_visible AND rendered_in_control
```

using the original supplied unified limbs, not raster keypoint parsing. Neck
is synthesized from two visible shoulders for renderer topology only; it has no
COCO identity and is never a PCK joint. Human-Art and COCO crowd recompute core
visibility plus `MIN_LIMBS=5`; Human-Art sidecar construction guarantees only
its prior `iscrowd==1` and `num_keypoints==0` removals. Those unavailable source
fields are not invented. COCO single uses only the requested sidecar annotation.

## Corrected geometry result

`scripts/reference_pose_gate.py geometry` passed, checking source-space
coordinates only for analytically renderer-represented joints against the
existing source control PNGs (tolerance 1.5 px):

- `real_human_humanart_17000000000288`: 34 joints, max 0.684 px.
- `coco_156320_crowd`: 39 joints, max 0 px.
- `coco_299468_426600`: 13 joints, max 0 px.
- `painting_humanart_10000000000838`: 45 joints, max 0.648 px.

The former left ankle is excluded from the last record's raster validation. The
persisted source-to-bucket resize/cover/center-crop transform is unchanged and
is used to construct bucket-space PCK references.

## One-sample real PCK smoke

Executed on the existing step-500 fixed-pose image for
`real_human_humanart_17000000000288` with torchvision Keypoint R-CNN COCO_V1,
confidence >= 0.5, deterministic Hungarian matching, and <= thresholds:

- PCK@0.05: `0.029411764705882353`
- PCK@0.10: `0.11764705882352941`
- PCK@0.20: `0.35294117647058826`
- reference/rendered reference/predicted/matched: `2/2/2/2`; unmatched `0/0`.
- source-visible/rendered/PCK-eligible/evaluated joints: `34/34/34/34`.
- joint evaluation coverage and generated-person detection coverage: `1.0`,
  `1.0`.

Exact GH200 command:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python scripts/reference_pose_gate.py smoke --device cuda
```

Exact command to print the smoke metrics:

```bash
UV_CACHE_DIR=/tmp/krea_uv_cache uv run python scripts/reference_pose_gate.py smoke --device cuda | python -m json.tool
```

## Files changed this session

- `pose_controlnet/reference_pose.py`
- `pose_controlnet/post500_evaluation.py`
- `scripts/reference_pose_gate.py`
- `tests/test_reference_pose.py`
- `tests/test_post500_evaluation.py`
- `docs/CODEX_HANDOFF.md`

## Commands run

- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python scripts/reference_pose_gate.py geometry`
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python scripts/reference_pose_gate.py smoke --device cpu`
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m unittest tests.test_post500_evaluation tests.test_evaluation tests.test_reference_pose` (29 tests)
- PASS: `UV_CACHE_DIR=/tmp/krea_uv_cache uv run python -m py_compile pose_controlnet/reference_pose.py pose_controlnet/post500_evaluation.py scripts/reference_pose_gate.py tests/test_reference_pose.py tests/test_post500_evaluation.py`

## Exact next recommended action

Review this one PCK-gate patch. Do not run full PCK, train, regenerate images,
commit, or push as part of this milestone.
