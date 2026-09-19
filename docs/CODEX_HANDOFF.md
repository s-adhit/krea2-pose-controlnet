# Project handoff

## Current objective

Training and final evaluation evidence are frozen for blog/release review. Do
not alter frozen evaluation artifacts, NFS checkpoints, release identity, or
hero-v2 winners. The next action is human review and freezing of blog figures
and tables; do not regenerate benchmark images without an explicit decision.

## Final release and training facts

- Release `mix-025` / `krea2-pose-control-lora-v1`: float32 interpolation of
  trainable `state['model']` only, `0.75 * parent-4000 + 0.25 * A4300`.
  Release SHA-256: `6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
- Endpoint: 215,488,512 float32 parameters / 450 tensors; expanded control
  projection plus rank-64 LoRA across eight targets in 28 blocks; backbone
  frozen. Objective: flow MSE plus `.04` normalized-coordinate Huber at
  shifted timesteps `[.10,.20]`.
- Training evidence: `docs/blog-evidence/TRAINING_METHODS_INFRA.md`,
  `training_methods_infra.json`, and `RUNTIME_AUDIT.md`.
- Normal-host record: ARM64 GH200; Python 3.10.12, torch 2.7.0/CUDA 12.8,
  cuDNN 9.8, Triton 3.3.0. The audit sandbox has no CUDA; this is not contrary
  evidence.

## Evaluation methodology and results frozen

- PCK uses thresholds `.05/.10/.20`, globally pools renderer-qualified
  reference joints, and applies inclusive Euclidean thresholds normalized by
  the reference person valid-joint extent diagonal. References are
  authoritative source annotations transformed into output geometry.
- Missing detector joints and unmatched eligible references remain denominator
  zeros; extra generated people are reported but do not directly add a PCK
  penalty. People use Hungarian matching on mean unnormalized distance over
  shared valid joints.
- Generated pose uses torchvision `keypointrcnn_resnet50_fpn:COCO_V1`; person
  and keypoint confidence thresholds are `.5`.
- CLIP is `openai/clip-vit-base-patch32`, image/prompt cosine similarity,
  averaged across complete generated sets independently from PCK matching.
  Prompt injection uses injected prompt text.
- Native uses persisted paired cached geometry. Dynamic-768 is a five-condition
  geometry/control-encoding ablation: aspect-ratio bucket, resize-to-cover,
  center crop, and control VAE encode, with no source RGB fallback.
- Full source locations, formulae, detector-default caveat, and limitations:
  `docs/blog-evidence/EVALUATION_METHODS.md` and
  `docs/blog-evidence/evaluation_methods.json`.
- Evaluation methodology and results are frozen for blog/release review;
  preserve all metrics, formulas, frozen artifacts, and methodology.

## Verified headline result facts

- Final-val: 48 images, 101 renderer-qualified people, 1,544 eligible joints.
  mix-025 PCK `.4520725389/.6042746114/.7240932642`, CLIP `.3369378586`, 95
  matched; parent-4000 `.4300518135/.5725388601/.7104922280`, CLIP
  `.3364916809`, 94; A4300 `.4404145078/.5939119171/.7169689119`, CLIP
  `.3369788169`, 93. Turbo baseline `.0356217617/.1101036269/.3387305699`,
  CLIP `.3440154150`, 95.
- Native/dynamic (five images each): native
  `.2920353982/.4026548673/.5884955752`, CLIP `.3074624562`, 14 matched;
  dynamic `.2477876106/.3584070796/.5309734513`, CLIP `.3096808381`, 14.
- Hard stress: single (8 images) `.3727272727/.5181818182/.6818181818`, CLIP
  `.3281741025`, 8; multi (4 images) `.2916666667/.4097222222/.5625000000`,
  CLIP `.3174141528`, 12.
- Prompt injection mix-025: 48 images, PCK
  `.4009067358/.5563471503/.6832901554`, CLIP `.3394259131`, 96 matched.
- Exact candidate/composition/count tables and raw source paths:
  `docs/blog-evidence/EVALUATION_RESULTS.md`.

## Verification this session

PASS:

```bash
uv run python scripts/audit_evaluation_results.py
uv run python -m unittest tests.test_turbo_evaluation
git diff --check
```

The audit checks raw NFS final-val/Turbo-baseline score payloads against the
committed rounded summary and summarizes native, hard-pose, and prompt results.

Known non-session failure:

```bash
uv run python -m unittest tests.test_hard_pose_multiperson_benchmark
```

It fails before assertions because the hard-pose frozen spec expects historical
`inference.py` SHA-256 `60992bba...f8c47a5b`, while current `inference.py` is
`38153edb...c804dd23`. Do not alter the historical hash/spec as part of this
documentation audit.

## Files changed this session

- `docs/blog-evidence/EVALUATION_METHODS.md`
- `docs/blog-evidence/evaluation_methods.json`
- `docs/blog-evidence/EVALUATION_RESULTS.md`
- `scripts/audit_evaluation_results.py`
- `docs/CODEX_HANDOFF.md`

Pre-existing/unrelated untracked file: `scripts/package_hero_v2_final.py`.
Do not commit or push without explicit authorization.

## Unresolved items

- Result payloads pin model identifiers and code behavior but not a separate
  detector-weight digest, original Transformers version, or expanded serialized
  CLIPProcessor transform configuration.
- Detector internal defaults are reconstructed from audit-time torchvision
  0.22.0; PCK output coordinates are postprocessed generated-image coordinates.

## Next recommended action

Review and freeze blog figures/tables from `EVALUATION_RESULTS.md` together
with its methodology/limitations; preserve raw result artifacts and release
contract unchanged.
