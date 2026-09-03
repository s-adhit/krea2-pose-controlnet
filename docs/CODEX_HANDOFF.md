# Project handoff

## Current objective

Run and inspect the isolated English / Chinese / Telugu fixed-pose multilingual v2 smoke on the GH200. Do not train, access the network from Codex, commit, push, modify canonical `inference.py`, modify training code, or alter frozen historical EN/ZH artifacts.

## Multilingual smoke v2 contract

- Immutable spec: `docs/evaluation/prompting-guide/multilingual_prompt_smoke_v2.jsonl`; SHA-256 `bc178f1e6c0559b3bfc92c7d48edbdd2a825e9451eb9d230aed794edaf23d9e5`.
- Ordered rows, all for `sculpture_humanart_14000000003803`:
  - `en`: `A single adult woman wearing a simple cream outfit in a quiet botanical courtyard, soft overcast daylight, natural textures.`
  - `zh`: `一位成年女性，穿着简洁的奶油色服装，身处安静的植物庭院中，柔和的阴天天光，自然真实的材质质感。`
  - `te`: `ఒక వయోజన మహిళ సరళమైన క్రీమ్ రంగు దుస్తులు ధరించి, నిశ్శబ్దమైన బొటానికల్ ప్రాంగణంలో ఉంది, మృదువైన మేఘావృత దినకాంతి, సహజమైన వాస్తవిక పదార్థాల స్పర్శ.`
- Every row uses the authoritative final-val sampling seed `8675987726486463627`, the same resolved pose control and SHA-256, `mix-025`, native/aspect-preserving cached-latent geometry, Krea-2 Turbo, 8 steps, CFG 0, `mu=1.15`, and control scale 1.0.
- The new `scripts/multilingual_prompt_smoke.py` scopes the existing frozen generation/PCK/CLIP mechanics to this v2 contract. It leaves `scripts/chinese_prompt_smoke.py`, `chinese_prompt_smoke.jsonl`, and completed EN/ZH artifacts unchanged. CLIP is the unchanged existing UTF-8 prompt path; it is not given Telugu-specific metric behavior.
- It writes immutable `language_smoke_provenance.json`, exactly three generations, `pck_clip_results.json`, `metrics_by_language.json`, `evaluation_summary.json`, `multilingual_comparison.png` (pose control | English | Chinese | Telugu), and `compact_summary.json`. Incomplete/drifted roots fail closed.

## Exact GH200 commands

Output root:

```bash
/lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/multilingual-smoke-v2
```

```bash
uv run python scripts/multilingual_prompt_smoke.py preflight --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/multilingual-smoke-v2
uv run python scripts/multilingual_prompt_smoke.py generate --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/multilingual-smoke-v2
uv run python scripts/multilingual_prompt_smoke.py score --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/multilingual-smoke-v2 --reference-sidecar docs/evaluation/final-val-benchmark-selection/final_val_benchmark_48_pose_targets_v3
uv run python scripts/multilingual_prompt_smoke.py report --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/multilingual-smoke-v2
uv run python scripts/multilingual_prompt_smoke.py summary --candidate mix-025 --output-root /lambda/nfs/adhit/krea2-pose/evaluation/prompting-guide/multilingual-smoke-v2
```

## README refresh

- Added current status and the frozen mix interpolation table; mix-025 remains a release candidate, not final.
- Added concise prompting, Style-LoRA composition, multilingual, evaluation, repository-tree, and GH200 helper coverage.
- Verified README links/images: final-val mix-025 contact sheet, prompting-study contact sheet, Style-LoRA strength-sweep contact sheet, trigger-correct and strength-sweep result directories, three curated showcase images, `prompting.md`, and `scripts/bootstrap_gh200.sh`.

## Files changed this session

- `README.md`
- `docs/evaluation/prompting-guide/multilingual_prompt_smoke_v2.jsonl`
- `scripts/multilingual_prompt_smoke.py`
- `tests/test_multilingual_prompt_smoke.py`
- `docs/CODEX_HANDOFF.md`

## Completed / green checks

PASS:

```bash
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m py_compile scripts/multilingual_prompt_smoke.py tests/test_multilingual_prompt_smoke.py scripts/chinese_prompt_smoke.py tests/test_chinese_prompt_smoke.py
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python -m unittest tests.test_multilingual_prompt_smoke tests.test_chinese_prompt_smoke tests.test_prompting_guide_study tests.test_style_lora_composition -v
# 30 tests passed
UV_CACHE_DIR=/tmp/krea2-uv-cache uv run python scripts/multilingual_prompt_smoke.py --help
```

Tests pin the v2 byte hash, exact Telugu UTF-8 text, EN/ZH/TE order, common seed/control/candidate/runtime, incomplete output refusal, score ordering, immutable drift failure, and unchanged legacy EN/ZH hash/rows. No generation or network access was performed in Codex.

## Next action

Run the Telugu multilingual smoke on GH200, inspect the comparison, then continue the remaining release evaluations.
