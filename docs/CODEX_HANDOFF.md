# Final pre-archive handoff

## Project state

Training, evaluation, release selection, and ComfyUI integration are complete.
The frozen release is `krea2-pose-control-lora-v1`, materialized artifact
`krea2-pose-control-mix025.safetensors`, SHA-256
`6d97e9c2e102e07928fc8864346401a0d2e6082d610ca6b037c4704102e3f8d1`.
Krea-2 Raw is training provenance; frozen inference is Krea-2 Turbo. The
targeted native-vs-dynamic and hard-pose blog comparison assets are complete
frozen evaluation outputs and must not be regenerated.

The architecture is a clean rendered-skeleton latent channel-concatenated with
the noisy image latent before `ControlInputLayer`; it is not a parallel
ControlNet branch. Trainable components are the control input and rank-64
LoRA. The frozen inference contract is 8 steps, CFG 0, `mu=1.15`, control
scale 1.0, native/aspect-preserving geometry, and no Style-LoRA.

## Appendix v1

The appendix is prepared for a fresh 48-image regeneration: 12 frozen controls
× four short, geometry-neutral visual families. It resolves every copied
control through `choose_bucket(source_size, REFERENCE_KREA_BUCKETS)` and
records the resulting positive, 16-divisible width and height. Seeds remain
deterministic. Preparation validates the release SHA, control PNGs, all output
paths, and prompt neutrality without loading a model.

Old appendix images whose sidecars contain the prior prompts are classified as
stale rather than reused. The next CUDA generation replaces each stale or
missing row, fails loudly on an error, and writes the contact sheet only after
all 48 rows succeed. The rebuilt sheet has rows `row label | control |
cinematic_fantasy_realism | painterly_storybook_fantasy |
polished_modern_anime | editorial_high_fashion_photography`; thumbnails are
letterboxed, never cropped.

## Verified this session

- PASS: `uv run python -m unittest tests.test_prepare_qualitative_appendix tests.test_inference tests.test_paired_preprocessing` — 27 tests.
- PASS: `python scripts/prepare_qualitative_appendix.py` — 48 rows planned,
  0 current-prompt successes, 48 pending CUDA; eight targeted evaluation
  outputs reused unchanged.
- INFO: `uv run python -c 'import torch; print(torch.cuda.is_available())'`
  reported `False`; no CUDA generation was launched in this audit shell.

## Files changed this session

- `scripts/prepare_qualitative_appendix.py`
- `tests/test_prepare_qualitative_appendix.py`
- `README.md`
- `comfyui/README.md`
- `docs/blog-assets/appendix-v1/README.md`
- `docs/blog-assets/appendix-v1/manifest.json`
- `docs/blog-assets/BLOG_ASSET_INDEX.md`
- `docs/blog-assets/SHA256SUMS.txt`
- `docs/CODEX_HANDOFF.md`

## Exact next action

```bash
cd /home/ubuntu/krea2-pose-controlnet
uv run python scripts/prepare_qualitative_appendix.py --generate-appendix
```

After that command succeeds, inspect the five-column appendix contact sheet,
run the appendix preparation command again, run `git diff --check`, and review
the generated manifest/checksums. Then make the README/docs commit, archive
evidence, verify the remote copy, clean NFS/GH200 working artifacts, and move
to the blog-only phase. Do not train, change release identity, alter frozen
metrics/runtime defaults, or regenerate targeted comparison assets.
