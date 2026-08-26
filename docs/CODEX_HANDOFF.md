# Phase 1 handoff

## Current objective

Complete GH200 environment Gate A while preserving the host-owned Torch/CUDA
stack. This audit shell does not expose CUDA, so Gate A is blocked; no
training, model download, or dataset download was started.

## Decisions in force

- Krea-2 Raw, skeleton control, channel concatenation, rank-64 LoRA, and all
  Phase-1 training decisions in `AGENTS.md` remain unchanged.
- Torch, torchvision, CUDA, cuDNN, Triton, and NVIDIA packages are host-owned
  and deliberately absent from `pyproject.toml`.
- The project venv uses system site packages via `scripts/create_uv_env.sh`.

## Verified environment facts

- Current host: Linux `aarch64`, Python 3.10.12, `uv 0.12.5`.
- Host stack visible from `/usr/bin/python3`: torch 2.7.0 (`cu128`),
  torchvision 0.22.0, Triton 3.3.0, cuDNN 90800.
- PASS: Torch CUDA build 12.8; cuDNN 90800 (9.8).
- BLOCKED: `uv lock` failed after retries because DNS could not resolve
  `pypi.org` while fetching numpy. No `uv.lock` was created; prohibited-package
  inspection could not be performed.
- BLOCKED: `bash scripts/create_uv_env.sh` created `.venv` with system site
  packages, then stopped because `uv.lock` is absent. Direct invocation reports
  `Permission denied` because the script is not executable; no mode change made.
- FAIL here: CUDA unavailable (`torch.cuda.is_available()` false, device count 0);
  `--require-cuda` exits accordingly.
- BLOCKED: BF16 CUDA allocation and cuDNN/SDPA/GQA smoke tests cannot run here.

## Exact checks

- venv version import — PASS for imports/versions; CUDA false.
- `scripts/check_environment.py --require-cuda` — FAIL: CUDA unavailable.
- BF16 allocation and CUDA SDPA/GQA — BLOCKED: CUDA unavailable.
- `git diff --check` — PASS.
- `git status --short` — `M docs/CODEX_HANDOFF.md`; generated env/cache ignored.

## Current blocker

Network DNS and CUDA are unavailable in this shell. Gate A remains incomplete.

## Files changed this session

`docs/CODEX_HANDOFF.md` only; earlier project edits are unchanged.

## Exact next recommended action

On a network-enabled GH200, run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv lock
rg -ni '(torch|torchvision|triton|cuda|cudnn|nvidia)' uv.lock
bash scripts/create_uv_env.sh
.venv/bin/python scripts/check_environment.py --require-cuda
```

Review the lockfile, then perform all Gate-A CUDA checks from the production
GH200 shell. Do not proceed to later gates until they pass.
