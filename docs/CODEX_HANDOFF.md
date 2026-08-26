# Phase 1 handoff

## Current objective

Establish a reproducible `uv` environment on the GH200 while retaining the
Lambda image's validated Torch/CUDA stack. No training, model download, or
dataset download was started.

## Decisions in force

- Krea-2 Raw, skeleton control, channel concatenation, rank-64 LoRA, and all
  Phase-1 training decisions in `AGENTS.md` remain unchanged.
- Torch, torchvision, CUDA, cuDNN, Triton, and NVIDIA packages are host-owned
  and deliberately absent from `pyproject.toml`.
- The project venv must be created with system site packages enabled. Use
  `scripts/create_uv_env.sh`; do not use `pip install`.

## Verified environment facts

- Current host: Linux `aarch64`, Python 3.10.12, `uv 0.12.5`.
- Host stack visible from `/usr/bin/python3`: torch 2.7.0 (`cu128`),
  torchvision 0.22.0, Triton 3.3.0, cuDNN 90800.
- CUDA is **not visible in this Codex session**, so this is not yet a green
  GH200 Gate A verification. No system package was changed.

## Completed / green checks

- Added `pyproject.toml` with the runtime/dev dependencies, Python 3.10
  constraint, and no Torch-family dependency.
- Added `scripts/create_uv_env.sh`, which uses an ignored project-local
  `.uv-cache`, creates `.venv` with `--system-site-packages`, and performs a
  locked sync.
- Expanded `scripts/check_environment.py` to report both project packages and
  the host-owned accelerator stack; `--require-cuda` is the GH200 gate.
- `python3 -m compileall -q scripts/check_environment.py`: PASS.
- `bash -n scripts/create_uv_env.sh`: PASS.
- `python3 scripts/check_environment.py`: PASS (report only; CUDA unavailable).

## Current blocker

`uv lock` cannot reach PyPI: DNS lookup for `pypi.org` fails in this session.
`uv lock --offline` also correctly fails because `diffusers` is not cached.
Consequently `uv.lock` and `.venv` were not created; the environment is not
yet fully reproducible until a network-enabled GH200/session generates and
reviews the lockfile.

## Files changed this session

`.gitignore`, `README.md`, `pyproject.toml`, `scripts/create_uv_env.sh`,
`scripts/check_environment.py`, and this handoff.

## Exact next recommended action

On a network-enabled GH200, run `UV_CACHE_DIR="$PWD/.uv-cache" uv lock`,
verify the resulting `uv.lock` contains no Torch/Triton/NVIDIA packages, then
run `scripts/create_uv_env.sh` and
`.venv/bin/python scripts/check_environment.py --require-cuda`. After that,
perform the Gate-A SDPA/attention smoke test without downloading models/data.
