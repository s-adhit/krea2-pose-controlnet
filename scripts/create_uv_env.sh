#!/usr/bin/env bash
# Create/sync the project environment without installing or replacing the
# CUDA-enabled PyTorch stack supplied by the GH200 image.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv"
host_python="${PYTHON:-/usr/bin/python3.10}"
uv_cache_dir="${UV_CACHE_DIR:-${project_root}/.uv-cache}"

cd "${project_root}"
export UV_CACHE_DIR="${uv_cache_dir}"

if [[ ! -x "${host_python}" ]]; then
  echo "Host Python not found: ${host_python}" >&2
  exit 1
fi

if [[ -e "${venv_dir}" && ! -f "${venv_dir}/pyvenv.cfg" ]]; then
  echo "Refusing to modify non-venv path: ${venv_dir}" >&2
  exit 1
fi

if [[ ! -d "${venv_dir}" ]]; then
  uv venv --python "${host_python}" --system-site-packages "${venv_dir}"
elif ! rg -q '^include-system-site-packages = true$' "${venv_dir}/pyvenv.cfg"; then
  echo "${venv_dir} does not inherit system site packages; recreate it with:" >&2
  echo "  rm -rf ${venv_dir}" >&2
  echo "  scripts/create_uv_env.sh" >&2
  exit 1
fi

# The lock intentionally contains no torch-family packages.  `--locked`
# prevents a deployment host from silently changing the resolved Python deps.
uv sync --active --locked --no-install-project

"${venv_dir}/bin/python" scripts/check_environment.py
