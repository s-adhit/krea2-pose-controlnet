#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${KREA2_REPO_URL:-https://github.com/s-adhit/krea2-pose-controlnet.git}"
REPO_DIR="${KREA2_REPO_DIR:-$HOME/krea2-pose-controlnet}"
NFS_ROOT="${KREA2_NFS_ROOT:-/lambda/nfs/adhit/krea2-pose}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/krea2-uv-cache}"

echo "== Krea-2 Pose Control GH200 bootstrap =="

command -v git >/dev/null || { echo "ERROR: git missing"; exit 1; }
[[ -d "$NFS_ROOT" ]] || { echo "ERROR: NFS missing: $NFS_ROOT"; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if [[ -z "$(git status --porcelain)" ]]; then
  git fetch origin
  git checkout main
  git pull --ff-only origin main
else
  echo "WARNING: dirty working tree; skipping git pull."
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [[ ! -d .venv ]]; then
  uv venv --system-site-packages .venv
fi

source .venv/bin/activate
export PYTHONPATH="$REPO_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"

uv sync --frozen

uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY

required=(
  "$NFS_ROOT/posebridge_hf"
  "$NFS_ROOT/posebridge_latents"
  "$NFS_ROOT/posebridge_latents_768"
  "$NFS_ROOT/text_conditioning"
  "$NFS_ROOT/pose_targets_v3"
  "$NFS_ROOT/pose_targets_v3_768"
  "$NFS_ROOT/models/krea-2-raw/raw.safetensors"
  "$NFS_ROOT/models/krea-2-turbo/turbo.safetensors"
  "$NFS_ROOT/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt"
  "$NFS_ROOT/checkpoints/pose-control-finish-control-4000-to4500/step_004300.pt"
)

missing=0
for p in "${required[@]}"; do
  if [[ -e "$p" ]]; then
    echo "OK      $p"
  else
    echo "MISSING $p"
    missing=$((missing+1))
  fi
done

check_hash() {
  p="$1"
  expected="$2"

  [[ -f "$p" ]] || { echo "MISSING $p"; return 1; }

  actual="$(sha256sum "$p" | awk '{print $1}')"

  if [[ "$actual" == "$expected" ]]; then
    echo "OK HASH $(basename "$p")"
  else
    echo "BAD HASH $p"
    echo " expected: $expected"
    echo " actual:   $actual"
    return 1
  fi
}

check_hash \
  "$NFS_ROOT/checkpoints/pose-control-production-cooldown-3000-to5000/step_004000.pt" \
  "0f10f708d12eb63bc2c17ff4556266005efaf57670886ffaf17e76c6980f7acd" \
  || missing=$((missing+1))

check_hash \
  "$NFS_ROOT/checkpoints/pose-control-finish-control-4000-to4500/step_004300.pt" \
  "17405082f5efd85967278e07ac94543d3c6e2d4b8da6763b817885f1216e27ff" \
  || missing=$((missing+1))

declare -A styles=(
  ["$NFS_ROOT/style_loras/darkbrush/darkbrush.safetensors"]="f476ad1c0679bc6b14c815187e78a6ece43248f6d232faeccbfed0c4f37f36de"
  ["$NFS_ROOT/style_loras/rainywindow/rainywindow.safetensors"]="7063a6f15ec6112ad3c06d79097b2a30a3ea7d9072821cb36021010d55989fe5"
  ["$NFS_ROOT/style_loras/retroanime/retroanime.safetensors"]="ca42107783d9e517c5d62cb9a9db9ab2ba4887d90e9dad97a9d1a7fe6ff14c56"
  ["$NFS_ROOT/style_loras/realism/krea2_realism_lora.safetensors"]="6c38a7934c54a56e0f67753660a4500a094d6dce28a0ee4a0d1dc9f4975d32d2"
)

for p in "${!styles[@]}"; do
  check_hash "$p" "${styles[$p]}" || missing=$((missing+1))
done

for p in \
  AGENTS.md \
  docs/CODEX_HANDOFF.md \
  prompting.md \
  scripts/style_lora_composition.py \
  scripts/chinese_prompt_smoke.py
do
  if [[ -e "$p" ]]; then
    echo "OK      $p"
  else
    echo "MISSING $p"
    missing=$((missing+1))
  fi
done

echo

if (( missing > 0 )); then
  echo "Bootstrap completed with $missing missing/invalid checks."
  exit 2
fi

echo "READY"
echo "Continue from: $REPO_DIR/docs/CODEX_HANDOFF.md"
echo
echo "Activate later with:"
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  export PYTHONPATH=."
echo "  export UV_CACHE_DIR=$UV_CACHE_DIR"
