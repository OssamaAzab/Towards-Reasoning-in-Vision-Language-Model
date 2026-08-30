#!/usr/bin/env bash
# Source this file from the repository root: source env.sh

_VLM_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$_VLM_ENV_ROOT"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"

if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  echo "Missing .venv. Create it with: python3.12 -m venv .venv" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"
unset _VLM_ENV_ROOT
echo "Environment ready: $PROJECT_ROOT"
