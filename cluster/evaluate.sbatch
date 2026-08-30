#!/usr/bin/env bash
#SBATCH --job-name=vlm_bridge_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.log

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the shared repository path}"
: "${VENV_PATH:?Set VENV_PATH to the Python virtual environment}"
: "${CHECKPOINT:?Set CHECKPOINT to a trained bridge file}"
: "${QIDS:?Set QIDS to one frozen endpoint JSON file}"

[[ -d "$REPO_ROOT" ]] || { echo "REPO_ROOT is not a directory" >&2; exit 2; }
[[ -x "$VENV_PATH/bin/python" ]] || { echo "VENV_PATH has no Python interpreter" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "CHECKPOINT does not exist" >&2; exit 2; }
[[ -f "$QIDS" ]] || { echo "QIDS does not exist" >&2; exit 2; }

cd "$REPO_ROOT"
mkdir -p outputs/logs outputs/eval

nvidia-smi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "Assigned GPU already has a compute process" >&2
  exit 4
fi
if pgrep -af '06c_train_bridge|07_evaluate' >/dev/null; then
  echo "Another project train/evaluate process is active" >&2
  exit 4
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

args=(
  scripts/07_evaluate.py
  --checkpoint "$CHECKPOINT"
  --qids "$QIDS"
  --out-dir outputs/eval
  --num-examples 0
)

if [[ -n "${STEM_SUFFIX:-}" ]]; then
  [[ "$STEM_SUFFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Unsafe STEM_SUFFIX" >&2; exit 2; }
  args+=(--stem-suffix "$STEM_SUFFIX")
fi

exec "$VENV_PATH/bin/python" "${args[@]}"
