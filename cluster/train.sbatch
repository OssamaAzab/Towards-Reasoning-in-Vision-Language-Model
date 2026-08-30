#!/usr/bin/env bash
#SBATCH --job-name=vlm_bridge_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.log

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the shared repository path}"
: "${VENV_PATH:?Set VENV_PATH to the Python virtual environment}"
: "${ENCODER:?Set ENCODER to clip, dinov2, or ijepa}"
: "${CKPT_TAG:?Set a fresh CKPT_TAG}"

BRIDGE_TYPE="${BRIDGE_TYPE:-mlp}"
LIMIT="${LIMIT:-150000}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
MLP_HIDDEN="${MLP_HIDDEN:-}"
STRIP_CLS="${STRIP_CLS:-0}"

[[ -d "$REPO_ROOT" ]] || { echo "REPO_ROOT is not a directory" >&2; exit 2; }
[[ -x "$VENV_PATH/bin/python" ]] || { echo "VENV_PATH has no Python interpreter" >&2; exit 2; }
[[ "$ENCODER" =~ ^(clip|dinov2|ijepa)$ ]] || { echo "Unsupported ENCODER" >&2; exit 2; }
[[ "$BRIDGE_TYPE" =~ ^(mlp|qformer|pool)$ ]] || { echo "Unsupported BRIDGE_TYPE" >&2; exit 2; }
[[ "$CKPT_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Unsafe CKPT_TAG" >&2; exit 2; }

cd "$REPO_ROOT"
mkdir -p outputs/logs outputs/checkpoints

if compgen -G "outputs/checkpoints/bridge_${ENCODER}_${CKPT_TAG}_*" >/dev/null; then
  echo "Checkpoint tag already exists: $CKPT_TAG" >&2
  exit 3
fi

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
  scripts/06c_train_bridge.py
  --encoder "$ENCODER"
  --bridge-type "$BRIDGE_TYPE"
  --limit "$LIMIT"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --lr 1e-4
  --weight-decay 0.01
  --batch-size 8
  --grad-accum 1
  --llm-precision bf16
  --prompt-format chatml_v1
  --supervise-eos
  --ckpt-tag "$CKPT_TAG"
)

if [[ "$BRIDGE_TYPE" =~ ^(mlp|pool)$ ]]; then
  : "${MLP_HIDDEN:?Set MLP_HIDDEN for an MLP or pool connector}"
  args+=(--mlp-hidden "$MLP_HIDDEN")
fi
if [[ "$STRIP_CLS" == "1" ]]; then
  args+=(--strip-cls)
fi

exec "$VENV_PATH/bin/python" "${args[@]}"
