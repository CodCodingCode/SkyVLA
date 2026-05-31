#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (parent of openfly/) so module imports work from anywhere.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

# Load Weights & Biases API key if present (kept out of git).
WANDB_KEY_FILE="$REPO_ROOT/.wandb_key"
if [ -f "$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi

# Default W&B project/entity (can be overridden by caller env).
export WANDB_PROJECT="${WANDB_PROJECT:-skyvla-subgoal-dit}"

# Source the shared env so OPENFLY_IMAGE_ROOT / OPENFLY_ANNOTATION_DIR are set
# (the dataset resolves image/annotation paths from these). Without it the RL
# dataset indexes 0 frames. activate.sh also conda-activates 'openfly'.
# shellcheck disable=SC1091
source "$REPO_ROOT/openfly/activate.sh"
cd "$REPO_ROOT"

export PYTHONFAULTHANDLER=1
# Reduce fragmentation OOMs on the 40GB A100 (two frozen PaliGemma-3B + trainable
# PixArt DiT + Adam + GRPO group all coresident). Recommended by the OOM message.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec python -u -m openfly.train_subgoal_dit_ddpo "$@"
