#!/usr/bin/env bash
# Source before OpenFly work (replaces activate_env.sh for outdoor track).
set -euo pipefail

export OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"
export OPENFLY_ANNOTATION_DIR="${OPENFLY_ANNOTATION_DIR:-$HOME/assets/OpenFly/Annotation}"
export OPENFLY_IMAGE_ROOT="${OPENFLY_IMAGE_ROOT:-$HOME/assets/OpenFly/images/Image}"
export DRONE_PROJECT="${DRONE_PROJECT:-$HOME/SkyVLA}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate openfly 2>/dev/null || true
fi

export PYTHONPATH="${DRONE_PROJECT}:${PYTHONPATH:-}"

echo "[openfly] OPENFLY_ROOT=$OPENFLY_ROOT"
echo "[openfly] OPENFLY_ANNOTATION_DIR=$OPENFLY_ANNOTATION_DIR"
echo "[openfly] OPENFLY_IMAGE_ROOT=$OPENFLY_IMAGE_ROOT"
echo "[openfly] DRONE_PROJECT=$DRONE_PROJECT"
if [[ -d "$OPENFLY_ROOT" ]]; then
  echo "[openfly] platform OK"
else
  echo "[openfly] WARN: run bash $DRONE_PROJECT/openfly/setup.sh"
fi
