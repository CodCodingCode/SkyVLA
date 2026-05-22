#!/usr/bin/env bash
set -euo pipefail
SCENE="${1:-env_airsim_16}"
OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"
DEST="$OPENFLY_ROOT/envs/airsim/$SCENE"

if [[ -d "$DEST" ]]; then
  echo "Scene already at $DEST"
  exit 0
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate openfly 2>/dev/null || true

mkdir -p "$OPENFLY_ROOT/envs/airsim"
echo "Downloading $SCENE from HuggingFace OpenFly_DataGen (this is large)..."

python - <<PY
from huggingface_hub import snapshot_download
import os
dest = os.path.expanduser("${DEST}")
parent = os.path.dirname(dest)
os.makedirs(parent, exist_ok=True)
path = snapshot_download(
    repo_id="IPEC-COMMUNITY/OpenFly_DataGen",
    repo_type="dataset",
    allow_patterns=["airsim/${SCENE}/**"],
    local_dir=parent,
)
print("Downloaded under", parent)
PY

echo "If files landed under airsim/$SCENE, verify: ls $DEST"
