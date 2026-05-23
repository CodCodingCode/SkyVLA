#!/usr/bin/env bash
# Download an OpenFly scene binary from HuggingFace OpenFly_DataGen.
# Supports the three unseen environments
# (env_game_gtav, env_ue_smallcity, env_gs_sjtu02) plus any AirSim
# scene name. The script picks the correct subdirectory of
# OPENFLY_ROOT/envs based on the scene prefix.
#
# Usage:
#   bash openfly/download_scene.sh env_game_gtav
#   bash openfly/download_scene.sh env_ue_smallcity
#   bash openfly/download_scene.sh env_gs_sjtu02
#   bash openfly/download_scene.sh env_airsim_16     # alias of download_airsim_scene.sh

set -euo pipefail

SCENE="${1:?Usage: $0 SCENE_NAME}"
OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"

case "$SCENE" in
  env_airsim_*) FAMILY="airsim" ;;
  env_game_*)   FAMILY="game" ;;
  env_ue_*)     FAMILY="ue" ;;
  env_gs_*)     FAMILY="gs" ;;
  *)
    echo "[download_scene] cannot infer family from scene name: $SCENE" >&2
    echo "  expected prefix env_airsim_/env_game_/env_ue_/env_gs_" >&2
    exit 1
    ;;
esac

DEST="$OPENFLY_ROOT/envs/$FAMILY/$SCENE"
if [[ -d "$DEST" ]]; then
  echo "[download_scene] $DEST already exists; skipping"
  exit 0
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate openfly 2>/dev/null || true

mkdir -p "$OPENFLY_ROOT/envs/$FAMILY"
echo "[download_scene] family=$FAMILY scene=$SCENE -> $DEST"
echo "[download_scene] downloading from IPEC-COMMUNITY/OpenFly_DataGen (large!)"

python - <<PY
from huggingface_hub import snapshot_download
import os
parent = os.path.expanduser("$OPENFLY_ROOT/envs/$FAMILY")
os.makedirs(parent, exist_ok=True)
snapshot_download(
    repo_id="IPEC-COMMUNITY/OpenFly_DataGen",
    repo_type="dataset",
    allow_patterns=["${FAMILY}/${SCENE}/**"],
    local_dir=parent,
)
print("Downloaded under", parent)
PY

echo "[download_scene] verify: ls $DEST"
