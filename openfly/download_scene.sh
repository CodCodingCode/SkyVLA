#!/usr/bin/env bash
# Download an OpenFly scene binary from HuggingFace OpenFly_DataGen.
#
# Layout on the HF dataset (verified 2026-05-26):
#   airsim/env_airsim_{16,18,23,26,gz,sh}.zip
#   ue/env_ue_{bigcity,smallcity}.zip       (+ *_default_low_quality.zip, *_windows.zip)
#   pcd_map/<scene>.pcd
# There is no game/ or gs/ tree on this dataset — env_game_gtav requires
# Windows + DeepGTAV (see upstream OpenFly-Platform README), and the
# env_gs_* Gaussian-splat scenes are not published yet ("Coming soon").
#
# Usage:
#   bash openfly/download_scene.sh env_airsim_16
#   bash openfly/download_scene.sh env_ue_smallcity
#   bash openfly/download_scene.sh env_ue_smallcity --quality low     # default_low_quality variant
#   bash openfly/download_scene.sh env_ue_smallcity --with-pcd        # also fetch the .pcd map

set -euo pipefail

SCENE="${1:?Usage: $0 SCENE_NAME [--quality low|windows] [--with-pcd]}"
shift || true

QUALITY=""
WITH_PCD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quality) QUALITY="${2:-}"; shift 2 ;;
    --with-pcd) WITH_PCD=1; shift ;;
    *) echo "[download_scene] unknown flag: $1" >&2; exit 2 ;;
  esac
done

OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"

case "$SCENE" in
  env_airsim_*) FAMILY="airsim" ;;
  env_ue_*)     FAMILY="ue" ;;
  env_game_*)
    cat >&2 <<EOF
[download_scene] $SCENE is a GTAV scene and is NOT distributed via HuggingFace.
  Upstream requires Windows + DeepGTAV; see:
  https://github.com/SHAILAB-IPEC/OpenFly-Platform#4-gtav-env_game_xxx-
EOF
    exit 1
    ;;
  env_gs_*)
    cat >&2 <<EOF
[download_scene] $SCENE is a Gaussian-splatting scene and is not yet published
  ("Coming soon" on the upstream README). You will also need SIBR_viewers
  built locally to render it.
EOF
    exit 1
    ;;
  *)
    echo "[download_scene] cannot infer family from scene name: $SCENE" >&2
    echo "  expected prefix env_airsim_ or env_ue_" >&2
    exit 1
    ;;
esac

case "$QUALITY" in
  "")      ZIP_NAME="${SCENE}.zip" ;;
  low)     ZIP_NAME="${SCENE}_default_low_quality.zip" ;;
  windows) ZIP_NAME="${SCENE}_windows.zip" ;;
  *) echo "[download_scene] --quality must be low or windows" >&2; exit 2 ;;
esac

DEST_DIR="$OPENFLY_ROOT/envs/$FAMILY"
DEST_ZIP="$DEST_DIR/$ZIP_NAME"
DEST_UNPACKED="$DEST_DIR/$SCENE"

mkdir -p "$DEST_DIR"

if [[ -d "$DEST_UNPACKED" ]]; then
  echo "[download_scene] $DEST_UNPACKED already exists; skipping"
  exit 0
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate openfly 2>/dev/null || true

echo "[download_scene] family=$FAMILY scene=$SCENE zip=$ZIP_NAME -> $DEST_DIR"
echo "[download_scene] downloading from IPEC-COMMUNITY/OpenFly_DataGen (multi-GB)"

# hf_hub_download is reliable for single-file fetches; snapshot_download with
# allow_patterns silently fetches zero files when the pattern doesn't match
# (the old bug this script had: "${FAMILY}/${SCENE}/**" never matches a .zip).
python - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(
    repo_id="IPEC-COMMUNITY/OpenFly_DataGen",
    repo_type="dataset",
    filename="${FAMILY}/${ZIP_NAME}",
)
dest = os.path.expanduser("${DEST_ZIP}")
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copy(src, dest)
print("downloaded", dest, round(os.path.getsize(dest)/1e9, 2), "GB")
PY

if [[ ! -s "$DEST_ZIP" ]]; then
  echo "[download_scene] ERROR: $DEST_ZIP missing or empty after download" >&2
  exit 1
fi

echo "[download_scene] unzipping into $DEST_DIR ..."
( cd "$DEST_DIR" && unzip -q -o "$ZIP_NAME" )

if [[ ! -d "$DEST_UNPACKED" ]]; then
  echo "[download_scene] WARNING: expected $DEST_UNPACKED after unzip; check contents of $DEST_DIR" >&2
fi

if [[ "$WITH_PCD" == "1" ]]; then
  echo "[download_scene] also fetching pcd_map/${SCENE}.pcd"
  python - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(
    repo_id="IPEC-COMMUNITY/OpenFly_DataGen",
    repo_type="dataset",
    filename="pcd_map/${SCENE}.pcd",
)
dest = os.path.expanduser("${OPENFLY_ROOT}/envs/pcd_map/${SCENE}.pcd")
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copy(src, dest)
print("downloaded", dest, round(os.path.getsize(dest)/1e6, 1), "MB")
PY
fi

echo "[download_scene] done. verify: ls $DEST_UNPACKED"
