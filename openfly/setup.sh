#!/usr/bin/env bash
# Install OpenFly-Platform + conda env + annotation JSON (outdoor VLN track).
set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"
ANNOTATION_DIR="${OPENFLY_ANNOTATION_DIR:-$HOME/assets/OpenFly/Annotation}"
CONDA_ENV="${OPENFLY_CONDA_ENV:-openfly}"

echo "=== OpenFly setup ==="
echo "  OPENFLY_ROOT=$OPENFLY_ROOT"
echo "  ANNOTATION_DIR=$ANNOTATION_DIR"

# --- conda ---
if [[ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  echo "Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CONDA_ENV} "; then
  conda create -n "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"

# --- clone OpenFly-Platform ---
if [[ ! -d "$OPENFLY_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/SHAILAB-IPEC/OpenFly-Platform.git "$OPENFLY_ROOT"
else
  echo "OpenFly-Platform already cloned at $OPENFLY_ROOT"
fi

pip install --upgrade pip
pip install numpy opencv-python
pip install msgpack-rpc-python
pip install airsim --no-build-isolation
pip install pyyaml unrealcv psutil requests pandas aiofiles scipy importlib_metadata
pip install huggingface_hub 'transformers>=4.45,<5' accelerate torch torchvision pillow timm einops

# flash-attn optional (GH200 may build; skip on failure)
pip install packaging ninja || true
pip install "flash-attn==2.5.5" --no-build-isolation 2>/dev/null || \
  echo "[openfly] flash-attn skipped (OpenFly-Agent will use default attention)"

# --- system deps (best-effort) ---
sudo apt-get update -qq
sudo apt-get install -y xvfb libgoogle-glog-dev 2>/dev/null || true

# --- annotations from HuggingFace ---
mkdir -p "$ANNOTATION_DIR"
download_annotation() {
  local file="$1"
  local dest="$ANNOTATION_DIR/$file"
  if [[ -f "$dest" ]]; then
    echo "  have $file"
    return
  fi
  echo "  downloading $file ..."
  python - <<PY
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id="IPEC-COMMUNITY/OpenFly",
    filename="Annotation/${file}",
    repo_type="dataset",
)
shutil.copy(path, "${dest}")
print("saved", "${dest}")
PY
}

download_annotation "unseen.json"
download_annotation "seen.json"
# train.json is large (~400 MB); skip with OPENFLY_SKIP_TRAIN=1
if [[ "${OPENFLY_SKIP_TRAIN:-0}" != "1" ]]; then
  download_annotation "train.json"
else
  echo "  skipping train.json (OPENFLY_SKIP_TRAIN=1)"
fi

# eval_test.json ships with the platform clone
if [[ ! -f "$OPENFLY_ROOT/configs/eval_test.json" ]]; then
  echo "WARN: missing eval_test.json in platform clone"
fi

# --- AirSim scene (smallest: env_airsim_16) ---
AIRSIM_ENV="$OPENFLY_ROOT/envs/airsim/env_airsim_16"
if [[ ! -d "$AIRSIM_ENV" ]]; then
  echo ""
  echo "=== AirSim scene download (large, ~several GB) ==="
  echo "Download env_airsim_16 from:"
  echo "  https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen/tree/main/airsim"
  echo "Extract to: $OPENFLY_ROOT/envs/airsim/env_airsim_16"
  echo ""
  echo "Optional auto-download (may take a while):"
  echo "  bash $DRONE_PROJECT/openfly/download_airsim_scene.sh env_airsim_16"
fi

mkdir -p "$DRONE_PROJECT/logs/benchmarks"

echo ""
echo "=== Done ==="
echo "  source $DRONE_PROJECT/openfly/activate.sh"
echo "  # after AirSim scene is installed:"
echo "  python -m openfly.eval_benchmark --split unseen --policy heuristic --max_episodes 3 --env_filter env_airsim_16"
