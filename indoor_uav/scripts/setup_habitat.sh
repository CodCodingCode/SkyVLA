#!/usr/bin/env bash
# Set up the isolated Habitat-Sim env + download indoor navigation scenes.
#
# Habitat-Sim lives in a DEDICATED conda env ('habitat') so its heavy / pinned
# deps never collide with the cu130 training stack in 'openfly'. The GS pipeline
# consumes Habitat output either in-process (from this env) or via dumped RGB-D
# frames (render_habitat.py), so the two envs stay decoupled.
set -euo pipefail

CONDA="${CONDA:-$HOME/miniconda3}"
SCENES_DIR="${SCENES_DIR:-$HOME/assets/indoor_scenes}"
mkdir -p "$SCENES_DIR"

if ! "$CONDA/bin/conda" env list | grep -q '^habitat'; then
  "$CONDA/bin/conda" create -n habitat -c aihabitat -c conda-forge \
    python=3.9 habitat-sim headless -y
fi

# Lightweight built-in test scenes (small) — good for bring-up.
"$CONDA/envs/habitat/bin/python" -m habitat_sim.utils.datasets_download \
  --uids habitat_test_scenes --data-path "$SCENES_DIR" || true

# HM3D minival (real Matterport indoor homes) needs a one-time Matterport token;
# see https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
# "$CONDA/envs/habitat/bin/python" -m habitat_sim.utils.datasets_download \
#   --uids hm3d_minival_v0.2 --data-path "$SCENES_DIR"

echo "[setup_habitat] scenes under: $SCENES_DIR"
find "$SCENES_DIR" -name "*.glb" 2>/dev/null | head
