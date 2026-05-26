#!/usr/bin/env bash
# Phase P3 — BC + subgoal tokens. Loads a frozen P2 SubgoalDiT and
# a P1 BC checkpoint, finetunes the policy under action CE with
# subgoal tokens (mix of oracle and DiT-generated) fed into its
# cross-attention. See docs/JOINT_TRAINING.md.
#
# Required:
#   --bc_init_ckpt logs/openfly/paligemma/<run>/last.pt
#   --dit_path     logs/openfly/subgoal_dit/<run>/best.pt
#
# Optional (PixArt-init world model):
#   --pretrained_path /path/to/pixart-sigma/snapshot
set -euo pipefail
DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"
cd "$DRONE_PROJECT"
python -m openfly.train_paligemma_subgoal "$@"
