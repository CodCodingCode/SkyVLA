#!/usr/bin/env bash
# Phase P3.5 — joint refine of (world model + policy). Unfreezes the
# P2 SubgoalDiT for a short joint-training run combining
# diffusion ε-MSE and action CE. See docs/JOINT_TRAINING.md.
#
# Required:
#   --p3_ckpt   logs/openfly/paligemma_subgoal/<run>/best.pt
#   --dit_path  logs/openfly/subgoal_dit/<run>/best.pt
#
# Optional (PixArt-init world model):
#   --pretrained_path /path/to/pixart-sigma/snapshot
#
# Typical run:
#   bash openfly/run_train_joint_refine.sh \
#     --p3_ckpt logs/openfly/paligemma_subgoal/<run>/best.pt \
#     --dit_path logs/openfly/subgoal_dit/<run>/best.pt \
#     --epochs 1 --batch_size 2 --ddim_steps 4 \
#     --lambda_mse 1.0 --lambda_ce 0.3
set -euo pipefail
DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"
cd "$DRONE_PROJECT"
python -m openfly.train_joint_refine "$@"
