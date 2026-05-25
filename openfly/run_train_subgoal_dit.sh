#!/usr/bin/env bash
# Phase P2 — pretrain the feature-space SubgoalDiT world model on OpenFly
# trajectories. PaliGemma is frozen; only the DiT trains.
#
# Quick smoke run (1 episode, debug):
#   bash openfly/run_train_subgoal_dit.sh \
#     --max_episodes 8 --epochs 1 --batch_size 4 --log_every 1
#
# Full pretrain (single A100, ~hours per epoch on train.json):
#   bash openfly/run_train_subgoal_dit.sh --epochs 5 --batch_size 8
set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.train_subgoal_dit "$@"
