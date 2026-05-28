#!/usr/bin/env bash
# Phase P2 — pretrain the feature-space SubgoalDiT world model on OpenFly
# trajectories. PaliGemma is frozen; only the DiT trains.
#
# --pretrained_path is REQUIRED. Random-init plateaus around val_cos≈0.6;
# starting from PixArt-Σ reaches the same point in a fraction of the
# steps and keeps climbing. The local snapshot lives at:
#   ~/assets/pretrained/hf_cache/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/snapshots/<hash>/transformer
#
# Quick smoke run (8 episodes, debug):
#   bash openfly/run_train_subgoal_dit.sh \
#     --pretrained_path <PIXART_DIR> \
#     --max_episodes 8 --epochs 1 --batch_size 4 --log_every 1
#
# Full pretrain (single A100, ~hours per epoch on train.json):
#   bash openfly/run_train_subgoal_dit.sh \
#     --pretrained_path <PIXART_DIR> \
#     --epochs 5 --batch_size 8
set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"

# PYTHONUNBUFFERED=1 (== ``python -u``) forces stdout/stderr to be
# line-buffered instead of the default block-buffered when not attached
# to a TTY. Without this, progress prints accumulate in memory and only
# appear in log files when the process exits cleanly — so any SIGTERM
# (timeout, OOM, manual kill) leaves the log looking like the script
# died silently before printing anything.
export PYTHONUNBUFFERED=1
# PYTHONFAULTHANDLER=1 prints a Python stack trace to stderr on segfault
# or fatal Python error — cheap to enable, free debugging when a CUDA
# extension or pretrained model crashes the interpreter.
export PYTHONFAULTHANDLER=1

python -u -m openfly.train_subgoal_dit "$@"
