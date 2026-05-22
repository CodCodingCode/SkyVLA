#!/usr/bin/env bash
# Wrap upstream OpenFly-Platform/train/train.py for the OpenFly-Agent
# (OpenVLA 7B) FSDP fine-tune.
#
# This script does NOT reimplement training in-repo. It documents and
# launches the official entry point. For a custom PaliGemma fine-tune use
# `openfly/run_train_paligemma.sh` instead.
#
# Required env (or pass --pretrained_checkpoint):
#   OPENFLY_ROOT                  default ~/OpenFly-Platform
#   OPENFLY_TFDS_DIR              TFDS build output (vln_norm)
#   OPENFLY_PRETRAINED_CKPT       default IPEC-COMMUNITY/openfly-agent-7b
#   OPENFLY_NPROC_PER_NODE        default 8 (matches upstream train.sh)
#
# Prerequisite (one-time): build the TFDS dataset from train.json
#   cd $OPENFLY_ROOT/train/dataset_builder/vln
#   tfds build --data_dir $OPENFLY_TFDS_DIR
# See openfly/README.md "Training track A" for full setup.

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"
PRETRAINED="${OPENFLY_PRETRAINED_CKPT:-IPEC-COMMUNITY/openfly-agent-7b}"
NPROC="${OPENFLY_NPROC_PER_NODE:-8}"

if [[ -z "${OPENFLY_TFDS_DIR:-}" ]]; then
  echo "ERROR: set OPENFLY_TFDS_DIR to your TFDS build output (vln_norm dataset)." >&2
  echo "See openfly/README.md 'Training track A: OpenFly-Agent' for the build steps." >&2
  exit 1
fi

cd "$OPENFLY_ROOT/train"

echo "[openfly] launching upstream FSDP training"
echo "  OPENFLY_ROOT=$OPENFLY_ROOT"
echo "  TFDS_DIR=$OPENFLY_TFDS_DIR"
echo "  PRETRAINED=$PRETRAINED"
echo "  nproc-per-node=$NPROC"

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node "$NPROC" \
  train.py \
  --grid_size 16 \
  --history_frames 2 \
  --pretrained_checkpoint "$PRETRAINED" \
  --data_root_dir "$OPENFLY_TFDS_DIR" \
  "$@"
