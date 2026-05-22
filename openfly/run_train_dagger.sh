#!/usr/bin/env bash
# DAgger fine-tuning between SFT and online RL.
#
# Usage:
#   bash ~/drone_project/openfly/run_train_dagger.sh \
#     --sft_ckpt logs/openfly/paligemma/<run>/last.pt \
#     --iterations 3 --episodes_per_iter 200
#
# For Track A (OpenFly-Agent 7B) we only collect a corrected JSONL:
#   bash ~/drone_project/openfly/run_train_dagger.sh \
#     --track openfly-agent --episodes_per_iter 100

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.train_dagger "$@"
