#!/usr/bin/env bash
# GRPO online RL on the PaliGemma BC policy.
#
# Required:
#   --init_ckpt PATH   PaliGemma checkpoint from DAgger (preferred) or SFT
#
# Example:
#   bash ~/drone_project/openfly/run_train_grpo.sh \
#     --init_ckpt logs/openfly/dagger/<run>/last.pt \
#     --steps 200 --group_size 4 --instructions_per_step 2

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.train_grpo_paligemma "$@"
