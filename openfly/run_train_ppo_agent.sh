#!/usr/bin/env bash
# PPO + LoRA + value head on OpenFly-Agent 7B.
#
# Heavy: needs ~24 GB VRAM for the 7B backbone in bf16 plus rollout +
# update compute. Run on the GH200 or an H100/A100 node.
#
# Example:
#   bash ~/drone_project/openfly/run_train_ppo_agent.sh \
#     --iterations 30 --episodes_per_iter 4 --ppo_epochs 2 --kl_coef 0.02

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.train_ppo_openfly_agent "$@"
