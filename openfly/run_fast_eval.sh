#!/usr/bin/env bash
# Fast sim SR eval wrapper. Same activation pattern as run_eval.sh.
#
# Usage:
#   ./openfly/run_fast_eval.sh                          # 20 unseen episodes, heuristic policy
#   ./openfly/run_fast_eval.sh paligemma <ckpt> unseen 20
#   ./openfly/run_fast_eval.sh paligemma <ckpt> seen 30
#
# Any extra args after the positional ones are forwarded to fast_sr_eval.py.

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"
cd "$DRONE_PROJECT"

POLICY="${1:-heuristic}"
CHECKPOINT="${2:-}"
SPLIT="${3:-unseen}"
N_EPISODES="${4:-20}"
shift $(($# < 4 ? $# : 4))

ARGS=(--policy "$POLICY" --split "$SPLIT" --n_episodes "$N_EPISODES")
if [[ "$POLICY" =~ ^(paligemma|grpo|vla|ppo|ppo-agent|openfly-agent-rl)$ ]]; then
  if [[ -z "$CHECKPOINT" ]]; then
    echo "policy=$POLICY requires a checkpoint as the second positional arg" >&2
    exit 2
  fi
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

python -m openfly.scripts.fast_sr_eval "${ARGS[@]}" "$@"
