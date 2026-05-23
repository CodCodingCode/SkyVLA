#!/usr/bin/env bash
# Evaluate one PaliGemma/DAgger/GRPO/PPO checkpoint on every OpenFly
# unseen environment plus the full seen split, producing one
# ``logs/benchmarks/openfly_*.json`` per call. The per-env JSON files
# can then be rolled up with ``openfly.scripts.aggregate_results``.
#
# Required:
#   --paligemma_ckpt PATH    (for paligemma|dagger|grpo policies)
#     or
#   --ppo_ckpt PATH          (for ppo)
#
# Optional:
#   --policy NAME            Defaults to "paligemma".
#   --max_episodes N         Episodes per env (default 50; seen uses 4x).
#   --max_steps N            Step budget per episode (default 100).
#   --tag NAME               Suffix appended to each output filename.
#
# Example:
#   bash openfly/run_per_env_eval.sh \
#     --policy grpo \
#     --paligemma_ckpt logs/openfly/curriculum/<run>/stage_hard/last.pt \
#     --tag b5_curriculum

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/SkyVLA}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

POLICY="paligemma"
PALIGEMMA_CKPT=""
PPO_CKPT=""
MAX_EPISODES=50
MAX_STEPS=100
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$2"; shift 2 ;;
    --paligemma_ckpt) PALIGEMMA_CKPT="$2"; shift 2 ;;
    --ppo_ckpt) PPO_CKPT="$2"; shift 2 ;;
    --max_episodes) MAX_EPISODES="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

CKPT_FLAG=()
case "$POLICY" in
  paligemma|dagger|grpo)
    if [[ -z "$PALIGEMMA_CKPT" ]]; then
      echo "--paligemma_ckpt is required for policy=$POLICY" >&2
      exit 1
    fi
    CKPT_FLAG=("--paligemma_ckpt" "$PALIGEMMA_CKPT")
    ;;
  ppo)
    if [[ -z "$PPO_CKPT" ]]; then
      echo "--ppo_ckpt is required for policy=ppo" >&2
      exit 1
    fi
    CKPT_FLAG=("--ppo_ckpt" "$PPO_CKPT")
    ;;
  heuristic|openfly-agent)
    : # no checkpoint flag needed
    ;;
  *)
    echo "unknown policy: $POLICY" >&2; exit 1 ;;
esac

cd "$DRONE_PROJECT"
mkdir -p logs/benchmarks

# Per-env unseen breakdown (the headline numbers).
for ENV_NAME in env_game_gtav env_ue_smallcity env_gs_sjtu02; do
  TS=$(date +%Y%m%d_%H%M%S)
  SUFFIX=""
  if [[ -n "$TAG" ]]; then
    SUFFIX="_$TAG"
  fi
  OUT="logs/benchmarks/openfly_unseen_${POLICY}_${ENV_NAME}${SUFFIX}_${TS}.json"
  echo
  echo "[per-env] policy=$POLICY env=$ENV_NAME -> $OUT"
  python -m openfly.eval_benchmark \
    --split unseen \
    --policy "$POLICY" \
    "${CKPT_FLAG[@]}" \
    --env_filter "$ENV_NAME" \
    --max_episodes "$MAX_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --output "$OUT"
done

# Full seen split as a dev-set check.
TS=$(date +%Y%m%d_%H%M%S)
SUFFIX=""
if [[ -n "$TAG" ]]; then
  SUFFIX="_$TAG"
fi
OUT="logs/benchmarks/openfly_seen_${POLICY}${SUFFIX}_${TS}.json"
echo
echo "[per-env] policy=$POLICY split=seen -> $OUT"
python -m openfly.eval_benchmark \
  --split seen \
  --policy "$POLICY" \
  "${CKPT_FLAG[@]}" \
  --max_episodes "$(( MAX_EPISODES * 4 ))" \
  --max_steps "$MAX_STEPS" \
  --output "$OUT"

echo
echo "[per-env] done. Aggregate with:"
echo "  python -m openfly.scripts.aggregate_results --logs_dir logs/benchmarks --per-env"
