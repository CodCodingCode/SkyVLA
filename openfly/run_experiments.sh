#!/usr/bin/env bash
# Orchestrate the full B0 -> B5 experiment matrix from docs/RESEARCH.md.
#
# Each stage writes its own checkpoints under logs/openfly/<track>/<run>/
# and its evaluation JSONs under logs/benchmarks/. The script is
# resumable: pass --skip to jump over stages whose checkpoints are
# already on disk.
#
# This is intentionally a thin sequencer; each underlying call already
# lives in its own shell wrapper so individual stages can be re-run by
# hand. Heavy GPU work (B1 SFT in particular) takes hours, so prefer
# running the script under ``tmux`` or ``nohup``.
#
# Required env: an x86_64 host with the AirSim UE binary launchable,
# the conda env ``openfly`` set up by ``openfly/setup.sh``, and
# OPENFLY_IMAGE_ROOT pointing at downloaded trajectory frames.
#
# Example:
#   bash openfly/run_experiments.sh --skip b0,b1 \
#       --sft_ckpt logs/openfly/paligemma/<run>/last.pt
#
# Stages:
#   B0  heuristic eval (no training)
#   B1  PaliGemma SFT
#   B2  DAgger from B1
#   B3  PPO dense baseline from B2
#   B4  GRPO cold sparse from B2
#   B5  GRPO curriculum (easy -> medium -> hard) from B2

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/SkyVLA}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"
cd "$DRONE_PROJECT"

SKIP=""
SFT_CKPT=""
DAGGER_CKPT=""
ENV_FILTER="env_airsim_16"
EVAL_EPISODES=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip) SKIP="$2"; shift 2 ;;
    --sft_ckpt) SFT_CKPT="$2"; shift 2 ;;
    --dagger_ckpt) DAGGER_CKPT="$2"; shift 2 ;;
    --env_filter) ENV_FILTER="$2"; shift 2 ;;
    --eval_episodes) EVAL_EPISODES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

skipped() { [[ ",$SKIP," == *",$1,"* ]]; }

# Pre-flight
bash openfly/scripts/verify_phase0.sh

# B0: heuristic eval
if skipped b0; then
  echo "[experiments] skip B0"
else
  echo "[experiments] B0 heuristic eval (per-env unseen + seen)"
  bash openfly/run_per_env_eval.sh --policy heuristic --tag b0_heuristic \
    --max_episodes "$EVAL_EPISODES"
fi

# B1: SFT
if skipped b1; then
  echo "[experiments] skip B1 (using --sft_ckpt=$SFT_CKPT)"
else
  echo "[experiments] B1 PaliGemma SFT"
  bash openfly/run_train_paligemma.sh --epochs 10 --batch_size 8
  SFT_CKPT="${SFT_CKPT:-$(ls -1t logs/openfly/paligemma/*/last.pt | head -1)}"
fi
if [[ -z "$SFT_CKPT" ]]; then
  echo "[experiments] cannot continue without --sft_ckpt" >&2
  exit 1
fi
bash openfly/run_per_env_eval.sh \
  --policy paligemma --paligemma_ckpt "$SFT_CKPT" --tag b1_sft \
  --max_episodes "$EVAL_EPISODES"

# B2: DAgger from B1
if skipped b2; then
  echo "[experiments] skip B2 (using --dagger_ckpt=$DAGGER_CKPT)"
else
  echo "[experiments] B2 DAgger from B1"
  bash openfly/run_train_dagger.sh --sft_ckpt "$SFT_CKPT" \
    --iterations 3 --episodes_per_iter 200
  DAGGER_CKPT="${DAGGER_CKPT:-$(ls -1t logs/openfly/dagger/*/last.pt | head -1)}"
fi
if [[ -z "$DAGGER_CKPT" ]]; then
  echo "[experiments] cannot continue without --dagger_ckpt" >&2
  exit 1
fi
bash openfly/run_per_env_eval.sh \
  --policy dagger --paligemma_ckpt "$DAGGER_CKPT" --tag b2_dagger \
  --max_episodes "$EVAL_EPISODES"

# B3: PPO dense baseline
if skipped b3; then
  echo "[experiments] skip B3"
else
  echo "[experiments] B3 PPO dense from DAgger"
  bash openfly/run_train_ppo_agent.sh \
    --reward_preset easy \
    --iterations 20 --episodes_per_iter 4
  PPO_CKPT=$(ls -1t logs/openfly/ppo_agent/*/last.pt | head -1)
  bash openfly/run_per_env_eval.sh \
    --policy ppo --ppo_ckpt "$PPO_CKPT" --tag b3_ppo_dense \
    --max_episodes "$EVAL_EPISODES"
fi

# B4: GRPO cold sparse
if skipped b4; then
  echo "[experiments] skip B4"
else
  echo "[experiments] B4 GRPO cold sparse from DAgger"
  bash openfly/run_train_grpo.sh \
    --init_ckpt "$DAGGER_CKPT" \
    --steps 200 --reward_preset hard
  GRPO_CKPT=$(ls -1t logs/openfly/grpo/*/last.pt | head -1)
  bash openfly/run_per_env_eval.sh \
    --policy grpo --paligemma_ckpt "$GRPO_CKPT" --tag b4_grpo_cold_sparse \
    --max_episodes "$EVAL_EPISODES"
fi

# B5: GRPO curriculum
if skipped b5; then
  echo "[experiments] skip B5"
else
  echo "[experiments] B5 GRPO curriculum from DAgger"
  bash openfly/run_train_curriculum.sh \
    --init_ckpt "$DAGGER_CKPT" \
    --env_filter "$ENV_FILTER" \
    --steps_easy 80 --steps_medium 60 --steps_hard 60
  CURR_CKPT=$(ls -1t logs/openfly/curriculum/*/stage_hard/last.pt | head -1)
  bash openfly/run_per_env_eval.sh \
    --policy grpo --paligemma_ckpt "$CURR_CKPT" --tag b5_curriculum \
    --max_episodes "$EVAL_EPISODES"
fi

echo
echo "[experiments] done. Roll up tables with:"
echo "  python -m openfly.scripts.aggregate_results --logs_dir logs/benchmarks --per-env --output docs/results_table.md"
echo "  python -m openfly.scripts.analyse_failures --inputs 'logs/benchmarks/openfly_unseen_*.json' --output docs/failure_modes.md"
