#!/usr/bin/env bash
# Run all available drone navigation benchmarks for drone_project.
set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

export PYTHONUNBUFFERED=1
OUT="$DRONE_PROJECT/logs/benchmarks"
BENCH_EXT="${BENCH_EXT:-$HOME/benchmarks_external}"
mkdir -p "$OUT"

echo "=== drone_project benchmark suite ==="
echo "Results -> $OUT"
date -Is | tee "$OUT/run_started.txt"

# --- 1) OpenFly (primary outdoor VLN benchmark) -------------------------
echo ""
echo "[1/2] OpenFly — heuristic + OpenFly-Agent"
ENV_FILTER="${OPENFLY_ENV_FILTER:-env_airsim_16}"
MAX_EPS="${OPENFLY_MAX_EPISODES:-5}"

bash "$DRONE_PROJECT/openfly/run_eval.sh" \
  --split unseen --policy heuristic \
  --env_filter "$ENV_FILTER" --max_episodes "$MAX_EPS" \
  --output "$OUT/openfly_unseen_heuristic.json" \
  || echo "WARN: openfly heuristic eval failed (AirSim scene installed?)"

if [[ "${RUN_OPENFLY_AGENT:-0}" == "1" ]]; then
  bash "$DRONE_PROJECT/openfly/run_eval.sh" \
    --split unseen --policy openfly-agent \
    --env_filter "$ENV_FILTER" --max_episodes "$MAX_EPS" \
    --output "$OUT/openfly_unseen_agent.json" \
    || echo "WARN: openfly-agent eval failed (flash-attn / model download?)"
fi

if [[ -n "${PALIGEMMA_CKPT:-}" && -f "$PALIGEMMA_CKPT" ]]; then
  bash "$DRONE_PROJECT/openfly/run_eval.sh" \
    --split unseen --policy paligemma \
    --paligemma_ckpt "$PALIGEMMA_CKPT" \
    --env_filter "$ENV_FILTER" --max_episodes "$MAX_EPS" \
    --output "$OUT/openfly_unseen_paligemma.json"
fi

# --- 2) CityNav oracle baseline (optional) -------------------------------
echo ""
echo "[2/2] CityNav oracle"
CITYNAV_ROOT="${CITYNAV_ROOT:-$BENCH_EXT/citynav}"
if [[ -d "$CITYNAV_ROOT/vlnce" && -d "$CITYNAV_ROOT/data" ]]; then
  python -m benchmarks.eval_citynav_oracle \
    --citynav_root "$CITYNAV_ROOT" \
    --split val_seen --max_episodes 100 2>&1 \
    | tee "$OUT/citynav_val_seen.txt"
else
  echo "SKIP: CityNav data not at $CITYNAV_ROOT (see benchmarks/setup_external.sh)"
  echo "status=skipped missing_data" > "$OUT/citynav_status.txt"
fi

echo ""
echo "Done. See $OUT/"
