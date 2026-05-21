#!/usr/bin/env bash
# Run all available drone navigation benchmarks for drone_project.
set -euo pipefail

source /home/ubuntu/miniconda3/bin/activate isaac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

DRONE=/home/ubuntu/drone_project
OUT="$DRONE/logs/benchmarks"
BENCH_EXT="${BENCH_EXT:-$HOME/benchmarks_external}"
mkdir -p "$OUT"

echo "=== drone_project benchmark suite ==="
echo "Results -> $OUT"
date -Is | tee "$OUT/run_started.txt"

# --- 1) HUGE-Bench (offline) -------------------------------------------
echo ""
echo "[1/4] HUGE-Bench — baselines (fast)"
for SPLIT in test_seen test_unseen; do
  python -m benchmarks.eval_huge \
    --backend waypoint_heuristic \
    --split "$SPLIT" --device cuda --batch_size 16 --num_workers 2 \
    --out_json "$OUT/huge_waypoint_heuristic_${SPLIT}.json"
done

BC_CKPT="${BC_CKPT:-}"
if [[ -z "$BC_CKPT" ]]; then
  BC_CKPT=$(ls -t "$DRONE"/logs/huge_bench/*/model_*.pt 2>/dev/null | head -1 || true)
fi
if [[ -n "$BC_CKPT" && -f "$BC_CKPT" ]]; then
  echo "[1/4] HUGE-Bench — trained BC ($BC_CKPT)"
  for SPLIT in test_seen test_unseen; do
    python -m benchmarks.eval_huge \
      --backend bc_checkpoint --checkpoint "$BC_CKPT" \
      --split "$SPLIT" --device cuda --batch_size 16 --num_workers 2 \
      --out_json "$OUT/huge_bc_${SPLIT}.json"
  done
else
  echo "[1/4] No HUGE BC checkpoint — run: MAX_STEPS=20000 bash huge_bench/run_train_bc.sh"
  echo "      Then re-run with BC_CKPT=logs/huge_bench/<run>/model_20000.pt"
fi

# --- 2) CityNav (oracle waypoint geometry) -----------------------------
echo ""
echo "[2/4] CityNav"
CITYNAV_ROOT="${CITYNAV_ROOT:-$BENCH_EXT/citynav}"
if [[ -d "$CITYNAV_ROOT/vlnce" && -d "$CITYNAV_ROOT/data" ]]; then
  python -m benchmarks.eval_citynav_oracle \
    --citynav_root "$CITYNAV_ROOT" \
    --split val_seen --max_episodes 100 2>&1 | tee "$OUT/citynav_val_seen.txt"
  python -m benchmarks.eval_citynav_oracle \
    --citynav_root "$CITYNAV_ROOT" \
    --split val_unseen --max_episodes 100 2>&1 | tee "$OUT/citynav_val_unseen.txt"
else
  echo "SKIP: CityNav data not at $CITYNAV_ROOT (see benchmarks/setup_external.sh)"
  echo "status=skipped missing_data" > "$OUT/citynav_status.txt"
fi

# --- 3) AirNav -----------------------------------------------------------
echo ""
echo "[3/4] AirNav"
AIRNAV_ROOT="${AIRNAV_ROOT:-$BENCH_EXT/AirNav}"
if [[ -f "$AIRNAV_ROOT/data/AirNav/val/airnav_val_seen.json" ]]; then
  echo "AirNav full eval requires NavGym + light_model_eval.py — not yet wired."
  echo "status=skipped needs_adapter" > "$OUT/airnav_status.txt"
else
  echo "SKIP: AirNav data not at $AIRNAV_ROOT/data/"
  echo "status=skipped missing_data" > "$OUT/airnav_status.txt"
fi

# --- 4) OpenFly ----------------------------------------------------------
echo ""
echo "[4/4] OpenFly — not open-sourced"
echo "status=unavailable" > "$OUT/openfly_status.txt"

# --- Summary -------------------------------------------------------------
python -m benchmarks.summarize --out_dir "$OUT" 2>/dev/null || true
echo ""
echo "Done. See $OUT/"
