#!/usr/bin/env bash
# Run Stages 5 -> 6 -> 7 -> 8 of the benchmark-aligned curriculum.
#
# Each stage gates on the previous stage's checkpoint, so re-running this
# script after a partial run will pick up where it left off (it does NOT
# re-run a stage that already produced output, unless FORCE=1 is set).
#
# Usage:
#     bash scripts/train_curriculum_benchmark.sh
#     bash scripts/train_curriculum_benchmark.sh --skip-stage7   # if HUGE sim missing
#     STAGE5_ITERS=200 STAGE6_STEPS=500 bash scripts/train_curriculum_benchmark.sh   # smoke
#     FORCE=1 bash scripts/train_curriculum_benchmark.sh         # ignore existing ckpts
set -euo pipefail

DRONE="${DRONE:-/home/ubuntu/drone_project}"
ISAACLAB="${ISAACLAB:-/home/ubuntu/IsaacLab}"
LOGROOT="${LOGROOT:-$DRONE/logs/curriculum}"
mkdir -p "$LOGROOT"

# Stage knobs (env overridable for smoke tests).
STAGE5_ITERS="${STAGE5_ITERS:-5000}"
STAGE5_NUM_ENVS="${STAGE5_NUM_ENVS:-256}"
STAGE6_STEPS="${STAGE6_STEPS:-5000}"
STAGE7_ITERS="${STAGE7_ITERS:-3000}"
STAGE7_NUM_ENVS="${STAGE7_NUM_ENVS:-64}"
STAGE7_SCENE_ID="${STAGE7_SCENE_ID:-1_office}"

SKIP_STAGE7=0
for arg in "$@"; do
    case "$arg" in
        --skip-stage7) SKIP_STAGE7=1 ;;
        *) echo "[curriculum] Unknown arg: $arg"; exit 1 ;;
    esac
done

source /home/ubuntu/miniconda3/bin/activate isaac
export OMNI_KIT_ACCEPT_EULA=yes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

latest_ckpt() {
    # $1: glob; print the most recent .pt path or empty if none.
    local glob="$1"
    ls -t $glob 2>/dev/null | head -1 || true
}

# ------------------------------------------------------------------
# Stage 5 — Core VLA RL on the empty arena
# ------------------------------------------------------------------
echo ""
echo "=== Stage 5: VLA core RL ==="
STAGE5_LOG_DIR="$DRONE/logs/rsl_rl/vla_drone_direct"
STAGE5_CKPT="$(latest_ckpt "$STAGE5_LOG_DIR/*/model_*.pt")"
if [[ -n "$STAGE5_CKPT" && -z "${FORCE:-}" ]]; then
    echo "[Stage5] reusing existing $STAGE5_CKPT (set FORCE=1 to re-run)"
else
    cd "$ISAACLAB"
    ./isaaclab.sh -p "$DRONE/vla/train.py" \
        --num_envs "$STAGE5_NUM_ENVS" --max_iterations "$STAGE5_ITERS" \
        --headless --enable_cameras --device cuda:0 \
        2>&1 | tee "$LOGROOT/stage5.log"
    STAGE5_CKPT="$(latest_ckpt "$STAGE5_LOG_DIR/*/model_*.pt")"
fi
if [[ -z "$STAGE5_CKPT" ]]; then
    echo "[Stage5] FAILED — no checkpoint produced; aborting." >&2
    exit 2
fi
echo "[Stage5] OK -> $STAGE5_CKPT"

# ------------------------------------------------------------------
# Stage 6 — HUGE high-level offline SFT
# ------------------------------------------------------------------
echo ""
echo "=== Stage 6: HUGE high-level offline SFT ==="
STAGE6_LOG_DIR="$DRONE/logs/huge_bench_highlevel"
STAGE6_CKPT="$(latest_ckpt "$STAGE6_LOG_DIR/*/model_*.pt")"
if [[ -n "$STAGE6_CKPT" && -z "${FORCE:-}" ]]; then
    echo "[Stage6] reusing existing $STAGE6_CKPT (set FORCE=1 to re-run)"
else
    cd "$DRONE"
    python -m huge_bench.train_vla_highlevel \
        --max_steps "$STAGE6_STEPS" \
        --resume_path "$STAGE5_CKPT" \
        2>&1 | tee "$LOGROOT/stage6.log"
    STAGE6_CKPT="$(latest_ckpt "$STAGE6_LOG_DIR/*/model_*.pt")"
fi
if [[ -z "$STAGE6_CKPT" ]]; then
    echo "[Stage6] FAILED — no checkpoint produced; aborting." >&2
    exit 3
fi
echo "[Stage6] OK -> $STAGE6_CKPT"

# ------------------------------------------------------------------
# Stage 7 — HUGE Isaac RL (scaffold; warehouse smoke until HUGE sim drops)
# ------------------------------------------------------------------
echo ""
echo "=== Stage 7: HUGE Isaac RL (scaffold) ==="
if [[ "$SKIP_STAGE7" == "1" ]]; then
    echo "[Stage7] skipped (--skip-stage7)"
    STAGE7_CKPT="$STAGE6_CKPT"
else
    STAGE7_LOG_DIR="$DRONE/logs/rsl_rl/vla_drone_huge"
    STAGE7_CKPT="$(latest_ckpt "$STAGE7_LOG_DIR/*/model_*.pt")"
    if [[ -n "$STAGE7_CKPT" && -z "${FORCE:-}" ]]; then
        echo "[Stage7] reusing existing $STAGE7_CKPT (set FORCE=1 to re-run)"
    else
        echo "[Stage7] HUGE Isaac sim is not publicly released; using warehouse_full as placeholder."
        echo "[Stage7] When HUGE drops sim, edit vla_huge/scenes.py to point at the real USD."
        cd "$ISAACLAB"
        ./isaaclab.sh -p "$DRONE/vla_huge/train.py" \
            --num_envs "$STAGE7_NUM_ENVS" --max_iterations "$STAGE7_ITERS" \
            --headless --enable_cameras --device cuda:0 \
            --scene_id "$STAGE7_SCENE_ID" \
            --resume_path "$STAGE6_CKPT" \
            2>&1 | tee "$LOGROOT/stage7.log"
        STAGE7_CKPT="$(latest_ckpt "$STAGE7_LOG_DIR/*/model_*.pt")"
        if [[ -z "$STAGE7_CKPT" ]]; then
            echo "[Stage7] WARNING — no checkpoint produced; falling back to Stage 6 ckpt for eval."
            STAGE7_CKPT="$STAGE6_CKPT"
        fi
    fi
    echo "[Stage7] OK -> $STAGE7_CKPT"
fi

# ------------------------------------------------------------------
# Stage 8 — Unified benchmark eval
# ------------------------------------------------------------------
echo ""
echo "=== Stage 8: unified benchmark eval ==="
cd "$DRONE"
VLA_CKPT="$STAGE7_CKPT" bash benchmarks/run_all.sh 2>&1 | tee "$LOGROOT/stage8.log"
echo ""
echo "Curriculum complete. Logs in $LOGROOT, benchmark JSONs in $DRONE/logs/benchmarks/"
