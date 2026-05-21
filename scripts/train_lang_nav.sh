#!/bin/bash
# =============================================================================
# Stage 3: CLIP Language-Grounded Navigation
# Run this on the GH200 after setup.sh completes.
#
# Prerequisites:
#   1. setup.sh completed successfully
#   2. Stage 2 waypoint checkpoint at ~/drone_project/checkpoints/stage2_waypoint.pt
#      (committed via Git LFS; if missing, copy your trained model into place)
#
# What this does:
#   1. Fixes the flatdict build issue (setuptools version conflict on ARM64)
#   2. Installs isaaclab core modules
#   3. Transfers Stage 2 waypoint weights into the Stage 3 lang_nav architecture
#   4. Trains lang_nav with CLIP text + image embeddings on 1024 parallel envs
#
# The drone learns to navigate to objects based on natural language commands
# like "fly to the red cube" or "go to the blue sphere" using dual CLIP
# embeddings (text command + onboard camera image).
#
# Expected training time: ~4-5 hours on H100, ~22 hours on A10
# Expected output: checkpoint at logs/rsl_rl/lang_drone_direct/<timestamp>/
# =============================================================================

set -e

echo "============================================"
echo "  Stage 3: CLIP Language Navigation Training"
echo "============================================"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate isaac

# -------------------------------------------------------------------
# 1. Fix flatdict build issue (ARM64 setuptools conflict)
# -------------------------------------------------------------------
echo "[1/4] Fixing dependencies..."
pip install "setuptools<81" --force-reinstall -q
pip install flatdict==4.0.1 -q 2>/dev/null || echo "  flatdict install failed (non-critical, continuing)"
pip install "setuptools>=82" -q

cd ~/IsaacLab
pip install -e source/isaaclab -q 2>/dev/null || pip install -e source/isaaclab --no-deps -q
pip install -e source/isaaclab_assets -q
pip install -e source/isaaclab_rl -q

./isaaclab.sh -p -c "import isaaclab; print('[1/4] isaaclab: OK')"

# -------------------------------------------------------------------
# 2. Check for Stage 2 waypoint checkpoint
# -------------------------------------------------------------------
echo "[2/4] Checking for Stage 2 waypoint checkpoint..."
CKPT="$HOME/drone_project/checkpoints/stage2_waypoint.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Stage 2 waypoint checkpoint not found at $CKPT"
    echo "If you have a freshly trained checkpoint, copy it into place:"
    echo "  cp logs/rsl_rl/waypoint_nav/<timestamp>/model_<iter>.pt $CKPT"
    echo "Or run 'git lfs pull' to fetch the committed pretrained weights."
    exit 1
fi
echo "[2/4] Checkpoint found: $CKPT"

# -------------------------------------------------------------------
# 3. Transfer Stage 2 weights into the Stage 3 lang_nav architecture
# -------------------------------------------------------------------
echo "[3/4] Transferring weights..."
cd ~/drone_project
mkdir -p logs/rsl_rl/lang_drone_direct

python scripts/transfer_waypoint_to_vla.py \
    --waypoint_checkpoint "$CKPT" \
    --output_path logs/rsl_rl/lang_drone_direct/lang_nav_init.pt

echo "[3/4] Weight transfer complete."

# -------------------------------------------------------------------
# 4. Train Stage 3: CLIP language navigation
# -------------------------------------------------------------------
echo "[4/4] Starting Stage 3 training..."
echo "  Envs: 1024 | Iterations: 3000 | Camera: enabled"
echo "  This will take ~4-5 hours on H100"
echo ""

cd ~/IsaacLab
./isaaclab.sh -p ~/drone_project/lang_nav/train.py \
    --num_envs 1024 \
    --max_iterations 3000 \
    --headless \
    --enable_cameras \
    --resume_path ~/drone_project/logs/rsl_rl/lang_drone_direct/lang_nav_init.pt

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo "Checkpoints saved to: ~/drone_project/logs/rsl_rl/lang_drone_direct/"
