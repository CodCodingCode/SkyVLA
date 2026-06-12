#!/bin/bash
# GRASP + LIFT phase (real physics, WIDE floored-scoop cage, no latch). Warm-started from
# the overhead-curriculum policy (model_900: lands the cage on the cube, sub-cm centred,
# user-verified on film). Reward: enclosure-gated exponential squeeze + 60x lift + full
# carry/deliver stack. cur_p 0.15 straddle demos teach squeeze value; reset_std 0.5
# restores grip-action exploration (taxed-shut all hover phases -> the close must be
# SAMPLED before it can be learned; wide 2.4cm window tolerates the extra hover jitter).
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/SkyVLA
cd /home/ubuntu/SkyVLA

DIR=/home/ubuntu/drone_project/logs/isaac/drone_snatch_grasp
INIT=/home/ubuntu/drone_project/logs/isaac/drone_snatch_overhead/model_900.pt
while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RESUME=${LATEST:-$INIT}
    python skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.15 \
        --no_latch --carry_demo 0.25 \
        --cube_mass 0.05 --side_spawn 5.0 \
        --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
        --entropy_coef 0.001 --latch_ready_coef 4.0 \
        --resume "$RESUME" --reset_std 0.2 \
        --num_envs 2048 --max_iterations 40000 --log_dir "$DIR"
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s"
    sleep 10
done
