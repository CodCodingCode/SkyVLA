#!/bin/bash
# OVERHEAD-FIRST curriculum (user-designed, 2026-06-11): from the NAV hover policy,
# (1) pay only for being DIRECTLY ABOVE the block (horiz -> 0) at an altitude setpoint
# starting 20cm up; (2) lower the setpoint rung-by-rung as centring is proven, down to
# nest depth. Real physics (floored scoop, no latch), gripper taxed (no grasp this phase).
# Restart loop per repo convention (Xid 43): resume latest, else the nav warm start.
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/SkyVLA
cd /home/ubuntu/SkyVLA

DIR=/home/ubuntu/drone_project/logs/isaac/drone_snatch_overhead
INIT=/home/ubuntu/drone_project/logs/isaac/drone_snatch_nav/model_250.pt
while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RESUME=${LATEST:-$INIT}
    python skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
        --overhead_first --no_latch \
        --cube_mass 0.05 --side_spawn 5.0 \
        --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
        --entropy_coef 0.002 --latch_ready_coef 4.0 \
        --resume "$RESUME" --reset_std 0.25 \
        --num_envs 2048 --max_iterations 40000 --log_dir "$DIR"
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s"
    sleep 10
done
