#!/bin/bash
# Staged curriculum with randomized SIDE spawns: navigate -> hover -> descend -> latch -> carry.
# Restart loop per repo convention (Xid 43 segfaults): resume from the latest checkpoint.
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/ubuntu/SkyVLA
cd /home/ubuntu/SkyVLA

DIR=/home/ubuntu/drone_project/logs/isaac/drone_snatch_nav
while true; do
    CKPT=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RES=""
    # reset_std also caps runaway exploration noise after every crash-resume
    [ -n "$CKPT" ] && RES="--resume $CKPT --reset_std 0.7"
    python skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
        --cube_mass 0.05 --side_spawn 5.0 --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 \
        --entropy_coef 0.001 --latch_ready_coef 4.0 \
        --num_envs 2048 --max_iterations 20000 --log_dir "$DIR" $RES
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s"
    sleep 10
done
