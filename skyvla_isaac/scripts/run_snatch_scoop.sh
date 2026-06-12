#!/bin/bash
# REAL-PHYSICS pick-and-place: floored-scoop cage, NO kinematic latch. Warm-started from
# the HOVER-mastery policy (nav model_250: cruise+settle+hunker, NO latch-era grasp habits
# to unlearn -- the m28800 warm start carried the early-snatch reflex; archived in
# drone_snatch_scoop_from28800/). Restart loop per repo convention (Xid 43).
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/SkyVLA
cd /home/ubuntu/SkyVLA

DIR=/home/ubuntu/drone_project/logs/isaac/drone_snatch_scoop
INIT=/home/ubuntu/drone_project/logs/isaac/drone_snatch_nav/model_250.pt
while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RESUME=${LATEST:-$INIT}
    # SURROUND-FIRST phase (user-directed): pure cage-around-cube precision, no grasp
    # rewards, all gripper use taxed (cage stays open). Next phase re-enables grip
    # (+ cur_p demos) from the surround expert this produces.
    python skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
        --surround_only \
        --cube_mass 0.05 --side_spawn 5.0 --no_latch \
        --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
        --entropy_coef 0.002 --latch_ready_coef 4.0 \
        --resume "$RESUME" --reset_std 0.4 \
        --num_envs 2048 --max_iterations 40000 --log_dir "$DIR"
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s"
    sleep 10
done
