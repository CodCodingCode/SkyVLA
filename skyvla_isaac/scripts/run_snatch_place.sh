#!/bin/bash
# STAGE 3 of 3 -- PLACE / DEPOSIT: lower the carried cube onto platform B and LET GO.
# Warm-started from the carry expert (stage 2), which already arrives over B holding the
# cube; this run only has to add the descent + release + "don't knock it off".
#
# --release_only flips what success MEANS: no longer "arrived still holding it" but
# "cube at rest on B's top, jaws open, drone not holding it" (see _get_dones). This is
# the first time in this repo's history that success does not require _held -- every
# previous 'place' metric, including the README's 86.1%, was a still-gripped hover.
#
# Kept SHORT-RANGE on purpose: the deposit is a precision skill, so learn it at 1.5-2 m
# where transit is nearly free, then re-run stage 2's distance ladder with this policy if
# you want deposit-at-distance. Nothing here depends on the separation.
# ENV: see run_snatch_carry.sh -- no conda on this host; Isaac lives in .venv311.
PY=${PY:-/home/ubuntu/SkyVLA/.venv311/bin/python}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/SkyVLA
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"
cd /home/ubuntu/SkyVLA

DIR=${DIR:-/home/ubuntu/SkyVLA/logs/isaac/drone_snatch_place}
INIT=${INIT:-/home/ubuntu/SkyVLA/logs/isaac/drone_snatch_carry/model_latest.pt}
SEP=${SEP:-1.5}
SEP_MAX=${SEP_MAX:-2.0}
mkdir -p "$DIR"

while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RESUME=${LATEST:-$INIT}
    "$PY" skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
        --two_platform --release_only --plat_sep "$SEP" --plat_sep_max "$SEP_MAX" \
        --no_latch --carry_demo 0.5 \
        --cube_mass 0.05 --side_spawn 5.0 --speed 1.5 \
        --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
        --entropy_coef 0.002 --latch_ready_coef 4.0 \
        --resume "$RESUME" --reset_std 0.35 \
        --num_envs 2048 --max_iterations 40000 --log_dir "$DIR"
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s"
    sleep 10
done
