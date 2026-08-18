#!/bin/bash
# STAGE 2 of 3 -- NAV / TRANSPORT: carry the grasped cube from platform A to platform B.
# Warm-started from the committed gentle-lift grasp expert (model_9250), which already
# flies in, cages and lifts; this run only has to add "and now haul it to B".
#
# Success here = ARRIVED over B with the cube still held (--release_only is stage 3).
#
# PURE TRANSPORT. --grip_closed welds the cage shut (a[4] forced to +1 every step), so this
# policy controls only [vx vy vz yaw_rate] and never has to maintain its own grasp. That
# forces --carry_demo 1.0: with the jaws shut a drone spawned away from the cube can never
# scoop it, so EVERY env starts already holding, at A, and the only task is the haul to B.
# Rationale: with the grasp in the loop, arrive_rate plateaued at ~4% for 380 iterations --
# most failures were the cube slipping mid-flight on a 49%-reliable friction grasp, not bad
# flying. Removing grasp maintenance isolates the transport skill. The real grasp is the
# snatch policy's job, and run_pipeline.py hands over to this one already holding.
#
# DISTANCE LADDER. env_spacing and episode_length are sized off --plat_sep_max at scene
# construction, so you cannot raise the ceiling mid-run: finish a rung, then relaunch the
# next with SEP/SEP_MAX raised and RESUME pointed at this run's latest checkpoint.
#   rung 1: 1.5 ->  4 m   (speed 1.5, the warm start's own regime)
#   rung 2:   4 -> 10 m   (speed 2.5)
#   rung 3:  10 -> 20 m   (speed 4.0)
#   rung 4:  20 -> 40 m   (speed 6.0)
# Raise --speed only ONE rung at a time: it rescales what action=1.0 means, so a big jump
# invalidates the warm start's learned action magnitudes.
# ENV: this host has NO `isaac` conda env (the other run_snatch_*.sh still source one and
# will fail here). Isaac Sim 5.1.0 / Isaac Lab 2.3.2 / rsl_rl 3.1.2 live in .venv311.
# LD_PRELOAD of libgomp is required on this aarch64 host or the interpreter aborts at import.
PY=${PY:-/home/ubuntu/SkyVLA/.venv311/bin/python}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/SkyVLA
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"
cd /home/ubuntu/SkyVLA

DIR=${DIR:-/home/ubuntu/SkyVLA/logs/isaac/drone_snatch_carry}
INIT=${INIT:-/home/ubuntu/SkyVLA/skyvla_isaac/snatch/checkpoints/model_9250.pt}
SEP=${SEP:-1.5}
SEP_MAX=${SEP_MAX:-4.0}
SPEED=${SPEED:-1.5}
mkdir -p "$DIR"

while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RESUME=${LATEST:-$INIT}
    "$PY" skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
        --two_platform --plat_sep "$SEP" --plat_sep_max "$SEP_MAX" \
        --grip_closed --no_latch --carry_demo 1.0 \
        --cube_mass 0.05 --side_spawn 5.0 --speed "$SPEED" \
        --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
        --entropy_coef 0.001 --latch_ready_coef 4.0 \
        --resume "$RESUME" --reset_std 0.2 \
        --num_envs 2048 --max_iterations 40000 --log_dir "$DIR"
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s"
    sleep 10
done
