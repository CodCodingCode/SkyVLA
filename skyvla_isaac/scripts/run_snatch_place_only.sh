#!/bin/bash
# PLACE-ONLY -- gentle deposit onto pad B from an ALREADY-HOLDING start.
#
# Unlike run_snatch_place.sh (stage 3 of the A->B transport chain, which needs a carry
# expert that can already arrive over B), this trains the deposit in ISOLATION: every
# episode begins with the cube seated in the closed cage, airborne within 0.8 m of pad B.
# The only skill learned is centre -> lower gently -> set down -> let go.
#
# WHY IT IS A SEPARATE REWARD (see _place_reward in snatch/pick_place_env.py):
# the stage>=1/2 reward pays an exponential height ladder (~153/step at the 15cm cap) plus
# 30*carry_up, against 80*success for a deposit. Holding the cube HIGH outearns putting it
# down and every step of the descent costs income -- which is exactly why the stage-2 carry
# run sat at arrive_rate 0.003 with the cube held at ~88cm. --place_only removes ALL
# altitude income: the only way to earn is to set the cube down on B and open the jaws.
#
# GEOMETRY FLOOR ON "GENTLE" (drone_snatch.urdf): the cage lips span tip-0.025..tip-0.017,
# so a shelf-seated cube's bottom is always 8mm above the cage's lowest point. The lips
# bottom out on the pad -> the cube is ALWAYS released from 8mm -> ~0.40 m/s contact is the
# geometric floor, not a training failure. What IS trained: the descent tapers to
# --gentle_v in the last 10cm, release happens from the lowest reachable pose with the
# drone's vz ~0, and any post-release cube motion is taxed.
#
# WARM START: the gentle-lift grasp expert model_9250. It already knows how to fly, hold
# the cube in the cage, and modulate the jaws -- it just has a hard-won habit of climbing,
# which term 4 of the place reward taxes directly. reset_std 0.35 gives it enough
# exploration to find the descent (the saved std is ~17 -> pure noise if not reset).
#
# ENV: no conda on this host; Isaac Sim 5.1.0 / Isaac Lab 2.3.2 / rsl_rl 3.1.2 in .venv311.
# LD_PRELOAD of libgomp is required on this aarch64 host or the interpreter aborts at import.
PY=${PY:-/home/ubuntu/SkyVLA/.venv311/bin/python}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/SkyVLA
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"
cd /home/ubuntu/SkyVLA

DIR=${DIR:-/home/ubuntu/SkyVLA/logs/isaac/drone_snatch_place_only}
INIT=${INIT:-/home/ubuntu/SkyVLA/skyvla_isaac/snatch/checkpoints/model_9250.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-20000}
mkdir -p "$DIR"

LOG=$DIR/train.log

while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)
    RESUME=${LATEST:-$INIT}
    # resume the start-ramp where it left off, otherwise a teardown segfault costs ~470
    # iterations of re-earned rungs. --reset_std only on the FIRST launch: after that the
    # policy's own std is the tuned one and stomping it would discard the progress.
    CURP=$(grep -o "difficulty=[0-9.]*" "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "$LATEST" ]; then STD=(); else STD=(--reset_std 0.30); fi
    "$PY" skyvla_isaac/scripts/train_snatch.py --no_cams --place_only \
        --plat_sep 1.5 --plat_sep_max 1.5 \
        --place_spawn_r 0.8 --place_spawn_h 0.55 0.85 \
        --gentle_v 0.15 --speed 1.5 --cube_mass 0.05 \
        --entropy_coef 0.002 --place_curr_start "${CURP:-0.0}" \
        --resume "$RESUME" "${STD[@]}" \
        --num_envs "$ENVS" --max_iterations "$ITERS" --log_dir "$DIR" 2>&1 | tee -a "$LOG"
    echo "[restart-loop] train exited ($?), resuming from latest checkpoint in 10s" | tee -a "$LOG"
    sleep 10
done
