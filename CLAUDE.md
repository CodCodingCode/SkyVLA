# SkyVLA — agent conventions

Rules for me (the agent) when working in this repo. Short and rule-focused. The
repo is the Isaac Sim / Isaac Lab gripper-drone work in `skyvla_isaac/` — a
free-floating quadrotor with a 4-jaw gripper doing pick-and-place (`tasks/pick_place_env.py`)
and Gaussian-map navigation (`gs/`). Training is PPO via rsl_rl (`scripts/train.py`).

## Environment — venvs, NOT conda

There is no conda on this host and no `isaac` / `habitat` env. Two virtualenvs:

| | path | has |
|---|---|---|
| Isaac (physics, training, eval) | `.venv311` (py3.11) | Isaac Sim 5.1.0, Isaac Lab 2.3.2, rsl_rl 3.1.2, torch 2.7.0+cu128 |
| Gaussian-map rendering | `.venv` (py3.10) | gsplat 1.5.3 |

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/ubuntu/SkyVLA
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1   # REQUIRED on this aarch64 host
/home/ubuntu/SkyVLA/.venv311/bin/python skyvla_isaac/scripts/<script>.py
```

**Never write `conda activate` into a script.** Four run scripts did; they resolved to
the system `python`, died on `ModuleNotFoundError: isaaclab`, and their restart loops
retried forever — looking alive in tmux while training nothing. They were deleted; their
hyperparameters live in the README's grasp-lineage section.

**No interpreter has both `isaaclab` and `gsplat`.** `gs_isaac_demo.py` and
`render_rollout_gs.py` import both and cannot run until `gsplat` is installed into
`.venv311`. `render_gs_cache.py` is the Isaac-free consumer and runs under `.venv`.

## W&B is on by default — don't disable it

`scripts/train.py` sets `logger="wandb"`, project `skyvla-isaac`. The API key
lives at `/home/ubuntu/SkyVLA/.wandb_key` (mode 600, gitignored). **The
`.wandb_key` file must never be committed** — `.gitignore` already covers
`.wandb_key` and `.env.local`. If a new credential file is needed, add it to
`.gitignore` first.

**Log only measurable training progress to W&B, nothing else** — the dashboard
exists to answer one question: "is training getting better?" For this task the
metrics that matter are the real success rates (`grasp_rate`, `lift_rate`,
`place_success`, `obj_to_goal`) plus mean episode reward. No per-step jitter, no
constants (e.g. `lr` after warmup), no operational counters. If a metric wouldn't
change what I'd do next on a chart, it goes to stdout, not W&B. Don't disable
W&B on long runs — we want the dashboard.

## After launching ANY training or eval run — ALWAYS give tail commands

**Hard rule, no exceptions.** Whenever I launch a training run, eval, or any
background process that writes to a log file, I immediately follow the launch
confirmation with the user-runnable tail commands, in a small code block right
after the "tmux session / log" lines (not buried later). Adapt the grep filters
to whatever the run prints.

```bash
tail -f <LOG>                                   # raw progress
tail -f <LOG> | grep -E "Iteration|reward|success"   # iteration summaries + rates
tmux attach -t <SESS>                           # live view (Ctrl-B then D to detach)
tmux ls                                         # find running sessions
```

## Long-running training runs

**Always launch long training runs (>15 min) inside a tmux session.** Never use
bare `nohup ... &` for PPO training — tmux is interactively attachable, has
cleaner process management, and the user explicitly asked for it so closing
their laptop never matters. Stop a run with `tmux kill-session -t <SESS>`
(preferred over `pkill -9`).

```bash
SESS=isaac_pickplace_$(date +%Y%m%d_%H%M%S)
LOG=/tmp/${SESS}.log
tmux new-session -d -s "$SESS" \
  "OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/ubuntu/SkyVLA \
   LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
   /home/ubuntu/SkyVLA/.venv311/bin/python skyvla_isaac/scripts/train.py \
   --num_envs 2048 --max_iterations 1500 2>&1 | tee $LOG"
```

The `run_snatch_carry.sh` / `run_snatch_place.sh` scripts already set all of this up
(`PY=`, `LD_PRELOAD`, restart loop); launch those directly rather than rebuilding the
command. Before trusting a restart loop, confirm the log shows real iterations — a loop
that reprints `[restart-loop]` every 10s is failing at import, not crashing mid-run.

`scripts/train.py` checkpoints every `save_interval` iterations to
`logs/isaac/drone_pick_place/`, so a crash loses at most that window — resume by
relaunching from the latest `model_*.pt`.

## Segfaults on this machine

This host is a **GH200 (aarch64)** running torch 2.7.0+**cu128**. The older note here
described an A100 with a cu130/CUDA-12.8 mismatch causing Xid 43 at ~50%/hour; that
hardware and that mismatch are both gone.

What is observed now is a **teardown segfault**: training completes normally, prints
`TRAIN_SNATCH_OK`, and *then* dies in `simulation_app.close()` with
`carb.graphics-vulkan.plugin: VkResult: ERROR_INCOMPATIBLE_DRIVER` (headless host, no
Vulkan). Intermittent — same command exits 0 on some runs, 139 on others. It is
harmless: it happens after checkpoints are written.

**Design for it anyway:** checkpoint often and wrap long launches in a restart loop.
But when a loop is spinning, check *where* it fails — a teardown segfault after real
iterations is benign; a loop that never logs an iteration is an import/env failure and
will never make progress.

## GS rendering is CACHED — never rebuild the splat room from scratch

The Gaussian-splat backdrop in `scripts/render_rollout_gs.py` is **static
geometry** (no policy/checkpoint dependence), so it is cached and reused — never
re-run the slow ~44-view RGB-D room orbit when a cache exists. Full reference:
[`skyvla_isaac/gs/CACHING.md`](skyvla_isaac/gs/CACHING.md).

Two caches under `skyvla_isaac/gs/cache/` (gitignored), via `gs/cache.py`:
- **scene** (`room_splat.pt`) — the splat + `K` + render W×H. Built once with
  Isaac; reused every run. `render_rollout_gs.py` loads it automatically and
  skips the orbit. Force a rebuild only with `--rebuild_gs` (changed room/res).
- **rollout** (`rollout_<ckpt>.npz`) — per-frame foreground + follow-cam pose for
  one checkpoint. Written by `render_rollout_gs.py --save_rollout`.

**To iterate on the render/camera look, do NOT boot Isaac.** Use the Isaac-free
consumer in `.venv` (~1.5 ms/frame):
```bash
PYTHONUTF8=1 PYTHONPATH=/home/ubuntu/SkyVLA \
.venv/bin/python skyvla_isaac/scripts/render_gs_cache.py \
    --rollout skyvla_isaac/gs/cache/rollout_<ckpt>.npz --out videos/replay.mp4
# or orbit the bare room:  render_gs_cache.py --out videos/gs_orbit.mp4
```
`PYTHONUTF8=1` is required (gsplat JITs `.cu` kernels via the locale codec). The
rollout cache is per-checkpoint; the scene cache is reused across all of them.
Clean `cache/rollout_*.npz` when done — disk is tight (see below).

## Disk hygiene

`/dev/vda1` is shared with `/tmp`. It is currently **3.9 TB at 2% used** — the old
"90%+ full" warning no longer applies, but PPO checkpoints and rendered mp4s still
add up. **Before a long run**, check `df -h /tmp` has headroom. A full root disk also
breaks the Claude Code harness (task-output dir can't be written).

## Don't break what works

The converged pick-place config is the result of careful reward balancing
(strong-but-capped lift gradient, dominant held-only placement reward, start-pose
curriculum). When changing `pick_place_env._get_rewards`, keep that structure —
getting lift vs. placement weights wrong collapses the whole task. When changing
the articulation or observation layout, keep old checkpoints loadable
(`load_state_dict(strict=False)` + shape filter) rather than silently breaking
every saved `model_*.pt`.
