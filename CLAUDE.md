# SkyVLA — agent conventions

Rules for me (the agent) when working in this repo. Short and rule-focused. The
repo is the Isaac Sim / Isaac Lab gripper-drone work in `skyvla_isaac/` — a
free-floating quadrotor with a 4-jaw gripper doing pick-and-place (`tasks/pick_place_env.py`)
and Gaussian-map navigation (`gs/`). Training is PPO via rsl_rl (`scripts/train.py`).

## Environment

All Isaac scripts run in the isolated `isaac` conda env (Python 3.10) and need:

```bash
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/ubuntu/SkyVLA
```

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
  "conda activate isaac && OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/ubuntu/SkyVLA \
   python skyvla_isaac/scripts/train.py --num_envs 2048 --max_iterations 1500 2>&1 | tee $LOG"
```

`scripts/train.py` checkpoints every `save_interval` iterations to
`logs/isaac/drone_pick_place/`, so a crash loses at most that window — resume by
relaunching from the latest `model_*.pt`.

## Xid 43 on this machine

The A100 on this host throws NVIDIA Xid 43 ("GPU stopped processing" / channel
reset) errors at roughly ~50% per hour of sustained training. They surface in
Python as `Fatal Python error: Segmentation fault` with no stack and no caught
exception, across PyTorch versions and architectures — no ECC errors, no
hardware fault. Suspected cause: a PyTorch cu130 binary on a system with CUDA
12.8 nvcc. **Design for it:** checkpoint often, and wrap long launches in a
restart loop so a mid-run segfault just resumes from the latest checkpoint
rather than losing the run.

**Don't rerun the cu128 reinstall without authorization.** We diagnosed Xid 43
and considered reinstalling PyTorch with cu128 wheels to match the system CUDA
toolkit; the user explicitly chose **not** to (preferred checkpoint + restart).
Don't propose the reinstall again unless the restart strategy starts failing in
a new way — the reinstall has a real blast radius (Isaac Lab / rsl_rl pinning).

## Disk hygiene

`/dev/vda1` is shared with `/tmp` and runs 90%+ full on this machine. PPO
checkpoints and rendered mp4s add up. **Before a long run**, check `df -h /tmp`
has headroom; delete stale checkpoints/videos first if not. A full root disk
also breaks the Claude Code harness (task-output dir can't be written), so this
matters more than usual.

## Don't break what works

The converged pick-place config is the result of careful reward balancing
(strong-but-capped lift gradient, dominant held-only placement reward, start-pose
curriculum). When changing `pick_place_env._get_rewards`, keep that structure —
getting lift vs. placement weights wrong collapses the whole task. When changing
the articulation or observation layout, keep old checkpoints loadable
(`load_state_dict(strict=False)` + shape filter) rather than silently breaking
every saved `model_*.pt`.
