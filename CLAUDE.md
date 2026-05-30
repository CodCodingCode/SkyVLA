# SkyVLA — agent conventions

Rules for me (the agent) when working in this repo. Short and rule-focused.

## W&B is on by default — don't disable it

The DiT trainer (`openfly/train_subgoal_dit.py`) auto-initializes a W&B run on every launch. The API key lives at `/home/ubuntu/SkyVLA/.wandb_key` (mode 600, gitignored); the shell wrapper exports it as `WANDB_API_KEY` before invoking Python, and the trainer falls back to reading the same file if the env var is missing.

Defaults:
- project: `skyvla-subgoal-dit`
- run name / id: `basename(out_dir)` — same run dir → same W&B run (auto-resumes via `resume="allow"`), so crash-loop relaunches stitch into one continuous series.
- mode: `online`

**Log only measurable training progress to W&B, nothing else.** No per-step jitter, no operational counters, no constants. The dashboard exists to answer one question: "is training getting better?" Every metric you add must move the needle on that question. Specifically:

DO log:
- **`val/cos_seen`, `val/cos_ood`** — THE metric. The whole reason we train. Per-epoch.
- **`val/cos_gap_seen_ood`** — generalization gap (seen − ood). Near zero = generalizing. Going up = overfit.
- **`val/best_cos`** — running max of `val/cos_seen`. Easy visual ceiling.
- **`val/loss_seen`, `val/loss_ood`** — secondary, but worth tracking the trend.
- **`epoch/train_cos_mean`** — denoised in-training direction quality (a leading indicator of what val_cos will be).
- **`epoch/train_loss`** — the headline objective.
- **`epoch/nan_skip_ratio`** — health check. > 5% means the run is fighting numerical instability and val numbers are fragile; investigate before drawing conclusions.
- **`train/*_ema`** (per step, EMA-smoothed) — `train/loss_ema`, `train/train_cos_ema`, `train/cos_loss_ema`, `train/repa_loss_ema`. EMA half-life ~50 steps. These are how you watch training "live" — raw per-batch numbers swing too wildly (`train_cos` jitters ±0.5 batch-to-batch).

DO NOT log:
- Raw per-step metrics (use EMA instead).
- `lr` (constant after warmup; not progress).
- `epoch_idx`, `step_in_epoch` (already encoded in gstep on the x-axis).
- `n_valid` per batch (debug noise).
- Operational counters that don't move with training quality (`nan_skip_count` cumulative is debug, `nan_skip_ratio` per epoch is signal).
- Anything with no monotonic / interpretable shape.

If you're tempted to add a new metric, ask: "would this on a chart change what I'd do next?" If the answer is no, it goes to stdout, not W&B.

To turn it off for a one-shot smoke run: `--wandb_mode disabled` (or `--wandb_project ""`). Don't disable on long runs — we want the dashboard.

The .wandb_key file must never be committed. .gitignore already covers `.wandb_key` and `.env.local`. If a new credential file is needed, add it to .gitignore first.

## After launching ANY training or eval run — ALWAYS give tail commands

**Hard rule, no exceptions.** Whenever you launch a training run, eval, ablation, or any background process that writes to a log file, immediately follow the launch confirmation with the user-runnable tail commands. The user has explicitly asked for this — they want to be able to watch progress without having to ask me where the log is.

Default set to surface (adapt grep filters to whatever metrics the run produces):

```bash
# Live training-step lines
tail -f <LOG> | grep -E "epoch [0-9]+ step"

# Clean tail (warnings filtered)
tail -f <LOG> | grep -vE "UserWarning|tensor_numpy|ascontiguous|FutureWarning"

# Epoch summaries + saves + restarts only (the moments that matter)
tail -f <LOG> | grep -E "epoch [0-9]+ →|saved best|periodic save|==== launch|EXIT reason"

# Attach to live tmux (interactive view; Ctrl-B D to detach)
tmux attach -t <SESS>
```

For eval scripts that don't emit per-step lines, default to a simpler set — just the clean tail + a one-shot "show results so far" pattern.

Phrasing: surface them in a small code block right after the "tmux session: X / log: Y" lines. Don't bury them later in the message; the user reads top-down.

## Long-running training runs

**Always launch long training runs (>15 min) inside a tmux session.** Never use bare `nohup ... &` for SFT / DiT / RL training — tmux is interactively attachable, has cleaner process management, and the user explicitly asked for it so that closing their laptop never matters.

**For any multi-hour DiT run, always use `--run_dir <pinned-path> --auto_resume --ckpt_every_steps 500`** so a Xid 43 / segfault / OOM doesn't lose the entire run. See the Xid 43 section below for why.

Pattern (resilient, with crash-loop wrapper):

```bash
RUN_DIR=/home/ubuntu/SkyVLA/logs/openfly/subgoal_dit/<descriptive_name>
SESS=<phase>_$(date +%Y%m%d_%H%M%S)
LOG=/tmp/${SESS}.log

# write a relaunch script that retries up to MAX_RESTARTS times on non-clean exit
RELAUNCH=/tmp/${SESS}_loop.sh
cat > "$RELAUNCH" <<EOF
#!/bin/bash
cd /home/ubuntu/SkyVLA
MAX_RESTARTS=8
RESTART=0
while true; do
  RESTART=\$((RESTART+1))
  echo "==== launch #\$RESTART at \$(date) ====" | tee -a $LOG
  ./openfly/run_train_<phase>.sh \\
    <your args> \\
    --ckpt_every_steps 500 \\
    --run_dir $RUN_DIR \\
    --auto_resume 2>&1 | tee -a $LOG
  if tail -200 $LOG | grep -q "EXIT reason=clean"; then
    LAST_EPOCH=\$(grep -oE "epoch [0-9]+ →" $LOG | tail -1 | grep -oE "[0-9]+")
    [ -n "\$LAST_EPOCH" ] && [ "\$LAST_EPOCH" -ge \$((EPOCHS-1)) ] && break
  fi
  [ \$RESTART -ge \$MAX_RESTARTS ] && break
  sleep 15
done
EOF
chmod +x "$RELAUNCH"
tmux new-session -d -s "$SESS" "$RELAUNCH"
```

For a one-off short smoke (<15 min) where a Xid 43 just means rerunning is fine, the simpler one-line tmux is OK:

```bash
tmux new-session -d -s "$SESS" "./openfly/run_train_<phase>.sh <args> 2>&1 | tee $LOG"
```

After launching, tell the user:
- `tmux attach -t <SESS>` to view live (Ctrl-B then D to detach without killing)
- `tail -f <LOG>` for file-based tail
- `tmux ls` to find running sessions

Stopping a run: `tmux kill-session -t <SESS>` (preferred over `pkill -9`).

## Xid 43 on this machine

The A100 on this host throws NVIDIA Xid 43 ("GPU stopped processing" / channel reset) errors at a rate of roughly ~50% per hour of sustained training. They surface in Python as `Fatal Python error: Segmentation fault` with no preceding stack and no caught exception. They happen across PyTorch versions, model architectures, and boot cycles — confirmed by multiple entries in `sudo dmesg` / `/var/log/kern.log` dating back days. No ECC errors, no hardware fault. Suspected cause: PyTorch 2.12+cu130 binary on a system with CUDA 12.8 nvcc, causing latent issues in triton JIT or cuDNN algorithm selection.

This means **any multi-hour DiT run will probably crash mid-way.** Don't fight this — design for it:

1. Pin the run directory with `--run_dir`.
2. Save state with `--ckpt_every_steps 500` (mid-epoch save bounds crash-loss to ~3 min on this hardware).
3. Auto-resume from `last.pt` with `--auto_resume`.
4. Wrap launch in a restart loop (see the tmux pattern above).

The trainer's `last.pt` carries `epoch`, `global_step`, `step_in_epoch`, optimizer, EMA — full state. Mid-epoch resume REPLAYS the current epoch from step 0 (preserves optimizer/EMA, redoes already-seen batches; the redo cost is ~10 min and is much simpler than rewinding the DataLoader). Clean epoch boundaries advance to the next epoch.

**Diagnostic output routing:** the trainer's faulthandler heartbeat (`dump_traceback_later(120, repeat=True)`) is routed to `<run_dir>/diagnostics.log`, NOT the main log. If the main log shows a thread-dump-looking block, it's the SIGTERM/SIGINT handler (one-shot at signal time) or an actual fatal — not the periodic heartbeat. Past confusion: a tail of the main log used to show repeating `Timeout (0:02:00)!` blocks every 2 min and looked like a crash; that was just the heartbeat.

## Don't rerun cu128 reinstall without authorization

We diagnosed the Xid 43 issue and considered reinstalling PyTorch with cu128 wheels to match the system CUDA toolkit. The user explicitly chose **not** to do that (preferred crash-loop + auto-resume instead). Don't propose the reinstall again unless the auto-resume strategy starts failing in a new way. The reinstall has a real blast radius (PaliGemma policy, transformers version pinning, accelerate, diffusers).

## Diagnostic logging in long runs

The trainer (`openfly/train_subgoal_dit.py`) already installs:
- `faulthandler.dump_traceback_later(120, repeat=True)` — periodic stack dumps so a hang isn't silent
- `atexit` handler printing `EXIT reason=clean|signal(N)|exception(...)`
- SIGTERM/SIGINT trap that dumps thread stacks before exiting

The shell wrapper (`openfly/run_train_subgoal_dit.sh`) sets `PYTHONUNBUFFERED=1` + `python -u` so prints flush in real time. **Don't pipe long runs through `tail -<N>` without `tee` first** — `tail` only emits its output buffer when its input closes, so a SIGTERM kills the pipe before any captured progress reaches the log.

## Val splits — never random_split

For any train/val split: load `train.json` for training, `seen.json` and/or `unseen.json` for val. **Never** `random_split(full_ds, ...)` on a single split — it leaks adjacent frames of the same trajectory across train/val and inflates metrics to look-too-good. See `docs/TRAIN.md` "Gotchas" for the full history.

## val_ddim_steps

Default is **4** in the trainer — matches the policy's inference (`PaliGemmaVLNPolicy.subgoal_sample_steps=4`). Past `val_cos≈0.61` numbers were measured at 20 steps and are deploy-inflated. Use 20 only as a one-off "denoising ceiling" diagnostic.

## Disk hygiene

Each DiT run writes `best.pt` (~2.5 GB) + `last.pt` (~10 GB). `/dev/vda1` is shared with `/tmp` and 95%+ full on this machine. **Before launching a multi-epoch run**, check `df -h /tmp` has ≥30 GB free. If not, delete old `last.pt` files first:

```bash
find /home/ubuntu/SkyVLA/logs/openfly/subgoal_dit -name "last.pt" -size +5G
# review then:
find /home/ubuntu/SkyVLA/logs/openfly/subgoal_dit -name "last.pt" -size +5G -delete
```

A full root disk also breaks the Claude Code harness (task-output dir can't be written), so this matters more than usual.

## Image data caveat — and why per-env balancing requires the new flag

**Only ~14% of `train.json` steps have local frames** in `~/assets/OpenFly/images/Image`. The dataset's `require_images=True` filter silently drops the rest. **Coverage is wildly uneven across envs** — env_ue_bigcity has 91% coverage (181k steps), env_gs_ecust has 0%, and most others sit at 1-10%. As of 2026-05-29 the breakdown is:

| env | usable steps | coverage |
|---|---|---|
| env_ue_bigcity | 181,792 | 91.4% |
| env_airsim_16 | 17,468 | 9.1% |
| env_airsim_26 | 6,913 | 4.0% |
| env_airsim_23 | 6,779 | 12.7% |
| env_gs_sjtu01 | 6,771 | 6.9% |
| env_airsim_sh | 6,410 | 2.0% |
| env_airsim_18 | 1,613 | 1.0% |
| env_airsim_gz | 1,612 | 0.9% |
| env_gs_nwpu02 | 509 | 0.4% |
| env_gs_nwpu01 | 57 | 0.1% |
| env_gs_ecust | 0 | 0.0% |

The 286k "missing" frames aren't a naming bug — the trajectory directories exist but are empty (download was incomplete). Trying to fix this by changing the path resolution in [openfly/dataset.py](openfly/dataset.py) won't help; the files genuinely aren't on disk.

**Why `--per_env_max_episodes` doesn't actually balance:** it caps EPISODES BEFORE the image-existence filter. Because bigcity has both more episodes AND higher per-episode coverage, any episode cap that's high enough to produce a usable training set is dominated by bigcity. Concretely `--per_env_max_episodes 2000` produces ~95% bigcity samples in the final dataset.

**Use `--per_env_max_index_samples N` for actual balance** — caps usable step-pairs per env AFTER image filtering. With N=10000 the dataset lands at ~50k step-pairs with bigcity at ~20%. Deterministic sampling (seed=0) so configs are reproducible. Always use this when "balanced multi-env training" is what you actually want — `--per_env_max_episodes` alone produces a heavily-skewed bigcity-mostly run.

## Training script gotchas to know about

* **Don't trust `history.json` mid-multi-launch run alone — the log has the truth.** Fixed 2026-05-29: resumes now load prior history.json so summaries from earlier launches survive. Before the fix, a Xid-43 mid-run dropped earlier epoch summaries from the file (they stayed in the stdout log though).
* The wrapper script `/openfly/run_train_subgoal_dit.sh` sets `PYTHONUNBUFFERED=1` and uses `python -u` — don't pipe long runs through `tail -N` without `tee` first, or progress prints get lost on signal.
* `args.json` is persisted to `out_dir` at startup so a mid-training crash still leaves the run config behind.

## Don't break what works

When making architectural changes to a model that has saved checkpoints, the `load_state_dict(strict=False)` + shape-filter pattern (see `openfly/policies.py:PaliGemmaOpenFlyPolicy.__init__`) keeps old checkpoints loadable with the new architecture. Apply the same pattern when adding/removing layers from `PaliGemmaVLNPolicy` or `PixArtSubgoalDiT`.
