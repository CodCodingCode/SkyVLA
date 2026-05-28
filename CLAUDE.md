# SkyVLA — agent conventions

Rules for me (the agent) when working in this repo. Short and rule-focused.

## Long-running training runs

**Always launch long training runs (>15 min) inside a tmux session.** Never use bare `nohup ... &` for SFT / DiT / RL training — tmux is interactively attachable, has cleaner process management, and the user explicitly asked for it so that closing their laptop never matters.

Pattern:

```bash
SESS=<phase>_$(date +%Y%m%d_%H%M%S)        # e.g. dit_20260528_192046, p3_..., grpo_...
LOG=/tmp/${SESS}.log
tmux new-session -d -s "$SESS" \
  "./openfly/run_train_<phase>.sh <args...> 2>&1 | tee $LOG"
echo "session=$SESS  log=$LOG"
```

After launching, tell the user:
- `tmux attach -t <SESS>` to view live (Ctrl-B then D to detach without killing)
- `tail -f <LOG>` for file-based tail
- `tmux ls` to find running sessions

Stopping a run: `tmux kill-session -t <SESS>` (preferred over `pkill -9`).

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

## Image data caveat

Only ~25% of `train.json` steps have local frames in `~/assets/OpenFly/images/Image`. The dataset's `require_images=True` filter silently drops the rest. The "train steps" the trainer prints is what's actually usable, not what's in the json.

## Don't break what works

When making architectural changes to a model that has saved checkpoints, the `load_state_dict(strict=False)` + shape-filter pattern (see `openfly/policies.py:PaliGemmaOpenFlyPolicy.__init__`) keeps old checkpoints loadable with the new architecture. Apply the same pattern when adding/removing layers from `PaliGemmaVLNPolicy` or `PixArtSubgoalDiT`.
