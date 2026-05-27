# Recording P3 policy demos in the real Unreal sim

End-to-end workflow for producing 1920×1080 H.264 MP4 videos of the
trained P3 policy (PaliGemma BC + subgoal DiT) navigating in real Unreal
Engine via the `env_ue_smallcity` CitySample scene.

This is validated — the exact commands below produced
[videos/p3_realsim.mp4](../videos/p3_realsim.mp4) (17.5 MB, 1920×1080,
120 frames) on 2026-05-27.

## TL;DR — copy-paste

```bash
# 1. Activate env (sets OPENFLY_ROOT, OPENFLY_IMAGE_ROOT, PYTHONPATH, conda env).
source ~/SkyVLA/openfly/activate.sh

# 2. Point at the PixArt-Σ snapshot the DiT was finetuned from.
export PIXART=/home/ubuntu/assets/pretrained/hf_cache/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/snapshots/76fb7eb5a9314bc1e4e479d2f13447517fca9be4

# 3. Record. Forks CitySample.sh, drifts to each episode, rolls the policy
#    for max_steps, tears the sim down on exit.
python -m openfly.scripts.record_p3_demo \
  --p3_ckpt   logs/openfly/paligemma_subgoal/20260527_163142/best.pt \
  --dit_path  logs/openfly/subgoal_dit/20260526_205830/best.pt \
  --pretrained_path "$PIXART" \
  --episodes 3 --max_steps 40 --ddim_steps 4 --drift_steps 12 \
  --image_size 256 --fps 5 \
  --out /home/ubuntu/SkyVLA/videos/p3_realsim.mp4
```

Outputs:

- `videos/p3_realsim.mp4` — FPV at the sim's native resolution (1920×1080),
  H.264, with HUD overlay (instruction, pose, action, running SR).
- `videos/p3_realsim_topdown.mp4` — 256×256 top-down trajectory map.
- `videos/p3_realsim_run.log` — full stdout including per-step UnrealCV
  diagnostics.

## What happens, in order

```
[record_p3_demo] using 3 episodes from split=unseen
[record_p3_demo] loading world model …          ← 5-10 s
[record_p3_demo] loading P3 policy …            ← 5-10 s (PaliGemma + LoRA)
[record_p3_demo] launching real sim … ~30-90s   ← UEBridge.__init__ forks
[record_p3_demo] real sim up in 32.2s             CitySample.sh + sleeps 15s

[record_p3_demo.preroll]      drift 1044m in 12 steps from (150,400) -> (-336,-522)
                                                ← drift_steps small hops
[record_p3_demo] native FPV=1920x1080             before the first teleport

[record_p3_demo] ep 0: steps=40 final_d=59.0m  success=False  (64.3s)
[record_p3_demo.ep1.preroll]  drift 1095m in 12 steps from (-336,-522) -> (590,62)
[record_p3_demo] ep 1: steps=40 final_d=114.0m success=False  (63.7s)
[record_p3_demo.ep2.preroll]  drift 1256m in 12 steps from (590,62) -> (-608,-314)
[record_p3_demo] ep 2: steps=40 final_d=125.0m success=False  (62.8s)
[record_p3_demo] tearing down real sim …
[record_p3_demo] DONE. SR=0.00% (0/3)
```

Total wall clock: ~6 min for 3×40-step episodes.

## Per-step cost breakdown (~1.6 s avg)

| Component                                    | ~time  |
| -------------------------------------------- | ------ |
| `bridge.get_camera_data('lit')` (TCP + JPEG) | 0.6-1.0 s |
| PaliGemma encode + 4-step DDIM + action head | 0.5-1.0 s |
| `bridge.set_camera_pose` (4 UnrealCV calls)  | ~0.3 s |

40 steps × ~1.6 s ≈ 60-65 s per episode. Each inter-episode preroll
adds another ~20 s.

## Why drifting matters

`env_ue_smallcity` is ~2 km across. Episode start poses are 500-1500 m
from `UEBridge`'s default init position `(150, 400, 15)`. **Direct
teleporting that far hangs UnrealCV indefinitely** — UE tries to stream
in an entirely new region while holding the single `vget /camera/1/lit
png` request open, and the Python client has no timeout.

`_drift_camera` breaks the long jump into `drift_steps=12` small hops,
fetching a frame after each. Each request's streaming delta stays
small enough to return promptly. Without this preroll the very first
`get_camera_data` after `make_bridge` hangs forever.

## Useful flags

| Flag                | Default            | Notes                                       |
| ------------------- | ------------------ | ------------------------------------------- |
| `--episodes N`      | 3                  | Number of episodes to record.               |
| `--max_steps N`     | 40                 | Per-episode step cap.                       |
| `--ddim_steps N`    | 4                  | DDIM steps for subgoal sampling.            |
| `--drift_steps N`   | 12                 | Hops per preroll drift. Larger = safer.     |
| `--fps N`           | 8                  | Output video framerate.                     |
| `--image_size N`    | 256                | Policy input size; FPV uses native res.     |
| `--sim_env NAME`    | `env_ue_smallcity` | Only env with a working launcher here.      |
| `--history_frames N`| 2                  | Past frames pooled into the policy.         |

For a more meaningful demo, bump `--max_steps` — at 40 steps × 9 m
max stride, a ~1 km episode can't reach the 20 m success radius even
with perfect heading. Try `--max_steps 100` for episodes that actually
have a chance of completing.

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `CitySample` dies seconds after start with `Killed` | Stale `Xvfb :9` from a prior run | `pkill -9 Xvfb` then retry |
| `get_camera_data` returns TypeError (bytes vs str) | Camera 1 not spawned in UE | Don't bypass `UEBridge.__init__` — use the default flow |
| First `vset` hangs forever | Sim not actually ready (port bound, world still streaming) | Same — let `UEBridge.__init__` do its 15s sleep |
| Device-side assert at GeLU in DiT | Out-of-range `last_action` embedding index | Already fixed: the recorder maps raw action ids → logit indices via `action_id_to_logit_index` |
| Recorder exits cleanly but `nvidia-smi` still shows GPU memory | Python proc didn't fully exit (rare; stuck atexit hook) | `kill -9 <pid>` then `nvidia-smi --query-gpu=memory.used` to confirm |

## Verifying outputs

```bash
ffprobe -v error -show_entries stream=width,height,codec_name,nb_frames,duration \
  -of default=noprint_wrappers=1 videos/p3_realsim.mp4
```

Expected for a 3-episode run at `--fps 5 --max_steps 40`:

```
codec_name=h264
width=1920
height=1080
duration=24.000000
nb_frames=120
```

## Manual sim teardown

If a recording fails midway and leaves the sim alive:

```bash
pkill -9 -f CitySample
pkill -9 -f CrashReport
pkill -9 Xvfb
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # should be 0 MiB
```

## Recording multiple takes

Each invocation does a fresh ~32 s cold start, runs all requested
episodes, then tears the sim down. There is no persistent-sim path —
`UEBridge.__init__`'s internal 15 s settle inside its sim-launch thread
is what keeps the first heavy `vset` from hanging, and you can't get
that without going through the full `make_bridge` cold start each time.

For a 10-episode comparison sweep:

```bash
python -m openfly.scripts.record_p3_demo \
  --p3_ckpt ... --dit_path ... --pretrained_path "$PIXART" \
  --episodes 10 --max_steps 100 \
  --out videos/p3_realsim_10ep.mp4
```

Cost: ~32 s cold start + 10 × (~20 s drift + ~160 s rollout) ≈ ~30 min.
