# SkyVLA

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5-76b900.svg)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1-76b900.svg)](https://isaac-sim.github.io/IsaacLab/)
[![W&B](https://img.shields.io/badge/W%26B-skyvla--isaac-yellow.svg)](https://wandb.ai/nathanyan2008p-personal/skyvla-isaac)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A free-flying quadrotor with a gripper that **picks things up, carries them, and puts them down** — and **maps the space it flies through**. Built on NVIDIA Isaac Sim 4.5 + Isaac Lab 2.1, with real PhysX contact grasping and thousands of parallel environments on one GPU.

<p align="center">
  <a href="videos/snatch_pickplace_victory.mp4">
    <img src="videos/demo.gif" alt="SkyVLA drone flying in, caging a cube, carrying it, and placing it" width="100%">
  </a>
</p>

<p align="center"><em>SNATCH policy (<code>model_9250</code>): grasp <b>87.4%</b> · lift <b>19.3 cm</b> · place <b>86.1%</b>. <a href="videos/snatch_pickplace_victory.mp4">▶ Full 1080p render.</a></em></p>

```
                 ┌─────────────────────────────────────────┐
   action (6) ──►│  free-floating quadrotor  (base wrench)   │
 [vx vy vz yaw   │      │                                    │
  lower grip]    │      └─► 2-DoF arm: lower + 4-jaw cage ────┼─► contact+friction grasp
                 │                                            │      (no attach hack)
  obs (17) ◄─────┤  drone pose · gripper state · object ·     │
                 │  target · tip→object · object→goal         │
                 └─────────────────────────────────────────┘
        PPO (rsl_rl), 2048 envs/GPU          GaussianMap (gsplat) ── flight → 3D map
```

## What it does

- **Manipulation — pick & place.** [`DronePickPlaceEnv`](skyvla_isaac/tasks/pick_place_env.py) is an Isaac Lab `DirectRLEnv` where the drone flies to a cube, **cages it with a 4-jaw gripper, and the cube is held by PhysX contact + friction** — a bad grasp slips, there is no kinematic attach. Trained end-to-end with PPO.
- **Navigation — live 3D mapping.** [`GaussianMap`](skyvla_isaac/gs/gaussian_map.py) fuses posed RGB-D frames into an explicit 3D-Gaussian map by closed-form back-projection (no per-step fitting), so coverage and novel-view rendering are cheap enough to drive a navigation reward. The [`isaac_camera`](skyvla_isaac/gs/isaac_camera.py) adapter is the entire sim port: feed it an Isaac Lab `Camera` instead of a Habitat sensor.

## Results

Real-physics pick-and-place, trained 3000 iterations across 2048 parallel envs (`model_2999.pt`).
No checkpoint for this first-generation task is committed; the numbers are from W&B.

| Metric | Rate |
|---|---:|
| Grasp (cube caged & lifted off the floor) | **79%** |
| Lift | **82%** |
| Place (carried to goal while held, < 18 cm) | **77%** |

> **What "place" measures.** Success is `_held & (d_goal < 0.18)` — the cube reaches the
> goal *while still gripped*. The goal itself is a waypoint sampled within 25 cm of the
> cube, floating 25 cm above the same platform. So this metric is **carry-and-hover, not
> deposit**: the drone never releases. Depositing onto a second platform is the separate
> `--two_platform` / `--release_only` task in the transport lineage below, and it is the
> only place in this repo where success does *not* require `_held`.

The hard part was reward shaping, not flight: lift must be a strong gradient so grasping is discovered, but capped low so "rocket to the ceiling" stops paying — placement only rewards while the cube is genuinely held. A curriculum anneals the start pose from straddling the cube (grasp discovery) to starting from altitude (the real task).

## Install

Two virtualenvs, because the Isaac stack and `gsplat` are built against different
Python versions. Everything runs from the repo root.

```bash
# Isaac (physics, training, eval) -- Isaac Sim 5.1 + Isaac Lab 2.3.2, Python 3.11
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$PWD
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1   # aarch64 hosts: required
.venv311/bin/python skyvla_isaac/scripts/<anything>.py

# Gaussian-map rendering only (gsplat, Python 3.10) -- no Isaac import
.venv/bin/python skyvla_isaac/scripts/render_gs_cache.py
```

`LD_PRELOAD` is not optional on aarch64: without it the interpreter aborts before
reaching any repo code. A convenience wrapper for the long form:

```bash
isaacpy() { OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" \
  LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 .venv311/bin/python "$@"; }
```

> **Known gap.** `gs_isaac_demo.py` and `render_rollout_gs.py` import *both*
> `isaaclab` and `gsplat`, and no single interpreter here has both — `gsplat` is
> not installed in `.venv311`. Those two scripts cannot run until it is. Every
> other script works.

W&B logging is on by default (project `skyvla-isaac`). On a new machine either
`wandb login`, set `WANDB_API_KEY`, or drop your key in a `.wandb_key` file at the
repo root (gitignored). With no credentials, training falls back to TensorBoard.

## Run

Using the `isaacpy` wrapper defined above (or the full env-var form in its place).

```bash
# 1. build the PhysX articulation from the URDF (one-time; lower/grip become real DoFs)
isaacpy skyvla_isaac/scripts/convert_urdf.py

# 2. smoke the env — N parallel envs, random actions, headless
isaacpy skyvla_isaac/scripts/smoke_env.py --num_envs 16

# 3. train pick-and-place (PPO, rsl_rl) — smoke, then full
isaacpy skyvla_isaac/scripts/train.py --num_envs 256  --max_iterations 3
isaacpy skyvla_isaac/scripts/train.py --num_envs 2048 --max_iterations 1500

# 4. evaluate a checkpoint — deterministic rollout + real success rate (+ optional mp4)
isaacpy skyvla_isaac/scripts/play.py --checkpoint logs/isaac/drone_pick_place/model_1499.pt --video

# 5. render a polished 3rd-person rollout to mp4
isaacpy skyvla_isaac/scripts/render_rollout.py \
  --checkpoint logs/isaac/drone_pick_place/model_1499.pt --out videos/isaac_pickplace.mp4

# 6. replay a cached Gaussian-map rollout (Isaac-free consumer, .venv)
.venv/bin/python skyvla_isaac/scripts/render_gs_cache.py --out videos/gs_orbit.mp4
```

## SNATCH — training recipe

SNATCH (Sim-trained Neural Aerial Transport and Capture) is the second-generation
task in [`snatch/`](skyvla_isaac/snatch/): a single-DOF caging gripper (one servo,
no lower/raise arm — the drone descends bodily to grasp), 5-dim action
`[vx vy vz yaw_rate grip]`, and modeled sim-to-real gaps (VIO drift, detection
noise; see [`snatch/DESIGN.md`](skyvla_isaac/snatch/DESIGN.md)). The state-based
variant (`--no_cams`) is the one that converges. Each stage warm-starts from the
previous one — none of them converges trained from scratch in one shot.

### Grasp lineage — how `model_9250` was produced

The committed checkpoints in [`snatch/checkpoints/`](skyvla_isaac/snatch/checkpoints/)
come from this chain. The three later stages warm-started from log dirs that no
longer exist on any current host, so **they cannot be re-run as-is** — the exact
flags are preserved here as the recipe of record. To continue the lineage today,
`--resume` from `snatch/checkpoints/model_9250.pt` instead.

```bash
# Stage A — NAV: randomized side spawns, so stage 0 is navigate->hover, not just descend.
isaacpy skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
  --cube_mass 0.05 --side_spawn 5.0 --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 \
  --entropy_coef 0.001 --latch_ready_coef 4.0 \
  --num_envs 2048 --max_iterations 20000 --log_dir logs/isaac/drone_snatch_nav

# Stage B — OVERHEAD-FIRST: pay only for being directly ABOVE the block, lowering the
# altitude setpoint rung by rung as centring is proven. Gripper taxed (no grasp yet).
isaacpy skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.0 \
  --overhead_first --no_latch --cube_mass 0.05 --side_spawn 5.0 \
  --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
  --entropy_coef 0.002 --latch_ready_coef 4.0 --reset_std 0.25 \
  --resume <stage-A model_250.pt> \
  --num_envs 2048 --max_iterations 40000 --log_dir logs/isaac/drone_snatch_overhead

# Stage C — GRASP + LIFT ("gentle-lift", the run model_9250 is a snapshot of).
# reset_std 0.5 restores grip-action exploration: the cage is taxed shut through every
# hover phase, so the close must be SAMPLED before it can be learned.
isaacpy skyvla_isaac/scripts/train_snatch.py --no_cams --staged_curriculum --cur_p 0.15 \
  --no_latch --carry_demo 0.25 --cube_mass 0.05 --side_spawn 5.0 \
  --stage_hover_thresh 0.70 --stage1_hover_anneal 10000 --start_stage 2 \
  --entropy_coef 0.001 --latch_ready_coef 4.0 --reset_std 0.2 \
  --resume <stage-B model_900.pt> \
  --num_envs 2048 --max_iterations 40000 --log_dir logs/isaac/drone_snatch_grasp

# Evaluate honestly (full fly-in, no straddle start) + the VIO-drift gap sweep:
isaacpy skyvla_isaac/scripts/eval_snatch.py --no_cams --no_latch --side_spawn 5.0 \
  --checkpoint skyvla_isaac/snatch/checkpoints/model_9250.pt
```

### Transport lineage — pick from A, carry to B, deposit on B

Architecture, stage contracts and handoff rules:
[`skyvla_isaac/snatch/THREE_POLICY.md`](skyvla_isaac/snatch/THREE_POLICY.md).

Three policies, each its own `.pt`, each warm-started from the one before. Enabled by
`--two_platform`, which adds delivery platform B and re-anchors the goal to *the cube
at rest on B* (without it, everything above behaves exactly as before).

```bash
# Stage 2 — CARRY: haul the grasped cube A->B. Success = arrived over B, still holding it.
bash skyvla_isaac/scripts/run_snatch_carry.sh                    # rung 1: 1.5 -> 4 m
SEP=4 SEP_MAX=10 SPEED=2.5 INIT=logs/isaac/drone_snatch_carry/model_XXXX.pt \
  DIR=logs/isaac/drone_snatch_carry_r2 bash skyvla_isaac/scripts/run_snatch_carry.sh

# Stage 3 — PLACE: lower onto B and let go. Success = cube at rest on B, jaws open.
INIT=logs/isaac/drone_snatch_carry/model_XXXX.pt \
  bash skyvla_isaac/scripts/run_snatch_place.sh

# Run all three end-to-end with state-based handoff (snatch -> carry -> place):
isaacpy skyvla_isaac/scripts/run_pipeline.py --no_cams \
  --snatch skyvla_isaac/snatch/checkpoints/model_9250.pt \
  --carry logs/isaac/drone_snatch_carry/model_XXXX.pt \
  --place logs/isaac/drone_snatch_place/model_XXXX.pt
```

**Distance ladder.** `env_spacing` and episode length are sized off `--plat_sep_max`
at scene construction, so the ceiling cannot be raised mid-run: finish a rung, then
relaunch with `SEP`/`SEP_MAX` raised and `INIT` pointed at the last checkpoint —
1.5→4 m at speed 1.5, 4→10 at 2.5, 10→20 at 4.0, 20→40 at 6.0. Raise `--speed` only
one rung at a time; it rescales what `action=1.0` means, so a large jump invalidates
the warm start.

Crash-restart any stage by `--resume`-ing the latest `model_*.pt` from its `--log_dir`
(checkpoints save every 50 iterations); both `run_snatch_*.sh` scripts do this
automatically in a restart loop. Status: the grasp lineage converges reliably (94.7%
end-to-end clean, ~51% under full modeled VIO drift). The **transport lineage is new
and unproven** — carry is in training, place has not been trained yet, and no
`--rc_distance_mode` fly-in beyond 10 m has ever converged. Exact run lineages with
metrics are on [W&B `skyvla-isaac`](https://wandb.ai/nathanyan2008p-personal/skyvla-isaac).

## Layout

```
skyvla_isaac/
  assets/
    drone_with_gripper.urdf      quadrotor + 2-DoF gripper (lower + 4-jaw cage)
    drone_with_gripper.usd       PhysX articulation (generated from the URDF)
  tasks/
    pick_place_env.py            DronePickPlaceEnv — DirectRLEnv, real contact grasping
  snatch/
    DESIGN.md                    SNATCH integration contract (action/obs/reward/DR spec)
    pick_place_env.py            DroneSnatchEnv — 1-DOF cage, curricula, VIO drift modeled
    rewards.py                   4-component shaping (nav/align/grasp/transport), pure torch
    randomization.py             domain randomization + VIO-drift / detection-noise models
    perception.py                dual depth-cam configs + ResNet-18 encoders (visuomotor)
  gs/
    gaussian_map.py              incremental 3D-Gaussian map (gsplat), sim-agnostic
    isaac_camera.py              Isaac Camera → (rgb, depth, pose, K) adapter
  scripts/
    convert_urdf.py              URDF → USD articulation (Isaac URDF importer)
    smoke_env.py                 build N envs, step random actions (sanity check)
    train.py                     PPO via rsl_rl, thousands of envs on one GPU
    train_snatch.py              SNATCH training entrypoint (all curriculum flags)
    eval_snatch.py               SNATCH eval + VIO-drift gap sweep
    play.py                      deterministic eval + success rate + video
    render_rollout.py            3rd-person Camera rollout → mp4
    run_snatch_carry.sh          stage 2 — A→B transport, restart loop + distance ladder
    run_snatch_place.sh          stage 3 — deposit on B (--release_only), restart loop
    run_pipeline.py              snatch → carry → place, state-based handoff between 3 .pt
    render_gs_cache.py           Isaac-free Gaussian-map replay (.venv)
    gs_isaac_demo.py             Gaussian-map navigation demo (needs gsplat in .venv311)
```

## How it works

- **Drone.** A single free-floating PhysX articulation. Instead of per-rotor thrust, a base wrench tracks a commanded velocity (`kv`) and an attitude controller (`k_att`, `k_damp`) keeps it upright — so the policy commands flight, the controller keeps it stable.
- **Gripper.** A 4-jaw cage (`grip_xl/xr/yl/yr`) closing along ±X and ±Y, plus a stiff `lower` arm. A flat two-finger pinch ejects a free cube; the cage boxes it in. High contact friction holds it during carry.
- **Action / obs.** 6 continuous actions `[vx, vy, vz, yaw_rate, lower, grip]`; 17-dim observation (drone pose, gripper state, object & target pose, tip→object and object→goal vectors). World is +Z up.
- **Navigation map.** Gaussian Splatting's cost is *fitting*; we skip it. `add_from_rgbd` splats in the Gaussians implied by each posed RGB-D frame in closed form, and `render` rasterizes a view in milliseconds — fast enough for an inner-loop coverage reward.

## Requirements

- NVIDIA GPU with **Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + rsl_rl 3.1.2**, in `.venv311` (Python 3.11).
- `OMNI_KIT_ACCEPT_EULA=YES` and `PYTHONPATH` set to the repo root (see Install).
- On aarch64 hosts, `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`.
- `gsplat` in `.venv` (Python 3.10) for Gaussian-map rendering.

Verified on a GH200 (aarch64) with torch 2.7.0+cu128. Earlier revisions of this
README specified Isaac Sim 4.5 / Isaac Lab 2.1 / Python 3.10 under conda envs named
`isaac` and `habitat`; that layout is gone and no script depends on it any more.

Agent/operator conventions for this repo (W&B logging, crash-resilient training on this host) live in [`CLAUDE.md`](CLAUDE.md).

## License

MIT — see [`LICENSE`](LICENSE).
