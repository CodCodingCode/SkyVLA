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

Real-physics pick-and-place, trained 3000 iterations across 2048 parallel envs (`model_2999.pt`):

| Metric | Rate |
|---|---:|
| Grasp (cube caged & lifted off the floor) | **79%** |
| Lift | **82%** |
| Place (carried to goal while held, < 18 cm) | **77%** |

The hard part was reward shaping, not flight: lift must be a strong gradient so grasping is discovered, but capped low so "rocket to the ceiling" stops paying — placement only rewards while the cube is genuinely held. A curriculum anneals the start pose from straddling the cube (grasp discovery) to starting from altitude (the real task).

## Install

```bash
conda activate isaac                  # Isaac Sim 4.5 + Isaac Lab 2.1, Python 3.10
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$PWD                # run all commands from the repo root
pip install gsplat                     # only needed for the Gaussian-map navigation demo
```

W&B logging is on by default (project `skyvla-isaac`). On a new machine either
`wandb login`, set `WANDB_API_KEY`, or drop your key in a `.wandb_key` file at the
repo root (gitignored). With no credentials, training falls back to TensorBoard.

## Run

```bash
# 1. build the PhysX articulation from the URDF (one-time; lower/grip become real DoFs)
python skyvla_isaac/scripts/convert_urdf.py

# 2. smoke the env — N parallel envs, random actions, headless
python skyvla_isaac/scripts/smoke_env.py --num_envs 16

# 3. train pick-and-place (PPO, rsl_rl) — smoke, then full
python skyvla_isaac/scripts/train.py --num_envs 256  --max_iterations 3
python skyvla_isaac/scripts/train.py --num_envs 2048 --max_iterations 1500

# 4. evaluate a checkpoint — deterministic rollout + real success rate (+ optional mp4)
python skyvla_isaac/scripts/play.py --checkpoint logs/isaac/drone_pick_place/model_1499.pt --video

# 5. render a polished 3rd-person rollout to mp4
python skyvla_isaac/scripts/render_rollout.py \
  --checkpoint logs/isaac/drone_pick_place/model_1499.pt --out videos/isaac_pickplace.mp4

# 6. navigation: fuse Isaac RGB-D into a GaussianMap and render a novel view
python skyvla_isaac/scripts/gs_isaac_demo.py
```

## SNATCH — training recipe

SNATCH (Sim-trained Neural Aerial Transport and Capture) is the second-generation
task in [`snatch/`](skyvla_isaac/snatch/): a single-DOF caging gripper (one servo,
no lower/raise arm — the drone descends bodily to grasp), 5-dim action
`[vx vy vz yaw_rate grip]`, and modeled sim-to-real gaps (VIO drift, detection
noise; see [`snatch/DESIGN.md`](skyvla_isaac/snatch/DESIGN.md)). The state-based
variant (`--no_cams`) is the one that converges; checkpoints are not committed, so
reproduce with the staged curriculum below. Each stage warm-starts from the
previous one — none of them converges trained from scratch in one shot.

```bash
# Stage 1 — base policy (default straddle-start curriculum): learn grasp + delivery
# with the drone starting near/over the cube. Good by ~iter 500-650 (watch
# grasp_rate / place_success on W&B).
python skyvla_isaac/scripts/train_snatch.py --no_cams \
  --num_envs 2048 --max_iterations 1500 --log_dir logs/isaac/drone_snatch

# Stage 2 — reverse curriculum (Florensa-style): spawn at the grasp pose, expand the
# start distribution as grasp survives. Light cube + slow speed make the bodily
# descent controllable. --reset_std/--entropy_coef tame the warm-started policy's
# exploration noise (the saved std is tuned to the old regime).
python skyvla_isaac/scripts/train_snatch.py --no_cams --reverse_curriculum \
  --cube_mass 0.05 --speed 0.6 \
  --resume logs/isaac/drone_snatch/model_650.pt --reset_std 0.6 --entropy_coef 0.005 \
  --num_envs 2048 --max_iterations 1500 --log_dir logs/isaac/drone_snatch_rc

# Stage 3 — distance curriculum: ramp the fly-in spawn distance toward 10 m.
# Env spacing and episode length auto-scale with --rc_dist_max.
python skyvla_isaac/scripts/train_snatch.py --no_cams --rc_distance_mode --rc_dist_max 10.0 \
  --cube_mass 0.05 --speed 0.35 \
  --resume logs/isaac/drone_snatch_rc/model_600.pt --rc_start 0.10 --reset_std 0.3 --entropy_coef 0.005 \
  --num_envs 2048 --max_iterations 100000 --log_dir logs/isaac/drone_snatch_dist

# Evaluate honestly (full fly-in, no straddle start) + the VIO-drift gap sweep:
python skyvla_isaac/scripts/eval_snatch.py --no_cams \
  --checkpoint logs/isaac/drone_snatch_dist/model_XXXX.pt
```

Crash-restart any stage by `--resume`-ing the latest `model_*.pt` from its
`--log_dir` and, for stage 3, setting `--rc_start` to the last difficulty printed
in the log so the expansion picks up where it left off (checkpoints save every 50
iterations). Status of this recipe: stages 1–2 converge reliably (94.7%
end-to-end pick→place clean, ~51% under full modeled VIO drift); stage 3 — full
10 m fly-in pickup — is the active frontier and is *not* solved at distance yet.
Exact run lineages with metrics are on
[W&B `skyvla-isaac`](https://wandb.ai/nathanyan2008p-personal/skyvla-isaac).

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
    gs_isaac_demo.py             end-to-end Gaussian-map navigation demo
```

## How it works

- **Drone.** A single free-floating PhysX articulation. Instead of per-rotor thrust, a base wrench tracks a commanded velocity (`kv`) and an attitude controller (`k_att`, `k_damp`) keeps it upright — so the policy commands flight, the controller keeps it stable.
- **Gripper.** A 4-jaw cage (`grip_xl/xr/yl/yr`) closing along ±X and ±Y, plus a stiff `lower` arm. A flat two-finger pinch ejects a free cube; the cage boxes it in. High contact friction holds it during carry.
- **Action / obs.** 6 continuous actions `[vx, vy, vz, yaw_rate, lower, grip]`; 17-dim observation (drone pose, gripper state, object & target pose, tip→object and object→goal vectors). World is +Z up.
- **Navigation map.** Gaussian Splatting's cost is *fitting*; we skip it. `add_from_rgbd` splats in the Gaussians implied by each posed RGB-D frame in closed form, and `render` rasterizes a view in milliseconds — fast enough for an inner-loop coverage reward.

## Requirements

- NVIDIA GPU with **Isaac Sim 4.5 + Isaac Lab 2.1** installed, in a conda env named `isaac` (Python 3.10).
- `OMNI_KIT_ACCEPT_EULA=YES` and `PYTHONPATH` set to the repo root (see Install).
- `gsplat` only for the navigation / Gaussian-map demo.

Agent/operator conventions for this repo (W&B logging, crash-resilient training on this host) live in [`CLAUDE.md`](CLAUDE.md).

## License

MIT — see [`LICENSE`](LICENSE).
