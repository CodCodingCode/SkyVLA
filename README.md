# SkyVLA

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5-76b900.svg)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1-76b900.svg)](https://isaac-sim.github.io/IsaacLab/)
[![W&B](https://img.shields.io/badge/W%26B-skyvla--isaac-yellow.svg)](https://wandb.ai/nathanyan2008p-personal/skyvla-isaac)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A free-flying quadrotor with a gripper that **picks things up, carries them, and puts them down** — and **maps the space it flies through**. Built on NVIDIA Isaac Sim 4.5 + Isaac Lab 2.1, with real PhysX contact grasping and thousands of parallel environments on one GPU.

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
export PYTHONPATH=/home/ubuntu/SkyVLA
pip install gsplat                     # only needed for the Gaussian-map navigation demo
```

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

## Layout

```
skyvla_isaac/
  assets/
    drone_with_gripper.urdf      quadrotor + 2-DoF gripper (lower + 4-jaw cage)
    drone_with_gripper.usd       PhysX articulation (generated from the URDF)
  tasks/
    pick_place_env.py            DronePickPlaceEnv — DirectRLEnv, real contact grasping
  gs/
    gaussian_map.py              incremental 3D-Gaussian map (gsplat), sim-agnostic
    isaac_camera.py              Isaac Camera → (rgb, depth, pose, K) adapter
  scripts/
    convert_urdf.py              URDF → USD articulation (Isaac URDF importer)
    smoke_env.py                 build N envs, step random actions (sanity check)
    train.py                     PPO via rsl_rl, thousands of envs on one GPU
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
