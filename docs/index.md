---
title: SkyVLA
---

# SkyVLA

A free-flying quadrotor with a gripper that **picks things up, carries them, and puts them down** — and **maps the space it flies through**. Built on NVIDIA Isaac Sim 4.5 + Isaac Lab 2.1, with real PhysX contact grasping and thousands of parallel environments on one GPU.

> Code, install, and run instructions live in the [repository README](https://github.com/CodCodingCode/SkyVLA#readme).

## What it does

- **Manipulation — pick & place.** The drone flies to a cube, cages it with a 4-jaw gripper, and the cube is held by **PhysX contact + friction** — a bad grasp slips; there is no kinematic attach. Trained end-to-end with PPO (`rsl_rl`).
- **Navigation — live 3D mapping.** Posed RGB-D frames are fused into an explicit 3D-Gaussian map (`gsplat`) by closed-form back-projection — no per-step fitting — so coverage and novel-view rendering are cheap enough to drive a navigation reward.

## Results

Real-physics pick-and-place, trained 3000 PPO iterations across 2048 parallel envs:

| Metric | Rate |
|---|---:|
| Grasp (cube caged & lifted off the floor) | **79%** |
| Lift | **82%** |
| Place (carried to goal while held, < 18 cm) | **77%** |

The hard part was reward shaping, not flight: lift must be a strong gradient so grasping is discovered, but capped low so "rocket to the ceiling" stops paying — placement only rewards while the cube is genuinely held. A curriculum anneals the start pose from straddling the cube to starting from altitude.

## How it works

- **Drone** — a single free-floating PhysX articulation; a base wrench tracks commanded velocity while an attitude controller keeps it upright.
- **Gripper** — a 4-jaw cage closing along ±X and ±Y plus a stiff lowering arm. A flat two-finger pinch ejects a free cube; the cage boxes it in, and high contact friction holds it during carry.
- **Action / obs** — 6 continuous actions `[vx, vy, vz, yaw_rate, lower, grip]`; a 17-dim observation (drone pose, gripper state, object & target pose, tip→object and object→goal vectors).
- **Navigation map** — Gaussian Splatting's cost is *fitting*, which we skip: each posed RGB-D frame splats in its implied Gaussians in closed form, and a view rasterizes in milliseconds.

---

<p><a href="https://github.com/CodCodingCode/SkyVLA">View the project on GitHub</a> · MIT License</p>
