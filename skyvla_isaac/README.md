# skyvla_isaac — Isaac Sim / Isaac Lab port

Migration of the drone manipulation work from Habitat-Sim to **NVIDIA Isaac Sim
4.5 + Isaac Lab 2.1**, for **real PhysX contact grasping** and massively parallel
RL. Runs in the isolated `isaac` conda env (Python 3.10).

```bash
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/ubuntu/SkyVLA
```

## Layout
- `assets/drone_with_gripper.urdf` — quadrotor + 2-DoF gripper (lower + grip).
- `assets/drone_with_gripper.usd` — PhysX articulation (generated from the URDF).
- `scripts/convert_urdf.py` — URDF → USD articulation (Isaac URDF importer).
- `tasks/pick_place_env.py` — `DronePickPlaceEnv` (DirectRLEnv): free-floating
  drone flown by a base wrench; gripper `lower`/`grip_l`/`grip_r` are actuated
  PhysX joints; the cube is gripped by **contact + friction** (no attach hack).
- `scripts/smoke_env.py` — build N envs, step random actions (verified ✓).
- `scripts/train.py` — PPO (rsl_rl), thousands of envs on one GPU.
- `gs/` — `GaussianMap` (sim-agnostic) + `isaac_camera.frame_from_camera`
  adapter (Isaac Camera → rgb/depth/pose/K). Needs `pip install gsplat` in the
  isaac env.

## Run
```bash
# regenerate the USD articulation from the URDF
python skyvla_isaac/scripts/convert_urdf.py
# smoke the env
python skyvla_isaac/scripts/smoke_env.py --num_envs 16
# train (smoke / full)
python skyvla_isaac/scripts/train.py --num_envs 256  --max_iterations 3
python skyvla_isaac/scripts/train.py --num_envs 2048 --max_iterations 1500
```

## Status
Working: install, URDF→USD articulation, parallel DirectRLEnv with real contact
grasping, rsl_rl PPO loop. **Reward shaping in `pick_place_env._get_rewards`
still needs tuning** for convergence (the smoke run's reward declines — expected
for an untrained policy against the time/effort costs).

## Not yet ported from Habitat (kept intact in `indoor_uav/`, `openfly/`)
- Gaussian-splat **navigation reward** loop (the `gs/` module is ported as a
  sim-agnostic consumer, but the camera-driven coverage env is not).
- OpenFly **VLN** benchmark + PaliGemma policy (large, scene/pose-bound).
These remain in the Habitat tree until ported — deleting them before porting
would remove un-replaced functionality.
