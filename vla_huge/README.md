# VLA Drone — HUGE-Bench Digital-Twin Scenes (Stage 7 scaffold)

This module is the placeholder home for Stage 7 of the curriculum: RL on
the four real-world digital-twin scenes that ship with HUGE-Bench.

## Status: blocked on upstream sim release

HUGE-Bench publishes its dataset (LeRobot trajectories) but the matching
Isaac Sim toolchain — the 3DGS-Mesh USDs, scene loaders, and benchmark
runners — is not publicly released yet. Track:

- HUGE-Bench project page: <https://jingyu198.github.io/HUGE_Bench>
- HuggingFace org `yu781986168` for sim assets

Until then, every entry in [`scenes.py`](scenes.py) resolves to the
`Simple_Warehouse/full_warehouse.usd` placeholder so Stage 7 acts as a
**resume smoke test** that validates the Stage 6 checkpoint round-trips
through the full RL loop. When HUGE drops, edit
`HugeScene.usd_path` in `scenes.py` and (optionally) add scene-specific
POI banks to differentiate the four digital-twin scenes properly.

## Files

| File | Purpose |
|---|---|
| `scenes.py` | HUGE scene metadata (scene_id, USD path, prompts). Edit when HUGE sim drops. |
| `vla_huge_env.py` | `VLAHugeDroneEnv` — subclasses `VLAWarehouseDroneEnv`; only overrides the instruction bank and per-scene spawn altitude. |
| `agents/rsl_rl_ppo_cfg.py` | PPO config (logs to `logs/rsl_rl/vla_drone_huge/`). |
| `train.py` | Shim — patches `sys.modules` so `vla/train.py` runs unchanged against this env. |
| `__init__.py` | Registers `Isaac-VLADrone-Huge-v0`. |

## Smoke (works today)

```bash
cd /home/ubuntu/IsaacLab
./isaaclab.sh -p /home/ubuntu/drone_project/vla_huge/train.py \
  --num_envs 4 --max_iterations 1 \
  --headless --enable_cameras \
  --scene_id 1_office \
  --resume_path /home/ubuntu/drone_project/logs/huge_bench_highlevel/<run>/model_5000.pt
```

This boots Isaac, swaps in the warehouse USD as a placeholder, loads
the Stage 6 checkpoint via the same shape-tolerant loader as
`vla/train.py`, and runs one PPO iteration. If this completes, the
weight pipeline is wired up correctly.

## Full run (when HUGE sim drops)

1. Edit [`scenes.py`](scenes.py) — replace the placeholder USD path
   with the real digital-twin asset and tune `env_spacing` /
   `spawn_altitude` per scene.
2. Optionally add per-scene POI banks (similar to
   `vla_warehouse/pois.py`) keyed on `scene_id`. Until then the
   warehouse POIs from `vla_warehouse/pois.py` are reused.
3. Launch:

```bash
./isaaclab.sh -p /home/ubuntu/drone_project/vla_huge/train.py \
  --num_envs 64 --max_iterations 3000 \
  --headless --enable_cameras \
  --scene_id 1_office \
  --resume_path /home/ubuntu/drone_project/logs/huge_bench_highlevel/<run>/model_5000.pt
```

`--num_envs 64` is the recommended outdoor-scale start (lower than
`vla/`'s 256-env empty arena because the cloned digital-twin scenes are
memory-heavy).

## What is trained vs frozen

Same as Stage 5/6: PaliGemma LoRA + cross-attn + depth_encoder + LSTM +
`target_mlp` are trainable; the Stage-2 waypoint MLP buffers stay
frozen. RL supervises the high-level target prediction via PPO reward
(distance / hover / success terms), so the action MLP is never
gradient-touched.
