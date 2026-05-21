"""Fast waypoint policy playback — no RTX cameras, physics-only headless.

Logs drone + goal trajectories, then stitches a lightweight top-down MP4
with matplotlib (seconds, not minutes).

Launch:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/drone_project/waypoint_nav/play_fast.py \
        --checkpoint ~/drone_project/model_2998_waypoint.pt --num_steps 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rsl_rl")
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaaclab.app import AppLauncher

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Fast waypoint nav playback (no RTX video).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=300, help="Simulation steps after reset.")
parser.add_argument("--output", type=str, default=None, help="Output .mp4 path (default: videos/waypoint_fast.mp4).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Physics-only: headless, NO cameras / RTX
args_cli.headless = True
if getattr(args_cli, "enable_cameras", None):
    args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import waypoint_nav  # noqa: F401

from waypoint_nav.waypoint_nav_env import WaypointNavEnvCfg
from waypoint_nav.agents.rsl_rl_ppo_cfg import WaypointNavPPORunnerCfg


def _make_video(drone_xy: np.ndarray, goal_xy: np.ndarray, out_path: str, fps: int = 30) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    all_x = np.concatenate([drone_xy[:, 0], goal_xy[:, 0]])
    all_y = np.concatenate([drone_xy[:, 1], goal_xy[:, 1]])
    pad = 0.5
    xmin, xmax = float(all_x.min() - pad), float(all_x.max() + pad)
    ymin, ymax = float(all_y.min() - pad), float(all_y.max() + pad)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Waypoint nav — model_2998 (top-down)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    (trail,) = ax.plot([], [], "b-", alpha=0.5, lw=1.5)
    (drone_dot,) = ax.plot([], [], "bo", ms=10)
    (goal_dot,) = ax.plot([], [], "r*", ms=14)

    def update(frame: int):
        trail.set_data(drone_xy[: frame + 1, 0], drone_xy[: frame + 1, 1])
        drone_dot.set_data([drone_xy[frame, 0]], [drone_xy[frame, 1]])
        goal_dot.set_data([goal_xy[frame, 0]], [goal_xy[frame, 1]])
        return trail, drone_dot, goal_dot

    anim = FuncAnimation(fig, update, frames=len(drone_xy), interval=1000 / fps, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[INFO] Video saved: {out_path}")


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    out_path = args_cli.output or os.path.join(_DRONE_PROJECT, "videos", "waypoint_fast.mp4")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Output:     {out_path}")

    env_cfg = WaypointNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.clone_in_fabric = False  # Fabric clone fails on single-env play
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    # Default decimation/render — no per-step RTX

    env = gym.make("Isaac-WaypointNav-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped

    agent_cfg = WaypointNavPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device

    from importlib.metadata import version as pkg_version

    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    drone_hist = []
    goal_hist = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            pos = base_env._robot.data.root_pos_w[0].cpu().numpy()
            goal = base_env._target_pos_w[0].cpu().numpy()
            drone_hist.append(pos[:2].copy())
            goal_hist.append(goal[:2].copy())

            if step % 50 == 0:
                dist = float(np.linalg.norm(pos[:3] - goal[:3]))
                print(f"  step {step:4d}/{args_cli.num_steps}  dist_to_goal={dist:.3f}m")

    elapsed = time.perf_counter() - t0
    steps_per_sec = args_cli.num_steps / elapsed
    print(f"[INFO] Rollout: {args_cli.num_steps} steps in {elapsed:.2f}s ({steps_per_sec:.1f} steps/s)")

    env.close()

    drone_xy = np.stack(drone_hist)
    goal_xy = np.stack(goal_hist)
    np.savez(
        out_path.replace(".mp4", ".npz"),
        drone_xy=drone_xy,
        goal_xy=goal_xy,
        drone_full=np.array([h for h in drone_hist]),
    )

    _make_video(drone_xy, goal_xy, out_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
