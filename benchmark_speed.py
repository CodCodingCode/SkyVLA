"""Benchmark Isaac Lab training throughput (steps/sec).

Launch:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/drone_project/benchmark_speed.py --task waypoint --num_envs 512 --num_iterations 3 --headless
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

_DRONE_PROJECT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Benchmark RL training steps/sec.")
parser.add_argument("--task", choices=["hover", "waypoint"], default="waypoint")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_iterations", type=int, default=3, help="PPO iterations to benchmark.")
parser.add_argument("--resume_path", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
if getattr(args_cli, "enable_cameras", None):
    args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

from importlib.metadata import version as pkg_version


def main():
    if args_cli.task == "hover":
        import hover  # noqa: F401

        from hover.hover_env import HoverEnvCfg
        from hover.agents.rsl_rl_ppo_cfg import HoverPPORunnerCfg

        env_id = "Isaac-Hover-Direct-v0"
        env_cfg = HoverEnvCfg()
        agent_cfg = HoverPPORunnerCfg()
        agent_cfg.logger = "tensorboard"
        agent_cfg.experiment_name = "benchmark_hover"
    else:
        import waypoint_nav  # noqa: F401

        from waypoint_nav.waypoint_nav_env import WaypointNavEnvCfg
        from waypoint_nav.agents.rsl_rl_ppo_cfg import WaypointNavPPORunnerCfg

        env_id = "Isaac-WaypointNav-Direct-v0"
        env_cfg = WaypointNavEnvCfg()
        agent_cfg = WaypointNavPPORunnerCfg()
        agent_cfg.logger = "tensorboard"
        agent_cfg.experiment_name = "benchmark_waypoint"

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    agent_cfg.max_iterations = args_cli.num_iterations
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed = 42

    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    print("=" * 60)
    print(f"  Benchmark: {args_cli.task}")
    print(f"  num_envs={args_cli.num_envs}  iterations={args_cli.num_iterations}")
    print(f"  num_steps_per_env={agent_cfg.num_steps_per_env}")
    print(f"  headless=True  cameras=False")
    print("=" * 60)

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    log_dir = os.path.join(_DRONE_PROJECT, "logs", "benchmark", args_cli.task)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.resume_path:
        runner.load(os.path.abspath(args_cli.resume_path))

    steps_per_iter = agent_cfg.num_steps_per_env * args_cli.num_envs

    # Warmup: first iteration includes extra overhead; report both
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=args_cli.num_iterations, init_at_random_ep_len=True)
    elapsed = time.perf_counter() - t0

    total_steps = steps_per_iter * args_cli.num_iterations
    steps_per_sec = total_steps / elapsed
    iters_per_sec = args_cli.num_iterations / elapsed

    print("=" * 60)
    print(f"  Total time:     {elapsed:.2f}s")
    print(f"  Total sim steps:{total_steps:,}")
    print(f"  Steps/sec:      {steps_per_sec:,.0f}")
    print(f"  Iters/sec:      {iters_per_sec:.3f}")
    print(f"  Per-iter (avg): {elapsed / args_cli.num_iterations:.2f}s")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
