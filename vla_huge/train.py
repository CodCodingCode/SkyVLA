"""Train the VLA drone in a HUGE-Bench scaffolded scene (Stage 7).

Thin shim over ``vla/train.py`` — same swap-via-sys.modules pattern as
``vla_warehouse/train.py`` so the 600-line training loop is reused
verbatim. Until the real HUGE Isaac Sim toolchain is released, the env
falls back to ``warehouse_full`` USD asset and the smoke path purely
verifies the resume-from-Stage-6 weights round-trip.

Launch (smoke):
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla_huge/train.py \
        --num_envs 4 --max_iterations 1 --headless --enable_cameras \
        --resume_path <stage6.pt>

Launch (full, after HUGE sim drops):
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla_huge/train.py \
        --num_envs 64 --max_iterations 3000 \
        --headless --enable_cameras --resume_path <stage6.pt>
"""

import argparse
import os
import sys

# Phase 0: AppLauncher must boot before any pxr / isaaclab imports.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train VLA drone in a HUGE-Bench scene (scaffold).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--scene_id", type=str, default="1_office",
                    help="HUGE scene id (1_office | 2_park | 3_campus | 4_lake).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_app_launcher = AppLauncher(args_cli)
_simulation_app = _app_launcher.app

# Phase 1: sys.modules patch — make `vla.vla_drone_env` resolve to the HUGE env.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_PROJECT = os.path.dirname(_HERE)
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla.vla_drone_env                      # noqa: F401 — load originals
import vla.agents.rsl_rl_ppo_cfg              # noqa: F401

import vla_huge.vla_huge_env as _huge_env_mod
import vla_huge.agents.rsl_rl_ppo_cfg as _huge_ppo_mod
sys.modules["vla.vla_drone_env"] = _huge_env_mod
sys.modules["vla.agents.rsl_rl_ppo_cfg"] = _huge_ppo_mod

# Phase 2: redirect gym.make to the HUGE task id.
import gymnasium as gym
import vla_huge  # noqa: F401 — registers Isaac-VLADrone-Huge-v0

_real_gym_make = gym.make


def _patched_gym_make(task_id, **kwargs):
    if task_id == "Isaac-VLADrone-Direct-v0":
        task_id = "Isaac-VLADrone-Huge-v0"
        cfg = kwargs.get("cfg", None)
        if cfg is not None and hasattr(cfg, "scene_id"):
            cfg.scene_id = args_cli.scene_id
    return _real_gym_make(task_id, **kwargs)


gym.make = _patched_gym_make

# Phase 3: neutralize the AppLauncher inside vla.train (we already booted one).
import isaaclab.app

_OriginalAppLauncher = isaaclab.app.AppLauncher


class _NoopAppLauncher:
    def __init__(self, *_args, **_kwargs):
        self.app = _simulation_app

    @staticmethod
    def add_app_launcher_args(parser):
        _OriginalAppLauncher.add_app_launcher_args(parser)


isaaclab.app.AppLauncher = _NoopAppLauncher

# Phase 4: defer to vla.train.
import vla.train  # noqa: F401


if __name__ == "__main__":
    vla.train.main()
    _simulation_app.close()
