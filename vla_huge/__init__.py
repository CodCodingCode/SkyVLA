"""VLA drone — HUGE-Bench digital-twin scenes (Stage 7 scaffold).

Until the HUGE-Bench Isaac Sim toolchain is publicly released, the env
falls back to ``vla_warehouse_full`` so that the resume-from-Stage-6
code path is exercisable today. See ``vla_huge/README.md``.
"""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-VLADrone-Huge-v0",
    entry_point=f"{__name__}.vla_huge_env:VLAHugeDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_huge_env:VLAHugeDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VLAHugePPORunnerCfg",
    },
)
