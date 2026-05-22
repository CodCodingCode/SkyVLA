"""Gymnasium-compatible environments wrapping OpenFly simulators.

Importing this module also registers ``OpenFly-AirSim-VLN-v0`` with
``gymnasium``'s registry when ``gymnasium`` is installed. The env class
itself can still be instantiated directly without going through
``gym.make`` if ``gymnasium`` is unavailable — useful for unit tests on
machines without AirSim.
"""

from __future__ import annotations

from openfly.envs.airsim_vln_env import AirSimVLNEnv, AirSimVLNEnvConfig

try:
    import gymnasium as _gym

    _gym.register(
        id="OpenFly-AirSim-VLN-v0",
        entry_point="openfly.envs.airsim_vln_env:AirSimVLNEnv",
    )
except ImportError:  # gymnasium optional at import time
    pass

__all__ = ["AirSimVLNEnv", "AirSimVLNEnvConfig"]
