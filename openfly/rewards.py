"""Episode-level reward functions for OpenFly online RL.

The primary reward is aligned with the OpenFly benchmark metrics
(success rate, SPL, and final navigation error). The shape mirrors the
ActiveVLN soft-success and ETP-R1 SPL/NE terms used in recent VLN-RL
work, but evaluated on OpenFly's 20 m success radius and the discrete
10-class macro action space defined in :mod:`openfly.actions`.

The dense step reward is optional. For GRPO we score whole trajectories
with :func:`compute_episode_reward` and let the group-relative baseline
absorb variance; for PPO we additionally use step-level progress shaping
:func:`compute_step_progress` so the value function can learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openfly.actions import distance3d, success_within


@dataclass(frozen=True)
class RewardConfig:
    """Tunable constants for the OpenFly reward.

    Defaults follow the values used during plan synthesis. The success
    radius matches the OpenFly eval harness; the SPL bonus is scaled so
    a path-optimal episode scores ~+16 while an episode that lands just
    inside the success radius scores ~+15.
    """

    success_dist: float = 20.0
    success_scale: float = 15.0
    spl_scale: float = 1.0
    overrun_penalty: float = 0.25
    ne_scale: float = 1.0 / 40.0
    progress_scale: float = 0.1
    collision_penalty: float = 0.5
    timeout_penalty: float = 0.0


DEFAULT_REWARD = RewardConfig()


def _path_length(positions: Sequence[Sequence[float]]) -> float:
    """Sum of segment lengths along a sequence of (x, y, z) poses."""
    total = 0.0
    for a, b in zip(positions[:-1], positions[1:]):
        total += distance3d(a, b)
    return total


def compute_episode_reward(
    *,
    trajectory_positions: Sequence[Sequence[float]],
    start: Sequence[float],
    goal: Sequence[float],
    stopped: bool,
    collided: bool = False,
    timed_out: bool = False,
    config: RewardConfig = DEFAULT_REWARD,
) -> dict[str, float]:
    """Score one episode using OpenFly-aligned terms.

    Args:
        trajectory_positions: Visited poses in order (first must equal
            ``start``). Length must be at least 1.
        start: Episode start pose (used for the SPL optimal-path proxy).
        goal: Episode goal pose.
        stopped: True if the agent emitted action 0 (stop) before the
            step budget ran out. Required for "success".
        collided: True if the rollout terminated on a sim crash or pose
            recovery failure (drains a small penalty).
        timed_out: True if the step budget elapsed without a stop.
        config: Reward weights.

    Returns:
        Dict with the total ``reward`` plus the named term contributions
        (``success_term``, ``spl_term``, ``overrun_term``, ``ne_term``,
        ``collision_term``, ``timeout_term``) and bookkeeping fields
        ``success``, ``d_final``, ``path_len``, ``optimal_len``.
    """
    if not trajectory_positions:
        raise ValueError("trajectory_positions must contain at least one pose")

    final = trajectory_positions[-1]
    d_final = distance3d(goal, final)
    success = bool(stopped and success_within(final, goal, config.success_dist))

    optimal_len = max(distance3d(start, goal), 1e-3)
    path_len = max(_path_length(trajectory_positions), 1e-3)

    success_term = 0.0
    spl_term = 0.0
    if success:
        # Soft success: scale by how deep inside the success radius we land,
        # so policies cannot game the binary metric by stopping at the edge.
        depth = max(0.0, 1.0 - d_final / config.success_dist)
        success_term = config.success_scale * depth
        spl_term = config.spl_scale * min(optimal_len / path_len, 1.0)

    overrun = max(0.0, path_len - optimal_len) / optimal_len
    overrun_term = -config.overrun_penalty * overrun

    ne_term = -config.ne_scale * d_final

    collision_term = -config.collision_penalty if collided else 0.0
    timeout_term = -config.timeout_penalty if timed_out and not success else 0.0

    total = (
        success_term
        + spl_term
        + overrun_term
        + ne_term
        + collision_term
        + timeout_term
    )

    return {
        "reward": total,
        "success_term": success_term,
        "spl_term": spl_term,
        "overrun_term": overrun_term,
        "ne_term": ne_term,
        "collision_term": collision_term,
        "timeout_term": timeout_term,
        "success": float(success),
        "d_final": d_final,
        "path_len": path_len,
        "optimal_len": optimal_len,
    }


def compute_step_progress(
    prev_pose: Sequence[float],
    new_pose: Sequence[float],
    goal: Sequence[float],
    config: RewardConfig = DEFAULT_REWARD,
) -> float:
    """Shaping reward: positive when the step shortens distance-to-goal.

    Useful for value-function fitting in PPO. GRPO does not need this.
    """
    before = distance3d(prev_pose, goal)
    after = distance3d(new_pose, goal)
    return config.progress_scale * (before - after)


__all__ = [
    "RewardConfig",
    "DEFAULT_REWARD",
    "compute_episode_reward",
    "compute_step_progress",
]
