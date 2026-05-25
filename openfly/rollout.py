"""Shared trajectory collection for GRPO and PPO.

Trainers should never call ``env.step`` directly: route through
:func:`collect_episode`. That keeps reward bookkeeping, OpenFly metric
calculation, and on-policy/expert label collection identical across the
RL pipelines so any improvements (e.g. better dense shaping) flow
to every trainer at once.

The collector is intentionally policy-agnostic: it accepts any callable
``policy_fn(obs, info) -> (action_id, extras)`` where ``extras`` is a
free-form dict (we expect ``logprob`` for GRPO/PPO and ``value`` for
PPO). Trainers wrap their PaliGemma/OpenFly-Agent forward call to
produce both an int action and the per-step logits/values.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from openfly.actions import distance3d, success_within

PolicyFn = Callable[
    [dict[str, np.ndarray], dict[str, Any]],
    tuple[int, dict[str, Any]],
]


@dataclass
class RolloutTrajectory:
    """One rollout. Captures everything the trainers need to update."""

    instruction: str = ""
    image_path: str = ""
    env_name: str = ""
    start: list[float] = field(default_factory=list)
    goal: list[float] = field(default_factory=list)
    optimal_len: float = 0.0

    actions: list[int] = field(default_factory=list)
    expert_actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    step_infos: list[dict[str, Any]] = field(default_factory=list)
    extras: list[dict[str, Any]] = field(default_factory=list)
    poses: list[list[float]] = field(default_factory=list)

    obs_rgb: list[np.ndarray] = field(default_factory=list)
    obs_history: list[np.ndarray] = field(default_factory=list)
    obs_poses: list[np.ndarray] = field(default_factory=list)

    # Episode-level fields populated on termination
    success: bool = False
    osr: bool = False
    ne_m: float = 0.0
    spl: float = 0.0
    pass_len: float = 0.0
    total_reward: float = 0.0
    steps: int = 0
    terminated_collision: bool = False
    truncated: bool = False
    elapsed_s: float = 0.0

    @property
    def episode_reward(self) -> float:
        return self.total_reward

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (drops large image arrays)."""
        return {
            "instruction": self.instruction[:200],
            "image_path": self.image_path,
            "env_name": self.env_name,
            "start": self.start,
            "goal": self.goal,
            "actions": self.actions,
            "expert_actions": self.expert_actions,
            "rewards": self.rewards,
            "poses": self.poses,
            "success": self.success,
            "osr": self.osr,
            "ne_m": self.ne_m,
            "spl": self.spl,
            "pass_len": self.pass_len,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "terminated_collision": self.terminated_collision,
            "truncated": self.truncated,
            "elapsed_s": self.elapsed_s,
        }


def collect_episode(
    env,
    policy_fn: PolicyFn,
    *,
    options: dict[str, Any] | None = None,
    capture_obs: bool = True,
    max_steps: int | None = None,
) -> RolloutTrajectory:
    """Run a single episode and return a :class:`RolloutTrajectory`.

    The environment must obey the :class:`openfly.envs.AirSimVLNEnv`
    info contract (``instruction``, ``goal``, ``distance_to_goal``,
    ``osr_flag``, and the episode-level ``success``/``d_final``/
    ``path_len`` fields on the terminating step).
    """
    t0 = time.time()
    obs, info = env.reset(options=options)

    traj = RolloutTrajectory(
        instruction=info.get("instruction", ""),
        image_path=info.get("image_path", ""),
        env_name=info.get("env_name", ""),
        start=list(info.get("start", [])),
        goal=list(info.get("goal", [])),
        optimal_len=float(info.get("optimal_len", 0.0)),
        expert_actions=list(info.get("expert_actions", [])),
    )
    traj.poses.append(list(obs.get("pose", [0.0, 0.0, 0.0, 0.0]).tolist()))

    osr = False
    step_budget = max_steps if max_steps is not None else getattr(env.cfg, "max_steps", 100)

    for _ in range(step_budget):
        action_id, extras = policy_fn(obs, info)

        if capture_obs:
            traj.obs_rgb.append(np.array(obs["rgb"], copy=True))
            if obs.get("rgb_history") is not None:
                traj.obs_history.append(np.array(obs["rgb_history"], copy=True))
            traj.obs_poses.append(np.array(obs["pose"], copy=True))

        obs, reward, terminated, truncated, info = env.step(int(action_id))
        traj.actions.append(int(action_id))
        traj.rewards.append(float(reward))
        traj.step_infos.append(info)
        traj.extras.append(extras)
        traj.poses.append(list(obs.get("pose", [0.0, 0.0, 0.0, 0.0]).tolist()))
        if info.get("osr_flag"):
            osr = True

        if terminated or truncated:
            traj.success = bool(info.get("success", False))
            traj.osr = osr or traj.success
            traj.ne_m = float(info.get("d_final", info.get("distance_to_goal", 0.0)))
            traj.pass_len = float(info.get("pass_len", 0.0))
            traj.total_reward = float(sum(traj.rewards))
            traj.steps = len(traj.actions)
            traj.terminated_collision = bool(info.get("collided", False))
            traj.truncated = bool(truncated and not traj.success)
            opt = max(traj.optimal_len, 1e-3)
            pass_len = max(traj.pass_len, 1e-3)
            traj.spl = (opt / pass_len) if traj.success else 0.0
            break

    if traj.steps == 0:
        # Hit the step budget without a terminator (very rare given env truncation).
        traj.steps = len(traj.actions)
        traj.total_reward = float(sum(traj.rewards))
    traj.elapsed_s = time.time() - t0
    return traj


def aggregate_metrics(
    trajectories: Sequence[RolloutTrajectory],
) -> dict[str, float]:
    """OpenFly-style summary over a batch of rollouts."""
    n = max(len(trajectories), 1)
    success = sum(int(t.success) for t in trajectories) / n
    osr = sum(int(t.osr) for t in trajectories) / n
    ne = sum(t.ne_m for t in trajectories) / n
    spl = sum(t.spl for t in trajectories) / n
    reward = sum(t.total_reward for t in trajectories) / n
    steps = sum(t.steps for t in trajectories) / n
    elapsed = sum(t.elapsed_s for t in trajectories) / n
    return {
        "n_episodes": float(len(trajectories)),
        "success_rate": success,
        "osr": osr,
        "mean_ne_m": ne,
        "mean_spl": spl,
        "mean_reward": reward,
        "mean_steps": steps,
        "mean_elapsed_s": elapsed,
    }


def save_trajectories_jsonl(
    trajectories: Sequence[RolloutTrajectory],
    path: str | Path,
) -> Path:
    """Append-friendly JSONL dump (one episode per line, no image data)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t.to_dict()) + "\n")
    return path


__all__ = [
    "PolicyFn",
    "RolloutTrajectory",
    "collect_episode",
    "aggregate_metrics",
    "save_trajectories_jsonl",
]
