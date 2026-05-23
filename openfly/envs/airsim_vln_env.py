"""Gymnasium environment around the OpenFly AirSim bridge.

The env replays the same loop the eval harness uses (``get_camera_data``
→ policy → ``apply_action`` → ``set_camera_pose``) but exposes it as a
standard ``gymnasium.Env`` so DAgger, GRPO, and PPO trainers can talk to
it with the usual ``reset``/``step``/``close`` contract.

One bridge maps to one ``AirSimVLNEnv`` instance. The class is **not**
thread-safe; vectorisation should use process-level parallelism. AirSim
is CPU-bound and the OpenFly platform launches a heavyweight UE scene
per bridge, so in practice you should run one or two envs per node.

Why teleportation, not physics control? OpenFly's official benchmark
uses kinematic pose updates (``simSetVehiclePose``) — we keep the same
behaviour so RL policies trained here stay comparable on the eval
harness. A future ``AirSimVLNVelocityEnv`` could expose continuous
controls for physics-based RL.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:  # pragma: no cover — gymnasium optional at import time
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYM = False

from openfly.actions import (
    ACTION_NAMES,
    apply_action,
    distance3d,
    success_within,
)
from openfly.episodes import group_by_env, load_episodes
from openfly.rewards import (
    DEFAULT_REWARD,
    RewardConfig,
    compute_episode_reward,
    compute_step_progress,
    get_reward_preset,
)


@dataclass
class AirSimVLNEnvConfig:
    """Configuration for :class:`AirSimVLNEnv`.

    Defaults mirror the eval harness: 100-step budget, 20 m success
    radius, single AirSim scene (``env_airsim_16``). The reward is
    sparse by default (episode-only) but ``dense_progress`` flips on
    the step-level shaping term in :mod:`openfly.rewards`.
    """

    split: str = "train"
    env_filter: str | None = "env_airsim_16"
    max_episodes: int = 0
    max_steps: int = 100
    image_size: int = 224
    history_frames: int = 2
    success_dist: float = 20.0
    reset_sleep_s: float = 0.2
    bridge_init_sleep_s: float = 3.0
    dense_progress: bool = False
    seed: int | None = None
    reward_config: RewardConfig = field(default_factory=lambda: DEFAULT_REWARD)
    # Curriculum knob: when set, looks up the preset in
    # ``openfly.rewards.REWARD_PRESETS`` and overrides ``reward_config``.
    # ``dense_progress`` is forced True for the ``easy`` preset and False
    # for ``medium`` / ``hard`` so the two knobs stay consistent.
    reward_preset: str | None = None

    def __post_init__(self) -> None:
        if self.reward_preset is not None:
            self.reward_config = get_reward_preset(self.reward_preset)
            self.dense_progress = self.reward_preset.lower() == "easy"


_AIRSIM_ENV_REGISTRY: dict[str, "AirSimVLNEnv"] = {}


class AirSimVLNEnv(gym.Env if _HAS_GYM else object):  # type: ignore[misc]
    """OpenFly VLN as a Gymnasium environment.

    Observation
    -----------
    Dict-space:

    * ``rgb``     — ``(H, W, 3)`` uint8 first-person view
    * ``rgb_history`` — ``(history_frames, H, W, 3)`` uint8 padded with
      copies of the first frame on reset
    * ``pose``    — ``(4,)`` float32 ``[x, y, z, yaw]``
    * ``goal``    — ``(3,)`` float32 goal xyz (constant during episode)
    * ``step_idx`` — int32 step counter
    * ``last_action`` — int64 previous action id (``-1`` on reset)

    The ``instruction`` text and per-episode metadata live in the
    ``info`` dict because Gymnasium spaces do not represent strings
    well; both DAgger and the RL trainers grab them from there.

    Action
    ------
    Discrete(10) — the OpenFly macro action ids in
    :data:`openfly.actions.ACTION_NAMES`.

    Reward
    ------
    Sparse episode reward from :func:`openfly.rewards.compute_episode_reward`
    delivered on the terminating step. With ``dense_progress=True`` the
    step reward also includes the small progress shaping term.
    """

    metadata = {"render_modes": [], "name": "OpenFly-AirSim-VLN"}

    def __init__(
        self,
        config: AirSimVLNEnvConfig | None = None,
        *,
        episodes: list[dict[str, Any]] | None = None,
        bridge: Any | None = None,
        eval_mod: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if not _HAS_GYM:
            raise RuntimeError(
                "gymnasium is required; install via `pip install gymnasium>=1.0`"
            )
        config = config or AirSimVLNEnvConfig()
        if kwargs:
            updates: dict[str, Any] = {}
            for k, v in kwargs.items():
                if not hasattr(config, k):
                    raise TypeError(f"Unknown AirSimVLNEnvConfig field: {k}")
                updates[k] = v
            config = AirSimVLNEnvConfig(**{**config.__dict__, **updates})
        self.cfg = config

        if episodes is None:
            episodes = load_episodes(
                config.split,
                max_episodes=config.max_episodes,
                env_filter=config.env_filter,
            )
        if not episodes:
            raise RuntimeError(
                f"No episodes found for split={config.split!r} "
                f"env_filter={config.env_filter!r}"
            )
        self._episodes = episodes
        self._groups = group_by_env(episodes)
        self._env_names = list(self._groups.keys())
        if len(self._env_names) > 1:
            print(
                "[openfly.env] WARN: multiple AirSim scenes in episode pool; "
                "switching scenes requires bridge restart and is slow. "
                "Filter to a single scene with env_filter."
            )

        self._np_random = np.random.default_rng(config.seed)
        self._bridge = bridge
        self._eval_mod = eval_mod
        self._current_env_name: str | None = None
        self._pos_ratio: float = 1.0

        # Episode state (set in reset)
        self._episode: dict[str, Any] | None = None
        self._pose: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._start_pose: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._goal: list[float] = [0.0, 0.0, 0.0]
        self._pitch: float = 0.0
        self._step_idx: int = 0
        self._last_action: int = -1
        self._trajectory: list[list[float]] = []
        self._action_history: list[int] = []
        self._frame_history: list[np.ndarray] = []
        self._last_rgb: np.ndarray = np.zeros(
            (config.image_size, config.image_size, 3), dtype=np.uint8
        )

        H = W = config.image_size
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
                "rgb_history": spaces.Box(
                    low=0,
                    high=255,
                    shape=(config.history_frames, H, W, 3),
                    dtype=np.uint8,
                ),
                "pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
                "goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "step_idx": spaces.Box(
                    low=0, high=np.iinfo(np.int32).max, shape=(), dtype=np.int32
                ),
                "last_action": spaces.Box(
                    low=-1, high=len(ACTION_NAMES) - 1, shape=(), dtype=np.int64
                ),
            }
        )
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

    # --- bridge lifecycle ------------------------------------------------

    def _ensure_bridge(self, env_name: str) -> None:
        """Lazy-construct the AirSim bridge for the requested scene.

        If a bridge was pre-supplied at construction time (e.g. a mock
        for testing, or a hand-built ``AirsimBridge`` so the caller
        controls scene launch lifecycle), it is honoured on the first
        reset and only torn down when the scene actually changes.
        """
        if self._bridge is not None and self._current_env_name is None:
            # Pre-supplied bridge: adopt it for this scene without replacing.
            self._current_env_name = env_name
            _AIRSIM_ENV_REGISTRY[env_name] = self
            return
        if self._bridge is not None and self._current_env_name == env_name:
            return
        if self._bridge is not None and self._current_env_name != env_name:
            self._teardown_bridge()

        from openfly.platform import load_eval_module, make_bridge, openfly_root

        if self._eval_mod is None:
            self._eval_mod = load_eval_module()
        os.chdir(openfly_root() / "train")
        time.sleep(self.cfg.bridge_init_sleep_s)
        bridge, pos_ratio = make_bridge(env_name, self._eval_mod)
        self._bridge = bridge
        self._pos_ratio = pos_ratio
        self._current_env_name = env_name
        _AIRSIM_ENV_REGISTRY[env_name] = self

    def _teardown_bridge(self) -> None:
        if self._eval_mod is None or self._current_env_name is None:
            self._bridge = None
            self._current_env_name = None
            return
        try:
            keywords = ["AirVLN", self._current_env_name]
            for kw in keywords:
                self._eval_mod.kill_env_process(kw)
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            print(f"[openfly.env] bridge teardown warning: {exc}")
        finally:
            _AIRSIM_ENV_REGISTRY.pop(self._current_env_name, None)
            self._bridge = None
            self._current_env_name = None

    # --- helpers ---------------------------------------------------------

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        H = W = self.cfg.image_size
        if rgb.shape[:2] == (H, W):
            return rgb.astype(np.uint8, copy=False)
        # Lazy import — cv2 is already a project dep but keep envs cheap.
        import cv2

        return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def _set_pose(self, pose: list[float]) -> None:
        assert self._bridge is not None
        self._bridge.set_camera_pose(
            pose[0] / self._pos_ratio,
            pose[1] / self._pos_ratio,
            pose[2] / self._pos_ratio,
            self._pitch,
            math.degrees(pose[3]),
            0,
        )

    def _grab_rgb(self) -> np.ndarray:
        assert self._bridge is not None
        rgb = self._bridge.get_camera_data()
        return self._resize(np.asarray(rgb))

    def _build_obs(self) -> dict[str, np.ndarray]:
        if self.cfg.history_frames > 0:
            while len(self._frame_history) < self.cfg.history_frames:
                self._frame_history.append(self._last_rgb)
            history = np.stack(
                self._frame_history[-self.cfg.history_frames :], axis=0
            )
        else:
            history = np.zeros(
                (0, self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8
            )
        return {
            "rgb": self._last_rgb,
            "rgb_history": history,
            "pose": np.asarray(self._pose, dtype=np.float32),
            "goal": np.asarray(self._goal, dtype=np.float32),
            "step_idx": np.int32(self._step_idx),
            "last_action": np.int64(self._last_action),
        }

    def _pitch_for(self, image_path: str) -> float:
        return -45.0 if "high" in image_path else 0.0

    def _pick_episode(self) -> dict[str, Any]:
        idx = int(self._np_random.integers(0, len(self._episodes)))
        return self._episodes[idx]

    # --- gym API ---------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        episode = (options or {}).get("episode") if options else None
        if episode is None:
            episode = self._pick_episode()
        self._episode = episode

        pos_list = episode["pos"]
        start_xyz = pos_list[0]
        goal_xyz = pos_list[-1]
        yaw0 = float(episode["yaw"][0])
        self._pose = [start_xyz[0], start_xyz[1], start_xyz[2], yaw0]
        self._start_pose = list(self._pose)
        self._goal = list(goal_xyz)
        self._pitch = self._pitch_for(episode["image_path"])

        self._step_idx = 0
        self._last_action = -1
        self._trajectory = [list(self._pose)]
        self._action_history = []
        self._frame_history = []

        env_name = episode["image_path"].split("/")[0]
        self._ensure_bridge(env_name)
        try:
            self._set_pose(self._pose)
            time.sleep(self.cfg.reset_sleep_s)
            self._last_rgb = self._grab_rgb()
        except Exception as exc:
            # Sim hiccup on reset — surface to caller with a black frame so
            # they can decide whether to retry.
            print(f"[openfly.env] reset error: {exc}", flush=True)
            self._last_rgb = np.zeros(
                (self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8
            )

        info: dict[str, Any] = {
            "instruction": episode.get("gpt_instruction", ""),
            "image_path": episode.get("image_path", ""),
            "env_name": env_name,
            "start": list(self._start_pose),
            "goal": list(self._goal),
            "optimal_len": max(distance3d(self._start_pose, self._goal), 1e-3),
            "expert_actions": list(episode.get("action", [])),
        }
        return self._build_obs(), info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("step() called before reset()")
        action_id = int(action)
        if action_id not in ACTION_NAMES:
            raise ValueError(f"Invalid action {action_id}")

        prev_pose = list(self._pose)
        new_pose = apply_action(self._pose, action_id)
        info: dict[str, Any] = {
            "instruction": self._episode.get("gpt_instruction", ""),
            "image_path": self._episode.get("image_path", ""),
            "env_name": self._current_env_name or "",
            "action_name": ACTION_NAMES[action_id],
        }

        collided = False
        terminated = False
        truncated = False
        reward = 0.0

        try:
            self._set_pose(new_pose)
            self._pose = new_pose
            self._trajectory.append(list(new_pose))
            self._action_history.append(action_id)
            self._last_action = action_id
            self._step_idx += 1
            if self._frame_history is not None and self.cfg.history_frames > 0:
                self._frame_history.append(self._last_rgb)
                if len(self._frame_history) > self.cfg.history_frames:
                    self._frame_history = self._frame_history[
                        -self.cfg.history_frames :
                    ]
            self._last_rgb = self._grab_rgb()
        except Exception as exc:
            print(f"[openfly.env] step error: {exc}", flush=True)
            collided = True
            terminated = True

        stopped = action_id == 0
        if stopped:
            terminated = True
        if self._step_idx >= self.cfg.max_steps:
            truncated = True

        if self.cfg.dense_progress and not terminated:
            reward += compute_step_progress(
                prev_pose, self._pose, self._goal, config=self.cfg.reward_config
            )

        d_to_goal = distance3d(self._goal, self._pose)
        info.update(
            {
                "distance_to_goal": d_to_goal,
                "pass_len": self._traj_length(),
                "osr_flag": int(d_to_goal < self.cfg.success_dist),
                "step_idx": self._step_idx,
                "collided": collided,
            }
        )

        if terminated or truncated:
            ep_info = compute_episode_reward(
                trajectory_positions=self._trajectory,
                start=self._start_pose,
                goal=self._goal,
                stopped=stopped,
                collided=collided,
                timed_out=truncated and not stopped,
                config=self.cfg.reward_config,
            )
            reward += ep_info["reward"]
            info.update(ep_info)
            info["expert_actions"] = list(
                self._episode.get("action", [])
            ) if self._episode else []
            info["trajectory"] = list(self._trajectory)
            info["action_history"] = list(self._action_history)

        return self._build_obs(), float(reward), terminated, truncated, info

    def _traj_length(self) -> float:
        total = 0.0
        for a, b in zip(self._trajectory[:-1], self._trajectory[1:]):
            total += distance3d(a, b)
        return total

    def close(self) -> None:
        self._teardown_bridge()


__all__ = ["AirSimVLNEnv", "AirSimVLNEnvConfig"]
