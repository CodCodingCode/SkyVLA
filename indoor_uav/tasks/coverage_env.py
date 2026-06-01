"""Gymnasium env: explore an indoor scene to maximize GS reconstruction coverage.

A thin wrapper over an :class:`~indoor_uav.sim.base.IndoorSim` backend
(SyntheticRoom now, HabitatRoom later) + an incremental
:class:`~indoor_uav.gs.GaussianMap`. The drone is a 6-DOF camera; each step it
moves (collision-checked against the sim's free-space query), renders RGB-D,
and the reward is the **GS exploration gain** of the new view — i.e. how much
previously-unreconstructed surface the move revealed. The frame is then splatted
into the map, so the same area can't be rewarded twice. This is the cheap,
analytic Gaussian-Splatting reward (a forward rasterize, no GS fitting in-loop).

Action space (Discrete 6): forward / backward / yaw-left / yaw-right /
ascend / descend. Translations are clipped to free space (collide-and-stay,
with a small bump penalty) — the drone cannot enter geometry.

Observation (Dict):
  * rgb   : (obs_res, obs_res, 3) uint8 — what the drone sees
  * state : (6,) float32 — [x,y,z normalised, sin(yaw), cos(yaw), coverage]

Episode ends after ``max_steps`` (pure exploration). A nav-to-landing variant
(terminate on reaching a target pose) is a small extension of ``_terminated``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("indoor_uav.tasks needs gymnasium (`pip install gymnasium`).") from exc

from indoor_uav.gs import GaussianMap, exploration_gain
from indoor_uav.sim.base import IndoorSim

# Action ids.
FWD, BACK, YAW_L, YAW_R, UP, DOWN = range(6)


class GSCoverageEnv(gym.Env):
    """Explore-for-reconstruction task over a single IndoorSim scene."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim: IndoorSim,
        *,
        max_steps: int = 64,
        step_m: float = 0.5,
        yaw_deg: float = 20.0,
        obs_res: int = 64,
        bump_penalty: float = 0.05,
        gs_stride: int = 4,
        max_depth: float = 10.0,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sim = sim
        self.device = device or sim.device
        self.max_steps = int(max_steps)
        self.step_m = float(step_m)
        self.yaw = float(yaw_deg) * math.pi / 180.0
        self.obs_res = int(obs_res)
        self.bump_penalty = float(bump_penalty)
        self.gs_stride = int(gs_stride)
        self.max_depth = float(max_depth)
        self._rng = np.random.default_rng(seed)

        self._K = sim.intrinsics()
        self._lo, self._hi = sim.bounds()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (obs_res, obs_res, 3), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
        })

        self._gmap: GaussianMap | None = None
        self._pos: torch.Tensor | None = None
        self._yaw_rad = 0.0
        self._t = 0
        self._coverage = 0.0

    # ------------------------------------------------------------------ #
    def _pose_c2w(self) -> torch.Tensor:
        """Build a camera-to-world matrix from (pos, yaw). OpenCV cam: +z fwd, +y down."""
        c, s = math.cos(self._yaw_rad), math.sin(self._yaw_rad)
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=self.device).float()
        flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=self.device).float()
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R @ flip
        T[:3, 3] = self._pos
        return T

    def _random_free_pos(self) -> torch.Tensor:
        lo, hi = self._lo, self._hi
        for _ in range(200):
            p = torch.tensor(self._rng.uniform(lo.cpu().numpy(), hi.cpu().numpy()),
                             device=self.device).float()
            if self.sim.is_free(p):
                return p
        return ((lo + hi) * 0.5).to(self.device).float()  # fallback: scene centre

    def _obs(self, frame) -> dict[str, np.ndarray]:
        # downsample rgb (H,W,3 float[0,1]) -> obs_res uint8
        rgb = frame.rgb
        if rgb.shape[0] != self.obs_res:
            rgb_t = rgb.permute(2, 0, 1).unsqueeze(0)
            rgb_t = torch.nn.functional.interpolate(
                rgb_t, size=(self.obs_res, self.obs_res), mode="bilinear", align_corners=False
            )
            rgb = rgb_t.squeeze(0).permute(1, 2, 0)
        rgb_u8 = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        span = (self._hi - self._lo).clamp(min=1e-3)
        pn = (2 * (self._pos - self._lo) / span - 1).cpu().numpy()  # [-1,1]
        state = np.array(
            [pn[0], pn[1], pn[2], math.sin(self._yaw_rad), math.cos(self._yaw_rad),
             2 * self._coverage - 1], dtype=np.float32,
        )
        return {"rgb": rgb_u8, "state": state}

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._gmap = GaussianMap(device=self.device)
        self._pos = self._random_free_pos()
        self._yaw_rad = float(self._rng.uniform(0, 2 * math.pi))
        self._t = 0
        self._coverage = 0.0
        frame = self.sim.render(self._pose_c2w())
        # seed the map with the starting view (no reward for the freebie)
        self._gmap.add_from_rgbd(frame.rgb, frame.depth, frame.pose_c2w, self._K,
                                 stride=self.gs_stride, max_depth=self.max_depth)
        return self._obs(frame), {}

    def _propose(self, action: int):
        """Return (new_pos, new_yaw, moved_into_wall)."""
        pos = self._pos.clone()
        yaw = self._yaw_rad
        if action in (YAW_L, YAW_R):
            yaw = yaw + (self.yaw if action == YAW_L else -self.yaw)
            return pos, yaw, False
        if action in (UP, DOWN):
            delta = torch.tensor([0, (self.step_m if action == UP else -self.step_m), 0],
                                 device=self.device).float()
        else:  # FWD / BACK along current heading (x-z plane)
            sign = 1.0 if action == FWD else -1.0
            delta = torch.tensor(
                [sign * self.step_m * math.sin(yaw), 0, sign * self.step_m * math.cos(yaw)],
                device=self.device).float()
        cand = pos + delta
        if self.sim.is_free(cand):
            return cand, yaw, False
        return pos, yaw, True  # blocked: stay put

    def step(self, action: int):
        action = int(action)
        new_pos, new_yaw, bumped = self._propose(action)
        self._pos, self._yaw_rad = new_pos, new_yaw
        self._t += 1

        frame = self.sim.render(self._pose_c2w())
        # geometry mask: only reward views that actually contain surface
        gmask = frame.depth > 1e-3
        # reward = NEW surface this view reveals, measured against the map BEFORE adding
        gain = exploration_gain(self._gmap, frame.pose_c2w, self._K,
                                self.sim.width, self.sim.height, geometry_mask=gmask)
        self._gmap.add_from_rgbd(frame.rgb, frame.depth, frame.pose_c2w, self._K,
                                 stride=self.gs_stride, max_depth=self.max_depth)
        self._coverage = min(1.0, self._coverage + gain * float(gmask.float().mean().item()))

        reward = float(gain) - (self.bump_penalty if bumped else 0.0)
        terminated = False  # pure-exploration variant; nav-to-landing sets this
        truncated = self._t >= self.max_steps
        info = {"gain": float(gain), "bumped": bumped,
                "gaussians": self._gmap.num_gaussians, "coverage": self._coverage}
        return self._obs(frame), reward, terminated, truncated, info
