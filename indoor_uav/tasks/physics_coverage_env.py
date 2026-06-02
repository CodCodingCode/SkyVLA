"""Gymnasium env: a PHYSICS drone explores a scene to maximize coverage.

Combines everything we built:
  * DronePhysics  — Bullet rigid body (mass, gravity, real wall collisions);
    the policy outputs VELOCITY commands, the controller + physics fly it.
  * CoverageMap   — top-down navigable/visited/frontier memory: the agent's
    knowledge of what it has and hasn't seen (so it can find the upstairs and
    not get stuck in one room).
  * onboard RGB-D + GS-style coverage reward.

Observation (Dict) — the (B) design (RGB + map memory + frontier + proprio):
  rgb   : (obs_res, obs_res, 3) uint8       — current onboard view
  map   : (3, map_res, map_res) float32     — navigable / visited / frontier
  state : (8,) float32 — pose(x,y,z norm), sin/cos yaw, vel(x,z), coverage

Action (Discrete 6): forward / back / yaw L / yaw R / up / down — issued as
VELOCITY commands to the physics controller (not teleports).

Reward = newly-covered navigable area this step (dense coverage gain)
       + small frontier-progress bonus (closer to nearest unexplored boundary)
       - collision penalty (it bumped a wall)  - small time cost.
Termination: coverage >= target, or no frontier left; truncate at max_steps.

NOTE: requires the habitat env (habitat_sim) AND, for RL, torch in the same
process. We run this from the habitat env with torch installed there (see
docs); the env itself is torch-free except the tiny rgb resize (done in numpy).
"""

from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("indoor_uav.tasks needs gymnasium.") from exc

from .coverage_map import CoverageMap

FWD, BACK, YAW_L, YAW_R, UP, DOWN = range(6)


class PhysicsCoverageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_glb: str,
        *,
        sim_res: int = 160,
        obs_res: int = 64,
        map_res: int = 96,
        hfov: float = 90.0,
        max_steps: int = 256,
        speed: float = 1.0,
        yaw_rate: float = 1.2,
        ctrl_dt: float = 1.0 / 15.0,
        target_coverage: float = 0.90,
        collision_penalty: float = 0.1,
        time_cost: float = 0.001,
        frontier_bonus: float = 0.05,
        reward_mode: str = "geometric",   # "geometric" (floor-grid) | "gs" (splat reconstruction)
        gs_stride: int = 6,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        import habitat_sim
        from indoor_uav.sim.drone_body import DronePhysics

        self._hs = habitat_sim
        bk = habitat_sim.SimulatorConfiguration()
        bk.scene_id = scene_glb
        bk.enable_physics = True   # rigid-body + real collisions
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [sim_res, sim_res]; rgb.hfov = hfov
        dep = habitat_sim.CameraSensorSpec()
        dep.uuid = "depth"; dep.sensor_type = habitat_sim.SensorType.DEPTH
        dep.resolution = [sim_res, sim_res]; dep.hfov = hfov
        ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb, dep]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))
        pf = self.sim.pathfinder
        ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
        ns.agent_radius = 0.25; ns.agent_height = 0.3; ns.agent_max_climb = 2.0; ns.cell_size = 0.07
        self.sim.recompute_navmesh(pf, ns)

        self.obs_res, self.map_res, self.sim_res = obs_res, map_res, sim_res
        self.max_steps = max_steps
        self.speed, self.yaw_rate, self.ctrl_dt = speed, yaw_rate, ctrl_dt
        self.target_coverage = target_coverage
        self.collision_penalty = collision_penalty
        self.time_cost = time_cost
        self.frontier_bonus = frontier_bonus
        self.reward_mode = reward_mode
        self.gs_stride = gs_stride
        self._rng = np.random.default_rng(seed)
        # GS reconstruction reward: maintain a GaussianMap of the scene the drone
        # is building from its RGB-D; reward = improvement in reconstruction.
        self._gmap = None
        self._K = None
        if reward_mode == "gs":
            import torch
            from indoor_uav.gs import GaussianMap
            self._torch = torch
            self._GaussianMap = GaussianMap
            self._gs_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            f = 0.5 * sim_res / np.tan(np.deg2rad(hfov) / 2)
            self._K = torch.tensor([[f, 0, sim_res / 2], [0, f, sim_res / 2], [0, 0, 1]],
                                   dtype=torch.float32, device=self._gs_dev)

        self.drone = DronePhysics(self.sim, mass=0.5, max_speed=max(speed * 1.5, 1.5))
        self._lo, self._hi = (np.array(x, np.float32) for x in pf.get_bounds())

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (obs_res, obs_res, 3), dtype=np.uint8),
            "map": spaces.Box(0.0, 1.0, (3, map_res, map_res), dtype=np.float32),
            "state": spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        })
        self._cov = None
        self._t = 0
        self._floor_y = 0.0
        self._prev_cov = 0.0

    # ------------------------------------------------------------------ #
    def _spawn(self):
        pf = self.sim.pathfinder
        ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(200)], np.float32)
        h, e = np.histogram(ys, 30); lo = e[int(h.argmax())]
        self._floor_y = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
        for _ in range(200):
            p = np.array(pf.get_random_navigable_point(), np.float32)
            if abs(float(p[1]) - self._floor_y) < 0.5:
                p[1] = self._floor_y + 1.0
                return p
        return np.array(pf.get_random_navigable_point(), np.float32) + np.array([0, 1, 0], np.float32)

    def _render(self):
        self.sim.get_agent(0).set_state(self.drone.camera_state())
        obs = self.sim.get_sensor_observations()
        return obs["rgb"][..., :3], obs["depth"]

    def _resize_rgb(self, rgb):
        if rgb.shape[0] == self.obs_res:
            return rgb.astype(np.uint8)
        idx = np.linspace(0, rgb.shape[0] - 1, self.obs_res).astype(int)
        return rgb[np.ix_(idx, idx)].astype(np.uint8)

    def _gs_pose_c2w(self):
        """Camera-to-world (OpenCV) for the GaussianMap, from drone pose+yaw."""
        import torch
        yaw = self.drone.yaw
        c, s = math.cos(yaw), math.sin(yaw)
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=self._gs_dev).float()
        # habitat (-z fwd, +y up) -> OpenCV (+z fwd, +y down): flip Y and Z, matching
        # _HAB2CV. (Was diag(1,-1,1), an improper reflection that mirrored the cloud
        # per frame and corrupted the cross-frame coverage reward.)
        flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=self._gs_dev).float()
        T = torch.eye(4, device=self._gs_dev)
        T[:3, :3] = R @ flip
        T[:3, 3] = torch.tensor(self.drone.position, device=self._gs_dev).float()
        return T

    def _gs_reconstruction_gain(self, rgb, depth):
        """Splat this RGB-D into the scene GaussianMap; reward = NEW reconstructed
        surface this view added (rendered alpha over real geometry, measured
        BEFORE adding so re-seeing earns ~0). This makes the reward 'improve the
        Gaussian-splat reconstruction', not 'cover the floor grid'."""
        from indoor_uav.gs.coverage import exploration_gain
        torch = self._torch
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self._gs_dev).float() / 255.0
        depth_t = torch.from_numpy(np.ascontiguousarray(depth)).to(self._gs_dev).float()
        pose = self._gs_pose_c2w()
        gmask = depth_t > 1e-3
        gain = exploration_gain(self._gmap, pose, self._K, depth_t.shape[1], depth_t.shape[0],
                                geometry_mask=gmask)
        self._gmap.add_from_rgbd(rgb_t, depth_t, pose, self._K,
                                 stride=self.gs_stride, max_depth=10.0)
        # weight by how much of the view actually had surface (avoid rewarding void)
        return float(gain) * float(gmask.float().mean().item())

    def _obs(self, rgb):
        p = self.drone.position; v = self.drone.velocity
        span = (self._hi - self._lo); span[span < 1e-3] = 1.0
        pn = np.clip(2 * (p - self._lo) / span - 1, -1, 1)
        state = np.array([
            pn[0], pn[1], pn[2],
            math.sin(self.drone.yaw), math.cos(self.drone.yaw),
            float(np.clip(v[0] / self.speed, -1, 1)), float(np.clip(v[2] / self.speed, -1, 1)),
            2 * self._cov.coverage_fraction() - 1,
        ], np.float32)
        return {"rgb": self._resize_rgb(rgb), "map": self._cov.observation(self.map_res), "state": state}

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        p = self._spawn()
        self.drone.reset(p, yaw=float(self._rng.uniform(0, 2 * math.pi)))
        self._cov = CoverageMap(self._lo, self._hi, res=self.map_res)
        self._cov.set_navigable_from_pathfinder(self.sim.pathfinder, height=self._floor_y)
        self._cov.mark_visited(p)
        self._t = 0
        self._prev_cov = self._cov.coverage_fraction()
        rgb, depth = self._render()
        if self.reward_mode == "gs":
            self._gmap = self._GaussianMap(device=self._gs_dev)
            self._gs_reconstruction_gain(rgb, depth)  # seed with start view (no reward)
        return self._obs(rgb), {}

    def step(self, action):
        a = int(action)
        vx = vz = vy = yr = 0.0
        if a == FWD:   vz = self.speed
        elif a == BACK: vz = -self.speed
        elif a == YAW_L: yr = self.yaw_rate
        elif a == YAW_R: yr = -self.yaw_rate
        elif a == UP:   vy = self.speed
        elif a == DOWN: vy = -self.speed
        tel = self.drone.step(vx_body=vx, vz_body=vz, vy=vy, yaw_rate=yr, dt=self.ctrl_dt)
        self._t += 1

        self._cov.mark_visited(self.drone.position)
        cov = self._cov.coverage_fraction()
        df = self._cov.dist_to_frontier(self.drone.position)  # frontier shaping (both modes)
        rgb, depth = self._render()

        if self.reward_mode == "gs":
            # (a) reward = improvement of the Gaussian-splat RECONSTRUCTION
            gain = self._gs_reconstruction_gain(rgb, depth)
            reward = (gain * 20.0
                      + self.frontier_bonus * (1.0 - df)
                      - (self.collision_penalty if tel["collided"] else 0.0)
                      - self.time_cost)
        else:
            # geometric floor-grid coverage (placeholder / fast baseline)
            gain = max(0.0, cov - self._prev_cov)
            reward = (gain * 100.0
                      + self.frontier_bonus * (1.0 - df)
                      - (self.collision_penalty if tel["collided"] else 0.0)
                      - self.time_cost)
        self._prev_cov = cov
        terminated = cov >= self.target_coverage or (self._cov.frontier().sum() == 0)
        truncated = self._t >= self.max_steps
        info = {"coverage": cov, "gain": gain, "collided": tel["collided"],
                "dist_frontier": df, "pos": self.drone.position.tolist()}
        return self._obs(rgb), float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()
