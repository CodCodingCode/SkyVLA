"""Gymnasium env: a drone with a 2-DoF gripper does indoor PICK-AND-PLACE.

Pure drone-based manipulation — there is no arm; the policy positions the whole
aircraft, extends/closes the gripper, and carries the object to a target zone.

Action (Box, 6, continuous in [-1,1]):
    [vx_body, vz_body, vy, yaw_rate, lower, grip]
  velocities scale to m/s; lower/grip map to [0,1] (grip>0.5 = closed).

Observation (Box, 19, float32) — compact state for fast RL:
    drone pos(3, norm) | sin/cos yaw(2) | drone vel(3) | lower,grip(2) |
    holding(1) | (object-tip)(3) | (target-object)(3) | d_tip_obj,d_obj_tgt(2)
  (RGB is available via ``render()`` for demos; obs is state-based for sample
   efficiency — see header note on switching to pixels.)

Reward (dense, shaped):
    + reach: progress of the gripper tip toward the object   (while not holding)
    + grasp bonus on a successful pick
    + carry: progress of the object toward the target        (while holding)
    + place bonus + SUCCESS when released inside the target zone
    - drop penalty for releasing away from the target
    - small collision + time costs
Termination: success; truncate at max_steps.

Run from the habitat env (needs habitat_sim + gymnasium; torch only for training).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("indoor_uav.tasks needs gymnasium.") from exc


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_glb: str,
        *,
        sim_res: int = 200,
        hfov: float = 90.0,
        max_steps: int = 220,
        speed: float = 1.0,
        yaw_rate: float = 1.5,
        ctrl_dt: float = 1.0 / 20.0,
        altitude: float = 1.3,
        obj_size: float = 0.10,
        target_radius: float = 0.6,
        place_dist: float = 8.0,          # max object/target spread from spawn
        grasp_bonus: float = 5.0,
        place_bonus: float = 15.0,
        drop_penalty: float = 2.0,
        collision_penalty: float = 0.05,
        time_cost: float = 0.002,
        reach_w: float = 1.0,
        carry_w: float = 1.5,
        load_urdf: bool = False,          # True for demos/video; False = faster headless
        seed: int | None = None,
    ) -> None:
        super().__init__()
        import habitat_sim
        from indoor_uav.sim.drone_body import DronePhysics
        from indoor_uav.sim.gripper import DroneGripper

        self._hs = habitat_sim
        bk = habitat_sim.SimulatorConfiguration()
        bk.scene_id = scene_glb
        bk.enable_physics = True
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

        self.sim_res = sim_res
        self.max_steps = max_steps
        self.speed, self.yaw_rate_max, self.ctrl_dt = speed, yaw_rate, ctrl_dt
        self.altitude = altitude
        self.obj_half = obj_size / 2
        self.target_radius = target_radius
        self.place_dist = place_dist
        self.grasp_bonus, self.place_bonus = grasp_bonus, place_bonus
        self.drop_penalty = drop_penalty
        self.collision_penalty, self.time_cost = collision_penalty, time_cost
        self.reach_w, self.carry_w = reach_w, carry_w
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            pf.seed(seed); self.sim.seed(seed)

        self.drone = DronePhysics(self.sim, mass=0.5, max_speed=max(speed * 1.6, 1.6))
        self.gripper = DroneGripper(self.sim, self.drone, load_urdf=load_urdf)
        self._lo, self._hi = (np.array(x, np.float32) for x in pf.get_bounds())
        self._obj = self._make_box("graspable", obj_size, (0.9, 0.2, 0.2), mass=0.05)
        self._target = None                   # visual ring marker (optional)

        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (19,), dtype=np.float32)

        self._t = 0; self._floor_y = 0.0
        self._target_xyz = np.zeros(3, np.float32)
        self._prev_reach = 0.0; self._prev_carry = 0.0
        self._was_holding = False

    # ------------------------------------------------------------------ #
    def _make_box(self, name, size, rgba, mass):
        otm = self.sim.get_object_template_manager()
        rom = self.sim.get_rigid_object_manager()
        base = otm.get_template_handles("cube")[0]
        t = otm.get_template_by_handle(base)
        t.scale = np.array([size, size, size], np.float32)
        t.mass = float(mass)
        otm.register_template(t, name)
        obj = rom.add_object_by_template_handle(name)
        obj.motion_type = self._hs.physics.MotionType.KINEMATIC   # we manage its pose
        return obj

    def _floor_and_point(self, near=None, rng_xz=(0.0, 999.0)):
        pf = self.sim.pathfinder
        for _ in range(400):
            p = np.array(pf.get_random_navigable_point(), np.float32)
            if abs(float(p[1]) - self._floor_y) > 0.5:
                continue
            if near is not None:
                d = float(np.linalg.norm((p - near)[[0, 2]]))
                if not (rng_xz[0] <= d <= rng_xz[1]):
                    continue
            return p
        return None

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        pf = self.sim.pathfinder
        ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(250)], np.float32)
        h, e = np.histogram(ys, 30); lo = e[int(h.argmax())]
        self._floor_y = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))

        sp = self._floor_and_point()
        start = (sp if sp is not None else np.array(pf.get_random_navigable_point(), np.float32)).copy()
        start[1] = self._floor_y + self.altitude
        op = self._floor_and_point(near=start, rng_xz=(1.5, self.place_dist))
        obj_p = op if op is not None else start.copy()
        tp = self._floor_and_point(near=obj_p, rng_xz=(1.5, self.place_dist))
        tgt_p = tp if tp is not None else obj_p.copy()

        yaw0 = float(np.arctan2((obj_p - start)[0], (obj_p - start)[2]))
        self.gripper.reset(start, yaw0)
        rest_y = self._floor_y + self.obj_half
        self._obj.translation = np.array([obj_p[0], rest_y, obj_p[2]], np.float32)
        self._target_xyz = np.array([tgt_p[0], rest_y, tgt_p[2]], np.float32)

        self._t = 0; self._was_holding = False
        self._prev_reach = float(np.linalg.norm(self.gripper.tip_world() - self._obj_xyz()))
        self._prev_carry = float(np.linalg.norm((self._obj_xyz() - self._target_xyz)[[0, 2]]))
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, np.float32).reshape(-1)
        vx, vz, vy = a[0] * self.speed, a[1] * self.speed, a[2] * self.speed
        yaw_rate = a[3] * self.yaw_rate_max
        lower = (a[4] + 1.0) * 0.5
        grip = (a[5] + 1.0) * 0.5

        tel = self.gripper.step(vx, vz, vy, yaw_rate, lower, grip,
                                dt=self.ctrl_dt, graspables=[self._obj])
        holding = self.gripper.held is not None
        if not holding:                       # manual settle: object rests on the floor
            o = np.asarray(self._obj.translation, np.float32)
            rest_y = self._floor_y + self.obj_half
            if o[1] > rest_y + 1e-3:
                o[1] = max(rest_y, o[1] - 2.0 * self.ctrl_dt)
                self._obj.translation = o

        # ---- reward ----
        tip = self.gripper.tip_world(); obj = self._obj_xyz()
        d_reach = float(np.linalg.norm(tip - obj))
        d_carry = float(np.linalg.norm((obj - self._target_xyz)[[0, 2]]))
        r = -self.time_cost
        if tel.get("collided"):
            r -= self.collision_penalty
        success = False
        if not self._was_holding and holding:           # just grasped
            r += self.grasp_bonus
        if self._was_holding and not holding:           # just released
            if d_carry < self.target_radius:
                r += self.place_bonus; success = True
            else:
                r -= self.drop_penalty
        if not holding and not success:                 # reach phase
            r += self.reach_w * (self._prev_reach - d_reach)
        if holding:                                     # carry phase
            r += self.carry_w * (self._prev_carry - d_carry)

        self._prev_reach, self._prev_carry = d_reach, d_carry
        self._was_holding = holding
        self._t += 1
        trunc = self._t >= self.max_steps
        info = {"holding": holding, "success": success,
                "d_reach": d_reach, "d_carry": d_carry}
        return self._obs(), float(r), bool(success), bool(trunc), info

    # ------------------------------------------------------------------ #
    def _obj_xyz(self):
        return np.asarray(self._obj.translation, np.float32)

    def _obs(self):
        p = np.asarray(self.drone.position, np.float32)
        v = np.asarray(self.drone.velocity, np.float32)
        span = (self._hi - self._lo); span[span < 1e-3] = 1.0
        pn = np.clip(2 * (p - self._lo) / span - 1, -1, 1)
        tip = self.gripper.tip_world(); obj = self._obj_xyz()
        sc = 5.0
        s = np.array([
            pn[0], pn[1], pn[2],
            math.sin(self.drone.yaw), math.cos(self.drone.yaw),
            float(np.clip(v[0] / self.speed, -2, 2)),
            float(np.clip(v[1] / self.speed, -2, 2)),
            float(np.clip(v[2] / self.speed, -2, 2)),
            self.gripper.lower, self.gripper.grip,
            1.0 if self.gripper.held is not None else 0.0,
            *((obj - tip) / sc),
            *((self._target_xyz - obj) / sc),
            float(np.linalg.norm(tip - obj) / sc),
            float(np.linalg.norm((obj - self._target_xyz)[[0, 2]]) / sc),
        ], np.float32)
        return s

    def render(self):
        self.sim.get_agent(0).set_state(self.drone.camera_state())
        obs = self.sim.get_sensor_observations()
        return obs["rgb"][..., :3].astype(np.uint8)

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass
