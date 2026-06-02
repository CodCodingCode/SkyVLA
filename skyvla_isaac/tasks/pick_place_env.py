"""Isaac Lab DirectRLEnv: drone + 2-DoF gripper PICK-AND-PLACE with REAL physics.

Unlike the Habitat version (kinematic-attach grasp), the gripper here grips by
PhysX contact + friction: the fingers (prismatic joints grip_l/grip_r) squeeze
the cube and friction holds it; a bad grasp slips. The whole drone is a
free-floating PhysX articulation flown by applying a wrench to its base to track
a commanded velocity.

Action (6, continuous, [-1,1]): [vx, vy, vz, yaw_rate, lower, grip]
  - vx,vy,vz,yaw_rate : commanded base velocity / yaw rate (tracked by a wrench)
  - lower             : gripper extension (DoF 1), maps to the `lower` joint
  - grip              : jaw closure (DoF 2), drives grip_l & grip_r symmetrically

Reward: reach (tip->object) + lift bonus (object off the floor = real grasp) +
carry (object->target) + place bonus; small effort/time costs.

Runs N envs in parallel on one GPU (Isaac Lab tiling). World is +Z up.
"""
from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

_USD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "drone_with_gripper.usd")

GRAV = 9.81


@configclass
class DronePickPlaceEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 12.0
    action_space = 6
    observation_space = 17
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=5.0,
                                                     replicate_physics=True)

    # the converted drone+gripper articulation; lower/grip joints are actuated
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.3),
            joint_pos={"lower": 0.0, "grip_l": 0.02, "grip_r": 0.02},
        ),
        actuators={
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["lower", "grip_l", "grip_r"],
                effort_limit=50.0, velocity_limit=2.0,
                stiffness=800.0, damping=60.0),
        },
    )

    # graspable cube
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.025)),
    )

    # flight controller gains + ranges
    speed: float = 1.5
    yaw_rate: float = 2.0
    kv: float = 8.0            # velocity-tracking force gain (per unit mass)
    max_drop: float = 0.5
    target_radius: float = 0.25
    lift_height: float = 0.15  # object this far off floor counts as grasped


class DronePickPlaceEnv(DirectRLEnv):
    cfg: DronePickPlaceEnvCfg

    def __init__(self, cfg: DronePickPlaceEnvCfg, render_mode: str | None = None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self._lower_i, _ = self.robot.find_joints("lower")
        self._gl_i, _ = self.robot.find_joints("grip_l")
        self._gr_i, _ = self.robot.find_joints("grip_r")
        self._base_i, _ = self.robot.find_bodies("base")
        # per-env target placement (set on reset), in env-local frame
        self._target = torch.zeros(self.num_envs, 3, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._mass = self.robot.root_physx_view.get_masses().sum(-1).to(self.device)

    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        a = self._actions
        # --- flight: wrench on the base to track a commanded velocity ---
        v_des = torch.zeros(self.num_envs, 3, device=self.device)
        v_des[:, 0] = a[:, 0] * self.cfg.speed
        v_des[:, 1] = a[:, 1] * self.cfg.speed
        v_des[:, 2] = a[:, 2] * self.cfg.speed
        v_now = self.robot.data.root_lin_vel_w
        force = (self.cfg.kv * (v_des - v_now) * self._mass.unsqueeze(-1))
        force[:, 2] += self._mass * GRAV                      # gravity compensation
        # clamp + sanitize so a flung env can't explode the controller -> NaN
        fmax = (4.0 * self._mass * GRAV).unsqueeze(-1)
        force = torch.nan_to_num(force, nan=0.0).clamp(-fmax, fmax)
        torque = torch.zeros(self.num_envs, 3, device=self.device)
        torque[:, 2] = a[:, 3] * self.cfg.yaw_rate * 0.5      # yaw
        self.robot.set_external_force_and_torque(
            force.unsqueeze(1), torque.unsqueeze(1), body_ids=self._base_i)
        # --- gripper joints ---
        lower = (a[:, 4] * 0.5 + 0.5) * self.cfg.max_drop     # [0, max_drop]
        jaw = (1.0 - (a[:, 5] * 0.5 + 0.5)) * 0.02            # grip=1 -> 0 (closed)
        tgt = torch.zeros(self.num_envs, 3, device=self.device)
        tgt[:, 0] = lower; tgt[:, 1] = jaw; tgt[:, 2] = jaw
        idx = torch.tensor(self._lower_i + self._gl_i + self._gr_i, device=self.device)
        self.robot.set_joint_position_target(tgt, joint_ids=idx.tolist())

    # ------------------------------------------------------------------ #
    def _tip_w(self):
        """Approx gripper tip in world: base minus lower extension along -Z."""
        base = self.robot.data.root_pos_w
        lower = self.robot.data.joint_pos[:, self._lower_i[0]]
        tip = base.clone()
        tip[:, 2] = tip[:, 2] - 0.05 - lower
        return tip

    def _get_observations(self) -> dict:
        base_p = self.robot.data.root_pos_w - self.scene.env_origins
        base_v = self.robot.data.root_lin_vel_w
        jpos = self.robot.data.joint_pos[:, self._lower_i + self._gl_i + self._gr_i]
        obj_p = self.object.data.root_pos_w - self.scene.env_origins
        tip = self._tip_w() - self.scene.env_origins
        obs = torch.cat([
            base_p, base_v,
            self.robot.data.root_ang_vel_w[:, 2:3],
            jpos,
            obj_p - tip,
            self._target - obj_p,
            (obj_p[:, 2:3]),                                  # object height
        ], dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        obj_p = self.object.data.root_pos_w - self.scene.env_origins
        tip = self._tip_w() - self.scene.env_origins
        d_reach = torch.norm(obj_p - tip, dim=-1)
        d_place = torch.norm(obj_p[:, :2] - self._target[:, :2], dim=-1)
        lifted = (obj_p[:, 2] > 0.025 + self.cfg.lift_height).float()
        placed = ((d_place < self.cfg.target_radius) & (lifted < 0.5)).float()
        # dense, bounded, always-positive shaping -> a clear hill for PPO to climb
        reach = torch.exp(-2.0 * d_reach)         # peaks (=1) at the object
        carry = torch.exp(-2.0 * d_place)         # peaks at the target
        r = (0.5 * reach                          # approach the object
             + 3.0 * lifted                       # real grasp: object off floor
             + 2.0 * lifted * carry               # carry toward target while held
             + 10.0 * placed                      # released in the target zone
             - 0.002)                             # small time cost
        return r

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        obj_p = self.object.data.root_pos_w - self.scene.env_origins
        base_p = self.robot.data.root_pos_w - self.scene.env_origins
        base_v = torch.norm(self.robot.data.root_lin_vel_w, dim=-1)
        # terminate (and reset) diverging envs so they can't NaN the batch
        oob = (obj_p[:, 2] < -0.5) | (base_p[:, 2] < -0.5) | (base_p[:, 2] > 4.0) \
            | (torch.norm(base_p[:, :2], dim=-1) > 5.0) | (base_v > 12.0)
        return oob, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        # drone reset above the object
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)
        # object on the floor at a random offset
        obj = self.object.data.default_root_state[env_ids].clone()
        rand = (torch.rand(n, 2, device=self.device) - 0.5) * 1.5
        obj[:, 0] += rand[:, 0]; obj[:, 1] += rand[:, 1]
        obj[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        # target placement (env-local)
        t = (torch.rand(n, 3, device=self.device) - 0.5)
        t[:, 0] *= 1.5; t[:, 1] *= 1.5; t[:, 2] = 0.025
        self._target[env_ids] = t
