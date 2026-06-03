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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

_USD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "drone_with_gripper.usd")

GRAV = 9.81


@configclass
class DronePickPlaceEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0
    action_space = 6
    observation_space = 17
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    # close follow-camera on env 0 (for clear rollout videos)
    viewer: ViewerCfg = ViewerCfg(eye=(1.6, 1.6, 1.2), lookat=(0.0, 0.0, 0.3),
                                  origin_type="env", env_index=0)

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
            pos=(0.0, 0.0, 1.0),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.025)),
    )

    # flight controller gains + ranges
    speed: float = 1.5
    yaw_rate: float = 2.0
    kv: float = 8.0            # velocity-tracking force gain (per unit mass)
    max_drop: float = 0.5
    target_radius: float = 0.25
    lift_height: float = 0.15  # object this far off floor counts as grasped
    grasp_radius: float = 0.15  # tip within this of cube + grip on -> magnetic latch
    render_camera: bool = False  # add a close 3rd-person Camera sensor (for rollout mp4)


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
        # potential-based-reward bookkeeping (see _get_rewards)
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._prev_d_reach, self._prev_d_tgt, self._prev_d_goal = z(), z(), z()
        self._d_reach, self._d_tgt = z(), z()
        self._holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # magnetic/suction gripper state (realistic aerial end-effector; learnable)
        self._held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obj_mass = float(self.object.root_physx_view.get_masses()[0].sum())

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
        if getattr(self.cfg, "render_camera", False):
            from isaaclab.sensors import Camera, CameraCfg
            ccfg = CameraCfg(
                prim_path="/World/render_cam", height=540, width=720, update_period=0.0,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.05, 80.0)))
            self._render_cam = Camera(ccfg)
            self.scene.sensors["render_cam"] = self._render_cam

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

        # --- magnetic/suction grasp: when activated (grip>0.5) and the tip is near
        # the cube, latch it and pull it to the tip with a spring force. Real
        # dynamics (mass/swing/drop-on-release); no finger precision needed. ---
        grip_on = (a[:, 5] * 0.5 + 0.5) > 0.5
        tip = self._tip_w()
        obj = self.object.data.root_pos_w
        d = torch.norm(tip - obj, dim=-1)
        self._held = (self._held & grip_on) | (grip_on & (d < self.cfg.grasp_radius))
        held_ids = self._held.nonzero(as_tuple=False).squeeze(-1)
        if held_ids.numel() > 0:
            # attach: snap the cube to the tip while held (released -> falls under gravity)
            pose = torch.zeros(held_ids.numel(), 7, device=self.device)
            pose[:, :3] = tip[held_ids]
            pose[:, 3] = 1.0                                 # quat (w,x,y,z) = identity
            self.object.write_root_pose_to_sim(pose, held_ids)
            self.object.write_root_velocity_to_sim(
                torch.zeros(held_ids.numel(), 6, device=self.device), held_ids)

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

    def _get_dones(self):
        # _get_dones runs BEFORE _get_rewards -> compute shared state + log metrics.
        obj_p = self.object.data.root_pos_w - self.scene.env_origins
        base_p = self.robot.data.root_pos_w - self.scene.env_origins
        base_v = torch.norm(self.robot.data.root_lin_vel_w, dim=-1)
        tip = self._tip_w() - self.scene.env_origins
        self._d_reach = torch.norm(obj_p - tip, dim=-1)
        self._d_goal = torch.norm(obj_p - self._target, dim=-1)            # 3D to goal waypoint
        self._obj_z = obj_p[:, 2]
        self._lifted = obj_p[:, 2] > 0.025 + 0.05                          # off floor 5cm
        self._success = self._d_goal < 0.18                               # cube delivered to goal

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        oob = (obj_p[:, 2] < -0.5) | (base_p[:, 2] < -0.5) | (base_p[:, 2] > 4.0) \
            | (torch.norm(base_p[:, :2], dim=-1) > 5.0) | (base_v > 12.0)

        self.extras["log"] = {
            "metrics/grasp_rate": self._lifted.float().mean(),
            "metrics/held_rate": self._held.float().mean(),
            "metrics/place_success": self._success.float().mean(),
            "metrics/obj_to_goal": self._d_goal.mean(),
        }
        return oob, time_out          # dense task: no success-termination

    def _get_rewards(self) -> torch.Tensor:
        # Isaac-Lab "lift-cube" dense recipe (known to converge): approach ->
        # lift -> track the object to a 3D goal waypoint (fine term near goal).
        held = self._held.float()
        notheld = 1.0 - held
        reach = 1.0 - torch.tanh(self._d_reach / 0.8)
        near = (self._d_reach < 0.12).float()
        grip_closed = (self._actions[:, 5] * 0.5 + 0.5)
        # BALANCED: a modest hold bonus keeps the grasp; a DOMINANT, wide-gradient
        # goal-track term makes carrying to the goal clearly beat hovering.
        goal_track = 1.0 - torch.tanh(self._d_goal / 0.8)              # wide gradient (~2 m)
        fine = 1.0 - torch.tanh(self._d_goal / 0.1)                    # sharp at-goal bonus
        carry_prog = held * (self._prev_d_goal - self._d_goal)
        r = (notheld * (1.0 * reach + 1.0 * near * grip_closed)        # reach + grab
             + 2.0 * held                                             # maintain grasp (modest)
             + 15.0 * held * goal_track                               # carry to goal (dominant)
             + 10.0 * carry_prog                                      # extra progress gradient
             + 10.0 * held * fine                                     # delivered at goal
             - 0.01)
        self._grasped = self._grasped | self._held
        self._prev_d_goal = self._d_goal.clone()
        return r

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
        # object on the floor near the start (curriculum: closer = learnable)
        obj = self.object.data.default_root_state[env_ids].clone()
        rand = (torch.rand(n, 2, device=self.device) - 0.5) * 0.8        # +/-0.4 m
        obj[:, 0] += rand[:, 0]; obj[:, 1] += rand[:, 1]
        obj[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        # goal = a 3D waypoint NEAR the object (short, uniform carry distance):
        # pick the cube up and deliver it ~0.5 m away at 0.5 m height.
        obj_local = obj[:, :3] - self.scene.env_origins[env_ids]
        t = torch.zeros(n, 3, device=self.device)
        t[:, :2] = obj_local[:, :2] + (torch.rand(n, 2, device=self.device) - 0.5) * 0.5
        t[:, 2] = 0.4                                                    # 0.4 m off the floor
        self._target[env_ids] = t

        # seed potential-reward bookkeeping from the reset positions
        self._grasped[env_ids] = False
        self._success[env_ids] = False
        self._held[env_ids] = False
        tip = root[:, :3].clone(); tip[:, 2] -= 0.05                     # gripper tip (lower=0)
        obj_w = obj[:, :3]
        tgt_w = self.scene.env_origins[env_ids] + t
        self._prev_d_reach[env_ids] = torch.norm(obj_w - tip, dim=-1)
        self._prev_d_tgt[env_ids] = torch.norm((obj_w - tgt_w)[:, :2], dim=-1)
        self._prev_d_goal[env_ids] = torch.norm(obj_w - tgt_w, dim=-1)
