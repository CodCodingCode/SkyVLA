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
from isaaclab.utils.math import quat_rotate

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

    # default contact material: HIGH friction so the (imported) gripper fingers
    # actually grip the cube instead of letting it slip out. combine=max -> the
    # higher of the two materials wins at each contact.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0, render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=1.8, friction_combine_mode="max"))
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
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={"lower": 0.0, "grip_xl": 0.0, "grip_xr": 0.0,
                       "grip_yl": 0.0, "grip_yr": 0.0},          # all 4 jaws OPEN
        ),
        actuators={
            # lower: soft positioning. grips: stiff + strong so they actually CLAMP.
            # STIFF lower joint: the arm must track its command rigidly, not telescope
            # and lag under flight accelerations (soft 400 made the gripper fling the
            # drone). High stiffness + damping + effort -> effectively a rigid arm.
            "lower": ImplicitActuatorCfg(joint_names_expr=["lower"],
                                         effort_limit=200.0, velocity_limit=2.0,
                                         stiffness=8000.0, damping=300.0),
            "grip": ImplicitActuatorCfg(joint_names_expr=["grip_.*"],
                                        effort_limit=80.0, velocity_limit=1.0,
                                        stiffness=2000.0, damping=10.0),
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
                static_friction=2.0, dynamic_friction=1.6, friction_combine_mode="max"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.025)),
    )

    # flight controller gains + ranges
    speed: float = 1.5
    yaw_rate: float = 2.0
    kv: float = 18.0            # velocity-tracking force gain (per unit mass)
    k_att: float = 4.0         # attitude-leveling torque gain (keeps drone upright)
    k_damp: float = 0.6        # angular-velocity damping (stops spin/tumble)
    max_drop: float = 0.5
    target_radius: float = 0.25
    curriculum_p: float = 0.7  # fraction of envs that start with the gripper straddling the cube
    render_camera: bool = False  # add a close 3rd-person Camera sensor (for rollout mp4)


class DronePickPlaceEnv(DirectRLEnv):
    cfg: DronePickPlaceEnvCfg

    def __init__(self, cfg: DronePickPlaceEnvCfg, render_mode: str | None = None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self._lower_i, _ = self.robot.find_joints("lower")
        self._grip_i, _ = self.robot.find_joints("grip_.*")   # all 4 cage jaws
        self._gl_i, _ = self.robot.find_joints("grip_xl")     # representative x-jaw (obs/test)
        self._gy_i, _ = self.robot.find_joints("grip_yl")     # representative y-jaw (obs)
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
        # ATTITUDE STABILIZATION: a real quad's controller keeps it level + damps
        # rotation. Without this the body free-spins (yaw doesn't affect reward).
        q = self.robot.data.root_quat_w
        wup = torch.zeros(self.num_envs, 3, device=self.device); wup[:, 2] = 1.0
        up_body = quat_rotate(q, wup)                          # body up-axis, in world
        level_err = torch.cross(up_body, wup, dim=-1)         # restoring axis (roll/pitch)
        ang = self.robot.data.root_ang_vel_w
        torque = self.cfg.k_att * level_err - self.cfg.k_damp * ang   # level + damp all spin
        torque = torch.nan_to_num(torque).clamp(-2.0, 2.0)
        self.robot.set_external_force_and_torque(
            force.unsqueeze(1), torque.unsqueeze(1), body_ids=self._base_i)
        # --- gripper joints: REAL physics. lower extends the arm down; grip drives
        # the finger joints CLOSED (target 0.03) so they physically clamp the cube.
        # No teleport/magnet -- the cube is held only by contact + friction. ---
        lower = (a[:, 4] * 0.5 + 0.5) * self.cfg.max_drop     # [0, max_drop]
        jaw = (a[:, 5] * 0.5 + 0.5) * 0.02                    # grip=1 -> 0.02 (all jaws closed)
        ng = len(self._grip_i)
        tgt = torch.zeros(self.num_envs, 1 + ng, device=self.device)
        tgt[:, 0] = lower
        tgt[:, 1:] = jaw.unsqueeze(-1)                        # all 4 cage jaws -> same closure
        self.robot.set_joint_position_target(tgt, joint_ids=self._lower_i + self._grip_i)

    # ------------------------------------------------------------------ #
    def _tip_w(self):
        """World position of the grasp point (between the fingers). The fingers sit
        at base - 0.33 - lower along -Z (lower-joint origin 0.03 + arm 0.30)."""
        base = self.robot.data.root_pos_w
        lower = self.robot.data.joint_pos[:, self._lower_i[0]]
        tip = base.clone()
        tip[:, 2] = tip[:, 2] - 0.33 - lower
        return tip

    def _get_observations(self) -> dict:
        base_p = self.robot.data.root_pos_w - self.scene.env_origins
        base_v = self.robot.data.root_lin_vel_w
        jpos = self.robot.data.joint_pos[:, self._lower_i + self._gl_i + self._gy_i]
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
        self._obj_p, self._tip = obj_p, tip
        self._d_reach = torch.norm(obj_p - tip, dim=-1)
        self._horiz = torch.norm((obj_p - tip)[:, :2], dim=-1)             # tip-cube horizontal
        self._d_goal = torch.norm(obj_p - self._target, dim=-1)            # 3D to goal waypoint
        self._obj_z = obj_p[:, 2]
        self._lifted = obj_p[:, 2] > 0.025 + 0.06                          # off floor >6cm
        # REAL grasp: cube off the floor AND between the fingers (no teleport).
        self._held = self._lifted & (self._d_reach < 0.10)
        self._success = self._d_goal < 0.18                               # cube delivered to goal

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        oob = (obj_p[:, 2] < -0.5) | (base_p[:, 2] < -0.5) | (base_p[:, 2] > 4.0) \
            | (torch.norm(base_p[:, :2], dim=-1) > 5.0) | (base_v > 12.0)

        self.extras["log"] = {
            "metrics/lift_rate": self._lifted.float().mean(),
            "metrics/grasp_rate": self._held.float().mean(),          # real contact grasp
            "metrics/place_success": self._success.float().mean(),
            "metrics/obj_to_goal": self._d_goal.mean(),
        }
        return oob, time_out          # dense task: no success-termination

    def _get_rewards(self) -> torch.Tensor:
        # REAL-physics grasp reward. The cube only leaves the floor if the fingers
        # physically clamp it, so cube_h (lift height) is the true grasp signal and
        # gets the dominant weight. Shaping guides: reach -> center -> close -> lift
        # -> carry -> place.
        held = self._held.float()                                     # REAL grasp (off floor + between fingers)
        grip_cmd = (self._actions[:, 5] * 0.5 + 0.5)                   # 0 open .. 1 closed
        reach = 1.0 - torch.tanh(self._d_reach / 0.5)                  # fingers to the cube
        align = 1.0 - torch.tanh(self._horiz / 0.06)                   # center over the cube
        at_cube = (self._d_reach < 0.07).float()
        grab = at_cube * grip_cmd                                      # close jaws ON the cube
        # Lift gate: STRONG gradient (so grasp is reliably discovered) but a LOW height
        # plateau (~15 cm). Capping low kills "rocket to the ceiling" while the strong
        # weight keeps grasp rate up; capping it too low (0.12*25) starved grasp -> 7%.
        cube_h = torch.clamp(self._obj_z - 0.025, 0.0, 0.15)          # REAL lift (contact only)
        # Placement is the DOMINANT objective and only pays while truly held, so the
        # big reward is unlockable ONLY through grasping-then-carrying to the 3D goal.
        place = held * (1.0 - torch.tanh(self._d_goal / 0.35))        # deliver cube to goal waypoint
        carry_prog = held * (self._prev_d_goal - self._d_goal)        # shaped progress toward goal
        success = (self._held & (self._d_goal < 0.18)).float()        # carried-and-delivered
        r = (1.0 * reach + 0.5 * align + 1.5 * grab                    # approach, center, clamp
             + 60.0 * cube_h                                          # grasp/lift gate (strong, low plateau)
             + 40.0 * place                                           # DOMINANT: deliver to goal
             + 25.0 * carry_prog                                      # progress toward goal
             + 80.0 * success                                         # sparse delivery bonus
             - 0.01)
        self._grasped = self._grasped | self._held
        self._prev_d_goal = self._d_goal.clone()
        return r

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        # --- object on the floor near the origin ---
        obj = self.object.data.default_root_state[env_ids].clone()
        rand = (torch.rand(n, 2, device=self.device) - 0.5) * 0.8        # +/-0.4 m
        obj[:, 0] += rand[:, 0]; obj[:, 1] += rand[:, 1]
        obj[:, :3] += origins
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        obj_local = obj[:, :3] - origins

        # --- CURRICULUM: a fraction start with the gripper STRADDLING the cube
        # (drone directly above at grasp height, jaws open) so the policy can
        # discover close->lift; the rest start at altitude and must fly in. ---
        near = torch.rand(n, device=self.device) < self.cfg.curriculum_p
        root = self.robot.data.default_root_state[env_ids].clone()       # (0,0,1.0), level, v=0
        root[near, 0] = obj_local[near, 0]
        root[near, 1] = obj_local[near, 1]
        root[near, 2] = 0.355                                            # tip (base-0.33) at cube
        root[:, :3] += origins
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()        # lower 0, jaws open (0)
        self.robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)

        # --- goal: a 3D waypoint near the cube (short, uniform carry) ---
        t = torch.zeros(n, 3, device=self.device)
        t[:, :2] = obj_local[:, :2] + (torch.rand(n, 2, device=self.device) - 0.5) * 0.5
        t[:, 2] = 0.4
        self._target[env_ids] = t

        # --- bookkeeping ---
        self._grasped[env_ids] = False
        self._success[env_ids] = False
        self._held[env_ids] = False
        tip = root[:, :3].clone(); tip[:, 2] -= 0.33                     # tip at fingers (lower=0)
        tgt_w = origins + t
        self._prev_d_goal[env_ids] = torch.norm(obj[:, :3] - tgt_w, dim=-1)
        self._prev_d_reach[env_ids] = torch.norm(obj[:, :3] - tip, dim=-1)
