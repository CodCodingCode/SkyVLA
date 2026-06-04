"""SNATCH integration env — DirectRLEnv wiring the four swarm-built modules.

  A1 assets/drone_snatch.usd  : single-DOF caging gripper (no lower DOF), dual cam mounts
  A2 snatch/perception        : top+bottom depth cameras + ResNet-18 encoders
  A3 snatch/randomization     : domain randomization + VIO drift + detection noise
  A4 snatch/rewards           : 4-component reward + heuristic grasp trigger

Action (5, [-1,1]): [vx, vy, vz, yaw_rate, gripper]  (direct velocity control; the
drone descends BODILY to grasp -- there is no lower/raise DOF).

Observation (flat, for rsl_rl MLP):
  [ top_latent(512), bottom_latent(512) ]  (only if cfg.use_cameras)
  + state(11): pos(3, VIO-drifted), lin_vel(3), gripper_state(1), gripper_torque(1),
               block_pos_est(3, detection-noised)

Physics 200Hz, policy 50Hz (decimation 4). World is +Z up.
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

from skyvla_isaac.snatch import rewards as snatch_rewards
from skyvla_isaac.snatch import randomization as snatch_rand

_USD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "configuration", "drone_snatch.usd")
GRAV = 9.81


@configclass
class DroneSnatchEnvCfg(DirectRLEnvCfg):
    decimation = 4                       # 200Hz physics / 4 -> 50Hz policy
    episode_length_s = 10.0
    action_space = 5                     # [vx,vy,vz,yaw_rate,gripper]
    observation_space = 1035             # 1024 latents + 11 state (set in __post_init__)
    state_space = 0

    use_cameras: bool = True             # visuomotor obs (dual depth + ResNet encoders)

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0, render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=1.8, friction_combine_mode="max"))
    viewer: ViewerCfg = ViewerCfg(eye=(1.6, 1.6, 1.2), lookat=(0.0, 0.0, 0.3),
                                  origin_type="env", env_index=0)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=6.0,
                                                     replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_USD, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={"grip_xl": 0.0, "grip_xr": 0.0, "grip_yl": 0.0, "grip_yr": 0.0}),
        actuators={
            "grip": ImplicitActuatorCfg(joint_names_expr=["grip_.*"],
                                        effort_limit=80.0, velocity_limit=1.0,
                                        stiffness=2000.0, damping=10.0)},
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=1.6, friction_combine_mode="max"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025)),
    )

    # flight + task params
    speed: float = 1.5                   # proven value; eases precise bodily descent-grasp
    yaw_rate_scale: float = 1.0
    kv: float = 18.0
    k_att: float = 4.0
    k_damp: float = 0.6
    obj_spawn_diam: float = 0.8          # proven distance for convergence (scale up later)
    goal_offset_diam: float = 0.5
    grasp_clear: float = 0.06            # cube off-floor height to count as lifted
    # Localization/perception noise: TRAIN CLEAN (0), EVAL sweeps the drift (the gap study).
    # Training under full drift corrupts the drone's own pose estimate -> it can't descend
    # onto the cube and grasp collapses. eval_snatch sets vio_drift_scale per sweep point.
    vio_drift_scale: float = 0.0         # headline sim2real knob (eval sweeps this)
    detection_noise_scale: float = 0.0   # block-position estimate noise (eval can sweep too)
    # curriculum: a fraction start straddling the cube (grasp discovery), annealed down
    # so the policy masters the full fly-in task. (Proven necessary for convergence.)
    curriculum_p_start: float = 0.85
    curriculum_p_end: float = 0.15
    anneal_steps: float = 60000.0

    def __post_init__(self):
        self.observation_space = (1024 if self.use_cameras else 0) + 14   # +3 for goal


class DroneSnatchEnv(DirectRLEnv):
    cfg: DroneSnatchEnvCfg

    def __init__(self, cfg: DroneSnatchEnvCfg, render_mode: str | None = None, **kw):
        cfg.observation_space = (1024 if cfg.use_cameras else 0) + 14   # recompute post-toggle (+goal)
        super().__init__(cfg, render_mode, **kw)
        self._grip_i, _ = self.robot.find_joints("grip_.*")
        self._base_i, _ = self.robot.find_bodies("base")
        self._mass = self.robot.root_physx_view.get_masses().sum(-1).to(self.device)
        self._target = torch.zeros(self.num_envs, 3, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._prev_d_goal = z()
        self._held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._carry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # per-env DR params (resampled on reset); vio_drift_scale is the eval-sweep knob
        self._dr = snatch_rand.sample_dr_params(
            self.num_envs, self.device, vio_drift_scale=self.cfg.vio_drift_scale,
            detection_noise_scale=self.cfg.detection_noise_scale)
        # perception encoders (frozen ResNet-18 feature extractors), lazily built
        self._enc_top = self._enc_bottom = None
        if self.cfg.use_cameras:
            from skyvla_isaac.snatch import perception as P
            self._enc_top = P.freeze_backbone(P.DepthEncoder().to(self.device)).eval()
            self._enc_bottom = P.freeze_backbone(P.DepthEncoder().to(self.device)).eval()
            self._P = P

    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)
        if self.cfg.use_cameras:
            from isaaclab.sensors import TiledCamera
            from skyvla_isaac.snatch import perception as P
            self._top_cam = TiledCamera(P.top_camera_cfg())
            self._bottom_cam = TiledCamera(P.bottom_camera_cfg())
            self.scene.sensors["top_cam"] = self._top_cam
            self.scene.sensors["bottom_cam"] = self._bottom_cam

    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0)
        self._train_steps = getattr(self, "_train_steps", 0) + 1

    def _apply_action(self):
        a = self._actions
        # velocity-tracking wrench on the base (direct velocity control)
        v_des = a[:, :3] * self.cfg.speed
        v_now = self.robot.data.root_lin_vel_w
        force = self.cfg.kv * (v_des - v_now) * self._mass.unsqueeze(-1)
        force[:, 2] += self._mass * GRAV
        # DR: wind + near-ground effect disturbances (A3)
        force[:, :2] += self._dr["wind"][:, :2]
        alt = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        force = force + snatch_rand.ground_effect_force(alt, self._dr)
        fmax = (4.0 * self._mass * GRAV).unsqueeze(-1)
        force = torch.nan_to_num(force, nan=0.0).clamp(-fmax, fmax)
        # attitude stabilization + commanded yaw rate
        q = self.robot.data.root_quat_w
        wup = torch.zeros(self.num_envs, 3, device=self.device); wup[:, 2] = 1.0
        up_body = quat_rotate(q, wup)
        level_err = torch.cross(up_body, wup, dim=-1)
        ang = self.robot.data.root_ang_vel_w
        torque = self.cfg.k_att * level_err - self.cfg.k_damp * ang
        torque[:, 2] += 2.0 * (a[:, 3] * self.cfg.yaw_rate_scale - ang[:, 2])  # yaw-rate track
        torque = torch.nan_to_num(torque).clamp(-2.0, 2.0)
        self.robot.set_external_force_and_torque(
            force.unsqueeze(1), torque.unsqueeze(1), body_ids=self._base_i)
        # gripper: all 4 cage jaws driven by the single gripper action
        jaw = (a[:, 4] * 0.5 + 0.5) * 0.02
        tgt = jaw.unsqueeze(-1).repeat(1, len(self._grip_i))
        self.robot.set_joint_position_target(tgt, joint_ids=self._grip_i)

    # ------------------------------------------------------------------ #
    def _tip_w(self):
        tip = self.robot.data.root_pos_w.clone()
        tip[:, 2] = tip[:, 2] - 0.33                # fingers rigid at base-0.33 (no lower DOF)
        return tip

    def _camera_latents(self):
        if not self.cfg.use_cameras:
            return torch.zeros(self.num_envs, 0, device=self.device)
        top = self._top_cam.data.output["distance_to_image_plane"][..., 0]      # (N,H,W)
        bot = self._bottom_cam.data.output["distance_to_image_plane"][..., 0]
        top = snatch_rand.add_depth_noise(torch.nan_to_num(top, posinf=5.0), self._dr)
        bot = snatch_rand.add_depth_noise(torch.nan_to_num(bot, posinf=5.0), self._dr)
        with torch.no_grad():
            return self._P.build_obs_latents(top, bot, self._enc_top, self._enc_bottom)

    def _get_observations(self) -> dict:
        base_p = self.robot.data.root_pos_w - self.scene.env_origins
        base_v = self.robot.data.root_lin_vel_w
        obj_p = self.object.data.root_pos_w - self.scene.env_origins
        # VIO-drifted pose estimate + detection-noised block estimate (the sim2real gap)
        pose_est = snatch_rand.apply_vio_drift(base_p, self.episode_length_buf.float(), self._dr)
        block_est = snatch_rand.apply_detection_noise(obj_p, self._dr)
        grip = self.robot.data.joint_pos[:, self._grip_i].mean(-1, keepdim=True)
        grip_tau = self.robot.data.applied_torque[:, self._grip_i].abs().mean(-1, keepdim=True)
        goal_rel = self._target - base_p                       # delivery target (drone-relative)
        state = torch.cat([pose_est, base_v, grip, grip_tau, block_est, goal_rel], dim=-1)  # (N,14)
        latents = self._camera_latents()
        obs = torch.cat([latents, state], dim=-1)
        return {"policy": torch.nan_to_num(obs)}

    # ------------------------------------------------------------------ #
    def _get_dones(self):
        obj_p = self.object.data.root_pos_w - self.scene.env_origins
        base_p = self.robot.data.root_pos_w - self.scene.env_origins
        tip = self._tip_w() - self.scene.env_origins
        self._d_reach = torch.norm(obj_p - tip, dim=-1)
        self._horiz = torch.norm((obj_p - tip)[:, :2], dim=-1)
        self._d_goal = torch.norm(obj_p - self._target, dim=-1)
        self._obj_p, self._base_p = obj_p, base_p
        self._lifted = obj_p[:, 2] > 0.025 + self.cfg.grasp_clear
        self._held = self._lifted & (self._d_reach < 0.10)
        self._carry = self._held
        self._success = self._held & (self._d_goal < 0.18)
        # aliases for eval_snatch's metric contract
        self._grasped = self._held
        self._placed = self._success
        self._grasp_pos_err = self._d_reach
        self._crashed = (base_p[:, 2] < 0.05) | (torch.norm(base_p[:, :2], dim=-1) > 5.0) \
            | (torch.norm(self.robot.data.root_lin_vel_w, dim=-1) > 12.0)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["log"] = {
            "metrics/grasp_rate": self._held.float().mean(),
            "metrics/place_success": self._success.float().mean(),
            "metrics/obj_to_goal": self._d_goal.mean(),
        }
        return self._crashed, time_out

    def _get_rewards(self) -> torch.Tensor:
        # bottom-cam centering proxy: horizontal tip-block offset -> pixels (~320 half-width)
        pixel_offset = (self._horiz / 0.5).clamp(max=2.0) * 320.0
        alt_above_block = (self._base_p[:, 2] - 0.33) - self._obj_p[:, 2] + 0.0  # tip alt over block
        s = dict(
            d_block=self._d_reach,
            pixel_offset=pixel_offset,
            alt_above_block=(self._base_p[:, 2] - self._obj_p[:, 2]),
            grasp_success=self._held,
            carrying=self._carry,
            d_goal=self._d_goal,
            place_success=self._success,
            crashed=self._crashed,
        )
        # PROVEN balance (drove the base task to 99.8%): dominant held-only placement so
        # the policy can't just "grab + hold high anywhere" (that made obj_to_goal blow up
        # to 6.5m with place~0). A4's spec reward is kept as a small auxiliary signal.
        held = self._held.float()
        grip_cmd = self._actions[:, 4] * 0.5 + 0.5
        reach = 1.0 - torch.tanh(self._d_reach / 0.5)
        align = 1.0 - torch.tanh(self._horiz / 0.06)
        grab = (self._d_reach < 0.07).float() * grip_cmd
        cube_h = torch.clamp(self._obj_p[:, 2] - 0.025, 0.0, 0.15)        # lift gate (discovery)
        place = held * (1.0 - torch.tanh(self._d_goal / 0.35))           # DOMINANT delivery
        carry_up = held * torch.clamp(self._obj_p[:, 2] - 0.15, 0.0, 0.25)
        carry_prog = held * (self._prev_d_goal - self._d_goal)
        success = (self._held & (self._d_goal < 0.18)).float()
        self._prev_d_goal = self._d_goal.clone()
        r = (1.0 * reach + 0.5 * align + 1.5 * grab + 60.0 * cube_h
             + 40.0 * place + 30.0 * carry_up + 25.0 * carry_prog + 80.0 * success - 0.01)
        return r + 0.1 * snatch_rewards.compute_reward(s)               # spec reward as aux

    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        # cube on the floor within obj_spawn_diam of origin
        obj = self.object.data.default_root_state[env_ids].clone()
        r = (torch.rand(n, 2, device=self.device) - 0.5) * self.cfg.obj_spawn_diam
        obj[:, 0] += r[:, 0]; obj[:, 1] += r[:, 1]
        obj[:, :3] += origins
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        obj_local = obj[:, :3] - origins
        # curriculum: a fraction start straddling the cube at grasp height (jaws open),
        # the rest from altitude; straddle fraction anneals high->low.
        prog = min(1.0, getattr(self, "_train_steps", 0) / self.cfg.anneal_steps)
        cur_p = self.cfg.curriculum_p_start + (self.cfg.curriculum_p_end - self.cfg.curriculum_p_start) * prog
        self._cur_p = cur_p
        near = torch.rand(n, device=self.device) < cur_p
        root = self.robot.data.default_root_state[env_ids].clone()
        root[near, 0] = obj_local[near, 0]
        root[near, 1] = obj_local[near, 1]
        root[near, 2] = 0.355                          # tip (base-0.33) at the cube
        root[:, :3] += origins
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)
        # goal drop-zone near the cube, at carry height
        t = torch.zeros(n, 3, device=self.device)
        t[:, :2] = obj_local[:, :2] + (torch.rand(n, 2, device=self.device) - 0.5) * self.cfg.goal_offset_diam
        t[:, 2] = 0.4
        self._target[env_ids] = t
        # resample DR for these envs (keep the vio_drift_scale knob fixed)
        fresh = snatch_rand.sample_dr_params(n, self.device, vio_drift_scale=self.cfg.vio_drift_scale,
                                             detection_noise_scale=self.cfg.detection_noise_scale)
        for k, v in fresh.items():
            if k in self._dr and self._dr[k].shape[0] == self.num_envs:
                self._dr[k][env_ids] = v
        self._prev_d_goal[env_ids] = torch.norm(obj[:, :3] - (origins + t), dim=-1)
        self._held[env_ids] = False
        self._carry[env_ids] = False
