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
    render_camera: bool = False          # add a 3rd-person RGB follow cam (for rollout mp4)
    render_cam_w: int = 1920             # demo-film resolution (render path only)
    render_cam_h: int = 1080

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.325)),   # rests on the table
    )

    # raised pick surface (table): the cage mounts FLUSH under the body, so to grasp
    # the drone hovers just above the table -- the body stays well clear of the floor
    # (a floor cube would force the body down ~9cm and the base scrapes the ground).
    surface_z: float = 0.30              # table top height
    # Solid table that RESTS ON THE FLOOR: a 1.0x1.0 m top, full height from the floor
    # (z=0) up to surface_z=0.30 -> the box spans [0, 0.30], bottom flush on the ground
    # (not hovering). Kinematic = immovable furniture (real static collision, like the
    # floor). The cube rests on top via gravity; the gripper grasps it there.
    platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),    # box [0,0.30], on the floor
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
    # STABLE curriculum (no aggressive anneal): a fixed straddle/fly-in mix. The
    # 0.85->0.15 anneal destabilized PPO on the harder table task (success peaked ~0.59
    # then DECLINED to 0.27 as the distribution shifted faster than the policy could
    # track -> catastrophic forgetting). A fixed 0.6 mix is a stationary target -> stable
    # convergence (high straddle + decent fly-in).
    curriculum_p_start: float = 0.6
    curriculum_p_end: float = 0.6
    anneal_steps: float = 60000.0
    # ANNEAL THE HOVER REWARD: the approach reward (earned just by being near/over the
    # cube) decays over training, so hovering stops paying and the only way to keep score
    # is to descend -> grab -> lift -> carry -> place. (Hover was a local optimum.)
    approach_w_start: float = 1.0        # early: guide the drone toward the cube
    approach_w_end: float = 0.2          # late: hovering barely pays -> must actually pick
    approach_anneal_steps: float = 100000.0

    # --- competence-gated curriculum + reward staging (OPT-IN; defaults off so the
    # converged fixed-0.6 config and model_650 path are unchanged) -----------------
    # The earlier 0.85->0.15 anneal failed because it was TIME-based: difficulty rose
    # on a fixed clock regardless of skill -> "shifted faster than the policy could
    # track" -> catastrophic forgetting. Here the anneal is gated on MEASURED fly-in
    # success: cur_p only ratchets down when the policy is already succeeding at the
    # current mix, and if a step hurts, the controller stalls and waits. Self-pacing.
    adaptive_curriculum: bool = False    # ratchet cur_p -> curr_floor as fly-in success rises
    reward_staging: bool = False         # learn pickup first, phase placement in after
    curr_grasp_thresh: float = 0.80      # fly-in grasp EMA needed to ramp placement reward
    curr_place_thresh: float = 0.65      # fly-in place EMA needed to lower cur_p one step
    curr_step: float = 0.05              # cur_p decrement per ratchet
    curr_floor: float = 0.0              # target floor (pure fly-in)
    curr_dwell: int = 4000               # env-steps to hold after any change (PPO re-settle)
    curr_ema: float = 0.99               # EMA decay over completed fly-in episodes
    place_gain_step: float = 0.10        # placement-reward ramp per ratchet (0 -> 1)

    # --- 3-stage reward curriculum (OPT-IN): hover -> grab -> carry/drop ----------
    # Decomposes the task so each sub-skill gets DENSE reward, fixing the "payoff only
    # after a grasp" sparsity that traps the policy in a hover. Stage 1 adds a dense
    # descent reward (pull the body to grasp height when aligned) + softens the
    # table-touch (small penalty, no episode-end) so the drone can EXPLORE the descent
    # instead of fearing the -50 crash. Stage transitions are competence-gated.
    # Run with --cur_p 0.0 (all fly-in) so it learns the descent from altitude.
    staged_curriculum: bool = False
    stage_dwell: int = 3000              # env-steps min per stage (let it settle)
    stage_hover_thresh: float = 0.70     # frac horizontally over cube (EMA) to leave stage 0
    stage_grasp_thresh: float = 0.45     # grasp_rate EMA to leave stage 1
    stage_ema: float = 0.995             # per-step EMA decay for stage metrics
    descend_coef: float = 6.0            # dense "get body to grasp height when aligned" reward
    table_touch_pen: float = 1.0         # soft table-touch penalty (stage>=1; replaces crash-end)

    # --- reverse curriculum (OPT-IN; Florensa et al. 2017) -----------------------
    # The literature-standard fix for "won't commit to the descent": start episodes AT
    # the grasp pose (trivial grasp) and expand the start distribution outward (spawn
    # height + horizontal offset) ONLY as grasp competence grows. Each env samples its
    # own difficulty in [0, ceiling] so easy starts are retained (no forgetting). This
    # replaces the binary straddle/altitude split (a cliff) with a smooth ramp. Uses the
    # original full reward + soft table-touch. Run WITHOUT staged_curriculum/cur_p.
    reverse_curriculum: bool = False
    rc_h_max: float = 0.75               # max spawn height ADDED above grasp pose (~full altitude)
    rc_r_max: float = 0.40               # max horizontal spawn offset from the cube
    rc_step: float = 0.05                # difficulty-ceiling increment per expansion
    rc_floor_thresh: float = 0.50        # grasp_rate EMA needed to expand the start distribution
    rc_dwell: int = 3000                 # env-steps to hold after each expansion
    rc_ema: float = 0.99                 # per-step EMA decay for the grasp gate

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
        # competence-gated curriculum / reward-staging controller state
        self._cur_p_dyn = float(self.cfg.curriculum_p_start)
        self._place_gain = 0.0 if self.cfg.reward_staging else 1.0
        self._grasp_ema = 0.0                # EMA of fly-in (far-start) grasp rate
        self._place_ema = 0.0                # EMA of fly-in (far-start) place success
        self._curr_ema_init = False          # seed EMAs on first real sample
        self._last_curr_change = 0           # env-step of last cur_p / place_gain change
        self._is_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 3-stage reward-curriculum controller (hover -> grab -> carry/drop)
        self._stage = 0
        self._hover_ema = 0.0; self._gr_ema = 0.0; self._pl_ema = 0.0
        self._stage_ema_init = False
        self._last_stage_change = 0
        # reverse-curriculum controller state
        self._rc_p = 0.0                     # start-distribution difficulty ceiling [0,1]
        self._rc_grasp_ema = 0.0; self._rc_ema_init = False; self._last_rc_change = 0
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
        self.platform = RigidObject(self.cfg.platform)
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["platform"] = self.platform
        if getattr(self.cfg, "render_camera", False):
            # cinematic 2-light rig for the demo film: cool sky-dome fill + a warm
            # directional key with a soft penumbra (angle>0) so the cube/table cast
            # readable, real-looking shadows. Render-path only -> the visuomotor
            # training cameras still see the plain bright dome below.
            dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.82, 0.88, 1.0))
            dome.func("/World/Light", dome)
            key = sim_utils.DistantLightCfg(intensity=3000.0, angle=2.0,
                                            color=(1.0, 0.95, 0.86))
            key.func("/World/KeyLight", key, orientation=(0.94, -0.342, 0.0, 0.0))
        else:
            light = sim_utils.DomeLightCfg(intensity=2000.0)
            light.func("/World/Light", light)
        if self.cfg.use_cameras:
            from isaaclab.sensors import TiledCamera
            from skyvla_isaac.snatch import perception as P
            self._top_cam = TiledCamera(P.top_camera_cfg())
            self._bottom_cam = TiledCamera(P.bottom_camera_cfg())
            self.scene.sensors["top_cam"] = self._top_cam
            self.scene.sensors["bottom_cam"] = self._bottom_cam
        if getattr(self.cfg, "render_camera", False):
            from isaaclab.sensors import Camera, CameraCfg
            ccfg = CameraCfg(
                prim_path="/World/snatch_render_cam",
                height=self.cfg.render_cam_h, width=self.cfg.render_cam_w,
                update_period=0.0, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.05, 80.0)))
            self._render_cam = Camera(ccfg)
            self.scene.sensors["render_cam"] = self._render_cam

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
        tip[:, 2] = tip[:, 2] - 0.07                # cage mounted directly under the body (base-0.07)
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
        self._lifted = obj_p[:, 2] > self.cfg.surface_z + 0.025 + self.cfg.grasp_clear
        self._held = self._lifted & (self._d_reach < 0.10)
        self._carry = self._held
        self._success = self._held & (self._d_goal < 0.18)
        # aliases for eval_snatch's metric contract
        self._grasped = self._held
        self._placed = self._success
        self._grasp_pos_err = self._d_reach
        # crash only at/below the table top (body physically rests at surface_z+0.025).
        # Was surface_z+0.05 = just 4.5cm under the 0.395 grasp hover -> from-altitude
        # descents kept tripping it, so success DECLINED as the curriculum added fly-ins.
        self._table_touch = base_p[:, 2] < self.cfg.surface_z
        out_of_bounds = (torch.norm(base_p[:, :2], dim=-1) > 5.0) \
            | (torch.norm(self.robot.data.root_lin_vel_w, dim=-1) > 12.0)
        if self.cfg.staged_curriculum or self.cfg.reverse_curriculum:
            # soft table-touch: bumping the table no longer ends the episode (penalized in
            # the reward instead) so the policy can explore the descent toward the cube.
            self._crashed = out_of_bounds
        else:
            self._crashed = self._table_touch | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # competence-gated curriculum + reward staging (opt-in; no-op when off)
        if self.cfg.adaptive_curriculum or self.cfg.reward_staging:
            self._update_curriculum(self._crashed | time_out)
        if self.cfg.staged_curriculum:
            self._update_stage()
        if self.cfg.reverse_curriculum:
            self._update_rc()
        self.extras["log"] = {
            "metrics/grasp_rate": self._held.float().mean(),
            "metrics/place_success": self._success.float().mean(),
            "metrics/obj_to_goal": self._d_goal.mean(),
        }
        if self.cfg.staged_curriculum:
            dev = self.device
            self.extras["log"].update({
                "stage/stage": torch.tensor(float(self._stage), device=dev),
                "stage/hover_ema": torch.tensor(self._hover_ema, device=dev),
                "stage/grasp_ema": torch.tensor(self._gr_ema, device=dev),
                "stage/place_ema": torch.tensor(self._pl_ema, device=dev),
            })
        if self.cfg.reverse_curriculum:
            dev = self.device
            self.extras["log"].update({
                "revcurr/difficulty": torch.tensor(self._rc_p, device=dev),
                "revcurr/grasp_ema": torch.tensor(self._rc_grasp_ema, device=dev),
            })
        if self.cfg.adaptive_curriculum or self.cfg.reward_staging:
            dev = self.device
            self.extras["log"].update({
                "curriculum/cur_p": torch.tensor(self._cur_p_dyn, device=dev),
                "curriculum/place_gain": torch.tensor(self._place_gain, device=dev),
                "curriculum/grasp_ema_flyin": torch.tensor(self._grasp_ema, device=dev),
                "curriculum/place_ema_flyin": torch.tensor(self._place_ema, device=dev),
            })
        return self._crashed, time_out

    # ------------------------------------------------------------------ #
    def _update_curriculum(self, finished: torch.Tensor):
        """Competence-gated controller. Tracks fly-in (far-start) grasp/place EMAs
        and (1) phases placement reward in once pickup is reliable, then (2) ratchets
        the straddle fraction cur_p down toward pure fly-in once placement is reliable.
        Each change is gated on EMA thresholds + a dwell window, so difficulty only
        rises with skill and a harmful step stalls further changes until recovery."""
        far = finished & (~self._is_near)
        n_far = int(far.sum().item())
        if n_far > 0:
            g = self._held[far].float().mean().item()
            p = self._success[far].float().mean().item()
            if not self._curr_ema_init:
                self._grasp_ema, self._place_ema, self._curr_ema_init = g, p, True
            else:
                w = self.cfg.curr_ema ** n_far     # more episodes finished -> faster blend
                self._grasp_ema = w * self._grasp_ema + (1 - w) * g
                self._place_ema = w * self._place_ema + (1 - w) * p
        step = getattr(self, "_train_steps", 0)
        if step - self._last_curr_change < self.cfg.curr_dwell:
            return
        changed = False
        if (self.cfg.reward_staging and self._place_gain < 1.0
                and self._grasp_ema >= self.cfg.curr_grasp_thresh):
            self._place_gain = min(1.0, self._place_gain + self.cfg.place_gain_step)
            changed = True
        elif (self.cfg.adaptive_curriculum and self._place_gain >= 1.0
                and self._cur_p_dyn > self.cfg.curr_floor
                and self._place_ema >= self.cfg.curr_place_thresh):
            self._cur_p_dyn = max(self.cfg.curr_floor, self._cur_p_dyn - self.cfg.curr_step)
            changed = True
        if changed:
            self._last_curr_change = step
            print(f"[curriculum] step={step} cur_p={self._cur_p_dyn:.2f} "
                  f"place_gain={self._place_gain:.2f} grasp_ema={self._grasp_ema:.2f} "
                  f"place_ema={self._place_ema:.2f}", flush=True)

    def _update_stage(self):
        """3-stage curriculum controller. EMAs of (hover over cube / grasp / place) and
        advances stage 0->1 once it reliably hovers over the cube, 1->2 once it grasps."""
        hov = (self._horiz < 0.08).float().mean().item()    # horizontally over the cube
        grb = self._held.float().mean().item()
        plc = self._success.float().mean().item()
        a = self.cfg.stage_ema
        if not self._stage_ema_init:
            self._hover_ema, self._gr_ema, self._pl_ema = hov, grb, plc
            self._stage_ema_init = True
        else:
            self._hover_ema = a * self._hover_ema + (1 - a) * hov
            self._gr_ema = a * self._gr_ema + (1 - a) * grb
            self._pl_ema = a * self._pl_ema + (1 - a) * plc
        step = getattr(self, "_train_steps", 0)
        if step - self._last_stage_change < self.cfg.stage_dwell:
            return
        if self._stage == 0 and self._hover_ema >= self.cfg.stage_hover_thresh:
            self._stage = 1; self._last_stage_change = step
            print(f"[stage] -> 1 GRAB  step={step} hover_ema={self._hover_ema:.2f}", flush=True)
        elif self._stage == 1 and self._gr_ema >= self.cfg.stage_grasp_thresh:
            self._stage = 2; self._last_stage_change = step
            print(f"[stage] -> 2 CARRY/DROP  step={step} grasp_ema={self._gr_ema:.2f}", flush=True)

    def _update_rc(self):
        """Reverse-curriculum controller (Florensa 2017): expand the start-state
        distribution outward (higher/farther spawns) only once grasp is reliable at the
        current spread. Self-paces -- if an expansion drops grasp below the gate, it
        stalls until the policy recovers."""
        g = self._held.float().mean().item()
        a = self.cfg.rc_ema
        if not self._rc_ema_init:
            self._rc_grasp_ema = g; self._rc_ema_init = True
        else:
            self._rc_grasp_ema = a * self._rc_grasp_ema + (1 - a) * g
        step = getattr(self, "_train_steps", 0)
        if step - self._last_rc_change < self.cfg.rc_dwell:
            return
        if self._rc_p < 1.0 and self._rc_grasp_ema >= self.cfg.rc_floor_thresh:
            self._rc_p = min(1.0, self._rc_p + self.cfg.rc_step)
            self._last_rc_change = step
            print(f"[revcurr] difficulty={self._rc_p:.2f} grasp_ema={self._rc_grasp_ema:.2f} "
                  f"step={step}", flush=True)

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
        cube_h = torch.clamp(self._obj_p[:, 2] - (self.cfg.surface_z + 0.025), 0.0, 0.15)  # lift off table
        place = held * (1.0 - torch.tanh(self._d_goal / 0.35))           # DOMINANT delivery
        carry_up = held * torch.clamp(self._obj_p[:, 2] - (self.cfg.surface_z + 0.085), 0.0, 0.25)
        carry_prog = held * (self._prev_d_goal - self._d_goal)
        success = (self._held & (self._d_goal < 0.18)).float()
        self._prev_d_goal = self._d_goal.clone()

        if self.cfg.staged_curriculum:
            # Stage 0: hover over cube (reach+align). Stage 1: + DENSE descent reward
            # (pull body to grasp height when horizontally aligned) + grab + lift, minus a
            # soft table-touch penalty. Stage 2: + carry + deliver. Dense signal per stage
            # -> the descent has a downward gradient instead of only a post-grasp payoff.
            grasp_z = self.cfg.surface_z + 0.025 + 0.07     # body height for tip-at-cube (~0.395)
            z_err = (self._base_p[:, 2] - grasp_z).abs()
            # gate by a SOFT "roughly over the cube" term (0.15 scale) -- the tight align
            # (0.06) was ~0 at the hover standoff, so the descent reward was switched off.
            over_cube = 1.0 - torch.tanh(self._horiz / 0.15)
            descend = over_cube * (1.0 - torch.tanh(z_err / 0.15))
            touch = self._table_touch.float()
            r = 1.0 * reach + 0.5 * align
            if self._stage >= 1:
                r = r + self.cfg.descend_coef * descend + 1.5 * grab + 60.0 * cube_h \
                    - self.cfg.table_touch_pen * touch
            if self._stage >= 2:
                r = r + 40.0 * place + 30.0 * carry_up + 25.0 * carry_prog + 80.0 * success
            return r - 0.01

        # reward staging: pickup terms always on; delivery ramps in via _place_gain.
        pg = self._place_gain
        # ANNEALED HOVER: the approach reward (close to the cube, incl. hovering above it)
        # decays start->end over training. Sharper 3D distance (/0.3) so it pulls the
        # gripper DOWN onto the cube, not just laterally over it. As it decays, lift+carry+
        # place (constant, large) dominate -> the policy must descend->grab->lift->deliver.
        prog = min(1.0, getattr(self, "_train_steps", 0) / self.cfg.approach_anneal_steps)
        w_app = self.cfg.approach_w_start + (self.cfg.approach_w_end - self.cfg.approach_w_start) * prog
        approach = 1.0 - torch.tanh(self._d_reach / 0.3)
        held_bonus = held * 5.0                                          # clear "you grabbed it" milestone
        r = (w_app * approach + 0.5 * align + 2.0 * grab + 60.0 * cube_h + held_bonus
             + pg * (40.0 * place + 30.0 * carry_up + 25.0 * carry_prog + 80.0 * success) - 0.01)
        return r

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
        grasp_z = self.cfg.surface_z + 0.025 + 0.07         # body height so tip is at the cube
        root = self.robot.data.default_root_state[env_ids].clone()
        if self.cfg.reverse_curriculum:
            # Florensa reverse curriculum: spawn AT the grasp pose, expand outward as the
            # ceiling self._rc_p grows. Per-env difficulty d ~ U[0, rc_p] -> a mix of easy
            # (at the cube) and hard (high+offset) starts at every stage (retains skills).
            d = torch.rand(n, device=self.device) * self._rc_p
            ang = torch.rand(n, device=self.device) * 6.2831853
            rad = d * self.cfg.rc_r_max
            root[:, 0] = obj_local[:, 0] + rad * torch.cos(ang)
            root[:, 1] = obj_local[:, 1] + rad * torch.sin(ang)
            root[:, 2] = grasp_z + d * self.cfg.rc_h_max
            self._is_near[env_ids] = d < 0.15
        else:
            # curriculum: a fraction start straddling the cube at grasp height (jaws open),
            # the rest from altitude; straddle fraction anneals high->low.
            if self.cfg.adaptive_curriculum:
                cur_p = self._cur_p_dyn                   # competence-gated (controller)
            else:
                prog = min(1.0, getattr(self, "_train_steps", 0) / self.cfg.anneal_steps)
                cur_p = self.cfg.curriculum_p_start + (self.cfg.curriculum_p_end - self.cfg.curriculum_p_start) * prog
            self._cur_p = cur_p
            near = torch.rand(n, device=self.device) < cur_p
            self._is_near[env_ids] = near                 # remember start stratum for the EMA gate
            root[near, 0] = obj_local[near, 0]
            root[near, 1] = obj_local[near, 1]
            root[near, 2] = grasp_z                       # tip (base-0.07) at the cube on the table
        root[:, :3] += origins
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)
        # goal drop-zone near the cube, at carry height
        t = torch.zeros(n, 3, device=self.device)
        t[:, :2] = obj_local[:, :2] + (torch.rand(n, 2, device=self.device) - 0.5) * self.cfg.goal_offset_diam
        t[:, 2] = self.cfg.surface_z + 0.25           # deliver up off the table
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
