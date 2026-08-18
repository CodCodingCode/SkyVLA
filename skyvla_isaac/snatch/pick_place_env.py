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

import math
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

    # --- TWO-PLATFORM transport (OPT-IN): pick from A, carry to B, deposit on B ----
    # Platform B is the DELIVERY pad. When enabled, the goal stops being a waypoint
    # floating 25cm over platform A and becomes "the cube at rest on B's top". B is
    # kinematic like A, and its pose is rewritten per-env at reset -> the separation
    # is a curriculum knob and the bearing is randomized (no fixed +X habit).
    two_platform: bool = False           # enables platform B + B-anchored goal
    plat_b: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PlatformB",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.38, 0.45))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.15)),
    )
    plat_sep: float = 1.5                # CURRENT separation A->B (curriculum start)
    plat_sep_max: float = 2.0            # ceiling; env_spacing/episode scale off THIS
    plat_sep_step: float = 0.25          # expansion increment per competence gate
    plat_sep_thresh: float = 0.55        # deposit/arrival EMA needed to expand
    plat_sep_dwell: int = 3000           # env-steps min between expansions
    # DEPOSIT success (place stage): cube at rest on B, jaws open, drone not holding it.
    place_radius: float = 0.20           # cube within this of B's centre (XY)
    place_settle_v: float = 0.10         # cube speed (m/s) below which it counts settled
    release_only: bool = False           # place stage: pay for the deposit, not the hold
    arrive_radius: float = 0.30          # nav stage: "over B" gate (XY, cube still held)
    # WELDED GRIP (carry stage): force a[4] = +1 every step, so the cage is commanded shut
    # for the whole ride and the policy effectively controls only [vx vy vz yaw_rate]. The
    # action vector stays 5-wide so warm starts and the 3-policy runner still load; the 5th
    # element is simply overridden. Pair with carry_demo_p = 1.0 -- with the jaws welded
    # shut a drone that spawns away from the cube can never scoop it, so every env must
    # start already holding. That turns the carry stage into pure transport.
    grip_closed: bool = False

    # --- PLACE-ONLY stage (OPT-IN): gentle deposit from an already-holding start ---
    # Every episode starts with the cube ALREADY seated in the closed cage, airborne
    # near platform B (see _reset_idx). The only skill trained is: centre over B, lower
    # gently, set down, open the jaws, don't knock it off. All the height income that
    # traps stage 2 in a hover (the exponential lift ladder + carry_up) is GONE here --
    # see _place_reward. Requires two_platform + grasp_latch=False + carry_demo_p>0.
    place_only: bool = False
    place_spawn_r: float = 0.8           # spawn disc radius around B's centre (XY, m)
    place_spawn_h_lo: float = 0.55       # spawn BODY altitude band (m). At the seat pose the
    place_spawn_h_hi: float = 0.85       # body is 0.395 -> this is 0.16-0.46 m of descent
    # COMPETENCE-GATED START RAMP. The end distribution is exactly the one above; this only
    # controls how fast we get there. At difficulty 0 the drone spawns dead over B with a
    # 5.5cm descent (a near-trivial deposit); the radius and altitude expand to the full
    # spec only as the deposit rate holds up. Per-env difficulty is U[0,p] with a frontier
    # bias, so easy starts are RETAINED at every stage -- the repo's proven pattern (a
    # time-based anneal caused catastrophic forgetting here before; a competence gate did
    # not). Set place_curriculum False to train the full distribution from step one.
    place_curriculum: bool = True
    place_h_easy: float = 0.45           # body altitude at difficulty 0
    place_curr_thresh: float = 0.25      # deposit-rate EMA needed to expand one step
    place_curr_step: float = 0.10        # difficulty increment per expansion
    place_curr_dwell: int = 3000         # env-steps to hold after an expansion (PPO re-settle)
    place_curr_ema: float = 0.99
    place_curr_start: float = 0.0        # resume point (set on crash-restart)
    gentle_v: float = 0.15               # safe descent speed inside place_taper_h of the seat
    place_taper_h: float = 0.10          # within this of the seat, the cap IS gentle_v
    place_taper_ramp: float = 0.30       # above place_taper_h, the cap ramps back to `speed`
                                         # over this height -> full cruise 40cm out, 0.15 m/s
                                         # for the last 10cm. (Scaling the cap by seat_err
                                         # directly instead only hit gentle_v in the last CM:
                                         # the descent entered the last 10cm still at 1.5 m/s.)
    # CAGE GEOMETRY (drone_snatch.urdf, base frame; tip = base - 0.07):
    #   wall spans tip+-0.025 ; inward lip spans tip-0.025 .. tip-0.017
    # so a shelf-seated cube's BOTTOM is lip_h above the cage's lowest point. The lips
    # bottom out on the pad, hence the cube is always released from lip_h -> an 8mm
    # free fall (~0.40 m/s) is the GEOMETRIC FLOOR of "gentle" for this gripper.
    lip_h: float = 0.008
    release_jaw: float = 0.014           # jaw pos below which the lip aperture exceeds the
                                         # cube (5cm) -> the cube is physically FREE. Deposit
                                         # success requires this: without it the policy scores
                                         # by lowering the cube to rest height and never
                                         # letting go (_held is already False once it rests).
    # RELEASE LATCH (place_only): hold the jaws open for this many policy steps once the
    # policy commands open. MEASURED: from the seated pose a forced a4=-1 takes exactly THREE
    # consecutive steps to clear release_jaw (jaw 28.99 -> 24.39 -> 9.59 mm). PPO explores with
    # per-step INDEPENDENT Gaussian noise, so the probability of the required 3-in-a-row is
    # p^3 = 2.6e-7 -> 0.21 discoveries per iteration across all 2048 envs, which is exactly the
    # trickle observed (deposit_rate below display resolution for 182 iterations). The latch
    # turns a 3-step motor program into a 1-step DECISION: p^3 -> p, ~5200 discoveries per
    # iteration. This is also the honest model of a real gripper, whose release is a commanded
    # sequence and not a per-20ms re-vote.  0 disables.
    release_latch: int = 4

    # flight + task params
    speed: float = 1.5                   # proven value; eases precise bodily descent-grasp
    yaw_rate_scale: float = 1.0
    kv: float = 18.0
    approach_brake: bool = False         # TRIED: cap horiz speed near cube to force centered grasp.
                                         # Verdict: HURT it (stuck at 16cm, grasp ~0.10 < baseline 0.13) --
                                         # the cap also blocks fine re-centering near the cube. Reverted.
    # GRASP LATCH: once the cube is inside the closed cage, hold it RIGIDLY to the gripper so it
    # can't slide/fall while the drone moves. The floorless friction-cage under-models a real grip
    # (off-center -> a wall loses contact -> cube drops out the bottom); this models a gripper that
    # actually closes on the object. Works regardless of off-center -> eliminates the slide that
    # capped distance. Released when the gripper is commanded open (for placement).
    grasp_latch: bool = True
    latch_r: float = 0.035               # cube center within this of cage center (+ gripper closing) -> latched
    grip_travel: float = 0.035           # jaw prismatic travel (m); must match the URDF joint limit
    # TERMINAL CENTERING: decouple cruise from the grab. The policy flies the drone toward the cube;
    # within term_r the horizontal command is blended toward a proportional pull onto the cube's xy,
    # cancelling the overshoot (drone arrives ~0.34m/s and sails off-center, esp. at distance). Unlike
    # the failed speed-cap brake, this ACTIVELY centers (drives toward the cube) instead of blocking it.
    terminal_center: bool = False
    term_r: float = 0.12                  # within this horizontal dist, centering assist ramps in
    term_kp: float = 5.0                  # proportional centering gain (-> v toward cube, capped at speed)
    pickup_speed: float = 0.12            # safe CONSTANT approach speed near the cube (user-requested):
                                          # speed target ramps from `speed` (far) down to this (near)
    w_decel: float = 4.0                  # penalty weight for exceeding the safe speed target (rushing in)
    k_att: float = 4.0
    k_damp: float = 0.6
    obj_spawn_diam: float = 0.8          # proven distance for convergence (scale up later)
    cube_mass: float = 0.05              # 0.05 is liftable by the cage; heavier slips out
    cube_size: float = 0.05              # smaller (e.g. 0.035) gives the cage CLEARANCE to descend
                                         # around it without clipping/knocking it (diagnosed failure)
    early_close_pen: float = 0.5         # penalty for closing the cage when NOT near the cube
    rc_horiz_start: float = 0.40         # reverse-curriculum: horizontal spawn offset only ramps in
                                         # for difficulty d > this (learn clean VERTICAL descent first)
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
    approach_anneal_steps: float = 20000.0   # faster: hovering must stop paying WHILE it learns descent

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
    # SIDE SPAWN (stage 0 = NAVIGATE -> hover, not just descend): spawn the drone to the
    # side of the cube at a random angle/radius/altitude so every episode starts with a
    # real cruise to the hover point. Pair with a lowered stage_hover_thresh -- cruise
    # time caps the achievable hover EMA.
    side_spawn_max: float = 0.0          # >0 enables; max horizontal spawn radius from the cube (m)
    side_spawn_min: float = 0.5          # min spawn radius (every episode keeps some approach)
    side_spawn_h_lo: float = 0.8         # spawn altitude range (m)
    side_spawn_h_hi: float = 1.6
    # HOVER ANNEAL (master-then-diminish): once stage 1 opens, decay the STANDOFF hover
    # payoff (reach+align) to 20% over this many env-steps, so sitting at the hover point
    # stops paying and the descent/grab terms take over. The nav cruise term is NOT
    # annealed -- with side spawns it is what pays for arriving at all.
    stage1_hover_anneal: float = 0.0     # env-steps; 0 disables (hover stays full-weight)
    # LATCH-READY reward (stage>=1): sharp bonus peaked at the EXACT pose the latch needs
    # (cube inside the cage footprint, tip at cube height). The broad descend term (0.15
    # scale) barely differentiates the last 5cm -- this puts a real gradient across it.
    latch_ready_coef: float = 0.0        # 0 disables
    start_stage: int = 0                 # start the 3-stage controller here (2 = all rewards on
                                         # from step 0; for warm-started expert policies)
    surround_only: bool = False          # train ONLY "cage surrounds the cube" (exponential horiz
                                         # precision at cube height); all grip use is taxed
    # OVERHEAD-FIRST curriculum (user-designed): the only paid spot is DIRECTLY ABOVE the
    # block at a target altitude that starts high and is LOWERED one rung at a time, gated
    # on proven centering at the current rung. horiz->0 is learned first by construction;
    # descent is doled out only as earned; ends at nest depth. Kills the lean exploit:
    # the cube is never inside the paid band until precision justifies it.
    overhead_first: bool = False
    oh_dz_start: float = 0.04            # initial setpoint: tip this far ABOVE cube centre (m).
                                         # (Was 0.20 -- the policy organically converged to a
                                         # centred hover at ~3cm and a 20cm rung across a reward
                                         # valley was unreachable; anchor the ladder where the
                                         # policy IS and walk the last cm to nest.)
    oh_dz_end: float = 0.0               # final setpoint: nest (tip at cube centre)
    oh_dz_step: float = 0.01             # one cm per earned rung
    oh_gate: float = 0.35                # EMA frac of in-band steps centred (gate_horiz) to earn a rung
    oh_gate_horiz: float = 0.008         # centring bar per rung. 0.3cm was tried and sat at the
                                         # physical noise floor (x,y floor ~0.6-0.7cm at std 0.25);
                                         # cage widened to a 2.4cm fit window instead -- 0.8cm bar
                                         # now carries ~3x geometric margin.
    oh_gate_dz: float = 0.010            # vertical alignment bar per rung (user: z must align too)
    oh_dwell: int = 2000                 # min env-steps between rungs
    lift_ramp_steps: float = 30000.0     # progress seesaw: env-steps over which squeeze pay
                                         # depreciates (8->4) and the lift ladder appreciates (x2)
    carry_demo_p: float = 0.0            # fraction of resets that spawn ALREADY CARRYING (cube
                                         # seated in the closed cage at altitude): a 15-step
                                         # coordinated climb cannot be sampled from white noise --
                                         # the carry's value must be experienced, like the squeeze

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
    rc_step: float = 0.025                # smooth, adaptable jumps
    rc_floor_thresh: float = 0.13         # push past 0.21m wall (stochastic under-reports ~2x; det ~0.22).
                                          # FLOOR 0.12 - 0.10 destroyed the grasp earlier; never go below 0.12
                                         # instantaneous grasp_rate, which under-reports the real
                                         # (deterministic) capability ~2x: at rc_p 0.10 the policy
                                         # grasps 77% deterministically but this metric reads ~0.34,
                                         # so 0.50 stalled the curriculum prematurely. 0.28 lets it
                                         # keep expanding while the policy is genuinely ~70% capable.
    rc_dwell: int = 6000                 # more time to learn deceleration at each new distance
    rc_ema: float = 0.99                 # per-step EMA decay for the grasp gate
    rc_start: float = 0.0                # starting difficulty (set on crash-restart to resume)

    # --- DISTANCE curriculum: slowly fly in from further and further (up to rc_dist_max) ---
    # Works AROUND the descent wall: keep the grasp short-range (the ~90% regime) but ramp
    # the horizontal FLY-IN distance from 0 toward rc_dist_max as grasp competence holds.
    # The drone learns to navigate in from far, then do the short grasp it's reliable at.
    rc_distance_mode: bool = False
    rc_dist_max: float = 10.0            # max spawn distance from the cube (m)
    rc_fixed_height: float = 0.02        # small grasp-friendly descent in distance mode (90% regime)
    nav_progress_w: float = 1.0          # dense reward for closing distance to the cube (any range)

    def __post_init__(self):
        self.observation_space = (1024 if self.use_cameras else 0) + 14   # +3 for goal


class DroneSnatchEnv(DirectRLEnv):
    cfg: DroneSnatchEnvCfg

    def __init__(self, cfg: DroneSnatchEnvCfg, render_mode: str | None = None, **kw):
        cfg.observation_space = (1024 if cfg.use_cameras else 0) + 14   # recompute post-toggle (+goal)
        cfg.object.spawn.mass_props.mass = cfg.cube_mass               # set pre-spawn
        cfg.object.spawn.size = (cfg.cube_size,) * 3                   # smaller cube -> cage clearance
        cfg.object.init_state.pos = (0.0, 0.0, cfg.surface_z + 0.5 * cfg.cube_size)  # rest on table for ANY cube size
        super().__init__(cfg, render_mode, **kw)
        self._grip_i, _ = self.robot.find_joints("grip_.*")
        self._base_i, _ = self.robot.find_bodies("base")
        self._mass = self.robot.root_physx_view.get_masses().sum(-1).to(self.device)
        self._target = torch.zeros(self.num_envs, 3, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._raw_actions = torch.zeros_like(self._actions)   # pre-clamp; see _pre_physics_step
        self._rel_latch = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._prev_d_goal = z()
        self._held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._carry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latch_off = torch.zeros(self.num_envs, 3, device=self.device)
        # two-platform transport state: B's env-local XY, arrival/deposit flags, and the
        # separation curriculum controller (expand A->B once arrival is reliable).
        self._plat_b_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._d_plat_b = torch.zeros(self.num_envs, device=self.device)
        self._arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._deposited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # place-stage bookkeeping: per-episode worst cube speed measured while the cube was
        # within a lip-height of resting (the impact), and the previous step's seated flag
        # (so the held->released transition can be priced on the exact action that let go).
        self._impact_v = z()
        self._prev_seated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cube_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cube_on_floor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_d_b = z()                 # baseline for the scale-free progress term
        # place start-ramp controller state
        self._place_p = float(self.cfg.place_curr_start)
        self._place_ema = 0.0
        self._place_ema_init = False
        self._last_place_change = 0
        self._sep = float(self.cfg.plat_sep)
        self._sep_ema = 0.0
        self._sep_ema_init = False
        self._last_sep_change = 0
        # competence-gated curriculum / reward-staging controller state
        self._cur_p_dyn = float(self.cfg.curriculum_p_start)
        self._place_gain = 0.0 if self.cfg.reward_staging else 1.0
        self._grasp_ema = 0.0                # EMA of fly-in (far-start) grasp rate
        self._place_ema = 0.0                # EMA of fly-in (far-start) place success
        self._curr_ema_init = False          # seed EMAs on first real sample
        self._last_curr_change = 0           # env-step of last cur_p / place_gain change
        self._is_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 3-stage reward-curriculum controller (hover -> grab -> carry/drop). start_stage
        # skips the gate-earning for warm-started experts (all rewards on from step 0;
        # also avoids re-earning gates after every crash-restart).
        self._stage = int(getattr(self.cfg, "start_stage", 0))
        self._stage1_step = 0                # hover-anneal clock (runs from boot if start_stage>=1)
        # overhead-first curriculum state: current altitude setpoint + competence EMA
        self._oh_dz = float(self.cfg.oh_dz_start)
        self._oh_ema = 0.0
        self._oh_last = 0
        self._hover_ema = 0.0; self._gr_ema = 0.0; self._pl_ema = 0.0
        self._stage_ema_init = False
        self._last_stage_change = 0
        # reverse-curriculum controller state
        self._rc_p = float(self.cfg.rc_start)   # start-distribution difficulty ceiling [0,1]
        self._rc_grasp_ema = 0.0; self._rc_ema_init = False; self._last_rc_change = 0
        self._prev_d_reach = torch.zeros(self.num_envs, device=self.device)  # nav-progress baseline
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
        if self.cfg.two_platform:
            self.plat_b = RigidObject(self.cfg.plat_b)
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["platform"] = self.platform
        if self.cfg.two_platform:
            self.scene.rigid_objects["plat_b"] = self.plat_b
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
        # KEEP THE UNCLAMPED ACTION. The clamp below is behaviour-preserving but
        # gradient-DESTROYING: every a4 above +1 drives the identical jaw target, so the actor's
        # grip mean can drift arbitrarily far out of range with nothing pushing back. Measured on
        # model_10900: grip mean +2.108 (sigma 0.282), i.e. the action that first moves the jaw is
        # 5.15 sigma away and a full release is 8.17 sigma -- the release was not "hard to
        # explore", it was outside the support of the policy. _place_reward taxes the overshoot.
        self._raw_actions = actions
        self._actions = actions.clamp(-1.0, 1.0)
        if self.cfg.grip_closed:
            # Weld the cage shut: whatever the actor emits on a[4] is discarded and the
            # jaws are driven fully closed every step. The policy is left with flight only.
            self._actions = self._actions.clone()
            self._actions[:, 4] = 1.0
        self._train_steps = getattr(self, "_train_steps", 0) + 1

    def _apply_action(self):
        a = self._actions
        # velocity-tracking wrench on the base (direct velocity control)
        v_des = a[:, :3] * self.cfg.speed
        # FORCE deceleration on approach: the policy ignores the decel reward and arrives at
        # ~0.34 m/s, overshooting so it grasps OFF-CENTER (centered only 39% at 22cm) -> the cube
        # isn't contained by the 7cm jaws and slides out. Cap the horizontal speed by the horizontal
        # distance to the cube so it MUST slow as it nears -> arrives centered -> firm cage. Far away
        # the cap is >= speed (no effect on fast fly-in), only the final approach is throttled.
        if self.cfg.approach_brake:
            tip = self._tip_w()
            d_h = torch.norm(self.object.data.root_pos_w[:, :2] - tip[:, :2], dim=-1)
            v_cap = (d_h * 2.0).clamp(min=0.10, max=self.cfg.speed)
            vh = v_des[:, :2]
            vmag = torch.norm(vh, dim=-1, keepdim=True).clamp(min=1e-6)
            v_des = torch.cat([vh * torch.clamp(v_cap.unsqueeze(-1) / vmag, max=1.0), v_des[:, 2:3]], dim=-1)
        if self.cfg.terminal_center:
            tip = self._tip_w()
            err = self.object.data.root_pos_w[:, :2] - tip[:, :2]          # toward cube (world xy)
            d_h = torch.norm(err, dim=-1, keepdim=True)
            w = (1.0 - d_h / self.cfg.term_r).clamp(0.0, 1.0)              # 1 at cube -> 0 at term_r
            v_center = (err * self.cfg.term_kp).clamp(-self.cfg.speed, self.cfg.speed)
            v_des = torch.cat([(1 - w) * v_des[:, :2] + w * v_center, v_des[:, 2:3]], dim=-1)
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
        jaw = (a[:, 4] * 0.5 + 0.5) * self.cfg.grip_travel
        if self.cfg.place_only and self.cfg.release_latch > 0:
            # commit to an open command for release_latch steps (see cfg.release_latch): the
            # jaw needs 3 consecutive steps of travel to free the cube and per-step Gaussian
            # exploration effectively never strings 3 together.
            asks_open = jaw < self.cfg.release_jaw
            self._rel_latch = torch.where(
                asks_open, torch.full_like(self._rel_latch, self.cfg.release_latch),
                (self._rel_latch - 1).clamp(min=0))
            jaw = torch.where(self._rel_latch > 0, torch.zeros_like(jaw), jaw)
        tgt = jaw.unsqueeze(-1).repeat(1, len(self._grip_i))
        self.robot.set_joint_position_target(tgt, joint_ids=self._grip_i)
        # GRASP LATCH: rigidly carry the caged cube with the gripper (no slide/fall while moving)
        if self.cfg.grasp_latch and bool(self._latched.any()):
            idx = self._latched.nonzero(as_tuple=False).squeeze(-1)
            tip_w = self._tip_w()
            pose = torch.zeros(len(idx), 7, device=self.device)
            pose[:, :3] = tip_w[idx] + self._latch_off[idx]
            pose[:, 3] = 1.0                                # keep the cube level (quat w=1)
            vel = torch.zeros(len(idx), 6, device=self.device)
            vel[:, :3] = self.robot.data.root_lin_vel_w[idx]   # match drone velocity -> rigid carry
            self.object.write_root_pose_to_sim(pose, idx)
            self.object.write_root_velocity_to_sim(vel, idx)

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
        # GRASP LATCH state update: engage when the cube is inside the cage footprint (horiz) and
        # roughly at cage height AND the gripper is commanded closed; release when commanded open.
        if self.cfg.grasp_latch:
            tip_w = self._tip_w()
            cube_w = self.object.data.root_pos_w
            inside = (self._horiz < self.cfg.latch_r) & ((cube_w[:, 2] - tip_w[:, 2]).abs() < 0.05)
            closing = self._actions[:, 4] > 0.0
            engage = inside & closing & (~self._latched)
            off = cube_w - tip_w
            off[:, :2] = off[:, :2].clamp(-0.012, 0.012)   # seat the cube near cage center
            self._latch_off[engage] = off[engage]
            # stays latched once engaged until the gripper is opened (a[4] < -0.2)
            self._latched = (self._latched | (inside & closing)) & (self._actions[:, 4] > -0.2)
        self._d_goal = torch.norm(obj_p - self._target, dim=-1)
        self._obj_p, self._base_p = obj_p, base_p
        self._lifted = obj_p[:, 2] > self.cfg.surface_z + 0.5 * self.cfg.cube_size + self.cfg.grasp_clear
        self._held = self._lifted & (self._d_reach < 0.10)
        self._carry = self._held
        self._success = self._held & (self._d_goal < 0.18)
        if self.cfg.two_platform:
            # ARRIVED (nav stage): still holding the cube, and over B's pad.
            d_b = torch.norm(obj_p[:, :2] - self._plat_b_xy, dim=-1)
            self._d_plat_b = d_b
            self._arrived = self._held & (d_b < self.cfg.arrive_radius)
            # DEPOSITED (place stage): cube at rest ON B and NOT in the cage. Note a cube
            # settled on B is automatically ~_lifted (B's top == surface_z), so ~_held
            # falls out of the same geometry the grasp stage already uses.
            # ...which is also why the JAWS-OPEN test below is required: ~_held goes true the
            # instant the cube reaches rest height, so without it "lower it and keep gripping"
            # scores a full deposit and the policy never releases anything.
            rest_z = self.cfg.surface_z + 0.5 * self.cfg.cube_size
            v_obj = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
            jaw_pos = self.robot.data.joint_pos[:, self._grip_i].mean(-1)
            self._released = jaw_pos < self.cfg.release_jaw     # lip aperture > cube -> free
            # HOLDING = "the cube is in the cage", which for the place stage CANNOT be _held:
            # _held requires the cube above surface_z + cube/2 + grasp_clear = 0.385, but the
            # seat pose puts it at 0.333, so _held goes false for the last 5cm of the descent.
            # Gating the approach/lower/seat/release terms on _held paid the policy to hover
            # 5cm short of the pad and made `seated` unreachable by construction. Judged
            # geometrically instead: cube at the cage tip, jaws not yet open.
            self._holding = (self._d_reach < 0.05) & (~self._released)
            self._deposited = ((d_b < self.cfg.place_radius)
                               & ((obj_p[:, 2] - rest_z).abs() < 0.5 * self.cfg.cube_size)
                               & (v_obj < self.cfg.place_settle_v)
                               & self._released
                               & (~self._held))
            # IMPACT: worst cube speed seen this episode while it was within a lip-height of
            # resting. This is the honest "how hard did it land" number; its floor is the
            # lip_h free fall (~0.40 m/s), not the commanded descent speed.
            # window = the whole lip_h fall (held seat is rest_z+lip_h, landing is rest_z).
            # Sampled at policy rate (20 ms) over a 40 ms fall, so it reads a LOWER BOUND.
            near_rest = (obj_p[:, 2] - rest_z).abs() < 2.0 * self.cfg.lip_h
            self._impact_v = torch.where(near_rest, torch.maximum(self._impact_v, v_obj),
                                         self._impact_v)
            # the stage being trained decides which one IS success
            self._success = self._deposited if self.cfg.release_only else self._arrived
        # aliases for eval_snatch's metric contract
        self._grasped = self._held
        self._placed = self._success
        self._grasp_pos_err = self._d_reach
        # crash only at/below the table top (body physically rests at surface_z+0.025).
        # Was surface_z+0.05 = just 4.5cm under the 0.395 grasp hover -> from-altitude
        # descents kept tripping it, so success DECLINED as the curriculum added fly-ins.
        self._table_touch = base_p[:, 2] < self.cfg.surface_z
        oob_r = (self.cfg.rc_dist_max + 2.0) if self.cfg.rc_distance_mode else 5.0
        if self.cfg.side_spawn_max > 0.0:
            # side spawns reach side_spawn_max from the CUBE (itself up to obj_spawn_diam/2
            # from origin) -- without this, the far spawn tail dies at birth as "OOB"
            oob_r = max(oob_r, self.cfg.side_spawn_max + 1.5)
        if self.cfg.two_platform:
            # B sits plat_sep_max from A on any bearing -- without this the transit itself
            # terminates as out-of-bounds the moment the gap exceeds the default 5 m.
            oob_r = max(oob_r, self.cfg.plat_sep_max + 2.0)
        out_of_bounds = (torch.norm(base_p[:, :2], dim=-1) > oob_r) \
            | (torch.norm(self.robot.data.root_lin_vel_w, dim=-1) > 12.0)
        if self.cfg.staged_curriculum or self.cfg.reverse_curriculum or self.cfg.place_only:
            # soft table-touch: bumping the table no longer ends the episode (penalized in
            # the reward instead) so the policy can explore the descent toward the cube.
            # (place_only NEEDS this: the deposit pose sits 9.5cm over the pad, so a
            # hard crash-end on pad contact would terminate every good set-down attempt.)
            self._crashed = out_of_bounds
        if self.cfg.place_only:
            # A cube out of the cage and AT REST that is not a valid deposit has decided the
            # episode: this policy starts holding and never learns to re-grasp. END it rather
            # than bleed per-step for the remaining seconds -- with a bleed, flying out of
            # bounds to stop the bleeding outscores loitering (terminations aren't
            # value-bootstrapped) and PPO finds that exploit. Priced as a one-time fine in
            # _place_reward instead.
            # Testing "below pad height" is NOT enough: the pads are 1x1 m, so a cube dropped
            # onto B off-centre stays above surface_z -> no termination, no deposit, and the
            # measured result was a -1/step bleed for ~225 steps (holding_frac 0.73 -> 0.17,
            # mean reward -310). "At rest and not deposited" covers both cases.
            self._cube_on_floor = self._obj_p[:, 2] < self.cfg.surface_z
            self._cube_lost = ((~self._holding) & (v_obj < self.cfg.place_settle_v)
                               & (~self._deposited))
            self._crashed = self._crashed | self._cube_lost
            if self.cfg.place_curriculum:
                self._update_place_curr()
        else:
            self._crashed = self._table_touch | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # competence-gated curriculum + reward staging (opt-in; no-op when off)
        if self.cfg.adaptive_curriculum or self.cfg.reward_staging:
            self._update_curriculum(self._crashed | time_out)
        if self.cfg.staged_curriculum:
            self._update_stage()
        if self.cfg.two_platform:
            self._update_sep()
        if self.cfg.reverse_curriculum:
            self._update_rc()
        if self.cfg.overhead_first:
            # lower the altitude setpoint one rung once centring at the CURRENT rung is
            # proven (EMA of "in band AND centred <1.2cm" over in-band time)
            dz_oh = (self._base_p[:, 2] - 0.07) - self._obj_p[:, 2]
            in_band = (dz_oh - self._oh_dz).abs() < 0.015
            ok = (in_band & (self._horiz < self.cfg.oh_gate_horiz)
                  & ((dz_oh - self._oh_dz).abs() < self.cfg.oh_gate_dz)).float().sum() \
                / in_band.float().sum().clamp(min=1)
            self._oh_ema = 0.995 * self._oh_ema + 0.005 * ok.item()
            step = getattr(self, "_train_steps", 0)
            if (step - self._oh_last >= self.cfg.oh_dwell and self._oh_ema >= self.cfg.oh_gate
                    and self._oh_dz > self.cfg.oh_dz_end):
                self._oh_dz = max(self.cfg.oh_dz_end, self._oh_dz - self.cfg.oh_dz_step)
                self._oh_last = step
                self._oh_ema = 0.0   # re-earn at the new rung
                print(f"[overhead] setpoint lowered -> {self._oh_dz*100:.0f}cm above cube "
                      f"(step={step})", flush=True)
        self.extras["log"] = {
            "metrics/grasp_rate": self._held.float().mean(),
            "metrics/place_success": self._success.float().mean(),
            "metrics/obj_to_goal": self._d_goal.mean(),
        }
        if self.cfg.two_platform:
            dev = self.device
            self.extras["log"].update({
                "metrics/arrive_rate": self._arrived.float().mean(),
                "metrics/deposit_rate": self._deposited.float().mean(),
                "curr/plat_sep": torch.tensor(self._sep, device=dev),
            })
        if self.cfg.place_only:
            # the four numbers that answer "is the placement getting better": does it let
            # go on the pad, how hard does the cube land, how far off centre, and is it
            # even reaching the set-down pose. deposit_rate above is the headline.
            rest_z = self.cfg.surface_z + 0.5 * self.cfg.cube_size
            seat_err = (self._obj_p[:, 2] - (rest_z + self.cfg.lip_h)).abs()
            seated = self._holding & (self._d_plat_b < self.cfg.place_radius) & (seat_err < 0.012)
            dep = self._deposited
            n_dep = dep.float().sum().clamp(min=1.0)
            self.extras["log"].update({
                "place/holding_frac": self._holding.float().mean(),
                "place/seated_frac": seated.float().mean(),
                "place/impact_v": (self._impact_v * dep.float()).sum() / n_dep,
                "place/landing_err_cm": (self._d_plat_b * dep.float()).sum() / n_dep * 100.0,
                "place/difficulty": torch.tensor(self._place_p, device=self.device),
            })
        if self.cfg.staged_curriculum:
            dev = self.device
            self.extras["log"].update({
                "stage/stage": torch.tensor(float(self._stage), device=dev),
                "stage/hover_ema": torch.tensor(self._hover_ema, device=dev),
                "stage/grasp_ema": torch.tensor(self._gr_ema, device=dev),
                "stage/place_ema": torch.tensor(self._pl_ema, device=dev),
            })
        if not self.cfg.grasp_latch and not self.cfg.place_only:
            # (skipped for place_only: these are GRASP-discovery diagnostics and the place
            # stage never grasps -- it starts holding. Keeping the dashboard answerable.)
            # CLOSE DIAGNOSTICS: is it closing in the right place? Envelope occupancy,
            # grip command inside vs outside it, and terminal accuracy near the cube.
            gc = self._actions[:, 4] * 0.5 + 0.5
            dz_t = (self._base_p[:, 2] - 0.07) - self._obj_p[:, 2]
            enc = ((self._horiz < 0.022) & (dz_t > -0.012) & (dz_t < 0.012)).float()
            near = (self._d_reach < 0.30).float()
            nz = near.sum().clamp(min=1)
            # position zones among near envs (d_reach<30cm): NESTED = cage around cube
            # (the goal); ON TOP = centred safely above; BESIDE = at cube height but
            # offset -> wall on cube, the knock zone.
            on_top = ((dz_t > 0.03) & (self._horiz < 0.02)).float() * near
            beside = ((dz_t.abs() < 0.03)
                      & (self._horiz > 0.024) & (self._horiz < 0.08)).float() * near
            self.extras["log"].update({
                "close/enclosed_frac": enc.mean(),
                "close/grip_in_envelope": (gc * enc).sum() / enc.sum().clamp(min=1),
                "close/grip_outside": (gc * (1 - enc)).sum() / (1 - enc).sum().clamp(min=1),
                "close/horiz_cm_near": (self._horiz * near).sum() / nz * 100.0,
                "close/tip_dz_cm_near": (dz_t * near).sum() / nz * 100.0,
                # per-axis breakdown: is each horizontal axis being optimized, and is
                # there a systematic bias hiding inside the combined horiz norm?
                "close/dx_cm_near": ((self._obj_p[:, 0] - self._base_p[:, 0]).abs() * near).sum() / nz * 100.0,
                "close/dy_cm_near": ((self._obj_p[:, 1] - self._base_p[:, 1]).abs() * near).sum() / nz * 100.0,
                "close/zone_on_top": on_top.sum() / nz,
                "close/zone_beside": beside.sum() / nz,
                # demo-vs-fly-in split: straddle demos dominate the enclosed pool, so the
                # combined grip_in hides whether closing has TRANSFERRED to fly-ins
                "close/grip_in_flyin": (gc * enc * (~self._is_near).float()).sum()
                                        / (enc * (~self._is_near).float()).sum().clamp(min=1),
                # the lift precursor: mean cube height off its rest pose among near envs.
                # grasp_ema only sees 6cm+ holds; this sees the first 5mm twitch-lifts.
                "close/lift_cm_near": (torch.clamp(self._obj_p[:, 2]
                                        - (self.cfg.surface_z + 0.5 * self.cfg.cube_size), min=0.0)
                                        * near).sum() / nz * 100.0,
            })
            if self.cfg.overhead_first:
                self.extras["log"].update({
                    "overhead/dz_target_cm": torch.tensor(self._oh_dz * 100.0, device=self.device),
                    "overhead/rung_ema": torch.tensor(self._oh_ema, device=self.device),
                })
        if self.cfg.reverse_curriculum:
            dev = self.device
            # actual fly-in distance in METERS (quadratic map) so the dashboard shows the
            # real "how far can it pick up from" directly, not just the 0-1 difficulty.
            dist_m = (self._rc_p ** 2) * self.cfg.rc_dist_max if self.cfg.rc_distance_mode \
                else self._rc_p * self.cfg.rc_r_max
            self.extras["log"].update({
                "revcurr/difficulty": torch.tensor(self._rc_p, device=dev),
                "revcurr/grasp_ema": torch.tensor(self._rc_grasp_ema, device=dev),
                "revcurr/distance_m": torch.tensor(float(dist_m), device=dev),
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
            self._stage1_step = step                     # hover-anneal clock starts here
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

    def _update_place_curr(self):
        """Place start-ramp controller. Expands the spawn disc + altitude band toward the
        full spec only once the deposit rate holds at the current spread. Self-pacing: if an
        expansion drops the deposit rate below the gate, it stalls until it recovers."""
        d = self._deposited.float().mean().item()
        a = self.cfg.place_curr_ema
        if not self._place_ema_init:
            self._place_ema, self._place_ema_init = d, True
        else:
            self._place_ema = a * self._place_ema + (1.0 - a) * d
        step = getattr(self, "_train_steps", 0)
        if step - self._last_place_change < self.cfg.place_curr_dwell:
            return
        if self._place_p < 1.0 and self._place_ema >= self.cfg.place_curr_thresh:
            self._place_p = min(1.0, self._place_p + self.cfg.place_curr_step)
            self._last_place_change = step
            print(f"[place] difficulty={self._place_p:.2f} -> spawn r<={self._place_p*self.cfg.place_spawn_r:.2f}m "
                  f"deposit_ema={self._place_ema:.2f} step={step}", flush=True)

    def _update_sep(self):
        """A->B separation curriculum. Expands the transport gap once the stage's own
        success (arrival for nav, deposit for place) holds up at the current distance."""
        step = getattr(self, "_train_steps", 0)
        a = self.cfg.stage_ema
        cur = float(self._success.float().mean().item())
        if not self._sep_ema_init:
            self._sep_ema = cur
            self._sep_ema_init = True
        else:
            self._sep_ema = a * self._sep_ema + (1.0 - a) * cur
        if step - self._last_sep_change < self.cfg.plat_sep_dwell:
            return
        if self._sep_ema >= self.cfg.plat_sep_thresh and self._sep < self.cfg.plat_sep_max:
            # PROPORTIONAL step: a fixed 0.25m increment needs 34 gates to walk 1.5 -> 10m,
            # and each gate costs a dwell plus re-earning the success EMA. Scaling with the
            # current gap keeps the *relative* jump constant (~15%), so the ladder reads
            # 1.5 -> 1.75 ... -> 5.0 (+0.75) -> ... -> 10.0 (+1.5): small where the policy is
            # fragile, larger once long-haul flight is already working.
            inc = max(self.cfg.plat_sep_step, 0.15 * self._sep)
            earned = self._sep_ema
            self._sep = min(self.cfg.plat_sep_max, self._sep + inc)
            self._last_sep_change = step
            self._sep_ema = 0.0          # re-earn the gate at the new distance
            print(f"[plat_sep] -> {self._sep:.2f} m (+{inc:.2f})  "
                  f"earned_ema={earned:.2f} step={step}", flush=True)

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
        cube_h = torch.clamp(self._obj_p[:, 2] - (self.cfg.surface_z + 0.5 * self.cfg.cube_size), 0.0, 0.15)  # lift off table
        place = held * (1.0 - torch.tanh(self._d_goal / 0.35))           # DOMINANT delivery
        carry_up = held * torch.clamp(self._obj_p[:, 2] - (self.cfg.surface_z + 0.5 * self.cfg.cube_size + self.cfg.grasp_clear), 0.0, 0.25)
        carry_prog = held * (self._prev_d_goal - self._d_goal)
        success = (self._held & (self._d_goal < 0.18)).float()
        if self.cfg.two_platform:
            # LONG-HAUL shaping. tanh(d/0.35) saturates to a flat 1.0 (zero gradient) once
            # B is metres away, so the potential term is rescaled to the CURRENT curriculum
            # separation. The scale must be ~sep, NOT a fraction of it: at 0.25*sep the
            # argument is 4.0 at the start of the haul, tanh -> 0.9998, and the whole term
            # is worth 0.01 reward -- measured dead over 690 iterations (arrive_rate flat at
            # 0.2%, d_goal DRIFTING UP 1.37 -> 1.67m as the policy grabbed and hovered).
            # At d_scale = sep the term pays ~9.6 mid-haul and ~37 on arrival: a real slope.
            d_scale = max(0.35, self._sep)
            place = held * (1.0 - torch.tanh(self._d_goal / d_scale))
            # LATERAL progress must outbid vertical hovering: carry_up pays up to 30*0.25
            # = 7.5 for merely holding the cube high, which was the local optimum the
            # policy found. 2.4x lifts carry_prog's weight 25 -> 60 for transport runs.
            carry_prog = carry_prog * 2.4
            if not (self.cfg.release_only or self.cfg.place_only):
                # STOP PAYING FOR MERELY HOLDING (carry stage). The raw potential above is
                # worth 40*(1-tanh(1.0)) = 9.5/step at the spawn separation -- ~5,700 over a
                # 600-step episode for standing still, against ~90 for completing the haul
                # (carry_prog telescopes to 60*sep). The policy correctly learned to hover:
                # arrive_rate rose to 4.7% then DECAYED to 0.9% over 1000 iterations while
                # mean reward climbed. Subtracting the spawn-distance baseline makes holding
                # at the start worth exactly 0, and only closing distance pays.
                p_base = 1.0 - math.tanh(max(self._sep, 1e-6) / d_scale)
                place = ((place - held * p_base) / max(1.0 - p_base, 1e-6)).clamp(min=0.0)
            success = self._success.float()
            over_b = 1.0 - torch.tanh(self._d_plat_b / 0.15)
            if self.cfg.release_only:
                # PLACE stage: pay for lowering the held cube onto B and then LETTING GO.
                # The open-jaw payment is gated on being seated over B, so "drop it early"
                # earns nothing; dropping the cube away from B is taxed outright.
                rest_z = self.cfg.surface_z + 0.5 * self.cfg.cube_size
                seat_err = (self._obj_p[:, 2] - rest_z).abs()
                lower = held * over_b * (1.0 - torch.tanh(seat_err / 0.12))
                ready = held * over_b * (seat_err < 0.04).float()
                release = ready * (1.0 - grip_cmd)               # jaws commanded open, seated
                stray = ((~self._held) & (self._d_plat_b > self.cfg.place_radius)).float()
                place = place + 15.0 * lower + 25.0 * release - 20.0 * stray
        self._prev_d_goal = self._d_goal.clone()

        if self.cfg.place_only:
            return self._place_reward()

        if self.cfg.staged_curriculum:
            # Stage 0: hover over cube (reach+align). Stage 1: + DENSE descent reward
            # (pull body to grasp height when horizontally aligned) + grab + lift, minus a
            # soft table-touch penalty. Stage 2: + carry + deliver. Dense signal per stage
            # -> the descent has a downward gradient instead of only a post-grasp payoff.
            grasp_z = self.cfg.surface_z + 0.5 * self.cfg.cube_size + 0.07     # body height for tip-at-cube
            z_err = (self._base_p[:, 2] - grasp_z).abs()
            # gate by a SOFT "roughly over the cube" term (0.15 scale) -- the tight align
            # (0.06) was ~0 at the hover standoff, so the descent reward was switched off.
            over_cube = 1.0 - torch.tanh(self._horiz / 0.15)
            descend = over_cube * (1.0 - torch.tanh(z_err / 0.15))
            touch = self._table_touch.float()
            # master-then-diminish: full hover payoff through stage 0; once stage 1 opens,
            # decay the standoff terms to 20% over stage1_hover_anneal env-steps so the
            # policy is pushed OFF the hover plateau and into the descent/grab rewards.
            w_hov = 1.0
            if self._stage >= 1 and self.cfg.stage1_hover_anneal > 0.0:
                p1 = min(1.0, (getattr(self, "_train_steps", 0) - getattr(self, "_stage1_step", 0))
                         / self.cfg.stage1_hover_anneal)
                w_hov = 1.0 - 0.8 * p1
            r = w_hov * (1.0 * reach + 0.5 * align)
            if self.cfg.side_spawn_max > 0.0:
                # navigation gradient: reach (0.5m scale) is saturated at side-spawn
                # distances, so the cruise needs its own long-range pull toward the cube.
                # NOT annealed: with side spawns this is the "get there" reward.
                r = r + 0.7 * (1.0 - torch.tanh(self._d_reach / 3.0))
            if self._stage >= 1:
                r = r + self.cfg.descend_coef * descend + 1.5 * grab + 60.0 * cube_h \
                    - self.cfg.table_touch_pen * touch
                if self.cfg.latch_ready_coef > 0.0 and not self.cfg.overhead_first:
                    # (disabled in overhead mode: a nest-peaked bonus is a "dive now"
                    # magnet that bypasses the altitude-rung ladder)
                    # peak reward at the exact capture pose. Under physics (no latch) the
                    # envelope is ~1.2cm, so the gradient scales sharpen to match (the
                    # broad descend term still covers the far field).
                    rs = 0.02 if not self.cfg.grasp_latch else self.cfg.latch_r
                    tip_z = self._base_p[:, 2] - 0.07
                    vert = 1.0 - torch.tanh((self._obj_p[:, 2] - tip_z).abs() / (rs * 0.857))
                    horiz_l = 1.0 - torch.tanh(self._horiz / rs)
                    r = r + self.cfg.latch_ready_coef * horiz_l * vert
                # keep the cage OPEN through the approach/descent (ported from the
                # non-staged branch): a permanently-closed cage "snatches early", shoves
                # the cube around, and only scores via the forgiving latch
                if self.cfg.grasp_latch:
                    early_close = grip_cmd * (self._d_reach > 0.10).float()
                    r = r - self.cfg.early_close_pen * early_close
                else:
                    # PHYSICS: the cube must be ENCLOSED before closing. Capture-envelope
                    # math (jaws close at 1 m/s -> 35ms; drone drifts 2-4mm during it):
                    #   horiz: open lip aperture +-3.9cm vs cube half 2.5cm -> centre
                    #     offset < 1.4cm geometric, 1.2cm after drift margin.
                    #   vert: lips sit 1.7-2.5cm BELOW the tip -> they only pass under the
                    #     cube midline for tip 0..+1.2cm above cube centre (table blocks
                    #     lower). +-2.5cm would pay for closing on the cube's TOP edge.
                    tip_z2 = self._base_p[:, 2] - 0.07
                    dz_tip = tip_z2 - self._obj_p[:, 2]               # tip above cube centre
                    # lower bound -1.2cm: a shelf-carried cube rests ~0.8cm ABOVE the tip
                    # (bottom on the lip tops) -- holding it closed must NOT be taxed. On
                    # the table the band below centre is unreachable anyway (table blocks).
                    # capture envelope for the WIDENED cage: lips at 4.9cm vs cube half
                    # 2.5cm -> 2.4cm geometric window; gate at 2.2cm with drift margin
                    enclosed = ((self._horiz < 0.022)
                                & (dz_tip > -0.012) & (dz_tip < 0.012)).float()
                    r = r - 1.5 * grab                    # remove the mis-gated grab paid above
                    # progress seesaw clock (defined FIRST -- used by the lift ladder below
                    # and the squeeze-depreciation block further down)
                    prog_l = min(1.0, getattr(self, "_train_steps", 0)
                                 / max(1.0, self.cfg.lift_ramp_steps))
                    # ALTITUDE ANCHOR FOLLOWS THE CUBE (not the fixed table height): the
                    # table-anchored descend term bled ~3.6/step over the first 6cm of
                    # ascent -- cancelling the lift bonus -- so "squeeze forever at the
                    # nest" was a stable optimum (grip 0.87, grasp 0). Cube-anchored,
                    # lifting together costs nothing and the 60x lift pays in the clear.
                    z_err_obj = (self._base_p[:, 2] - (self._obj_p[:, 2] + 0.07)).abs()
                    r = r - self.cfg.descend_coef * descend
                    r = r + self.cfg.descend_coef * (1.0 - 0.6 * prog_l) * over_cube * (1.0 - torch.tanh(z_err_obj / 0.15))
                    # GENTLE LIFT (probe-measured): carry succeeds 98% at <=0.3 m/s ascent,
                    # 9% at full speed -- the seat breaks under jerk. While engaged
                    # (enclosed + squeezing), a gentle climb pays; a violent one is taxed.
                    vz_d = self.robot.data.root_lin_vel_w[:, 2]
                    engaged = enclosed * (grip_cmd > 0.8).float()
                    r = r + 4.0 * engaged * torch.clamp(vz_d, 0.0, 0.3) / 0.3
                    r = r - 2.0 * engaged * torch.clamp(vz_d - 0.45, min=0.0)
                    # EXPONENTIAL HEIGHT LADDER (user-designed): linear height pay gave
                    # constant marginal reward against RISING marginal risk of losing the
                    # seat -> optimal policy dabbled low and collapsed (bursts at 0.2-0.5cm,
                    # first 6cm holds at 0.07% then retreat). Each cm now pays more than
                    # the last: ~1.6/step at 2cm, ~9 at 6cm, ~26 at 10cm, ~76 at 15cm (cap).
                    h_lift = torch.clamp(self._obj_p[:, 2]
                                         - (self.cfg.surface_z + 0.5 * self.cfg.cube_size),
                                         0.0, 0.15)
                    r = r + 4.0 * (1.0 + prog_l) * (torch.exp(h_lift / 0.05) - 1.0)
                    # explicit DROP penalty: one-time fine at the held->lost transition so
                    # credit lands on the exact action that broke the seat (the ongoing
                    # income cliff already punishes the aftermath)
                    if not hasattr(self, "_prev_held_buf"):
                        self._prev_held_buf = torch.zeros_like(self._held)
                    dropped = (self._prev_held_buf & ~self._held).float()
                    r = r - 10.0 * dropped
                    self._prev_held_buf = self._held.clone()
                    if self.cfg.overhead_first:
                        # OVERHEAD-FIRST (user-designed): one paid spot in the world --
                        # dead-centre at the current altitude setpoint. Lean/contact paths
                        # pay nothing and are taxed; descent arrives only via the
                        # competence-gated setpoint in the controller above.
                        r = r - self.cfg.descend_coef * descend          # no generic descend
                        # THREE centering scales so the gradient reaches from anywhere:
                        # coarse (8cm) pulls 25cm->5cm, fine (2cm) pulls 5->1, exponential
                        # (1cm) pays the bullseye -- only at the altitude setpoint band.
                        center_c = 1.0 - torch.tanh(self._horiz / 0.08)
                        center = 1.0 - torch.tanh(self._horiz / 0.02)
                        band = 1.0 - torch.tanh((dz_tip - self._oh_dz).abs() / 0.02)
                        r = r + 2.0 * center_c
                        r = r + 3.0 * center
                        # the setpoint altitude pays a BASE rate even off-centre (it must be
                        # "the place to be" before centring there can be learned); centring
                        # multiplies it up via the exponential
                        r = r + 1.5 * band
                        r = r + 8.0 * torch.exp(-self._horiz / 0.01) * band
                        cube_v = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
                        r = r - 3.0 * torch.clamp(cube_v, max=1.0)       # DON'T TOUCH THE CUBE
                        # contact-possible band taxed at ANY off-centre offset: walls span
                        # tip+-2.5cm and the cube top is +2.5cm, so clipping is possible the
                        # moment dz_tip < 5cm. Off-centre there = squish/top-clip territory.
                        # The only untaxed entry into the final 5cm of descent is centred
                        # inside the WIDENED 2.4cm fit window (lips at 4.9cm, cube half 2.5).
                        beside = ((dz_tip > -0.03) & (dz_tip < 0.05)
                                  & (self._horiz > 0.024)).float()
                        r = r - 2.0 * beside                             # the lean/knock lane
                        r = r - self.cfg.early_close_pen * grip_cmd      # cage stays open
                    elif self.cfg.surround_only:
                        # SURROUND-FIRST (user-directed): no grasp rewards at all. CHOREOGRAPHY
                        # FIX: the generic descend term (15cm-soft centering gate) paid the
                        # policy to sit at cube height while 2-3cm OFF -- i.e. wall against
                        # cube, knocking it. Order is now CENTER -> SINK -> NEST:
                        #   1) centering pays at ANY height (so it lines up while still high)
                        #   2) descending pays ONLY once centered (tight 2cm gate)
                        #   3) nested surround pays exponentially at cube height
                        #   4) the cube MOVING at all is penalized (it should never be touched)
                        r = r - self.cfg.descend_coef * descend          # retract loose descend
                        center = 1.0 - torch.tanh(self._horiz / 0.02)
                        vert_band = ((dz_tip > -0.012) & (dz_tip < 0.012)).float()
                        r = r + 3.0 * center
                        r = r + self.cfg.descend_coef * center * (1.0 - torch.tanh(z_err / 0.15))
                        r = r + 8.0 * torch.exp(-self._horiz / 0.01) * vert_band
                        cube_v = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
                        r = r - 3.0 * torch.clamp(cube_v, max=1.0)       # DON'T TOUCH THE CUBE
                        # BESIDE-ZONE PENALTY (user-directed): being at cube height while
                        # 1.4-7cm off-centre = a cage wall overlapping the cube = the knock
                        # zone. Explicitly taxed -- the only paid route down is the centred
                        # column. (Geometry: open walls at 4.4cm, lips 3.9cm, cube half 2.5.)
                        beside = ((dz_tip.abs() < 0.03)
                                  & (self._horiz > 0.024) & (self._horiz < 0.08)).float()
                        r = r - 2.0 * beside
                        # TWO ALTITUDE SETPOINTS (user-directed), never "go down":
                        # centred -> nest height (paid above); NOT centred -> hold the safe
                        # waiting band ABOVE the cube (tip 4-15cm over cube centre) while
                        # lining up. Off-centre loitering now has a correct place to be.
                        off_c = (self._horiz > 0.02).float()
                        wait_band = ((dz_tip > 0.04) & (dz_tip < 0.15)).float()
                        r = r + 1.0 * off_c * wait_band
                        r = r - self.cfg.early_close_pen * grip_cmd
                    else:
                        # PRICED TO ATTEMPT + EXPONENTIAL PRECISION (user-designed): squeezing
                        # only pays inside the (widened, 2.2cm) envelope, exponentially more
                        # dead-centre (12mm scale matches the new fit window).
                        # PROGRESS SEESAW (user-designed): as iterations grow, squeeze-camping
                        # pay DEPRECIATES (8 -> 4) while the lift tier APPRECIATES (x1 -> x2
                        # below) -- the income mix migrates toward grasping and lifting.
                        prog_l = min(1.0, getattr(self, "_train_steps", 0)
                                     / max(1.0, self.cfg.lift_ramp_steps))
                        prec = torch.exp(-self._horiz / 0.012)
                        r = r + (8.0 - 7.0 * prog_l) * prec * enclosed * grip_cmd
                        r = r - self.cfg.early_close_pen * grip_cmd * (1.0 - enclosed)
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
        # keep the cage OPEN through the descent: penalize commanding close while still far
        # from the cube, so it descends AROUND the cube instead of clamping/knocking it.
        early_close = grip_cmd * (self._d_reach > 0.10).float()
        r = (w_app * approach + 0.5 * align + 2.0 * grab + 60.0 * cube_h + held_bonus
             - self.cfg.early_close_pen * early_close
             + pg * (40.0 * place + 30.0 * carry_up + 25.0 * carry_prog + 80.0 * success) - 0.01)
        if self.cfg.rc_distance_mode:
            # PROXIMITY navigation (gradient across the full 0-10m range; rewards arriving and
            # STAYING near the cube, not speed -> no overshoot incentive). Plus a SETTLE term:
            # penalize horizontal speed when close, so it decelerates to actually grasp instead
            # of zooming past. (nav_progress rewarded speed -> the drone overshot and never grasped.)
            nav = 1.0 - torch.tanh(self._d_reach / 3.0)              # pull toward the cube
            hspeed = torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)
            # SPEED PROFILE: go FAST when far, SLOW when near. Desired speed scales with distance
            # to the cube, so the drone accelerates across open space and decelerates to ~0 right
            # over the cube -> it ARRIVES slow enough to grasp precisely. Track it both ways:
            # penalize being too fast near AND too slow far (so navigation stays quick).
            # USER-REQUESTED deceleration: the closer to the cube, the slower it must go -- down to a
            # safe CONSTANT pickup speed. v_target ramps from `speed` (far) to pickup_speed (near).
            # Penalize ONLY over-speed (rushing in faster than the safe target) so the drone can still
            # slow further / stop to grasp -> kills the overshoot without discouraging a careful stop.
            v_target = torch.clamp(self._d_reach * 1.5, self.cfg.pickup_speed, self.cfg.speed)
            over_speed = (hspeed - v_target).clamp(min=0.0)
            r = r + 2.0 * self.cfg.nav_progress_w * nav - self.cfg.w_decel * over_speed
        self._prev_d_reach = self._d_reach.clone()
        return r

    # ------------------------------------------------------------------ #
    def _place_reward(self) -> torch.Tensor:
        """PLACE stage: gentle deposit onto platform B from an already-holding start.

        Deliberately does NOT reuse the stage>=1/2 pile. That reward pays an exponential
        height ladder (~153/step at the 15cm cap, line ~934) plus 30*carry_up (7.5/step)
        against 80*success for a deposit -- i.e. holding the cube HIGH outearns putting it
        down, and every step of the descent costs income. That is exactly the hover the
        stage-2 carry run is stuck in (arrive_rate 0.003 at iter 9790, cube held at ~88cm).
        Here there is NO altitude income at all: the only way to earn is to set the cube
        down on B and let go.

        Seat geometry (cage lips bottom out on the pad, see cfg.lip_h): a HELD cube can get
        no lower than seat_z = pad_top + lip_h + cube/2; a RELEASED one rests at rest_z =
        pad_top + cube/2. So `lower` is scored against seat_z (reachable, error -> 0) while
        success is scored against rest_z.
        """
        cfg = self.cfg
        rest_z = cfg.surface_z + 0.5 * cfg.cube_size          # cube at rest ON the pad
        seat_z = rest_z + cfg.lip_h                            # lowest a HELD cube reaches
        obj_z = self._obj_p[:, 2]
        d_b = self._d_plat_b                                   # cube->B centre (XY)
        seat_err = (obj_z - seat_z).abs()
        drop_h = torch.clamp(obj_z - rest_z, min=0.0)          # cube bottom over the pad
        vz_d = self.robot.data.root_lin_vel_w[:, 2]
        hspeed = torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)
        v_cube = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        held_f = self._holding.float()          # see _get_dones: geometric, NOT _held
        # 1. GET TO B. A tanh potential alone does NOT work here: at a 0.4m scale its
        #    gradient is ~0.05 per metre by the time the drone is 0.9m out, and the measured
        #    result was a policy that drifted to 0.9m and simply stayed in that flat region
        #    (obj_to_goal 0.73 -> 0.99, deposit 0). So the long-range work is done by a
        #    SCALE-FREE per-step progress difference -- the same term that is the only reason
        #    the carry stage has gradient across metres (see the two_platform block above).
        #    Telescoping: closing then re-opening the same distance nets zero, so it cannot
        #    be farmed by orbiting.
        prog_b = held_f * (self._prev_d_b - d_b)
        r = 30.0 * prog_b
        self._prev_d_b = d_b.clone()
        #    REPRICED: at the spawn distribution the curriculum actually runs (difficulty 0 =
        #    dead over B), d_b is ~0 from step one, so these two paid a flat 10/step for merely
        #    existing over the pad -- 22% of the hover income, earned for nothing. They exist
        #    for the expanded disc, so they stay, but they must not bankroll a parked drone.
        r = r + 2.0 * held_f * (1.0 - torch.tanh(d_b / 1.0))    # broad basin, gradient to ~2m
        over_b = 1.0 - torch.tanh(d_b / 0.15)                   # fine centring at the pad
        r = r + 2.0 * held_f * over_b
        # 2. LOWER -- gated on being near B (0.25m scale, matched to place_radius 0.20) so it
        #    cannot be farmed over the floor or over pad A, but is not so tight that a drone
        #    20cm out gets no descent signal at all. Zero at the reachable seat pose.
        lower_gate = 1.0 - torch.tanh(d_b / 0.25)
        r = r + 12.0 * held_f * lower_gate * (1.0 - torch.tanh(seat_err / 0.15))
        # 3. GENTLE DESCENT PROFILE (the user's bar): the safe descent speed tapers from
        #    `speed` far out down to gentle_v within place_taper_h of the seat. Only EXCESS
        #    speed is taxed -- descending slower, or stopping dead, is never penalized.
        taper_k = (cfg.speed - cfg.gentle_v) / max(cfg.place_taper_ramp, 1e-3)
        vz_safe = torch.clamp(cfg.gentle_v + (seat_err - cfg.place_taper_h) * taper_k,
                              cfg.gentle_v, cfg.speed)
        r = r - 10.0 * held_f * torch.clamp(-vz_d - vz_safe, min=0.0)
        #    same idea laterally: don't arrive at the pad sideways at cruise speed
        h_safe = torch.clamp(d_b * 1.5, cfg.gentle_v, cfg.speed)
        r = r - 4.0 * held_f * torch.clamp(hspeed - h_safe, min=0.0)
        # 4. NO ALTITUDE INCOME, and climbing while holding is taxed outright -- the warm
        #    start arrives with a hard-won "grab it and rocket up" habit that has to be
        #    unlearned before any descent can be discovered.
        r = r - 2.0 * held_f * torch.clamp(vz_d, min=0.0)
        # 5. SEATED: cube centred over B and at the reachable seat height, still gripped.
        seated = held_f * over_b * (seat_err < 0.012).float()
        #    REPRICED 15->6 / 8->3. Being seated is a WAYPOINT, not the job. At 23/step it was
        #    the largest single block of income available, it is collectable indefinitely (the
        #    episode never ends while holding), and the measured result was seated_frac 0.992 /
        #    holding_frac 1.000 / episode length 399.00 / deposit_rate 0.0000 for 644 straight
        #    iterations. Keep it positive -- a bleed here makes fleeing the arena outscore
        #    loitering (see term 9) -- but small enough that term 6 dominates it.
        r = r + 6.0 * seated
        #    hold still once seated (this is what makes the release itself soft)
        r = r + 3.0 * seated * (1.0 - torch.tanh(vz_d.abs() / 0.10))
        #    and specifically STOP DESCENDING once seated: continuing to push down presses
        #    the lips onto the pad, and the contact friction locks the jaw joints so the cube
        #    can never be released (measured: 61% of scripted set-downs failed exactly here)
        r = r - 6.0 * seated * torch.clamp(-vz_d, min=0.0)
        # 6. RELEASE: open the jaws, but ONLY while seated. Scored on the MEASURED jaw
        #    position, never the command -- if the drone keeps pressing the lips into the pad
        #    the contact friction locks the prismatic joints, and a command-scored version
        #    pays full price for an "open" that never happens (verified: the cube stays
        #    sitting on the lips 5-13mm up, 61% of scripted attempts). ~0 at the 0.026 hold
        #    pose, 1 once the lip aperture actually clears the cube.
        #    SCALE 0.004 -> 0.012 (the load-bearing fix). The held pose sits at jaw ~0.026 and
        #    release_jaw is 0.014 -- a 12mm gap read through a 4mm tanh is THREE scale-lengths,
        #    so openness at the held pose was 1-tanh(3.0) = 0.005 and this whole term paid
        #    0.2/step against ~45/step for staying shut. Flat, not just small: the gradient
        #    only appeared below jaw 0.022, which is a ~3 sigma move on a4 at action_std 0.25,
        #    sustained for several steps while the joint travels. Exploration never got there
        #    (30 accidental deposits in 1650 iterations, none after 10048). At 0.012 the held
        #    pose reads 1-tanh(1.0) = 0.238 and the ramp is monotone all the way to release.
        jaw_now = self.robot.data.joint_pos[:, self._grip_i].mean(-1)
        openness = 1.0 - torch.tanh(torch.clamp(jaw_now - cfg.release_jaw, min=0.0) / 0.012)
        #    60 (was 40): starting to open must beat the entire seated block on the FIRST step
        #    of the move, or the policy re-derives the hover. 60*0.238 = 14.3/step at the held
        #    pose vs 9/step for seated+still.
        r = r + 60.0 * seated * openness
        #    releasing anywhere else drops the cube from height / off the pad
        early_open = held_f * openness * (seat_err > 0.03).float()
        r = r - 12.0 * early_open
        # 7. DEPOSITED: persistent income, and it must beat everything available while
        #    holding (peak hold income ~35/step) or the policy just hovers over B forever.
        dep = self._deposited.float()
        #    GENTLENESS IS MOST OF THE INCOME, NOT A BONUS ON TOP. Previously this paid a flat
        #    120 plus a 60 gentleness bonus on a 0.30 tanh -- and the 0.30 scale saturates by
        #    ~1.0 m/s, so at the 1.37 m/s the policy actually converged to it paid 0.25 of 60
        #    with a gradient of -1.7 per m/s. Flat, i.e. invisible: exactly the failure the 4mm
        #    `openness` scale had. FILMED RESULT of that combination: the policy opens the jaws
        #    on step ONE from 35cm up and lets the cube's inherited horizontal velocity carry it
        #    into the 20cm ring -- 61/64 episodes "deposit" by THROWING (videos/place_deposit.mp4).
        #    Scale 0.30 -> 1.2 puts real gradient at the speeds it flies at, and moving 50 of the
        #    flat 120 into the gentleness-scaled term makes a throw earn ~109/step against ~240
        #    for a set-down instead of 133 against 240.
        r = r + 70.0 * dep
        r = r + 60.0 * dep * (1.0 - torch.tanh(d_b / 0.10))     # land it mid-pad
        r = r + 110.0 * dep * (1.0 - torch.tanh(
            torch.clamp(self._impact_v - (cfg.gentle_v + 0.30), min=0.0) / 1.2))
        # 8. RELEASE HEIGHT: pay for letting go from as low as the lips allow (floor lip_h).
        #    Measured at the transition so it prices the action, not the aftermath.
        just_let_go = (self._prev_seated & self._released & (~self._holding)).float()
        #    Scale 0.01 -> 0.05: a 1cm scale reads every real release height as identical zero
        #    (the filmed take let go at 34.7cm, i.e. 27 scale-lengths out), so it priced nothing.
        r = r + 25.0 * just_let_go * (1.0 - torch.tanh((drop_h - cfg.lip_h).clamp(min=0.0) / 0.05))
        r = r - 20.0 * just_let_go * torch.clamp(-vz_d - 0.10, min=0.0)   # no downward throw
        self._prev_seated = (seated > 0.5)
        # 9. PENALTIES -- priced as EVENTS, not as an ongoing bleed. Every terminal case ends
        #    the episode the same step, so none can be gamed by ending it early: a botched
        #    set-down costs 10-30, fleeing the arena costs 50, and simply holding position
        #    pays ~+18/step, so quitting is never the better option.
        #    Graded fine: landing on the pad but outside place_radius is a near-miss (10);
        #    putting the cube on the floor is a real failure (30).
        r = r - (10.0 + 20.0 * self._cube_on_floor.float()) * self._cube_lost.float()
        r = r - 50.0 * (self._crashed & (~self._cube_lost)).float()
        #    bounce/knock: any cube motion once it is out of the cage, capped so a hard
        #    landing is a bounded fine rather than a reason to end the episode
        r = r - 15.0 * (1.0 - held_f) * torch.clamp(v_cube - cfg.place_settle_v, 0.0, 1.0)
        #    small standing pressure to not just sit there having dropped it near the pad
        r = r - 1.0 * ((~self._holding) & (~self._deposited)).float()
        r = r - cfg.table_touch_pen * self._table_touch.float()
        # 10. SATURATION TAX on the grip channel. Nothing else in this reward can see an action
        #     of +2.1 -- the clamp makes it identical to +1.0 -- so the actor's grip mean has no
        #     force returning it to the usable range and it ran to 5 sigma past the jaw-moving
        #     threshold. Priced on the RAW action so the gradient exists, and only on the
        #     overshoot so behaviour inside [-1,1] is untouched.
        r = r - 1.0 * torch.clamp(self._raw_actions[:, 4].abs() - 1.0, min=0.0)
        return r - 0.01

    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        self._rel_latch[env_ids] = 0          # a new episode starts with the jaws under policy control
        origins = self.scene.env_origins[env_ids]
        # cube on the floor within obj_spawn_diam of origin
        obj = self.object.data.default_root_state[env_ids].clone()
        r = (torch.rand(n, 2, device=self.device) - 0.5) * self.cfg.obj_spawn_diam
        obj[:, 0] += r[:, 0]; obj[:, 1] += r[:, 1]
        obj[:, :3] += origins
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        obj_local = obj[:, :3] - origins
        grasp_z = self.cfg.surface_z + 0.5 * self.cfg.cube_size + 0.07         # body height so tip is at the cube
        root = self.robot.data.default_root_state[env_ids].clone()
        if self.cfg.two_platform:
            # Platform B at the CURRENT curriculum separation, on a random bearing around
            # A (no fixed +X habit), rewritten per-env because B is kinematic furniture.
            # Placed HERE, before the drone spawn, because the place stage spawns the drone
            # relative to B (see the carry-demo block below).
            ang_b = torch.rand(n, device=self.device) * 2.0 * math.pi
            sep = self._sep * (0.85 + 0.15 * torch.rand(n, device=self.device))
            b_xy = torch.stack([sep * torch.cos(ang_b), sep * torch.sin(ang_b)], dim=-1)
            self._plat_b_xy[env_ids] = b_xy
            b_pose = torch.zeros(n, 7, device=self.device)
            b_pose[:, :2] = b_xy + origins[:, :2]
            b_pose[:, 2] = 0.15                       # box [0,0.30], flush on the floor
            b_pose[:, 3] = 1.0                        # identity quat (w,x,y,z)
            self.plat_b.write_root_pose_to_sim(b_pose, env_ids)
        if self.cfg.reverse_curriculum:
            # Florensa reverse curriculum: spawn AT the grasp pose, expand outward as the
            # ceiling self._rc_p grows. Per-env difficulty d ~ U[0, rc_p] -> a mix of easy
            # (at the cube) and hard (high+offset) starts at every stage (retains skills).
            # frontier-biased difficulty (d = ceiling * sqrt(U)): mass near the current
            # frontier so the policy practices the HARD descents and grasp_ema reflects
            # frontier competence -- uniform[0,ceiling] let easy spawns mask the stall.
            d = self._rc_p * torch.sqrt(torch.rand(n, device=self.device))
            ang = torch.rand(n, device=self.device) * 6.2831853
            if self.cfg.rc_distance_mode:
                # DISTANCE curriculum: quadratic map reaches 10m near rc_p=1. ANTI-FORGETTING:
                # ~40% of spawns stay in the reliable CLOSE grasp zone (<=0.17m) so the precise
                # grasp is NOT forgotten as distance grows (it was collapsing 0.37 -> 0.09 when
                # all spawns chased the frontier); the other 60% push the frontier rc_p^2*10m.
                far_rad = (d * d) * self.cfg.rc_dist_max
                fmax = (self._rc_p ** 2) * self.cfg.rc_dist_max
                close_rad = torch.minimum(0.17 * torch.sqrt(torch.rand(n, device=self.device)),
                                          torch.full_like(d, float(fmax)))
                rad = torch.where(torch.rand(n, device=self.device) < 0.4, close_rad, far_rad)
                root[:, 0] = obj_local[:, 0] + rad * torch.cos(ang)
                root[:, 1] = obj_local[:, 1] + rad * torch.sin(ang)
                root[:, 2] = grasp_z + self.cfg.rc_fixed_height
            else:
                # decouple: HEIGHT ramps with d from the start; horizontal offset only ramps in
                # for d > rc_horiz_start, so it masters a clean vertical descent onto a centered
                # cube before having to correct sideways mid-drop (the diagnosed knock cause).
                h_frac = ((d - self.cfg.rc_horiz_start) / (1.0 - self.cfg.rc_horiz_start)).clamp(min=0.0)
                rad = h_frac * self.cfg.rc_r_max
                root[:, 0] = obj_local[:, 0] + rad * torch.cos(ang)
                root[:, 1] = obj_local[:, 1] + rad * torch.sin(ang)
                root[:, 2] = grasp_z + d * self.cfg.rc_h_max
            self._is_near[env_ids] = d < 0.15
            self._prev_d_reach[env_ids] = rad             # navigation-progress baseline
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
            far = ~near
            if self.cfg.side_spawn_max > 0.0 and bool(far.any()):
                # SIDE SPAWN: random angle + area-uniform radius around the CUBE, random
                # altitude -> the policy must navigate to the hover point, then descend.
                ang = torch.rand(n, device=self.device) * 6.2831853
                lo, hi = self.cfg.side_spawn_min, self.cfg.side_spawn_max
                rad = torch.sqrt(torch.rand(n, device=self.device) * (hi * hi - lo * lo) + lo * lo)
                z = self.cfg.side_spawn_h_lo + torch.rand(n, device=self.device) \
                    * (self.cfg.side_spawn_h_hi - self.cfg.side_spawn_h_lo)
                root[far, 0] = obj_local[far, 0] + (rad * torch.cos(ang))[far]
                root[far, 1] = obj_local[far, 1] + (rad * torch.sin(ang))[far]
                root[far, 2] = z[far]
        # CARRY DEMOS: a fraction spawn ALREADY AIRBORNE WITH THE CUBE SEATED in the
        # closed cage (drone at 0.50-0.65m, cube at the tip, jaws shut). The full
        # carry/delivery income flows from step one -> the critic learns what holding
        # high is worth; fly-in episodes inherit the motive to lift. (Same medicine
        # that broke the squeeze-discovery wall: experience first, behavior follows.)
        carry = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self.cfg.carry_demo_p > 0.0 and not self.cfg.grasp_latch:
            carry = torch.rand(n, device=self.device) < self.cfg.carry_demo_p
            if bool(carry.any()):
                if self.cfg.place_only:
                    # PLACE STAGE beginning frame: hold the cube ABOVE THE DELIVERY PAD, not
                    # above pad A. Area-uniform disc of radius place_spawn_r around B on a
                    # random bearing, body altitude in [h_lo, h_hi] -> every episode starts
                    # with a real approach + descent onto B from a random direction. This is
                    # a superset of "dead over B", so the pure vertical set-down is retained.
                    # per-env difficulty, frontier-biased (sqrt) so most episodes practise
                    # near the current frontier while easy ones are still sampled
                    dd = torch.ones(n, device=self.device)
                    if self.cfg.place_curriculum:
                        dd = self._place_p * torch.sqrt(torch.rand(n, device=self.device))
                    ang_s = torch.rand(n, device=self.device) * 2.0 * math.pi
                    rad_s = (dd * self.cfg.place_spawn_r
                             * torch.sqrt(torch.rand(n, device=self.device)))
                    z_full = (self.cfg.place_spawn_h_lo
                              + (self.cfg.place_spawn_h_hi - self.cfg.place_spawn_h_lo)
                              * torch.rand(n, device=self.device))
                    z_s = self.cfg.place_h_easy + dd * (z_full - self.cfg.place_h_easy)
                    root[carry, 0] = (b_xy[:, 0] + rad_s * torch.cos(ang_s))[carry]
                    root[carry, 1] = (b_xy[:, 1] + rad_s * torch.sin(ang_s))[carry]
                    root[carry, 2] = z_s[carry]
                else:
                    root[carry, 0] = obj_local[carry, 0]
                    root[carry, 1] = obj_local[carry, 1]
                    # 0.38-0.65m: some demos start barely above the nest (short, survivable
                    # dives -> retention learned at easy altitude first), others at goal
                    # height -- the exp ladder pulls survivors upward along a smooth value slope
                    root[carry, 2] = 0.38 + 0.27 * torch.rand(n, device=self.device)[carry]
                obj[carry, 0] = root[carry, 0] + origins[carry, 0]
                obj[carry, 1] = root[carry, 1] + origins[carry, 1]
                # SHELF-SEATED, not tip-centred: cube bottom ON the lip tops (centre =
                # tip + 0.8cm). Tip-centred with open jaws = 5.4cm lip aperture under a
                # 5.0cm cube -> free-fall at spawn before the squeeze can arrive.
                obj[carry, 2] = root[carry, 2] - 0.062 + origins[carry, 2]
                obj[carry, 7:] = 0.0
                self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
                self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        root[:, :3] += origins
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        if bool(carry.any()):
            # 0.026: shelf aperture 4.6cm < 5cm cube -> the shelves SUPPORT the seated
            # cube from step one; walls (2.8cm half) clear its sides; nothing penetrates
            jpos[carry] = 0.026
        self.robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)
        if self.cfg.two_platform:
            # goal = the cube AT REST ON B's top (not a waypoint floating over A).
            # B's pose itself was written near the top of this method (the place stage
            # spawns the drone relative to it, so it has to exist before the spawn).
            t = torch.zeros(n, 3, device=self.device)
            t[:, :2] = b_xy
            if self.cfg.release_only or self.cfg.place_only:
                # DEPOSIT stages: the goal IS the cube at rest on B's top.
                t[:, 2] = self.cfg.surface_z + 0.5 * self.cfg.cube_size
            else:
                # CARRY stage: a hover point ABOVE B, not the resting height. At resting
                # height the goal pulled the cube DOWN to 0.325 while carry_up paid to lift
                # it UP to 0.635 -- two terms fighting over 31cm of altitude that _arrived
                # does not even score (it is XY-only). Same 0.25m standoff the original
                # single-platform goal used. Descending onto B is the PLACE stage's job.
                t[:, 2] = self.cfg.surface_z + 0.25
        else:
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
        self._arrived[env_ids] = False
        self._deposited[env_ids] = False
        self._released[env_ids] = False
        self._impact_v[env_ids] = 0.0
        self._prev_seated[env_ids] = False
        if self.cfg.two_platform:
            # progress baseline = the spawn distance, so step 1 doesn't bank a phantom gain
            self._prev_d_b[env_ids] = torch.norm(obj[:, :2] - (origins[:, :2] + b_xy), dim=-1)
        self._latched[env_ids] = False
