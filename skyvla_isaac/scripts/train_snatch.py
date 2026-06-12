"""Train the SNATCH aerial pick-and-place policy with PPO (rsl_rl) on Isaac Lab.

SNATCH = Sim-trained Neural Aerial Transport and Capture. Free-flying quadrotor
with a single-DOF caging gripper, dual-camera visuomotor policy, 5-action direct
velocity control (see snatch/DESIGN.md). Same rsl_rl stack as scripts/train.py.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
    python skyvla_isaac/scripts/train_snatch.py --num_envs 2048 --max_iterations 1500
  # smoke:
    python skyvla_isaac/scripts/train_snatch.py --num_envs 256 --max_iterations 3

NOTE: the SNATCH env (snatch/pick_place_env.py) is assembled by the orchestrator;
this entrypoint is written against that import path and the 5-action / dict-obs
contract, so it may not run end-to-end until that env lands.
"""
import argparse
import os

from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--log_dir", type=str,
                    default=os.path.join(_REPO, "logs/isaac/drone_snatch"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_cams", action="store_true",
                    help="state-based variant (no cameras): tractable convergence + VIO-gap study")
parser.add_argument("--cur_p", type=float, default=None,
                    help="fixed straddle-start fraction (overrides cfg curriculum; e.g. 0.9 for a clean demo)")
parser.add_argument("--adaptive_curriculum", action="store_true",
                    help="competence-gated anneal of straddle-start fraction toward pure fly-in (cur_p->0)")
parser.add_argument("--reward_staging", action="store_true",
                    help="learn pickup first, then phase in the placement/delivery reward")
parser.add_argument("--staged_curriculum", action="store_true",
                    help="3-stage reward curriculum: hover -> grab(+dense descent, soft table) -> carry/drop")
parser.add_argument("--side_spawn", type=float, default=None,
                    help="randomized SIDE-spawn radius (m): stage 0 becomes navigate->hover, not just descend. "
                         "Auto-scales env_spacing and episode length to fit the cruise")
parser.add_argument("--stage_hover_thresh", type=float, default=None,
                    help="stage 0->1 hover-EMA gate override (lower it with --side_spawn: cruise time caps the EMA)")
parser.add_argument("--stage1_hover_anneal", type=float, default=None,
                    help="master-then-diminish: env-steps over which the standoff hover payoff decays to 20% "
                         "after stage 1 opens (nav cruise term stays full)")
parser.add_argument("--latch_ready_coef", type=float, default=None,
                    help="stage>=1 bonus peaked at the exact latch-engage pose (cube in cage footprint, "
                         "tip at cube height); puts gradient across the last 5cm the descend term misses")
parser.add_argument("--no_latch", action="store_true",
                    help="REAL PHYSICS grasp: disable the kinematic latch; the floored-scoop cage must "
                         "physically capture, hold, and carry the cube (contact forces only)")
parser.add_argument("--carry_demo", type=float, default=None,
                    help="fraction of resets spawning ALREADY CARRYING (cube seated in closed cage at "
                         "altitude): teaches the carry's value -- white noise cannot sample a 15-step climb")
parser.add_argument("--start_stage", type=int, default=None,
                    help="start the staged controller at this stage (2 = all rewards on from step 0; "
                         "use when warm-starting an already-expert policy)")
parser.add_argument("--surround_only", action="store_true",
                    help="train ONLY cage-surrounds-cube precision (exponential horiz at cube height); "
                         "all gripper use taxed -- grasp training comes later from the surround expert")
parser.add_argument("--overhead_first", action="store_true",
                    help="overhead curriculum: pay ONLY dead-centre at an altitude setpoint that starts "
                         "20cm above the cube and lowers rung-by-rung as centring is proven; ends at nest")
parser.add_argument("--reverse_curriculum", action="store_true",
                    help="Florensa reverse curriculum: spawn at grasp pose, expand start distribution as grasp improves")
parser.add_argument("--rc_start", type=float, default=None,
                    help="reverse-curriculum starting difficulty (use on crash-restart to resume the expansion)")
parser.add_argument("--cube_mass", type=float, default=None,
                    help="cube mass kg (heavier e.g. 0.15 resists being knocked by an imperfect descent)")
parser.add_argument("--cube_size", type=float, default=None,
                    help="cube edge length m (smaller e.g. 0.035 gives the cage clearance to descend around it)")
parser.add_argument("--speed", type=float, default=None,
                    help="max velocity m/s (lower e.g. 0.6 -> gentler, controllable descent into the cage)")
parser.add_argument("--rc_distance_mode", action="store_true",
                    help="DISTANCE curriculum: slowly ramp the fly-in distance to the cube up to rc_dist_max")
parser.add_argument("--rc_dist_max", type=float, default=10.0,
                    help="max fly-in spawn distance from the cube (m) for the distance curriculum")
parser.add_argument("--curr_start", type=float, default=None,
                    help="adaptive-curriculum starting cur_p (use on crash-restart to resume the anneal point)")
parser.add_argument("--resume", type=str, default=None,
                    help="warm-start from a checkpoint (e.g. logs/isaac/drone_snatch_state/model_650.pt)")
parser.add_argument("--reset_std", type=float, default=None,
                    help="after warm-start, override the policy exploration std (e.g. 0.25). model_650's "
                         "saved std is ~17 -> stochastic rollouts saturate to random; reset for a real head start")
parser.add_argument("--entropy_coef", type=float, default=0.01,
                    help="PPO entropy bonus; lower (e.g. 0.002) when refining a warm-started policy so std stays sane")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_cams   # SNATCH is visuomotor; state-based for the gap study
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import (  # noqa: E402
    DroneSnatchEnv, DroneSnatchEnvCfg)

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = not args.no_cams
if args.cur_p is not None:
    cfg.curriculum_p_start = cfg.curriculum_p_end = args.cur_p
cfg.adaptive_curriculum = args.adaptive_curriculum
cfg.reward_staging = args.reward_staging
cfg.staged_curriculum = args.staged_curriculum
cfg.reverse_curriculum = args.reverse_curriculum
if args.rc_start is not None:
    cfg.rc_start = args.rc_start
if args.cube_mass is not None:
    cfg.cube_mass = args.cube_mass
if args.cube_size is not None:
    cfg.cube_size = args.cube_size
if args.speed is not None:
    cfg.speed = args.speed
if args.rc_distance_mode:
    cfg.reverse_curriculum = True
    cfg.rc_distance_mode = True
    cfg.rc_dist_max = args.rc_dist_max
    # the drone must have room to fly in from far without entering a neighbour env, and time
    # to actually reach the cube from rc_dist_max at the (slow) grasp speed.
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 2.5 * args.rc_dist_max)
    cfg.episode_length_s = max(cfg.episode_length_s, args.rc_dist_max / max(cfg.speed, 0.3) + 6.0)
if args.side_spawn is not None:
    cfg.side_spawn_max = args.side_spawn
    # room to cruise in from the side without entering a neighbour env, and time to
    # reach the cube from the far edge of the spawn annulus at cruise speed
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 2.0 * args.side_spawn + 2.0)
    cfg.episode_length_s = max(cfg.episode_length_s, 10.0 + args.side_spawn / max(cfg.speed, 0.3))
if args.stage_hover_thresh is not None:
    cfg.stage_hover_thresh = args.stage_hover_thresh
if args.stage1_hover_anneal is not None:
    cfg.stage1_hover_anneal = args.stage1_hover_anneal
if args.latch_ready_coef is not None:
    cfg.latch_ready_coef = args.latch_ready_coef
if args.no_latch:
    cfg.grasp_latch = False
if args.carry_demo is not None:
    cfg.carry_demo_p = args.carry_demo
if args.start_stage is not None:
    cfg.start_stage = args.start_stage
if args.surround_only:
    cfg.surround_only = True
if args.overhead_first:
    cfg.overhead_first = True
if args.curr_start is not None:
    cfg.curriculum_p_start = args.curr_start    # adaptive anneal resumes from here
cfg.scene.num_envs = args.num_envs
if hasattr(cfg, "seed"):
    cfg.seed = args.seed
env = DroneSnatchEnv(cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)

# W&B auth from the gitignored key (repo convention). On a machine with no W&B
# credentials (env var, key file, or `wandb login`), fall back to tensorboard
# instead of crashing at logger init.
_kf = os.path.join(_REPO, ".wandb_key")
if "WANDB_API_KEY" not in os.environ and os.path.exists(_kf):
    os.environ["WANDB_API_KEY"] = open(_kf).read().strip()
_netrc = os.path.expanduser("~/.netrc")
_use_wandb = ("WANDB_API_KEY" in os.environ
              or (os.path.exists(_netrc) and "api.wandb.ai" in open(_netrc).read()))
if not _use_wandb:
    print("[train_snatch] no W&B credentials found -> logging to tensorboard only")

# PPO hyperparams from the SNATCH spec: lr=3e-4, clip=0.2, entropy_coef=0.01,
# ~64 steps/rollout, 10 epochs/update.
agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=64,                 # ~64 steps/rollout (spec)
    max_iterations=args.max_iterations,
    save_interval=50,
    experiment_name="drone_snatch",
    seed=args.seed,
    logger="wandb" if _use_wandb else "tensorboard",  # W&B on by default (project convention)
    wandb_project="skyvla-isaac",
    empirical_normalization=True,         # normalize obs -> stability
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",            # std = exp(log_std) -> always > 0
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,                  # spec
        num_learning_epochs=10,          # 10 epochs/update (spec)
        num_mini_batches=4,
        learning_rate=3.0e-4,            # spec
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        entropy_coef=args.entropy_coef,  # lower when refining a warm-started policy
    ),
)

runner = OnPolicyRunner(
    env, class_to_dict(agent_cfg), log_dir=args.log_dir, device=env.unwrapped.device)
if args.resume is not None:
    # reset_std => fresh optimizer (loaded momentum is tuned to the huge-std regime and
    # would fight the reset); keep the good policy mean + obs normalizer.
    runner.load(args.resume, load_optimizer=(args.reset_std is None))
    print(f"[train_snatch] warm-started from {args.resume}")
    if args.reset_std is not None:
        import math, torch  # noqa: E402
        with torch.no_grad():
            pol = runner.alg.policy
            if hasattr(pol, "log_std"):                       # noise_std_type="log"
                pol.log_std.fill_(math.log(args.reset_std))
            elif hasattr(pol, "std"):
                pol.std.fill_(args.reset_std)
        print(f"[train_snatch] reset exploration std -> {args.reset_std} "
              f"(was ~17; entropy_coef={args.entropy_coef})")
runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
print("TRAIN_SNATCH_OK")
env.close()
sim_app.close()
