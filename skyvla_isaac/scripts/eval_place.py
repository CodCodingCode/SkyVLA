"""Evaluate a PLACE-stage policy: does it deposit the cube on pad B, and how gently?

Deterministic (mean-action) rollouts on the place_only env at FULL difficulty -- the
start-ramp is disabled so every episode uses the real spec distribution (spawn holding the
cube anywhere in a 0.8 m disc around B, body 0.55-0.85 m) rather than the easy rungs the
policy may still be training on.

Per-EPISODE metrics (not per-step, so they can't be inflated by a deposit that simply sits
there earning for the rest of the episode):
  deposit %        episodes where the cube ended at rest on B with the jaws open
  contact speed    cube speed at the lip_h landing. Its FLOOR is the 8mm free fall
                   (~0.40 m/s) -- the cage lips bottom out on the pad, so the cube is
                   always released from lip_h. Sampled at policy rate over a 40ms fall,
                   so it reads a lower bound.
  landing error    cube distance from B's centre at rest
  release height   cube bottom over the pad at the moment the jaws let go (floor = 8mm)
  gentle %         deposits whose contact speed stayed within --gentle_bar

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
    .venv311/bin/python skyvla_isaac/scripts/eval_place.py \
      --checkpoint logs/isaac/drone_snatch_place_only/model_XXXX.pt
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=1500,
                    help="stop once at least this many episodes have finished")
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--spawn_r", type=float, default=0.8)
parser.add_argument("--spawn_h", type=float, nargs=2, default=[0.55, 0.85])
parser.add_argument("--plat_sep", type=float, default=1.5)
parser.add_argument("--gentle_bar", type=float, default=0.45,
                    help="contact-speed bar for the gentle%% column (default = the 8mm-fall floor "
                         "0.40 m/s plus a small margin)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False          # place stage is the state-based (--no_cams) variant
sim_app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402


def make_env():
    cfg = DroneSnatchEnvCfg()
    cfg.use_cameras = False
    cfg.place_only = True
    cfg.two_platform = True
    cfg.release_only = True
    cfg.grasp_latch = False
    cfg.carry_demo_p = 1.0
    cfg.curriculum_p_start = cfg.curriculum_p_end = 0.0
    cfg.place_curriculum = False            # FULL spec distribution, no easy rungs
    cfg.place_spawn_r = args.spawn_r
    cfg.place_spawn_h_lo, cfg.place_spawn_h_hi = args.spawn_h
    cfg.plat_sep = cfg.plat_sep_max = args.plat_sep
    cfg.episode_length_s = 8.0
    cfg.scene.env_spacing = max(cfg.scene.env_spacing,
                                2.0 * (cfg.plat_sep_max + cfg.place_spawn_r) + 2.0)
    cfg.scene.num_envs = args.num_envs
    return RslRlVecEnvWrapper(DroneSnatchEnv(cfg, render_mode=None))


def build_runner(env, device):
    """Net dims must match training or the checkpoint won't load."""
    agent_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=64, max_iterations=1, save_interval=50,
        experiment_name="drone_snatch", logger="tensorboard",
        empirical_normalization=True,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0, noise_std_type="log",
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64], activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.01, num_learning_epochs=10, num_mini_batches=4,
            learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0),
    )
    return OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=device)


env = make_env()
base = env.unwrapped
dev = base.device
runner = build_runner(env, dev)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=dev)
print(f"[eval_place] checkpoint={args.checkpoint}")
print(f"[eval_place] spawn disc r<={args.spawn_r}m  body z {args.spawn_h}  pad sep {args.plat_sep}m")

PAD = base.cfg.surface_z
REST = PAD + 0.5 * base.cfg.cube_size
n = base.num_envs
z = lambda: torch.zeros(n, device=dev)                                    # noqa: E731
ever_dep = torch.zeros(n, dtype=torch.bool, device=dev)
prev_rel = torch.zeros(n, dtype=torch.bool, device=dev)
rel_h = torch.full((n,), float("nan"), device=dev)                        # release height
# accumulators over FINISHED episodes
tot = dep = gentle = 0
sum_impact = sum_land = sum_relh = 0.0
n_relh = 0

obs, _ = env.get_observations()
for step in range(args.max_steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, dones, _ = env.step(act)
    cube = base.object.data.root_pos_w - base.scene.env_origins
    # capture the release height on the step the jaws first free the cube
    newly_rel = base._released & (~prev_rel)
    if bool(newly_rel.any()):
        rel_h[newly_rel] = (cube[:, 2] - 0.5 * base.cfg.cube_size - PAD)[newly_rel]
    prev_rel = base._released.clone()
    ever_dep |= base._deposited

    dm = dones.bool()
    if bool(dm.any()):
        d = dm & ever_dep
        tot += int(dm.sum().item())
        dep += int(d.sum().item())
        if bool(d.any()):
            sum_impact += float(base._impact_v[d].sum().item())
            sum_land += float(base._d_plat_b[d].sum().item())
            gentle += int((base._impact_v[d] <= args.gentle_bar).sum().item())
            good_h = d & torch.isfinite(rel_h)
            if bool(good_h.any()):
                sum_relh += float(rel_h[good_h].sum().item())
                n_relh += int(good_h.sum().item())
        ever_dep[dm] = False
        prev_rel[dm] = False
        rel_h[dm] = float("nan")
    if tot >= args.episodes:
        break

d1 = max(1, dep)
print("\n=== PLACE eval (deterministic, full spec distribution) ===")
print(f"episodes                 {tot}")
print(f"deposit rate             {dep / max(1, tot) * 100:.1f}%")
print(f"contact speed  (m/s)     {sum_impact / d1:.3f}    [8mm-fall floor ~0.40]")
print(f"landing error  (cm)      {sum_land / d1 * 100:.1f}     [gate: <{base.cfg.place_radius*100:.0f}]")
print(f"release height (cm)      {(sum_relh / max(1, n_relh)) * 100:.1f}     [floor {base.cfg.lip_h*100:.1f}]")
print(f"gentle (<={args.gentle_bar} m/s)      {gentle / d1 * 100:.1f}% of deposits")
print("EVAL_PLACE_OK")
env.close()
sim_app.close()
