"""Phase 1 of filming a deposit: roll out the PLACE policy headlessly and dump the best
successful episode's trajectory to .npz.

Split from the drawing phase on purpose. render_place.py uses Isaac's RTX camera, which
CANNOT run on this host -- vkCreateInstance fails (`Vulkan 1.1 is not supported`) so the
Camera sensor hangs forever inside env construction. Headless physics is unaffected, so the
simulation here is the real one; only the drawing moves to matplotlib (film_place_draw.py).

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
  .venv311/bin/python skyvla_isaac/scripts/film_place_rollout.py \
      --checkpoint logs/isaac/drone_snatch_place_only_v4/model_20000.pt --out /tmp/take.npz
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--spawn_r", type=float, default=0.55)
parser.add_argument("--spawn_h", type=float, nargs=2, default=[0.60, 0.75])
parser.add_argument("--tail", type=int, default=45, help="frames to keep after the landing")
parser.add_argument("--min_approach", type=int, default=8, help="min frames held before release")
parser.add_argument("--min_settle", type=int, default=12, help="min frames after the landing")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False          # <- the whole point: no RTX on this host
app = AppLauncher(args).app

import numpy as np, torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False
cfg.place_only = cfg.two_platform = cfg.release_only = True
cfg.grasp_latch = False
cfg.carry_demo_p = 1.0
cfg.curriculum_p_start = cfg.curriculum_p_end = 0.0
cfg.place_curriculum = False
cfg.place_spawn_r = args.spawn_r
cfg.place_spawn_h_lo, cfg.place_spawn_h_hi = args.spawn_h
cfg.plat_sep = cfg.plat_sep_max = 1.5
cfg.episode_length_s = 10.0
cfg.scene.env_spacing = max(cfg.scene.env_spacing, 2.0 * (cfg.plat_sep_max + cfg.place_spawn_r) + 2.0)
cfg.scene.num_envs = args.num_envs
env = DroneSnatchEnv(cfg, render_mode=None)
wenv = RslRlVecEnvWrapper(env)

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50,
    experiment_name="drone_snatch", logger="tensorboard", empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
                                  actor_hidden_dims=[256, 128, 64],
                                  critic_hidden_dims=[256, 128, 64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True,
                                   clip_param=0.2, entropy_coef=0.01, num_learning_epochs=10,
                                   num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
                                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0))
runner = OnPolicyRunner(wenv, class_to_dict(agent_cfg), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
print(f"[film] loaded {args.checkpoint}", flush=True)

n = env.num_envs
org = env.scene.env_origins
rec = {k: [] for k in ("drone", "cube", "jaw", "released", "deposited", "holding", "done", "impact")}
_o = wenv.get_observations()
obs = _o[0] if isinstance(_o, tuple) else _o
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    _s = wenv.step(act)
    obs, dones = _s[0], _s[2]
    rec["drone"].append((env.robot.data.root_pos_w - org).cpu().numpy())
    rec["cube"].append((env.object.data.root_pos_w - org).cpu().numpy())
    rec["jaw"].append(env.robot.data.joint_pos[:, env._grip_i].mean(-1).cpu().numpy())
    rec["released"].append(env._released.cpu().numpy())
    rec["deposited"].append(env._deposited.cpu().numpy())
    rec["holding"].append(env._holding.cpu().numpy())
    rec["impact"].append(env._impact_v.cpu().numpy())
    rec["done"].append(dones.bool().cpu().numpy())
R = {k: np.stack(v) for k, v in rec.items()}          # (T, N, ...)

# pick the best take: deposits, survives to the end of its episode, and had a real approach
best, best_score = None, -1e9
for e in range(n):
    dep = R["deposited"][:, e]
    if not dep.any():
        continue
    t_dep = int(np.argmax(dep))
    d_before = np.where(R["done"][:t_dep, e])[0]
    t0 = int(d_before[-1]) + 1 if len(d_before) else 0        # start of THIS episode
    rel = np.where(R["released"][t0:, e])[0]
    if not len(rel):
        continue
    t_rel = t0 + int(rel[0])
    if t_rel - t0 < args.min_approach:                        # want a visible approach first
        continue
    end = min(args.steps, t_dep + args.tail)
    d_after = np.where(R["done"][t_dep:end, e])[0]
    if len(d_after):
        end = t_dep + int(d_after[0]) + 1
    if end - t_dep < args.min_settle:                         # want to watch it settle
        continue
    score = -abs(float(R["cube"][t_dep, e, 0]))               # prefer well-centred landings
    if score > best_score:
        best_score, best = score, (e, t0, t_rel, t_dep, end)

stats = []
for e in range(n):
    dep = R["deposited"][:, e]
    if not dep.any():
        continue
    t_dep = int(np.argmax(dep))
    d_before = np.where(R["done"][:t_dep, e])[0]
    t0 = int(d_before[-1]) + 1 if len(d_before) else 0
    rel = np.where(R["released"][t0:, e])[0]
    if len(rel):
        stats.append((t_rel := t0 + int(rel[0])) - t0)
if stats:
    a = np.array(stats)
    print(f"[film] {len(a)}/{n} envs deposited | frames held before release: "
          f"min {a.min()} median {int(np.median(a))} max {a.max()}", flush=True)
else:
    print(f"[film] 0/{n} envs deposited", flush=True)

if best is None:
    print("[film] NO usable take -- rerun with more envs/steps", flush=True)
    import os; os._exit(1)
e, t0, t_rel, t_dep, end = best
pad_b = (env.plat_b.data.root_pos_w - org)[e].cpu().numpy()
pad_a = (env.platform.data.root_pos_w - org)[e].cpu().numpy()
sl = slice(t0, end)
np.savez(args.out,
         drone=R["drone"][sl, e], cube=R["cube"][sl, e], jaw=R["jaw"][sl, e],
         released=R["released"][sl, e], deposited=R["deposited"][sl, e],
         holding=R["holding"][sl, e], impact=R["impact"][sl, e],
         pad_a=pad_a, pad_b=pad_b, surface_z=cfg.surface_z, cube_size=cfg.cube_size,
         place_radius=cfg.place_radius, lip_h=cfg.lip_h,
         t_rel=t_rel - t0, t_dep=t_dep - t0, dt=0.02)
land_err = float(np.linalg.norm(R["cube"][t_dep, e, :2] - pad_b[:2]))
print(f"[film] env {e}: {end-t0} frames | release @{t_rel-t0} | deposit @{t_dep-t0}", flush=True)
print(f"[film] contact speed {float(R['impact'][t_dep, e]):.3f} m/s   "
      f"landing error {land_err*100:.1f} cm   [gate <{cfg.place_radius*100:.0f}]", flush=True)
print(f"[film] release height {(float(R['cube'][t_rel, e, 2]) - cfg.surface_z - 0.5*cfg.cube_size)*100:.1f} cm "
      f"above rest   [floor {cfg.lip_h*100:.1f}]", flush=True)
print("FILM_ROLLOUT_OK", flush=True)
import os; os._exit(0)
