"""Phase 1 of filming a PICK-UP: roll out the grasp expert headlessly and dump the best
successful grasp+lift to .npz. Snatch-task sibling of film_place_rollout.py.

Same reason for the split: Isaac's RTX camera cannot run on this host (vkCreateInstance
fails), so the physics is recorded headlessly and drawn afterwards by film_place_draw.py.

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
  .venv311/bin/python skyvla_isaac/scripts/film_snatch_rollout.py \
      --checkpoint skyvla_isaac/snatch/checkpoints/model_9250.pt --out /tmp/pick.npz
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--side_spawn", type=float, default=1.6, help="fly-in radius from the cube (m)")
parser.add_argument("--tail", type=int, default=55, help="frames to keep after the lift registers")
parser.add_argument("--min_hold", type=int, default=25, help="lift must survive this many frames")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app = AppLauncher(args).app

import numpy as np, torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False
# the grasp task exactly as π_snatch was evaluated: single platform, REAL friction cage
# (no kinematic latch), honest full fly-in from altitude with no straddle-start freebies
cfg.place_only = cfg.two_platform = cfg.release_only = False
cfg.grasp_latch = False
cfg.carry_demo_p = 0.0
cfg.curriculum_p_start = cfg.curriculum_p_end = 0.0
cfg.side_spawn_max = args.side_spawn
cfg.staged_curriculum = True          # soft table-touch, as in training
cfg.episode_length_s = 12.0
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
rec = {k: [] for k in ("drone", "cube", "jaw", "held", "lifted", "reach", "done")}
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
    rec["held"].append(env._held.cpu().numpy())
    rec["lifted"].append(env._lifted.cpu().numpy())
    rec["reach"].append(env._d_reach.cpu().numpy())
    rec["done"].append(dones.bool().cpu().numpy())
R = {k: np.stack(v) for k, v in rec.items()}

held_any = R["held"].any(0).sum()
print(f"[film] {held_any}/{n} envs grasped and lifted", flush=True)

best, best_score = None, -1e9
for e in range(n):
    h = R["held"][:, e]
    if not h.any():
        continue
    t_h = int(np.argmax(h))
    d_before = np.where(R["done"][:t_h, e])[0]
    t0 = int(d_before[-1]) + 1 if len(d_before) else 0
    if t_h - t0 < 20:                                  # want the whole fly-in on screen
        continue
    end = min(args.steps, t_h + args.tail)
    d_after = np.where(R["done"][t_h:end, e])[0]
    if len(d_after):
        end = t_h + int(d_after[0]) + 1
    if not R["held"][t_h:min(t_h + args.min_hold, end), e].all():
        continue                                       # grasp must SURVIVE, not flicker
    lift = float(R["cube"][end-1, e, 2] - cfg.surface_z - 0.5 * cfg.cube_size)
    if lift > best_score:
        best_score, best = lift, (e, t0, t_h, end)

if best is None:
    print("[film] NO usable take -- rerun with more envs/steps", flush=True)
    import os; os._exit(1)
e, t0, t_h, end = best
plat = (env.platform.data.root_pos_w - org)[e].cpu().numpy()
sl = slice(t0, end)
np.savez(args.out, mode="snatch",
         drone=R["drone"][sl, e], cube=R["cube"][sl, e], jaw=R["jaw"][sl, e],
         held=R["held"][sl, e], lifted=R["lifted"][sl, e], reach=R["reach"][sl, e],
         pad_a=plat, surface_z=cfg.surface_z, cube_size=cfg.cube_size,
         grasp_clear=cfg.grasp_clear, t_grasp=t_h - t0, dt=0.02)
d0 = float(np.linalg.norm(R["drone"][t0, e, :2] - R["cube"][t0, e, :2]))
print(f"[film] env {e}: {end-t0} frames | fly-in from {d0:.2f} m | grasp+lift @{t_h-t0}", flush=True)
print(f"[film] lift height {best_score*100:.1f} cm above the table", flush=True)
print("FILM_ROLLOUT_OK", flush=True)
import os; os._exit(0)
