"""Smoke test: build the Isaac Lab drone pick-place env (N parallel envs) and
step random actions headless. Confirms the articulation, cube, contact physics,
observations, and rewards all wire up."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
from skyvla_isaac.tasks.pick_place_env import DronePickPlaceEnv, DronePickPlaceEnvCfg  # noqa: E402

cfg = DronePickPlaceEnvCfg()
cfg.scene.num_envs = args.num_envs
env = DronePickPlaceEnv(cfg)
obs, _ = env.reset()
print(f"[smoke] obs policy shape: {tuple(obs['policy'].shape)} (expect (*, 17))")
print(f"[smoke] num_envs={env.num_envs} device={env.device} action_dim={env.cfg.action_space}")
rew_acc = 0.0
for i in range(args.steps):
    a = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
    obs, rew, term, trunc, info = env.step(a)
    rew_acc += float(rew.mean())
print(f"[smoke] stepped {args.steps}x; mean reward/step={rew_acc/args.steps:.4f}")
print(f"[smoke] object z range across envs: "
      f"{float(env.object.data.root_pos_w[:,2].min()):.2f}..{float(env.object.data.root_pos_w[:,2].max()):.2f}")
print("SMOKE_ENV_OK")
env.close()
sim_app.close()
