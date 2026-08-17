"""Evaluate a trained pick-place policy: deterministic rollout, real success
rate, and (optional) video.

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES
  .venv311/bin/python skyvla_isaac/scripts/play.py --checkpoint logs/isaac/drone_pick_place/model_1499.pt
  .venv311/bin/python skyvla_isaac/scripts/play.py --checkpoint <ckpt> --video   # also save mp4
"""
import argparse
import os

from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--cur_p", type=float, default=None,
                    help="pin straddle-start fraction (no anneal); default keeps cfg default")
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.tasks.pick_place_env import DronePickPlaceEnv, DronePickPlaceEnvCfg  # noqa: E402

cfg = DronePickPlaceEnvCfg()
if args.cur_p is not None:
    cfg.curriculum_p_start = cfg.curriculum_p_end = args.cur_p   # pin (no anneal during eval)
cfg.scene.num_envs = 1 if args.video else args.num_envs
env = DronePickPlaceEnv(cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    env = gym.wrappers.RecordVideo(
        env, video_folder=os.path.join(_REPO, "videos/isaac_play"),
        step_trigger=lambda s: s == 0, video_length=500, disable_logger=True)
env = RslRlVecEnvWrapper(env)
base = env.unwrapped

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50,
    experiment_name="drone_pick_place", logger="tensorboard",
    empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
                                  actor_hidden_dims=[256, 128, 64],
                                  critic_hidden_dims=[256, 128, 64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0),
)
runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=base.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=base.device)
print(f"[play] loaded {args.checkpoint}")

obs, _ = env.get_observations()
n = base.num_envs
ever = torch.zeros(n, dtype=torch.bool, device=base.device)   # delivered at least once this episode
ep_succ, ep_cnt, deliver_steps, grasp_steps = 0, 0, 0.0, 0.0
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, dones, _ = env.step(act)
    ever |= base._success
    deliver_steps += float(base._success.float().sum().item())
    grasp_steps += float(base._held.float().sum().item())
    dm = dones.bool()
    if dm.any():
        ep_succ += int((ever & dm).sum().item())
        ep_cnt += int(dm.sum().item())
        ever[dm] = False

ep_rate = ep_succ / max(1, ep_cnt)
print(f"[play] episodes={ep_cnt}  EPISODE_SUCCESS_RATE={ep_rate:.1%}  "
      f"delivered_frac={deliver_steps/(args.steps*n):.1%}  grasp_frac={grasp_steps/(args.steps*n):.1%}")
print("PLAY_DONE")
env.close()
sim_app.close()
