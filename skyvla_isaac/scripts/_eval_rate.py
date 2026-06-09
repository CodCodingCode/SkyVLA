"""Quick success-rate eval: deterministic policy over many envs at fixed reverse-curriculum
difficulties. Reports the fraction that GRASP (lift+hold) and PLACE (deliver) per episode."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--diffs", type=str, default="0.0,0.05,0.075")
parser.add_argument("--cube_size", type=float, default=0.05, help="match the trained policy's cube edge (m)")
parser.add_argument("--cube_mass", type=float, default=0.05, help="match the trained policy's cube mass (kg)")
parser.add_argument("--speed", type=float, default=0.6, help="match the trained policy's max speed (m/s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False
cfg.reverse_curriculum = True
cfg.cube_size = args.cube_size
cfg.cube_mass = args.cube_mass
cfg.speed = args.speed
cfg.scene.num_envs = args.num_envs
env = DroneSnatchEnv(cfg); wenv = RslRlVecEnvWrapper(env)
agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50, experiment_name="drone_snatch",
    logger="tensorboard", empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
                                  actor_hidden_dims=[256,128,64], critic_hidden_dims=[256,128,64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                   entropy_coef=0.01, num_learning_epochs=10, num_mini_batches=4,
                                   learning_rate=3e-4, schedule="adaptive", gamma=0.99, lam=0.95,
                                   desired_kl=0.01, max_grad_norm=1.0))
runner = OnPolicyRunner(wenv, class_to_dict(agent_cfg), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
print(f"[eval] loaded {args.checkpoint}, num_envs={args.num_envs}")

ep_len = int(env.max_episode_length)
obs, _ = wenv.get_observations()
for diff in [float(x) for x in args.diffs.split(",")]:
    env._rc_p = diff
    # warm up 1.5 episodes so EVERY env has auto-reset at this difficulty, then measure the
    # steady-state holding/placed fraction over a full episode (same definition as training).
    for t in range(int(ep_len * 1.5)):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = wenv.step(act)
    g = p = 0.0
    for t in range(ep_len):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = wenv.step(act)
        g += env._held.float().mean().item()
        p += env._success.float().mean().item()
    print(f"[eval] difficulty={diff:.3f}  grasp_held={g/ep_len:.3f}  "
          f"placed={p/ep_len:.3f}  (steady-state, n={args.num_envs})", flush=True)
print("EVAL_RATE_DONE")
app.close()
