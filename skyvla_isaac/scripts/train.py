"""Train the drone pick-and-place policy with PPO (rsl_rl) on Isaac Lab.

Massively parallel: thousands of PhysX envs on one GPU. Real contact grasping.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES \
    python skyvla_isaac/scripts/train.py --num_envs 2048 --max_iterations 1500
  # smoke:
    python skyvla_isaac/scripts/train.py --num_envs 256 --max_iterations 3
"""
import argparse
import os

from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--log_dir", type=str, default=os.path.join(_REPO, "logs/isaac/drone_pick_place"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.tasks.pick_place_env import DronePickPlaceEnv, DronePickPlaceEnvCfg  # noqa: E402

cfg = DronePickPlaceEnvCfg()
cfg.scene.num_envs = args.num_envs
env = DronePickPlaceEnv(cfg, render_mode=None)
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
    print("[train] no W&B credentials found -> logging to tensorboard only")

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=args.max_iterations,
    save_interval=50,
    experiment_name="drone_pick_place",
    logger="wandb" if _use_wandb else "tensorboard",  # W&B on by default (project convention)
    wandb_project="skyvla-isaac",
    empirical_normalization=True,        # normalize observations -> stability
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",           # std = exp(log_std) -> always > 0 (fixes std>=0 crash)
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    ),
)

runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=args.log_dir, device=env.unwrapped.device)
runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
print("TRAIN_SMOKE_OK")
env.close()
sim_app.close()
