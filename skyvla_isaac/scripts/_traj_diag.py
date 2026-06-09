"""Distance-mode trajectory diagnostic: spawn at a fly-in distance, log closest approach,
final distance, horizontal speed near the cube, and grasp -> tells us if it navigates,
overshoots, or reaches-but-cannot-settle."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--rc_p", type=float, default=0.1)   # 0.1 -> ~1m
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,  # noqa
                                RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa
cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False; cfg.reverse_curriculum = True; cfg.rc_distance_mode = True
cfg.rc_dist_max = 10.0; cfg.cube_mass = 0.05; cfg.speed = 0.6; cfg.rc_start = args.rc_p
cfg.scene.num_envs = args.num_envs; cfg.scene.env_spacing = 25.0; cfg.episode_length_s = 22.0
env = DroneSnatchEnv(cfg); wenv = RslRlVecEnvWrapper(env)
ac = RslRlOnPolicyRunnerCfg(num_steps_per_env=24, max_iterations=1, save_interval=50,
    experiment_name="d", logger="tensorboard", empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
        actor_hidden_dims=[256,128,64], critic_hidden_dims=[256,128,64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=10, num_mini_batches=4, learning_rate=3e-4,
        schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0))
runner = OnPolicyRunner(wenv, class_to_dict(ac), log_dir=None, device=env.device)
runner.load(args.checkpoint); policy = runner.get_inference_policy(device=env.device)
obs, _ = wenv.get_observations()
# warm up one episode so all envs reset at this difficulty
for _ in range(int(env.max_episode_length)):
    with torch.no_grad(): a = policy(obs)
    obs, _, _, _ = wenv.step(a)
# measure one episode
min_d = torch.full((args.num_envs,), 1e9, device=env.device)
hspeed_at_min = torch.zeros(args.num_envs, device=env.device)   # horiz speed at closest approach
ever_held = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
ever_close = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)  # tip within 8cm
ever_centered = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)  # horiz<3cm over cube
for t in range(int(env.max_episode_length)):
    with torch.no_grad(): a = policy(obs)
    obs, _, _, _ = wenv.step(a)
    hsp = torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=-1)
    newmin = env._d_reach < min_d
    hspeed_at_min = torch.where(newmin, hsp, hspeed_at_min)
    min_d = torch.minimum(min_d, env._d_reach)
    ever_held |= env._held
    ever_close |= (env._d_reach < 0.08)
    ever_centered |= (env._horiz < 0.03)
dist_m = args.rc_p ** 2 * 10.0
print(f"[traj] rc_p={args.rc_p} -> ~{dist_m:.2f}m fly-in, n={args.num_envs}")
print(f"[traj] closest approach d_reach: mean={min_d.mean():.3f}m median={min_d.median():.3f}m min={min_d.min():.3f}m")
print(f"[traj] horiz speed AT closest approach: mean={hspeed_at_min.mean():.3f}m/s (does it slow down?)")
print(f"[traj] reached cube(<8cm): {ever_close.float().mean():.3f}  centered(<3cm): {ever_centered.float().mean():.3f}  grasped+held: {ever_held.float().mean():.3f}")
print("TRAJ_DIAG_DONE"); app.close()
