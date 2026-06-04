"""Render a SNATCH (state-based) pick-and-place rollout to MP4 with a 3rd-person cam.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1
  python skyvla_isaac/scripts/render_snatch.py \
     --checkpoint logs/isaac/drone_snatch_state/model_650.pt --out videos/snatch_pickplace.mp4
"""
import argparse, os, subprocess, tempfile
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default="/home/ubuntu/SkyVLA/videos/snatch_pickplace.mp4")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--cur_p", type=float, default=0.0)   # from-altitude demo
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import torch, numpy as np, imageio.v2 as imageio  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False          # state-based policy (model_650)
cfg.render_camera = True         # 3rd-person RGB cam for the video
cfg.curriculum_p_start = cfg.curriculum_p_end = args.cur_p
cfg.scene.num_envs = 1
env = DroneSnatchEnv(cfg); wenv = RslRlVecEnvWrapper(env)

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
print(f"[snatch-render] loaded {args.checkpoint}")

obs, _ = wenv.get_observations()
cam = env._render_cam
off_eye = torch.tensor([1.9, 1.9, -0.18], device=env.device)   # side-on, slightly below drone
off_tgt = torch.tensor([0.0, 0.0, -0.20], device=env.device)   # so the cube hanging below is clear
def follow():
    dp = env.robot.data.root_pos_w[0]
    cam.set_world_poses_from_view((dp + off_eye).unsqueeze(0), (dp + off_tgt).unsqueeze(0))
follow()
tmp = tempfile.mkdtemp(prefix="snatch_"); nf = 0
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, _, _ = wenv.step(act)
    follow(); cam.update(env.sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), rgb); nf += 1
print(f"[snatch-render] captured {nf} frames")
cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-crf", "21", args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-800:])
print(f"[snatch-render] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
print("SNATCH_RENDER_DONE")
app.close()
