"""Render a trained pick-and-place rollout to MP4 from a close Isaac Camera.

Runs the deterministic policy in 1 env with a 3rd-person Camera sensor framing
the workspace, captures RGB each step, encodes an mp4 (libx264, QuickTime-ready).

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES
  .venv311/bin/python skyvla_isaac/scripts/render_rollout.py \
     --checkpoint logs/isaac/drone_pick_place/model_1499.pt --out videos/isaac_pickplace.mp4
"""
import argparse
import os
import subprocess
import tempfile

from isaaclab.app import AppLauncher


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default=os.path.join(_REPO, "videos/isaac_pickplace.mp4"))
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--cur_p", type=float, default=None,
                    help="pin straddle-start fraction (0.0 = full fly-in-from-altitude task)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.tasks.pick_place_env import DronePickPlaceEnv, DronePickPlaceEnvCfg  # noqa: E402

cfg = DronePickPlaceEnvCfg()
if args.cur_p is not None:
    cfg.curriculum_p_start = cfg.curriculum_p_end = args.cur_p   # pin start (no anneal)
cfg.scene.num_envs = 1
cfg.render_camera = True
env = DronePickPlaceEnv(cfg)
wenv = RslRlVecEnvWrapper(env)

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50,
    experiment_name="drone_pick_place", logger="tensorboard", empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
                                  actor_hidden_dims=[256, 128, 64],
                                  critic_hidden_dims=[256, 128, 64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0),
)
runner = OnPolicyRunner(wenv, class_to_dict(agent_cfg), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
print(f"[render] loaded {args.checkpoint}")

obs, _ = wenv.get_observations()
cam = env._render_cam
off_eye = torch.tensor([1.3, 1.3, 0.55], device=env.device)   # 3/4 close follow offset
off_tgt = torch.tensor([0.0, 0.0, -0.2], device=env.device)   # aim a bit below (gripper/cube)

def follow():
    dp = env.robot.data.root_pos_w[0]                          # drone world pos (env0 @ origin)
    cam.set_world_poses_from_view((dp + off_eye).unsqueeze(0), (dp + off_tgt).unsqueeze(0))

follow()
tmp = tempfile.mkdtemp(prefix="rollout_")
nf = 0
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, _, _ = wenv.step(act)
    follow()                                            # camera tracks the drone
    rgb = cam.data.output["rgb"][0, ..., :3]
    img = rgb.detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), img)
    nf += 1
print(f"[render] captured {nf} frames")

cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-crf", "21", args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-800:])
print(f"[render] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
print("RENDER_DONE")
sim_app.close()
