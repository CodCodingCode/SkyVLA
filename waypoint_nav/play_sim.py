"""Record waypoint playback MP4 from Isaac Sim (RTX chase camera).

Launch:
    cd ~/IsaacLab
    source ~/drone_project/activate_env.sh
    ./isaaclab.sh -p ~/drone_project/waypoint_nav/play_sim.py \\
        --checkpoint ~/drone_project/checkpoints/stage2_waypoint.pt \\
        --num_steps 150 --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import cv2
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rsl_rl")
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaaclab.app import AppLauncher

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Record waypoint nav in Isaac Sim to MP4.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=200, help="Frames recorded after warmup.")
parser.add_argument("--output", type=str, default=None, help="Output .mp4 path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import waypoint_nav  # noqa: F401
import waypoint_nav.play_sim_env  # noqa: F401

from waypoint_nav.waypoint_nav_env import WaypointNavEnvCfg
from waypoint_nav.agents.rsl_rl_ppo_cfg import WaypointNavPPORunnerCfg


def _finalize_mp4_for_mac(path: str) -> None:
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        print("[WARN] ffmpeg not found — MP4 may not play on Mac", flush=True)
        return
    tmp = path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-profile:v", "high", "-level", "4.0", "-crf", "23",
        tmp,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[WARN] ffmpeg failed: {r.stderr[-500:]}", flush=True)
        return
    os.replace(tmp, path)
    print("[INFO] Re-encoded to H.264 (Mac / QuickTime compatible)", flush=True)


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    out_path = args_cli.output or os.path.join(_DRONE_PROJECT, "videos", "waypoint_sim_arena.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Output:     {out_path}")

    env_cfg = WaypointNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.clone_in_fabric = False
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 1

    from isaaclab.sim import RenderCfg
    env_cfg.sim.render = RenderCfg(
        enable_dl_denoiser=True,
        antialiasing_mode="DLAA",
        dome_light_upper_lower_strategy=4,
        enable_direct_lighting=True,
        samples_per_pixel=2,
    )

    env = gym.make("Isaac-WaypointNav-Direct-ChaseCam-v0", cfg=env_cfg)
    env_unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env)

    agent_cfg = WaypointNavPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device
    from importlib.metadata import version as pkg_version
    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    warmup = 60
    fps = 30
    writer = None
    frames_written = 0
    total = warmup + args_cli.num_steps

    obs, _ = env.reset()
    print(f"[INFO] RTX chase camera warmup={warmup} record={args_cli.num_steps}", flush=True)

    with torch.inference_mode():
        for step in range(total):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if step >= warmup:
                frame_bgr = env_unwrapped.get_chase_frame_bgr()
                if frame_bgr is not None:
                    if writer is None:
                        h, w = frame_bgr.shape[:2]
                        writer = cv2.VideoWriter(
                            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        print(f"[INFO] Recording {w}x{h}", flush=True)
                    cv2.putText(
                        frame_bgr,
                        f"step {step - warmup}",
                        (20, frame_bgr.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    writer.write(frame_bgr)
                    frames_written += 1

            if step % 25 == 0:
                print(f"  step {step}/{total}  frames={frames_written}", flush=True)

    env.close()

    if writer is not None:
        writer.release()
        _finalize_mp4_for_mac(out_path)
        print(f"[INFO] Saved {out_path} ({frames_written} frames)", flush=True)
    else:
        print("[ERROR] No frames captured.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
