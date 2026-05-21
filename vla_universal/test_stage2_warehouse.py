"""Smoke test: frozen stage2_waypoint.pt flies to a warehouse POI.

No semantic map, no PaliGemma — only UniversalDroneEnv + WaypointController.
Use this to verify Stage 2 generalizes to warehouse geometry or is arena-only.

  bash ~/drone_project/vla_universal/run_test_stage2.sh \
      --scene warehouse_full --poi forklift_main
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test stage2 waypoint in warehouse.")
parser.add_argument("--scene", type=str, default="warehouse_full")
parser.add_argument("--poi", type=str, default="forklift_main",
                    help="POI name from vla_warehouse/pois.py")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Default: checkpoints/stage2_waypoint.pt")
parser.add_argument("--timeout_s", type=float, default=90.0)
parser.add_argument("--success_dist", type=float, default=1.0)
parser.add_argument("--record_video", action="store_true", default=False,
                    help="Needs RTX cameras; off by default on headless servers")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
args.headless = True
if args.record_video:
    args.enable_cameras = True
else:
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla  # noqa: F401
import vla_warehouse  # noqa: F401
import vla_universal  # noqa: F401

from vla_warehouse.pois import SCENES
from vla_universal.universal_env import UniversalDroneEnvCfg
from vla_universal.waypoint_controller import WaypointController
from vla_universal.math_utils import quat_rotate_inverse_np


def main() -> int:
    if args.scene not in SCENES:
        raise SystemExit(f"Unknown scene {args.scene!r}. Keys: {list(SCENES)}")

    pois = SCENES[args.scene]["pois"]
    poi = next((p for p in pois if p.name == args.poi), None)
    if poi is None:
        names = [p.name for p in pois]
        raise SystemExit(f"POI {args.poi!r} not in {args.scene}. Available: {names}")

    ckpt = args.checkpoint or os.path.join(_DRONE_PROJECT, "checkpoints", "stage2_waypoint.pt")
    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log(f"[test] scene={args.scene} poi={args.poi} ({poi.cls})")
    _log(f"[test] checkpoint={ckpt}")

    env_cfg = UniversalDroneEnvCfg()
    env_cfg.scene_name = args.scene
    env_cfg.scene.num_envs = 1
    env_cfg.spawn_xy_radius = 0.0
    env_cfg.physics_only = not args.record_video
    env_cfg.sim.device = "cuda:0"
    if args.record_video:
        env_cfg.camera_every_n = 1
        env_cfg.sim.render_interval = 1
        env_cfg.decimation = 1

    env = gym.make("Isaac-VLADrone-Universal-v0", cfg=env_cfg)
    env_impl = env.unwrapped
    zero = torch.zeros((1, 4), device="cuda:0")
    for _ in range(15):
        env.step(zero)

    origin = env_impl._terrain.env_origins[0].detach().cpu().numpy()
    target_local = np.array([poi.x, poi.y, poi.z], dtype=np.float32)
    target_xyz = origin + target_local
    _log(f"[test] env_origin={origin}")
    _log(f"[test] target_local={target_local} -> world={target_xyz}")

    controller = WaypointController(ckpt_path=ckpt, device="cuda:0")
    target_range = 3.0

    writer = None
    video_path = None
    if args.record_video:
        import cv2
        video_dir = os.path.join(_DRONE_PROJECT, "videos", "vla_universal")
        os.makedirs(video_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(
            video_dir, f"test_stage2_{args.scene}_{args.poi}_{stamp}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 25.0, (448, 224))
        _log(f"[test] recording {video_path}")

    max_steps = int(args.timeout_s * 50)
    converged_ticks = 0
    final_dist = float("inf")

    for step in range(max_steps):
        fs = env_impl.get_flight_state()
        drone_pos, drone_quat = env_impl.get_drone_pose()
        pos_err_w = (target_xyz - drone_pos).astype(np.float32)
        target_body = np.clip(
            quat_rotate_inverse_np(drone_quat, pos_err_w),
            -target_range, target_range,
        )
        action = controller.act(fs, target_body, pos_err_w)
        env.step(action.unsqueeze(0).float().clamp(-1.0, 1.0))

        final_dist = float(np.linalg.norm(pos_err_w))
        speed = float(np.linalg.norm(fs[:3]))

        if writer is not None and step % 2 == 0:
            import cv2
            batch = env_impl.get_camera_batch()
            front = (batch["rgb"][0] * 255).clip(0, 255).astype("uint8")
            front = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            right = (batch["rgb"][1] * 255).clip(0, 255).astype("uint8")
            right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            cv2.putText(front, f"dist {final_dist:.2f}m spd {speed:.2f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            writer.write(np.concatenate([front, right], axis=1))

        if final_dist < 0.5 and speed < 0.3:
            converged_ticks += 1
            if converged_ticks >= 50:
                _log(f"[test] PASS converged step={step} dist={final_dist:.2f}m speed={speed:.2f}")
                break
        else:
            converged_ticks = 0

        if step % 100 == 0:
            _log(f"[test] step {step:5d} pos=({drone_pos[0]:+.1f},{drone_pos[1]:+.1f},"
                 f"{drone_pos[2]:+.1f}) dist={final_dist:.2f}m speed={speed:.2f}")
    else:
        _log(f"[test] TIMEOUT final_dist={final_dist:.2f}m")

    success = final_dist <= args.success_dist
    _log(f"[test] result: {'SUCCESS' if success else 'FAIL'} "
         f"(dist={final_dist:.2f}m, threshold={args.success_dist}m)")
    if writer is not None:
        writer.release()
        _log(f"[test] video: {video_path}")

    env.close()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)
