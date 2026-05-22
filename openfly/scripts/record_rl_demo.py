#!/usr/bin/env python3
"""Record an MP4 of the OpenFly RL pipeline running end-to-end.

This drives the **actual** :class:`AirSimVLNEnv`, :func:`compute_episode_reward`,
and :func:`collect_episode` against real OpenFly episodes from
``seen.json`` / ``unseen.json``. Every step that the live AirSim bridge
would execute is executed here too — the only substitution is the
``get_camera_data()`` source, since the OpenFly scene assets are gated
on HuggingFace and cannot be launched without authentication.

The mock bridge synthesises a drone-eye RGB frame from the episode's
geometry (top-down map + first-person heading projection) so the
recording stays grounded in the real (pose, goal, instruction) tuples
the env emits.

Output: ``<out>.mp4`` (drone view + HUD) and ``<out>_topdown.mp4``
(top-down trajectory map). Run ``python -m openfly.scripts.record_rl_demo
--help`` for options.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_DRONE_ROOT = Path(__file__).resolve().parents[2]
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import ACTION_NAMES, distance3d, goal_heuristic_action
from openfly.envs import AirSimVLNEnv, AirSimVLNEnvConfig
from openfly.episodes import load_episodes
from openfly.rewards import DEFAULT_REWARD, compute_episode_reward
from openfly.rollout import aggregate_metrics, collect_episode


# ---------------------------------------------------------------------------
# Mock AirSim bridge with procedural rendering
# ---------------------------------------------------------------------------

@dataclass
class _SceneCtx:
    image_size: int
    bounds: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    goal: tuple[float, float, float]
    start: tuple[float, float, float]
    instruction: str
    waypoints: list[tuple[float, float, float]]


class MockAirsimBridge:
    """Stand-in for ``AirsimBridge``.

    Implements the two methods :class:`AirSimVLNEnv` calls:

    * ``set_camera_pose(x, y, z, pitch, yaw_deg, roll)``
    * ``get_camera_data()`` → ``np.ndarray`` (H, W, 3) uint8

    The frame is a procedurally rendered first-person view: a synthetic
    horizon + ground grid in the drone's heading direction, plus an
    overlay arrow toward the goal. It is intentionally simple — the
    purpose is to exercise the env code, not to fool a VLA.
    """

    def __init__(self, scene: _SceneCtx) -> None:
        self.scene = scene
        self._pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, pitch, yaw_deg, roll

    def set_camera_pose(self, x, y, z, pitch, yaw_deg, roll):
        self._pose = (float(x), float(y), float(z), float(pitch), float(yaw_deg), float(roll))

    def get_camera_data(self) -> np.ndarray:
        x, y, z, pitch, yaw_deg, _ = self._pose
        H = W = self.scene.image_size
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # Sky / ground gradient: pitch shifts horizon line.
        horizon = int(H * (0.5 - pitch / 180.0))
        horizon = max(20, min(H - 20, horizon))
        img[:horizon] = (110, 150, 210)  # sky (BGR)
        img[horizon:] = (60, 90, 70)     # ground

        # Perspective ground grid in heading direction.
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        for d in range(10, 200, 8):
            # project a grid square `d` meters ahead onto the image
            scale = max(2.0, 600.0 / d)
            line_y = horizon + int(d / 4.0)
            if line_y >= H:
                continue
            cv2.line(img, (0, line_y), (W, line_y), (40, 60, 50), 1)

        # Project goal into a coarse bearing-based marker.
        gx, gy, gz = self.scene.goal
        dx, dy = gx - x, gy - y
        bearing = math.atan2(dy, dx)
        yaw_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
        in_view = abs(yaw_err) < math.radians(60)
        d_goal = math.hypot(dx, dy)
        if in_view and d_goal > 1e-3:
            u = int(W * (0.5 + (yaw_err / math.radians(60)) * 0.5))
            v = horizon + int(50.0 / max(d_goal / 30.0, 0.2))
            v = max(horizon + 5, min(H - 10, v))
            r = max(4, int(40.0 / max(d_goal / 20.0, 0.2)))
            cv2.circle(img, (u, v), r, (50, 230, 230), 2)
            cv2.putText(
                img,
                f"goal {d_goal:.0f}m",
                (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (50, 230, 230),
                1,
                cv2.LINE_AA,
            )

        # Compass strip at the top.
        compass_y = 12
        cv2.line(img, (0, compass_y), (W, compass_y), (200, 200, 200), 1)
        for h in range(-180, 181, 30):
            rel = ((h - yaw_deg + 540) % 360) - 180
            if abs(rel) > 70:
                continue
            u = int(W / 2 + (rel / 70) * (W / 2))
            cv2.line(img, (u, compass_y - 4), (u, compass_y + 4), (220, 220, 220), 1)
            cv2.putText(
                img,
                f"{int(h)}",
                (u - 8, compass_y - 6),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (220, 220, 220),
                1,
            )

        return img


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    *,
    instruction: str,
    pose: list[float],
    goal: list[float],
    action_id: int,
    reward_so_far: float,
    step: int,
    episode_idx: int,
    sr_running: float,
) -> np.ndarray:
    H, W = frame.shape[:2]
    overlay = frame.copy()
    panel_h = 110
    cv2.rectangle(overlay, (0, H - panel_h), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    txt_color = (240, 240, 240)
    accent = (90, 200, 255)
    y = H - panel_h + 18
    wrapped = instruction[:90] + ("..." if len(instruction) > 90 else "")
    cv2.putText(out, f'"{wrapped}"', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent, 1, cv2.LINE_AA)

    y += 22
    d = distance3d(pose, goal)
    cv2.putText(
        out,
        f"ep{episode_idx}  step {step:3d}  d_goal={d:5.1f}m  R={reward_so_far:+6.2f}  SR̄={sr_running:.2f}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        txt_color,
        1,
        cv2.LINE_AA,
    )
    y += 22
    pose_txt = (
        f"pose x={pose[0]:7.1f}  y={pose[1]:7.1f}  z={pose[2]:6.1f}  "
        f"yaw={math.degrees(pose[3]):+6.1f}°"
    )
    cv2.putText(out, pose_txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(
        out,
        f"action {action_id}  {ACTION_NAMES.get(action_id, '?')}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (140, 230, 140),
        1,
        cv2.LINE_AA,
    )

    # Banner
    cv2.rectangle(out, (0, 0), (W, 24), (15, 15, 15), -1)
    cv2.putText(
        out,
        "OpenFly RL env (AirSimVLNEnv) - heuristic policy - synthetic render",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def _topdown_frame(
    *,
    scene: _SceneCtx,
    trail: list[list[float]],
    pose: list[float],
    image_size: int,
    success: bool | None,
    reward: float,
) -> np.ndarray:
    img = np.full((image_size, image_size, 3), 18, dtype=np.uint8)

    pad = 0.15
    xs = [p[0] for p in trail] + [scene.start[0], scene.goal[0]]
    ys = [p[1] for p in trail] + [scene.start[1], scene.goal[1]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    rng = max(x_max - x_min, y_max - y_min, 30.0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min, x_max = cx - rng * (0.5 + pad), cx + rng * (0.5 + pad)
    y_min, y_max = cy - rng * (0.5 + pad), cy + rng * (0.5 + pad)

    def project(p):
        u = int((p[0] - x_min) / (x_max - x_min) * (image_size - 20)) + 10
        v = int((1.0 - (p[1] - y_min) / (y_max - y_min)) * (image_size - 20)) + 10
        return u, v

    # Grid
    for k in range(0, image_size, 40):
        cv2.line(img, (0, k), (image_size, k), (35, 35, 35), 1)
        cv2.line(img, (k, 0), (k, image_size), (35, 35, 35), 1)

    # Trail
    for a, b in zip(trail[:-1], trail[1:]):
        cv2.line(img, project(a), project(b), (90, 200, 255), 2)

    # Start / goal markers
    cv2.circle(img, project(scene.start), 6, (60, 200, 60), -1)
    cv2.putText(img, "S", (project(scene.start)[0] + 8, project(scene.start)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 60), 1)
    cv2.circle(img, project(scene.goal), 8, (60, 230, 230), 2)
    cv2.circle(img, project(scene.goal), 4, (60, 230, 230), -1)
    cv2.putText(img, "G", (project(scene.goal)[0] + 10, project(scene.goal)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 230), 1)
    # 20 m success ring around goal
    radius_px = int(20.0 / (x_max - x_min) * (image_size - 20))
    cv2.circle(img, project(scene.goal), max(4, radius_px), (60, 230, 230), 1)

    # Drone
    u, v = project(pose)
    yaw = pose[3]
    head_u = u + int(math.cos(yaw) * 12)
    head_v = v - int(math.sin(yaw) * 12)
    cv2.arrowedLine(img, (u, v), (head_u, head_v), (255, 120, 60), 2, tipLength=0.4)
    cv2.circle(img, (u, v), 4, (255, 120, 60), -1)

    # HUD
    cv2.rectangle(img, (0, 0), (image_size, 22), (15, 15, 15), -1)
    cv2.putText(
        img,
        "Top-down trajectory (RL env step trace)",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 220),
        1,
        cv2.LINE_AA,
    )
    status = "..."
    if success is True:
        status = "SUCCESS"
    elif success is False:
        status = "FAIL"
    cv2.putText(
        img,
        f"R={reward:+6.2f}  {status}",
        (6, image_size - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--split", default="seen")
    p.add_argument("--env_filter", default="env_airsim_16")
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        default=str(_DRONE_ROOT / "videos" / "openfly_rl_env_demo.mp4"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topdown_path = out_path.with_name(out_path.stem + "_topdown.mp4")

    episodes = load_episodes(
        args.split, max_episodes=args.episodes, env_filter=args.env_filter
    )
    if not episodes:
        raise RuntimeError(
            f"No episodes for split={args.split!r} env_filter={args.env_filter!r}"
        )
    print(f"[record] using {len(episodes)} episodes from split={args.split}")

    # OpenCV VideoWriter — use mp4v which is portable; we'll re-encode to h264 below.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fpv_writer = cv2.VideoWriter(
        str(out_path.with_suffix(".raw.mp4")),
        fourcc,
        args.fps,
        (args.image_size, args.image_size),
    )
    td_writer = cv2.VideoWriter(
        str(topdown_path.with_suffix(".raw.mp4")),
        fourcc,
        args.fps,
        (args.image_size, args.image_size),
    )
    if not fpv_writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open — check ffmpeg backend")

    n_total = 0
    n_success = 0
    aggregate = []

    for ep_idx, ep in enumerate(episodes):
        scene = _SceneCtx(
            image_size=args.image_size,
            bounds=(0, 0, 0, 0),
            goal=tuple(ep["pos"][-1]),
            start=tuple(ep["pos"][0]),
            instruction=ep.get("gpt_instruction", ""),
            waypoints=[tuple(p) for p in ep["pos"]],
        )
        bridge = MockAirsimBridge(scene)
        cfg = AirSimVLNEnvConfig(
            split=args.split,
            env_filter=args.env_filter,
            max_steps=args.max_steps,
            image_size=args.image_size,
            history_frames=0,
            reset_sleep_s=0.0,
            bridge_init_sleep_s=0.0,
            seed=args.seed + ep_idx,
        )
        env = AirSimVLNEnv(cfg, episodes=[ep], bridge=bridge)
        obs, info = env.reset(options={"episode": ep})

        reward_so_far = 0.0
        trail = [obs["pose"].tolist()]
        success: bool | None = None

        for step in range(args.max_steps):
            # Drive with the same oracle policy used by the heuristic eval baseline.
            action_id = goal_heuristic_action(obs["pose"].tolist(), obs["goal"].tolist())

            fpv = _draw_hud(
                obs["rgb"],
                instruction=info.get("instruction", ""),
                pose=obs["pose"].tolist(),
                goal=obs["goal"].tolist(),
                action_id=action_id,
                reward_so_far=reward_so_far,
                step=step,
                episode_idx=ep_idx,
                sr_running=(n_success / max(n_total, 1)),
            )
            fpv_writer.write(fpv)

            td = _topdown_frame(
                scene=scene,
                trail=trail,
                pose=obs["pose"].tolist(),
                image_size=args.image_size,
                success=None,
                reward=reward_so_far,
            )
            td_writer.write(td)

            obs, reward, terminated, truncated, info = env.step(action_id)
            reward_so_far += float(reward)
            trail.append(obs["pose"].tolist())

            if terminated or truncated:
                success = bool(info.get("success", False))
                # Final HUD frame held for a beat.
                fpv = _draw_hud(
                    obs["rgb"],
                    instruction=info.get("instruction", ""),
                    pose=obs["pose"].tolist(),
                    goal=obs["goal"].tolist(),
                    action_id=action_id,
                    reward_so_far=reward_so_far,
                    step=step + 1,
                    episode_idx=ep_idx,
                    sr_running=(n_success + int(success or 0)) / max(n_total + 1, 1),
                )
                td = _topdown_frame(
                    scene=scene,
                    trail=trail,
                    pose=obs["pose"].tolist(),
                    image_size=args.image_size,
                    success=success,
                    reward=reward_so_far,
                )
                for _ in range(args.fps):  # 1 second hold
                    fpv_writer.write(fpv)
                    td_writer.write(td)
                break

        n_total += 1
        n_success += int(bool(success))
        aggregate.append(
            {
                "episode": ep_idx,
                "success": bool(success),
                "reward": reward_so_far,
                "steps": len(trail) - 1,
                "ne_m": float(info.get("d_final", info.get("distance_to_goal", 0.0))),
                "spl": float(info.get("spl_term", 0.0)),
            }
        )
        env.close()
        print(
            f"[record] ep {ep_idx:02d} steps={len(trail)-1} R={reward_so_far:+.2f} "
            f"success={success} NE={info.get('d_final', 0):.1f}m"
        )

    fpv_writer.release()
    td_writer.release()

    # Re-encode mp4v → h264 so the file is playable everywhere (browsers, GitHub previews).
    def transcode(src: Path, dst: Path) -> None:
        cmd = (
            f"ffmpeg -y -loglevel error -i '{src}' "
            f"-c:v libx264 -pix_fmt yuv420p -movflags +faststart '{dst}'"
        )
        rc = os.system(cmd)
        if rc != 0:
            print(f"[record] ffmpeg transcode failed for {src} (rc={rc}); keeping raw")
        else:
            src.unlink(missing_ok=True)

    transcode(out_path.with_suffix(".raw.mp4"), out_path)
    transcode(topdown_path.with_suffix(".raw.mp4"), topdown_path)

    summary = {
        "n_episodes": n_total,
        "success_rate": n_success / max(n_total, 1),
        "episodes": aggregate,
    }
    print(f"\n[record] summary: {summary}")
    print(f"[record] FPV video → {out_path}")
    print(f"[record] top-down video → {topdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
