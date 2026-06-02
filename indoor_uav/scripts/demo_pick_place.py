"""Scripted pick-and-place demo — verifies the gripper/grasp/place mechanic and
renders a video. A hand-coded controller (no trained policy) drives the env so we
can confirm the DroneGripper + PickPlaceEnv work end to end:

  APPROACH object -> DESCEND + close gripper (PICK) -> CARRY to target ->
  open gripper (PLACE) -> DONE.

Renders side-by-side:  LEFT = drone FPV + status ;  RIGHT = top-down (drone,
object, target) so you can watch the object get carried and placed.

  conda activate habitat; export CUDA_HOME=/usr; export PYTHONUTF8=1
  python -m indoor_uav.scripts.demo_pick_place --scene <glb>
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile

import numpy as np


def _topdown(res, bounds, traj, drone_p, yaw, obj, tgt, tgt_r, status):
    from PIL import Image, ImageDraw
    xmin, xmax, zmin, zmax = bounds
    im = Image.new("RGB", (res, res), (18, 18, 22)); dr = ImageDraw.Draw(im)

    def px(p):
        u = (p[0] - xmin) / (xmax - xmin + 1e-6) * (res - 20) + 10
        v = (p[2] - zmin) / (zmax - zmin + 1e-6) * (res - 20) + 10
        return (u, v)
    if len(traj) > 1:
        dr.line([px(p) for p in traj], fill=(90, 110, 160), width=2)
    gx, gy = px(tgt); rpx = tgt_r / (xmax - xmin + 1e-6) * (res - 20)
    dr.ellipse([gx - rpx, gy - rpx, gx + rpx, gy + rpx], outline=(80, 230, 120), width=2)
    ox, oy = px(obj)
    dr.ellipse([ox - 6, oy - 6, ox + 6, oy + 6], fill=(235, 70, 70))      # object
    dx, dy = px(drone_p)
    hx = dx + 15 * math.sin(yaw); hy = dy + 15 * math.cos(yaw)
    dr.line([dx, dy, hx, hy], fill=(120, 180, 255), width=3)
    dr.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill=(120, 180, 255))    # drone
    dr.rectangle([0, 0, res, 16], fill=(0, 0, 0)); dr.text((3, 3), status, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/home/ubuntu/SkyVLA/videos/pick_place_demo.mp4")
    ap.add_argument("--max_steps", type=int, default=320)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from indoor_uav.tasks.pick_place_env import PickPlaceEnv

    env = PickPlaceEnv(args.scene, max_steps=args.max_steps, load_urdf=True, seed=args.seed)
    env.reset()
    spd = env.speed; yrm = env.yaw_rate_max
    floor = env._floor_y

    def goto(world_xz_target, des_alt, lower, grip):
        """Normalized action: HOLONOMIC move toward a world target + gripper cmd.
        Commands the body-frame error directly (strafe + forward), so the drone
        settles ON the point instead of orbiting it (pure yaw-pursuit circles a
        near target). Gently faces the target when far, for a sensible FPV."""
        p = np.asarray(env.drone.position, np.float32)
        ex, ez = (np.asarray(world_xz_target, np.float32) - p)[[0, 2]]
        yaw = env.drone.yaw; c, s = math.cos(yaw), math.sin(yaw)
        vx_b = c * ex - s * ez                      # world error -> body frame
        vz_b = s * ex + c * ez
        mag = math.hypot(vx_b, vz_b) + 1e-6
        desired = min(mag * 1.5, 1.0) * spd         # proportional, capped -> hover at goal
        vx_b *= desired / mag; vz_b *= desired / mag
        des_yaw = math.atan2(ex, ez)
        yaw_err = (des_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_cmd = float(np.clip(yaw_err * 2.0 / yrm, -1, 1)) if mag > 0.6 * spd else 0.0
        vy = float(np.clip((des_alt - p[1]) * 2.0, -spd, spd))
        a = np.array([vx_b / spd, vz_b / spd, vy / spd, yaw_cmd,
                      lower * 2 - 1, grip * 2 - 1], np.float32)
        return np.clip(a, -1, 1)

    obj0 = env._obj_xyz().copy(); tgt = env._target_xyz.copy()
    pad = 2.5
    bounds = (min(obj0[0], tgt[0]) - pad, max(obj0[0], tgt[0]) + pad,
              min(obj0[2], tgt[2]) - pad, max(obj0[2], tgt[2]) + pad)

    tmp = tempfile.mkdtemp(prefix="pickplace_")
    traj = []; phase = "APPROACH"; carry_alt = floor + env.altitude
    grab_alt = floor + 0.55; done_hold = 0
    print(f"[demo] obj={obj0}  target={tgt}  floor={floor:.2f}")

    for i in range(args.max_steps):
        p = np.asarray(env.drone.position, np.float32)
        obj = env._obj_xyz()
        d_obj_xz = float(np.linalg.norm((p - obj)[[0, 2]]))
        d_tgt_xz = float(np.linalg.norm((p - tgt)[[0, 2]]))
        holding = env.gripper.held is not None

        if phase == "APPROACH":
            a = goto(obj, carry_alt, lower=0.0, grip=0.0)
            if d_obj_xz < 0.35:
                phase = "DESCEND"
        elif phase == "DESCEND":
            # descend with the gripper extended; once low + roughly over the
            # object, close the jaws — the grasp radius tolerates small offset.
            low = p[1] < grab_alt + 0.25
            grip = 1.0 if (low and d_obj_xz < 0.4) else 0.0
            a = goto(obj, grab_alt, lower=1.0, grip=grip)
            if holding:
                phase = "LIFT"
        elif phase == "LIFT":
            a = goto(tgt, carry_alt, lower=0.0, grip=1.0)   # rise + head to target
            if p[1] > carry_alt - 0.2:
                phase = "CARRY"
        elif phase == "CARRY":
            a = goto(tgt, carry_alt, lower=0.0, grip=1.0)
            if d_tgt_xz < 0.35:
                phase = "PLACE"
        elif phase == "PLACE":
            a = goto(tgt, carry_alt, lower=0.3, grip=0.0)   # open -> release over target
            done_hold += 1
        else:
            a = np.zeros(6, np.float32)

        _, r, term, trunc, info = env.step(a)
        traj.append(p.copy())
        if info["success"] and phase != "DONE":
            phase = "DONE"; print(f"[demo] PLACED successfully at step {i}")

        rgb = env.render()
        status = f"{phase}  hold={holding}  d_obj={d_obj_xz:.2f} d_tgt={d_tgt_xz:.2f}"
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 16], fill=(0, 0, 0)); dr.text((3, 3), status, fill=(255, 255, 255))
        left = np.asarray(im)
        right = _topdown(env.sim_res, bounds, traj, p, env.drone.yaw, obj, tgt,
                         env.target_radius, "top-down")
        gap = np.full((env.sim_res, 4, 3), 255, np.uint8)
        imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), np.concatenate([left, gap, right], axis=1))
        if phase == "DONE" and done_hold > args.fps:
            break
        if term:
            break

    env.close()
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
           "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-crf", "20", args.out]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    if rr.returncode != 0:
        print("ffmpeg error:\n", rr.stderr[-1000:]); return 1
    print(f"[demo] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
