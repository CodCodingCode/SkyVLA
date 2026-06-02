"""Demo: VLM bounding box -> 3D goal -> the drone flies to the object.

Closed loop each step:
  1. render the drone's FPV (RGB + metric depth)
  2. DETECT: a VLM (or keyless oracle) returns a pixel box for the text query
  3. LIFT: back-project the box centre via depth+pose -> a 3D world goal
     (or, if the target is beyond depth range, a bearing to servo toward)
  4. FLY: steer toward the goal; stop at a standoff distance

Renders a side-by-side mp4:  LEFT = FPV with the box drawn + status;
RIGHT = top-down map (drone, heading, trajectory, target, goal).

Run (habitat env):
  conda activate habitat; export CUDA_HOME=/usr; export PATH=$CONDA_PREFIX/bin:$PATH
  # keyless geometry demo (no API key):
  python -m indoor_uav.scripts.demo_fly_to_object --scene <glb> --detector synthetic
  # real VLM (after you put a FRESH key in /home/ubuntu/SkyVLA/.openai_key):
  python -m indoor_uav.scripts.demo_fly_to_object --scene <glb> --detector openai \
        --query "the wooden chair"
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile

import numpy as np


def _build_sim(scene, res, hfov):
    import habitat_sim
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = scene; bk.enable_physics = True
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [res, res]; rgb.hfov = hfov
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid = "depth"; dep.sensor_type = habitat_sim.SensorType.DEPTH
    dep.resolution = [res, res]; dep.hfov = hfov
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb, dep]
    return habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))


def _pose_c2w(pos, yaw, device, torch):
    """OpenCV camera-to-world for the drone (corrected flip diag(1,-1,-1), det=+1)."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device).float()
    flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device).float()
    T = torch.eye(4, device=device); T[:3, :3] = R @ flip
    T[:3, 3] = torch.tensor(pos, device=device).float()
    return T


def _draw_fpv(rgb, box, status, query, color=(0, 255, 0)):
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb.astype(np.uint8)); dr = ImageDraw.Draw(im)
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        for w in range(3):
            dr.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        dr.line([cx - 6, cy, cx + 6, cy], fill=color); dr.line([cx, cy - 6, cx, cy + 6], fill=color)
    dr.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    dr.text((3, 3), f'"{query}"  |  {status}', fill=(255, 255, 255))
    return np.asarray(im)


def _draw_topdown(res, bounds, traj, drone_p, drone_yaw, target, goal):
    from PIL import Image, ImageDraw
    xmin, xmax, zmin, zmax = bounds
    im = Image.new("RGB", (res, res), (18, 18, 22)); dr = ImageDraw.Draw(im)

    def to_px(p):
        u = (p[0] - xmin) / (xmax - xmin + 1e-6) * (res - 20) + 10
        v = (p[2] - zmin) / (zmax - zmin + 1e-6) * (res - 20) + 10
        return (u, v)
    if len(traj) > 1:
        dr.line([to_px(p) for p in traj], fill=(90, 110, 160), width=2)
    if target is not None:                                  # target = red star-ish
        tx, ty = to_px(target)
        dr.ellipse([tx - 7, ty - 7, tx + 7, ty + 7], outline=(255, 70, 70), width=3)
        dr.line([tx - 10, ty, tx + 10, ty], fill=(255, 70, 70))
        dr.line([tx, ty - 10, tx, ty + 10], fill=(255, 70, 70))
    if goal is not None:                                    # goal = green ring
        gx, gy = to_px(goal)
        dr.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], outline=(80, 230, 120), width=2)
    dx, dy = to_px(drone_p)                                 # drone = blue + heading
    hx = dx + 16 * math.sin(drone_yaw); hy = dy + 16 * math.cos(drone_yaw)
    dr.line([dx, dy, hx, hy], fill=(120, 180, 255), width=3)
    dr.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill=(120, 180, 255))
    dr.text((6, 6), "top-down", fill=(180, 180, 180))
    return np.asarray(im)


def _open_indoor_start(sim, pf, np_, habitat_sim, seed):
    """Reuse the seeded indoor + open-space logic from the GS film script."""
    np_.random.seed(seed); pf.seed(seed); sim.seed(seed)

    def indoor(p, max_h=10.0):
        h = sim.cast_ray(habitat_sim.geo.Ray(np_.asarray(p, np_.float32),
                                             np_.array([0, 1, 0], np_.float32)), max_distance=max_h)
        return h.has_hits and len(h.hits) > 0

    pts = np_.array([pf.get_random_navigable_point() for _ in range(1200)], np_.float32)
    ipts = pts[np_.array([indoor(p) for p in pts], bool)]
    ys = ipts[:, 1]; hh, e = np_.histogram(ys, 40); lo = e[int(hh.argmax())]
    floor_y = float(np_.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
    fp = ipts[np_.abs(ipts[:, 1] - floor_y) < 0.5]

    def clearance(p, n=8, max_r=8.0):
        best = max_r
        for k in range(n):
            a = 2 * math.pi * k / n
            d = np_.array([math.cos(a), 0, math.sin(a)], np_.float32)
            h = sim.cast_ray(habitat_sim.geo.Ray(np_.asarray(p, np_.float32), d), max_distance=max_r)
            if h.has_hits and len(h.hits) > 0:
                best = min(best, float(h.hits[0].ray_distance))
        return best
    clr = np_.array([clearance(p) for p in fp], np_.float32)
    fp = fp[np_.argsort(-clr)]
    return fp, floor_y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/home/ubuntu/SkyVLA/videos/fly_to_object.mp4")
    ap.add_argument("--detector", choices=["synthetic", "openai"], default="synthetic")
    ap.add_argument("--query", default="the target object")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--max_steps", type=int, default=140)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_speed", type=float, default=1.5)
    ap.add_argument("--altitude", type=float, default=1.8, help="m above floor (clear clutter)")
    ap.add_argument("--standoff", type=float, default=1.5)
    ap.add_argument("--detect_every", type=int, default=1, help="VLM call cadence (steps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim
    import imageio.v2 as imageio
    import torch
    import importlib.util as ilu
    here = os.path.dirname(__file__)
    sp = ilu.spec_from_file_location("db", os.path.join(here, "../sim/drone_body.py"))
    db = ilu.module_from_spec(sp); sp.loader.exec_module(db)
    from indoor_uav.perception import bbox_to_world, OpenAIDetector, SyntheticDetector

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = _build_sim(args.scene, args.res, args.hfov)
    pf = sim.pathfinder
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.25; ns.agent_height = 0.3; ns.agent_max_climb = 2.0; ns.cell_size = 0.07
    sim.recompute_navmesh(pf, ns)

    f = 0.5 * args.res / np.tan(np.deg2rad(args.hfov) / 2)
    K = torch.tensor([[f, 0, args.res / 2], [0, f, args.res / 2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)
    K_np = K.cpu().numpy()

    fp, floor_y = _open_indoor_start(sim, pf, np, habitat_sim, args.seed)
    alt = floor_y + args.altitude                  # fly above the furniture clutter
    start = fp[0].copy(); start[1] = alt
    # the "object": the most-open point (fp is clearance-sorted) that is a real but
    # reachable distance away, so the drone has a clear lane down the hall.
    d_xz = np.linalg.norm((fp - start)[:, [0, 2]], axis=1)
    cand = fp[(d_xz > 5.0) & (d_xz < 11.0)]
    far = (cand[0] if len(cand) else fp[int(np.argmax(d_xz))]).copy()
    far[1] = alt
    target = far.astype(np.float32)
    d_far = float(np.linalg.norm((target - start)[[0, 2]]))

    drone = db.DronePhysics(sim, mass=0.5, max_speed=args.max_speed)
    drone.reset(start)
    # start facing AWAY from the target so the "search" phase is visible
    to0 = target - start
    drone._yaw = float(np.arctan2(to0[0], to0[2]) + np.pi)

    if args.detector == "openai":
        det = OpenAIDetector(model=args.model)
        print(f"[demo] OpenAI detector ({args.model}); query={args.query!r}")
    else:
        det = SyntheticDetector(target)
        args.query = args.query if args.query != "the target object" else "the marked object"
        print(f"[demo] SYNTHETIC oracle (keyless); target={target}")
    # ground-truth target is only known in synthetic mode; with a real VLM we
    # don't know where the object is, so don't draw a (misleading) target marker.
    target_marker = None if args.detector == "openai" else target

    tmp = tempfile.mkdtemp(prefix="flydemo_")
    traj = []; goal = None; state = "SEARCH"; dt = 1.0 / args.fps
    last_box = None
    # lock-on: confirm the object across 2 agreeing detections, COMMIT the 3D
    # point, then stop detecting and fly to it. Without this the goal teleports
    # every time the VLM re-boxes a different object and the drone never settles.
    pending = None; pending_n = 0; locked = False
    print(f"[demo] start={start}  target={target}  dist={d_far:.1f}m")

    for i in range(args.max_steps):
        sim.get_agent(0).set_state(drone.camera_state())
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][..., :3].astype(np.uint8); depth = obs["depth"]
        pose = _pose_c2w(drone.position, drone.yaw, dev, torch).cpu().numpy()
        traj.append(np.asarray(drone.position, np.float32))

        # DETECT — only until the goal is locked (then the goal can't teleport).
        if not locked and i % args.detect_every == 0:
            try:
                last_box = det.detect(rgb, args.query, pose, K_np)
            except Exception as ex:
                print(f"  [detect] error: {ex}"); last_box = None
            if last_box is not None:
                res = bbox_to_world(last_box, depth, pose, K_np,
                                    max_depth=10.0, standoff=args.standoff)
                if res["kind"] == "point":
                    g = res["goal"]
                    if pending is not None and np.linalg.norm(g - pending) < 2.0:
                        pending_n += 1                 # consistent -> more confident
                    else:
                        pending_n = 1                  # new candidate
                    pending = g; state = "CONFIRM"
                    if pending_n >= 2:                 # two agreeing views -> commit
                        goal = pending; locked = True; state = "LOCKED"
                        print(f"  LOCKED goal at step {i}: {goal}")
                else:                                  # beyond depth range: bearing servo
                    goal = drone.position + res["dir"] * 4.0; state = "SERVO"

        # CONTROL
        if (locked or state == "SERVO") and goal is not None:
            to = goal - drone.position; dist = float(np.linalg.norm(to))
            if locked and dist < 0.5:
                state = "ARRIVED"
            if state != "ARRIVED":
                des_yaw = float(np.arctan2(to[0], to[2]))
                yaw_err = (des_yaw - drone.yaw + np.pi) % (2 * np.pi) - np.pi
                fwd = args.max_speed if abs(yaw_err) < 0.5 else 0.2
                drone.step(0.0, fwd, float(np.clip(to[1], -1.0, 1.0)),
                           float(np.clip(yaw_err * 3.0, -2.0, 2.0)), dt=dt)
            else:
                drone.step(0.0, 0.0, 0.0, 0.0, dt=dt)
        elif pending is not None:                      # CONFIRM: hold, look again
            drone.step(0.0, 0.0, 0.0, 0.0, dt=dt)
        else:                                          # SEARCH: rotate to find it
            drone.step(0.0, 0.0, 0.0, 1.2, dt=dt)

        status = state + (f"  d={np.linalg.norm(goal - drone.position):.1f}m"
                          if goal is not None else "")
        col = (90, 230, 120) if state == "ARRIVED" else (0, 255, 0)
        left = _draw_fpv(rgb, last_box, status, args.query, color=col)
        # auto-fit the top-down to the trajectory + goal (+ target, if known)
        marks = list(traj) + ([goal] if goal is not None else []) \
            + ([target_marker] if target_marker is not None else [])
        mk = np.asarray(marks, np.float32)
        cx0, cz0 = float(mk[:, 0].min()), float(mk[:, 2].min())
        cx1, cz1 = float(mk[:, 0].max()), float(mk[:, 2].max())
        span = max(cx1 - cx0, cz1 - cz0, 6.0); h = span / 2 + 2.0
        mxc, mzc = 0.5 * (cx0 + cx1), 0.5 * (cz0 + cz1)
        bounds = (mxc - h, mxc + h, mzc - h, mzc + h)
        right = _draw_topdown(args.res, bounds, traj, drone.position, drone.yaw,
                              target_marker, goal)
        gap = np.full((args.res, 4, 3), 255, np.uint8)
        imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"),
                        np.concatenate([left, gap, right], axis=1))
        if state == "ARRIVED":
            print(f"  ARRIVED at step {i}");
            for _ in range(args.fps):               # hold the final frame ~1s
                pass
            break

    sim.close()
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
           "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-crf", "20", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg error:\n", r.stderr[-1000:]); return 1
    print(f"[demo] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
