"""Film the Gaussian-splat reconstruction BUILDING UP as the drone flies.

Side-by-side mp4:
  LEFT  = the drone's real onboard camera (ground-truth scene)
  RIGHT = the GaussianMap rendered from the SAME pose — the splat the drone has
          reconstructed SO FAR. It starts empty and fills in as the drone sees
          more of the scene. This is the actual reward signal made visible.

Physics drone (Bullet) flies an autopilot tour; each step we splat its RGB-D into
the GaussianMap and render the map from the drone's viewpoint. Requires the
habitat env with torch + gsplat (CUDA backend compiles on first use).

Usage:
  conda activate habitat; export CUDA_HOME=/usr; export PATH=$CONDA_PREFIX/bin:$PATH
  python -m indoor_uav.scripts.film_gs_buildup --scene <glb> --out /tmp/gs.mp4
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
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device).float()
    flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float()
    T = torch.eye(4, device=device); T[:3, :3] = R @ flip
    T[:3, 3] = torch.tensor(pos, device=device).float()
    return T


def _orbit_pose_c2w(center, radius, theta, elev, device, torch):
    """External camera on an orbit, LOOKING AT `center`. Lets us watch the whole
    GS reconstruction from outside (not the drone's nose-cam). OpenCV cam: +z fwd."""
    import numpy as _np
    eye = _np.array([
        center[0] + radius * _np.cos(elev) * _np.sin(theta),
        center[1] - radius * _np.sin(elev),          # above the scene (y-down world)
        center[2] + radius * _np.cos(elev) * _np.cos(theta),
    ], _np.float32)
    fwd = (_np.asarray(center, _np.float32) - eye); fwd /= (_np.linalg.norm(fwd) + 1e-8)
    up = _np.array([0, -1, 0], _np.float32)          # world up is -y here
    right = _np.cross(fwd, up); right /= (_np.linalg.norm(right) + 1e-8)
    up = _np.cross(fwd, right)
    T = torch.eye(4, device=device)
    Rm = _np.stack([right, up, fwd], axis=1)
    T[:3, :3] = torch.tensor(Rm, device=device).float()
    T[:3, 3] = torch.tensor(eye, device=device).float()
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/tmp/gs_buildup.mp4")
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--waypoints", type=int, default=6)
    ap.add_argument("--max_speed", type=float, default=1.5)
    ap.add_argument("--gs_stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim
    import imageio.v2 as imageio
    import torch
    import importlib.util as ilu
    here = os.path.dirname(__file__)

    def load(name, rel):
        sp = ilu.spec_from_file_location(name, os.path.join(here, rel))
        m = ilu.module_from_spec(sp); sp.loader.exec_module(m); return m
    DronePhysics = load("db", "../sim/drone_body.py").DronePhysics
    from indoor_uav.gs import GaussianMap

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = _build_sim(args.scene, args.res, args.hfov)
    pf = sim.pathfinder
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.25; ns.agent_height = 0.3; ns.agent_max_climb = 2.0; ns.cell_size = 0.07
    sim.recompute_navmesh(pf, ns)

    f = 0.5 * args.res / np.tan(np.deg2rad(args.hfov) / 2)
    K = torch.tensor([[f, 0, args.res / 2], [0, f, args.res / 2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)

    # dominant floor + goals
    ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(300)], np.float32)
    h, e = np.histogram(ys, 40); lo = e[int(h.argmax())]
    floor_y = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
    goals = []
    while len(goals) < args.waypoints and len(goals) < 300:
        p = np.array(pf.get_random_navigable_point(), np.float32)
        if abs(float(p[1]) - floor_y) < 0.5:
            p[1] = floor_y + 1.0; goals.append(p)
    goals = np.array(goals, np.float32)

    drone = DronePhysics(sim, mass=0.5, max_speed=args.max_speed)
    drone.reset(goals[0])
    gmap = GaussianMap(device=dev)

    # external-orbit camera params: center on the floor, radius from scene extent
    blo, bhi = (np.array(x, np.float32) for x in pf.get_bounds())
    # center on the drone's working area (its goals), not the whole multi-floor bbox,
    # so the orbit frames the part actually being reconstructed.
    center = goals.mean(axis=0).astype(np.float32)
    spread = float(np.linalg.norm(goals.max(0) - goals.min(0)))
    orbit_r = max(3.0, 0.9 * spread)

    n_frames = int(args.seconds * args.fps); dt = 1.0 / args.fps
    tmp = tempfile.mkdtemp(prefix="gsbuild_")
    print(f"filming GS build-up: {n_frames} frames, splat stride={args.gs_stride}")
    gi = 1
    for i in range(n_frames):
        tgt = goals[min(gi, len(goals) - 1)]
        to = tgt - drone.position; dist = float(np.linalg.norm(to))
        if dist < 0.5 and gi < len(goals) - 1:
            gi += 1; tgt = goals[gi]; to = tgt - drone.position; dist = float(np.linalg.norm(to))
        des_yaw = float(np.arctan2(to[0], to[2])) if dist > 1e-3 else drone.yaw
        yaw_err = (des_yaw - drone.yaw + np.pi) % (2 * np.pi) - np.pi
        fwd = args.max_speed if abs(yaw_err) < 0.6 else 0.3
        drone.step(0.0, fwd, float(np.clip(to[1], -args.max_speed, args.max_speed)),
                   float(np.clip(yaw_err * 3.0, -2.0, 2.0)), dt=dt)

        # real onboard view
        sim.get_agent(0).set_state(drone.camera_state())
        obs = sim.get_sensor_observations()
        real = obs["rgb"][..., :3].astype(np.uint8)
        depth = obs["depth"]

        # splat this RGB-D into the GaussianMap from the drone's pose
        pose = _pose_c2w(drone.position, drone.yaw, dev, torch)
        rgb_t = torch.from_numpy(np.ascontiguousarray(real)).to(dev).float() / 255.0
        dep_t = torch.from_numpy(np.ascontiguousarray(depth)).to(dev).float()
        gmap.add_from_rgbd(rgb_t, dep_t, pose, K, stride=args.gs_stride, max_depth=10.0)

        # render the WHOLE reconstruction from an EXTERNAL orbiting camera (not the
        # drone's nose-cam) so you watch the full 3D splat fill in. The camera
        # slowly rotates around the scene as the map grows.
        theta = 2 * math.pi * (i / max(n_frames - 1, 1))
        orbit = _orbit_pose_c2w(center, orbit_r, theta, elev=0.35, device=dev, torch=torch)
        splat_rgb, _ = gmap.render(orbit, K, args.res, args.res)
        splat = (splat_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        gap = np.zeros((args.res, 4, 3), np.uint8); gap[:] = 255
        frame = np.concatenate([real, gap, splat], axis=1)  # LEFT drone cam, RIGHT external splat
        imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), frame)
        if i % 40 == 0:
            print(f"  frame {i}/{n_frames}  gaussians={gmap.num_gaussians}", flush=True)
    sim.close()
    print(f"final gaussians={gmap.num_gaussians}")

    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg error:\n", r.stderr[-1200:]); return 1
    print(f"DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)  LEFT=camera  RIGHT=GS reconstruction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
