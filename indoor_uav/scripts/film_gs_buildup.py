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
    # habitat camera (-z fwd, +y up) -> OpenCV camera (+z fwd, +y down): flip Y and Z.
    # Must match _HAB2CV in sim/habitat_room.py. (Was diag(1,-1,1): an improper
    # reflection, det=-1, which placed the cloud mirrored front-to-back per frame.)
    flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device).float()
    T = torch.eye(4, device=device); T[:3, :3] = R @ flip
    T[:3, 3] = torch.tensor(pos, device=device).float()
    return T


def _overhead_pose_c2w(center, height, device, torch):
    """FIXED top-down camera `height` ABOVE center, looking straight down.
    World is +y UP (see _HAB2CV), so 'above' = center.y + height. Returns an
    OpenCV camera-to-world with basis columns [right, down, fwd] (det = +1)."""
    import numpy as _np
    eye = _np.asarray(center, _np.float32) + _np.array([0, height, 0], _np.float32)
    # right=+x, down=+z (image-down points world +z), fwd=-y (look straight down)
    Rm = _np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], _np.float32)
    T = torch.eye(4, device=device)
    T[:3, :3] = torch.tensor(Rm, device=device).float()
    T[:3, 3] = torch.tensor(eye, device=device).float()
    return T


def _orbit_pose_c2w(center, radius, theta, elev, device, torch):
    """External camera on an orbit around `center`, LOOKING AT it.
    World is +y UP. OpenCV basis [right, down, fwd] with right x down = fwd, so
    the rotation is proper (det = +1) and the view is upright."""
    import numpy as _np
    c = _np.asarray(center, _np.float32)
    eye = c + _np.array([
        radius * _np.cos(elev) * _np.sin(theta),
        radius * _np.sin(elev),                      # +y is UP -> +sin(elev) = above
        radius * _np.cos(elev) * _np.cos(theta),
    ], _np.float32)
    fwd = c - eye; fwd /= (_np.linalg.norm(fwd) + 1e-8)
    world_up = _np.array([0, 1, 0], _np.float32)
    right = _np.cross(fwd, world_up); right /= (_np.linalg.norm(right) + 1e-8)
    down = _np.cross(fwd, right)                      # right x down = fwd (OpenCV)
    Rm = _np.stack([right, down, fwd], axis=1)
    T = torch.eye(4, device=device)
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

    # --- deterministic, INDOOR flight -------------------------------------
    # `--seed` was previously ignored: get_random_navigable_point()'s RNG was
    # never seeded, so every run flew a different random path, and goals were
    # drawn from ANYWHERE on the navmesh (incl. the courtyard/rooftop) — which is
    # how the drone ended up outside the building. Seed the RNG and keep goals to
    # INDOOR points (geometry overhead) on one floor, near the start.
    np.random.seed(args.seed)
    pf.seed(args.seed)
    sim.seed(args.seed)

    def _indoor(p, max_h: float = 10.0) -> bool:
        """True if there is a ceiling (geometry) above p within max_h metres.
        Outdoor courtyard / rooftop points see open sky -> rejected."""
        ray = habitat_sim.geo.Ray(np.asarray(p, np.float32),
                                  np.array([0, 1, 0], np.float32))  # +y is up
        hit = sim.cast_ray(ray, max_distance=max_h)
        return hit.has_hits and len(hit.hits) > 0

    # sample the navmesh once, keep only indoor points, pick the dominant INDOOR
    # floor (so we never lock onto an outdoor floor as "dominant").
    pts = np.array([pf.get_random_navigable_point() for _ in range(1200)], np.float32)
    ipts = pts[np.array([_indoor(p) for p in pts], bool)]
    if len(ipts) < 10:
        raise RuntimeError(f"too few indoor navigable points ({len(ipts)}); "
                           "check the scene / navmesh")
    ys = ipts[:, 1]
    hh, e = np.histogram(ys, 40); lo = e[int(hh.argmax())]
    floor_y = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
    floor_pts = ipts[np.abs(ipts[:, 1] - floor_y) < 0.5]

    # rank floor points by horizontal wall-clearance, so the drone flies through
    # OPEN space (the middle of the hall) and the camera sees the room at a
    # distance — instead of being pressed against a wall/ceiling, which makes both
    # the FPV and the splat just one flat slab.
    def _clearance(p, n: int = 8, max_r: float = 8.0) -> float:
        best = max_r
        for k in range(n):
            a = 2 * math.pi * k / n
            dvec = np.array([math.cos(a), 0.0, math.sin(a)], np.float32)
            hit = sim.cast_ray(habitat_sim.geo.Ray(np.asarray(p, np.float32), dvec),
                               max_distance=max_r)
            if hit.has_hits and len(hit.hits) > 0:
                best = min(best, float(hit.hits[0].ray_distance))
        return best

    clr = np.array([_clearance(p) for p in floor_pts], np.float32)
    order = np.argsort(-clr); floor_pts = floor_pts[order]; clr = clr[order]
    start = floor_pts[0].copy()                       # most open point = hall centre
    # goals: other reasonably-open points within a bounded radius of the centre.
    d_xz = np.linalg.norm((floor_pts - start)[:, [0, 2]], axis=1)
    cand = floor_pts[(d_xz < 12.0) & (clr > max(1.0, 0.5 * float(clr[0])))]
    np.random.shuffle(cand)
    goals = np.concatenate([[start], cand])[: max(2, args.waypoints)].astype(np.float32)
    goals[:, 1] = floor_y + 1.0
    print(f"flight: {len(goals)} open indoor goals on floor_y={floor_y:.2f} "
          f"(centre clearance={float(clr[0]):.1f}m, {len(ipts)}/{len(pts)} indoor pts)")

    drone = DronePhysics(sim, mass=0.5, max_speed=args.max_speed)
    drone.reset(goals[0])
    gmap = GaussianMap(device=dev)

    # external-orbit camera params: center on the floor, radius from scene extent
    blo, bhi = (np.array(x, np.float32) for x in pf.get_bounds())
    # center on the drone's working area (its goals), not the whole multi-floor bbox,
    # so the orbit frames the part actually being reconstructed.
    center = goals.mean(axis=0).astype(np.float32)
    spread = float(np.linalg.norm(goals.max(0) - goals.min(0)))
    # external orbit: radius pulled back so the working area fits with margin.
    orbit_r = max(8.0, 2.2 * spread)
    orbit_elev = np.deg2rad(22.0)

    # convention sanity check: the external extrinsic must be a proper rotation.
    _chk = _orbit_pose_c2w(center, orbit_r, 0.0, orbit_elev, dev, torch)
    _det = float(torch.linalg.det(_chk[:3, :3]).item())
    assert abs(_det - 1.0) < 1e-3, f"orbit R is not a proper rotation (det={_det:.3f})"
    print(f"orbit cam det(R)={_det:.4f} (want +1)  center={center}  radius={orbit_r:.1f}")

    n_frames = int(args.seconds * args.fps); dt = 1.0 / args.fps
    tmp = tempfile.mkdtemp(prefix="gsbuild_")
    traj = []
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
        traj.append(np.asarray(drone.position, np.float32))
        rgb_t = torch.from_numpy(np.ascontiguousarray(real)).to(dev).float() / 255.0
        dep_t = torch.from_numpy(np.ascontiguousarray(depth)).to(dev).float()
        gmap.add_from_rgbd(rgb_t, dep_t, pose, K, stride=args.gs_stride, max_depth=10.0)

        # render the WHOLE reconstruction from an EXTERNAL orbiting camera (not the
        # drone's nose-cam) so you watch the full 3D splat fill in. Aim at the LIVE
        # cloud centroid and size the radius to the cloud extent so the growing
        # reconstruction stays framed (the goal centroid is a poor proxy — it sits
        # ~6m above the captured geometry).
        if gmap.num_gaussians > 1000:
            m = gmap.means
            ctr = m.mean(0).detach().cpu().numpy()
            ext = float((m.max(0).values - m.min(0).values).max().item())
            r = float(min(max(1.4 * ext, 8.0), 45.0))
        else:
            ctr, r = center, orbit_r
        theta = 2 * np.pi * (i / max(1, n_frames - 1))
        cam = _orbit_pose_c2w(ctr, r, theta, orbit_elev, device=dev, torch=torch)
        splat_rgb, _ = gmap.render(cam, K, args.res, args.res)
        splat = (splat_rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        gap = np.zeros((args.res, 4, 3), np.uint8); gap[:] = 255
        frame = np.concatenate([real, gap, splat], axis=1)  # LEFT drone cam, RIGHT external splat
        imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), frame)
        if i % 40 == 0:
            print(f"  frame {i}/{n_frames}  gaussians={gmap.num_gaussians}", flush=True)
    sim.close()
    print(f"final gaussians={gmap.num_gaussians}")
    # diagnostic: cloud AABB vs flown trajectory — they should overlap. A cloud
    # far from the trajectory (or a degenerate/tiny AABB) means the splat poses
    # are still wrong.
    if gmap.num_gaussians > 0:
        mn = gmap.means.min(0).values.cpu().numpy()
        mx = gmap.means.max(0).values.cpu().numpy()
        traj = np.asarray(traj, np.float32)
        print(f"cloud  AABB min={mn}  max={mx}  size={mx - mn}")
        print(f"trajectory bbox min={traj.min(0)}  max={traj.max(0)}")

    # -profile:v main + -movflags +faststart -> plays in macOS QuickTime/Preview
    # (default High profile with the moov atom at EOF can fail to open there).
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
           "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-crf", "20", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg error:\n", r.stderr[-1200:]); return 1
    print(f"DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)  LEFT=camera  RIGHT=GS reconstruction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
