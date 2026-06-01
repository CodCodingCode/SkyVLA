"""Film a smooth drone fly-through of an HM3D scene -> mp4 (run in 'habitat' env).

Plans a path of collision-free 3D waypoints (true 3D clearance, any height /
floor — not the 2.5D navmesh), fits a smooth Catmull-Rom spline through them,
and flies the camera along it rendering every frame. The camera yaws to look in
its direction of travel (with a little easing) so it reads like real drone
footage. Frames -> ffmpeg -> H.264 mp4.

Usage:
  python -m indoor_uav.scripts.film_flight --scene <X.basis.glb> --out /tmp/drone.mp4 \
      --waypoints 8 --seconds 20 --fps 30 --res 480
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np


def _build_sim(scene, res, hfov):
    import habitat_sim
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = scene; bk.enable_physics = False
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [res, res]; rgb.hfov = hfov
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb]
    return habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))


def clearance_3d(sim, p, radius):
    """True if a drone-radius sphere at p clears geometry in all directions (3D)."""
    import habitat_sim
    for v in np.linspace(-1, 1, 5):
        h = np.sqrt(max(0.0, 1 - v * v))
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            d = np.array([h * np.cos(a), v, h * np.sin(a)], np.float32)
            hit = sim.cast_ray(habitat_sim.geo.Ray(np.array(p, np.float32), d),
                               max_distance=radius * 1.6)
            if hit.has_hits and len(hit.hits) > 0 and hit.hits[0].ray_distance < radius:
                return False
    return True


def segment_clear(sim, a, b, radius):
    """True if the straight path a->b clears geometry.

    Two free waypoints can still have a wall BETWEEN them, so we (1) raycast
    a->b and require the first hit to be beyond the segment, and (2) sample
    clearance spheres densely along it. This is the edge check the previous
    version was missing (it only validated the endpoints).
    """
    import habitat_sim
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    seg = b - a; L = float(np.linalg.norm(seg))
    if L < 1e-4:
        return True
    d = seg / L
    hit = sim.cast_ray(habitat_sim.geo.Ray(a, d), max_distance=L)
    if hit.has_hits and len(hit.hits) > 0 and hit.hits[0].ray_distance < L:
        return False  # a wall sits directly on the line of travel
    n = max(2, int(L / (radius * 0.75)))  # dense sphere samples along the edge
    for t in np.linspace(0.0, 1.0, n):
        if not clearance_3d(sim, a + seg * t, radius):
            return False
    return True


def catmull_rom(pts, samples_per_seg):
    """Smooth spline through control points (clamped ends)."""
    P = np.asarray(pts, np.float32)
    P = np.vstack([P[0], P, P[-1]])  # phantom endpoints
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2*p0 - 5*p1 + 4*p2 - p3) * t2
                              + (-p0 + 3*p1 - 3*p2 + p3) * t3))
    out.append(P[-2])
    return np.asarray(out, np.float32)


def look_at_habitat(eye, fwd):
    """Return (position, quaternion) for habitat agent looking along `fwd` from `eye`."""
    import quaternion
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    # OpenCV cam basis (z fwd, y down), then convert to habitat (y up, z back)
    up = np.array([0, -1, 0], np.float32)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-8)
    up = np.cross(fwd, right)
    Rcv = np.stack([right, up, fwd], axis=1)
    Rhab = Rcv @ np.diag([1, -1, -1]).astype(np.float32)
    pos_hab = np.array([eye[0], eye[1], eye[2]], np.float32) * np.array([1, 1, 1], np.float32)
    # eye is already in habitat world coords (we plan in habitat coords below)
    return pos_hab, quaternion.from_rotation_matrix(Rhab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/tmp/drone_flight.mp4")
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--waypoints", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--drone_radius", type=float, default=0.18, help="clearance margin (m)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim  # noqa: F401
    import imageio.v2 as imageio

    sim = _build_sim(args.scene, args.res, args.hfov)
    lo, hi = sim.pathfinder.get_bounds(); lo = np.array(lo); hi = np.array(hi)
    ext = hi - lo
    rng = np.random.default_rng(args.seed)

    # ---- plan collision-free 3D waypoints, biased to interior heights ----
    wps = []
    tries = 0
    # start from a navigable floor point lifted up (a believable take-off spot)
    start = np.array(sim.pathfinder.get_random_navigable_point(), np.float32)
    start[1] += 1.0
    if clearance_3d(sim, start, args.drone_radius):
        wps.append(start)
    while len(wps) < args.waypoints and tries < 20000:
        tries += 1
        p = lo + rng.random(3).astype(np.float32) * ext
        p[1] = lo[1] + rng.uniform(0.15, 0.85) * ext[1]  # interior height band
        if not clearance_3d(sim, p, args.drone_radius):
            continue
        # the connecting SEGMENT to the last waypoint must be clear, and not a
        # teleport jump — this is what stops the path cutting through a wall.
        if wps:
            if np.linalg.norm(p - wps[-1]) > 0.4 * np.linalg.norm(ext):
                continue
            if not segment_clear(sim, wps[-1], p, args.drone_radius):
                continue
        wps.append(p)
    if len(wps) < 3:
        print(f"only found {len(wps)} connectable free waypoints; scene may be tight.")
        sim.close(); return 1
    print(f"planned {len(wps)} waypoints with collision-free connecting segments")

    # ---- smooth spline ----
    n_frames = int(args.seconds * args.fps)
    sps = max(2, n_frames // (len(wps)))
    path = catmull_rom(wps, sps)[:n_frames]
    if len(path) < n_frames:
        path = np.vstack([path] + [path[-1]] * (n_frames - len(path)))

    # ---- VALIDATE the actual spline: the curve can still bow into geometry
    #      between control points. Clamp any unsafe frame to the last safe one
    #      so the rendered camera never sits inside a wall. ----
    safe = path.copy()
    bad = 0
    for i in range(len(safe)):
        if not clearance_3d(sim, safe[i], args.drone_radius * 0.9):
            safe[i] = safe[i - 1] if i > 0 else safe[i]
            bad += 1
    path = safe
    print(f"spline validated: {bad}/{len(path)} frames clamped (were inside geometry)")

    tmpdir = tempfile.mkdtemp(prefix="flight_")
    print(f"rendering {len(path)} frames @ {args.res}px ...")
    for i in range(len(path)):
        eye = path[i]
        nxt = path[min(i + 3, len(path) - 1)]
        fwd = nxt - eye
        if np.linalg.norm(fwd) < 1e-3:
            fwd = np.array([0, 0, 1], np.float32)
        pos, quat = look_at_habitat(eye, fwd)
        st = habitat_sim.AgentState(); st.position = pos; st.rotation = quat
        sim.get_agent(0).set_state(st)
        frame = sim.get_sensor_observations()["rgb"][..., :3]
        imageio.imwrite(os.path.join(tmpdir, f"f{i:05d}.png"), frame)
        if i % 60 == 0:
            print(f"  frame {i}/{len(path)}", flush=True)
    sim.close()

    # ---- encode with ffmpeg ----
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(tmpdir, "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg error:\n", r.stderr[-1500:]); return 1
    sz = os.path.getsize(args.out) / 1e6
    print(f"\nDONE -> {args.out}  ({sz:.1f} MB, {len(path)} frames, {args.seconds}s @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
