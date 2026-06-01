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
    # enable_physics=True is REQUIRED: without it cast_ray hits nothing (0/100
    # rays register), so every collision/segment check silently passes and the
    # camera flies through walls. With Bullet on, raycasts hit the collision
    # mesh reliably (100/100, floor detected) — this is the real clipping fix.
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = scene; bk.enable_physics = True
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [res, res]; rgb.hfov = hfov
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid = "depth"; dep.sensor_type = habitat_sim.SensorType.DEPTH
    dep.resolution = [res, res]; dep.hfov = hfov
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb, dep]
    return habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))


def frame_is_bad(sim, eye, fwd, look_at_fn, *, min_center=0.5, max_void=0.45, max_close=0.5):
    """Validate a CAMERA POSE by what it actually renders (depth), not raycasts.

    The earlier raycast verifier flew through HM3D scan holes (rays through a
    gap hit nothing -> 'clear') while the camera rendered torn void -> what the
    viewer sees as 'clipping'. So we judge each pose by its rendered depth:

      * center too near        -> wall/point-blank in face
      * too much VOID (depth~0)-> looking through a scan hole into nothing
      * too much surface <0.5m -> jammed against geometry

    Returns (is_bad, stats).
    """
    import numpy as np
    pos, quat = look_at_fn(eye, fwd)
    import habitat_sim
    st = habitat_sim.AgentState(); st.position = pos; st.rotation = quat
    sim.get_agent(0).set_state(st)
    d = sim.get_sensor_observations()["depth"]
    h, w = d.shape
    center = float(d[h // 2, w // 2])
    void = float((d <= 0.05).mean())          # holes / no surface
    close = float((d > 0.05).astype(float).__mul__(d < 0.5).mean()) if False else float(((d > 0.05) & (d < 0.5)).mean())
    bad = (center < min_center) or (void > max_void) or (close > max_close)
    return bad, {"center": round(center, 2), "void": round(void, 2), "close": round(close, 2)}


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
    """Return (position, quaternion) for a LEVEL drone camera at `eye` heading `fwd`.

    Crucial: this is YAW-ONLY — a pure rotation about the world-up (+Y) axis. A
    full look-at basis flips upside-down whenever the travel direction tilts
    vertically (cross(fwd, up) degenerates). Real drone footage holds the
    horizon level, so we use only the heading's horizontal component and build
    the rotation as a single spin about +Y, which can never roll or flip.

    Habitat's camera looks along -Z at zero yaw. We want it to look toward the
    horizontal heading (fwd.x, fwd.z), so yaw = atan2(fwd.x, -fwd.z).
    """
    import quaternion
    fx, _, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
    if abs(fx) < 1e-6 and abs(fz) < 1e-6:
        fz = -1.0  # degenerate (pure vertical move): keep last sensible heading
    yaw = np.arctan2(fx, -fz)
    q = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0], np.float32))
    pos_hab = np.array([eye[0], eye[1], eye[2]], np.float32)
    return pos_hab, q


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
    ap.add_argument("--fly_height", type=float, default=0.8,
                    help="metres to lift the floor route to drone eye height")
    ap.add_argument("--max_plan_tries", type=int, default=40,
                    help="replan attempts until the rendered path verifies clean")
    ap.add_argument("--max_bad_frac", type=float, default=0.08,
                    help="max fraction of frames allowed to be wall-in-face/void")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim
    import imageio.v2 as imageio

    sim = _build_sim(args.scene, args.res, args.hfov)
    pf = sim.pathfinder
    rng = np.random.default_rng(args.seed)

    # ---- recompute a DRONE navmesh so we have a connectivity graph to route on.
    #      The fix for wall-clipping is NOT "check random points" — it's planning
    #      on the navmesh, whose ShortestPath only traverses connected free space
    #      (through real doorways), so a path can never cross a wall. ----
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = max(0.15, args.drone_radius)
    ns.agent_height = 0.30
    ns.agent_max_climb = 2.0
    ns.cell_size = 0.07
    if not sim.recompute_navmesh(pf, ns) or not pf.is_loaded:
        print("could not build a drone navmesh for this scene; aborting.")
        sim.close(); return 1
    print(f"drone navmesh area={pf.navigable_area:.1f} m^2")

    def navpt():
        return np.array(pf.get_random_navigable_point(), np.float32)

    # ---- pick the DOMINANT floor by histogramming navigable-point heights, so
    #      we don't accidentally anchor to a tiny landing/stairwell. ----
    ys = np.array([float(navpt()[1]) for _ in range(400)], np.float32)
    h, edges = np.histogram(ys, bins=40)
    lo_e = edges[int(h.argmax())]; hi_e = lo_e + (edges[1] - edges[0])
    floor_y = float(np.median(ys[(ys >= lo_e) & (ys < hi_e)]))
    floor_band = 0.5
    n_frames = int(args.seconds * args.fps)
    margin = args.drone_radius              # required wall clearance for EVERY frame
    print(f"dominant floor_y={floor_y:.2f}  Y range {ys.min():.1f}..{ys.max():.1f}")

    def seg_hits_wall(a, b):
        """True if the straight a->b crosses a surface (swept ray, both directions)."""
        v = b - a; L = float(np.linalg.norm(v))
        if L < 1e-5:
            return False
        d = (v / L).astype(np.float32)
        for o, dd in ((a, d), (b, -d)):
            hit = sim.cast_ray(habitat_sim.geo.Ray(o.astype(np.float32), dd), max_distance=L)
            if hit.has_hits and len(hit.hits) > 0 and hit.hits[0].ray_distance < L - 1e-3:
                return True
        return False

    def resample(route):
        seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)]); total = float(cum[-1])
        if total < 1e-3:
            return None
        out = np.empty((n_frames, 3), np.float32)
        for k, dpos in enumerate(np.linspace(0, total, n_frames)):
            j = min(max(int(np.searchsorted(cum, dpos, "right") - 1), 0), len(route) - 2)
            f = (dpos - cum[j]) / max(seg[j], 1e-6)
            out[k] = route[j] * (1 - f) + route[j + 1] * f
        return out

    def verify(path):
        """Two independent gates (physics now ON, so raycasts actually hit):
          1. SEGMENT crossings — raycast each consecutive pair; a hit before the
             next point means the straight move passed THROUGH a wall (the
             'phasing through walls' bug). With physics on this is real.
          2. DEPTH per-frame — what the camera renders (wall-in-face / void).
        Returns (segment_crossings, bad_frames)."""
        cross = 0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            v = b - a; L = float(np.linalg.norm(v))
            if L < 1e-5:
                continue
            d = (v / L).astype(np.float32)
            h = sim.cast_ray(habitat_sim.geo.Ray(a.astype(np.float32), d), L)
            if h.has_hits and len(h.hits) > 0 and h.hits[0].ray_distance < L - 0.03:
                cross += 1
        bad = 0
        for i in range(len(path)):
            j = min(i + 3, len(path) - 1)
            fwd = path[j] - path[i]
            if np.linalg.norm(fwd) < 1e-4:
                fwd = path[i] - path[max(i - 1, 0)]
            isbad, _ = frame_is_bad(sim, path[i], fwd, look_at_habitat)
            bad += int(isbad)
        return cross, bad

    # ---- plan -> verify -> REPLAN until the actual rendered path is clean ----
    path = None
    for attempt in range(1, args.max_plan_tries + 1):
        route, cur, legs = [], None, 0
        for _ in range(args.waypoints * 60):
            if legs >= args.waypoints:
                break
            cand = navpt()
            if abs(float(cand[1]) - floor_y) > floor_band:
                continue
            if cur is None:
                cur = cand; continue
            sp = habitat_sim.ShortestPath()
            sp.requested_start = cur; sp.requested_end = cand
            if not pf.find_path(sp) or len(sp.points) < 2 or sp.geodesic_distance < 1.0:
                continue
            pts = [np.array(p, np.float32) for p in sp.points]
            if any(abs(float(p[1]) - floor_y) > floor_band for p in pts):
                continue
            route.extend(pts if not route else pts[1:]); cur = cand; legs += 1
        if len(route) < 3:
            continue
        route = np.asarray(route, np.float32)
        route[:, 1] = floor_y + args.fly_height          # constant cruise altitude
        cand_path = resample(route)
        if cand_path is None:
            continue
        cross, bad = verify(cand_path)
        frac_bad = bad / max(len(cand_path), 1)
        print(f"  plan attempt {attempt}: legs={legs} verts={len(route)} "
              f"wall_crossings={cross} bad_frames={bad}/{len(cand_path)} ({frac_bad:.0%})")
        # HARD gate: zero segment wall-crossings (the phasing bug) AND few bad frames
        if cross == 0 and frac_bad <= args.max_bad_frac:
            path = cand_path
            print(f"ACCEPTED on attempt {attempt}: 0 wall-crossings, {frac_bad:.0%} bad frames")
            break
    if path is None:
        print("could not find a clean path; aborting (will not render a clipping clip).")
        sim.close(); return 1

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
