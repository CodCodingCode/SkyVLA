"""Show the drone-camera scale vs the room, and FLY it to chosen 3D areas.

Two things:

1) SCALE REPORT — prints the drone's physical size next to the scene's real
   dimensions, so "how big is the camera relative to the room" is concrete. A
   real indoor quadrotor is ~0.10-0.15 m; the camera itself is a point (a
   pinhole) — it has no volume, only a collision *radius* we assign for safety.

2) FLY-TO — moves the camera to arbitrary 3D waypoints (full 6-DOF, any height,
   across floors) and renders what it sees from each. Unlike the navmesh path,
   this does NOT confine the drone to a 2.5D floor surface: it flies through the
   true 3D volume. Free-space is checked with a 3D clearance test (sphere of
   `drone_radius` around the target must clear geometry via raycasts), which is
   the right model for a tiny flyer in a multi-floor house.

Run (habitat env), saves RGB frames as a contact sheet:
  python -m indoor_uav.scripts.fly_camera --scene <scene.basis.glb> --out /tmp/fly.png
"""
from __future__ import annotations

import argparse

import numpy as np


def _build_sim(scene, res, hfov):
    import habitat_sim
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = scene; bk.enable_physics = False
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [res, res]; rgb.hfov = hfov
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid = "depth"; dep.sensor_type = habitat_sim.SensorType.DEPTH
    dep.resolution = [res, res]; dep.hfov = hfov
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb, dep]
    return habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))


def clearance_3d(sim, p, radius, n_rays=14):
    """True if a sphere of `radius` at world point p is collision-free in 3D.

    Casts rays in many directions; if the nearest hit in every direction is
    farther than `radius`, the drone fits there. This is a real 3D free-space
    test (works mid-air, between floors), unlike the 2.5D navmesh.
    """
    import habitat_sim
    dirs = []
    for v in (np.linspace(-1, 1, 5)):
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            h = np.sqrt(max(0.0, 1 - v * v))
            dirs.append([h * np.cos(a), v, h * np.sin(a)])
    for d in dirs:
        ray = habitat_sim.geo.Ray(np.array(p, dtype=np.float32),
                                  np.array(d, dtype=np.float32))
        hit = sim.cast_ray(ray, max_distance=radius * 1.5)
        if hit.has_hits and len(hit.hits) > 0 and hit.hits[0].ray_distance < radius:
            return False
    return True


def look_at(eye, target):
    """4x4 camera-to-world (OpenCV: +z fwd, +y down) looking from eye to target."""
    eye = np.asarray(eye, np.float32); target = np.asarray(target, np.float32)
    fwd = target - eye; fwd /= (np.linalg.norm(fwd) + 1e-8)
    up = np.array([0, -1, 0], np.float32)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-8)
    up = np.cross(fwd, right)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right; T[:3, 1] = up; T[:3, 2] = fwd; T[:3, 3] = eye
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/tmp/fly.png")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--drone_radius", type=float, default=0.12,
                    help="real indoor quadrotor half-width (m); ~0.10-0.15")
    ap.add_argument("--waypoints", type=int, default=6)
    args = ap.parse_args()

    import habitat_sim  # noqa: F401
    sim = _build_sim(args.scene, args.res, args.hfov)
    pf = sim.pathfinder
    lo, hi = pf.get_bounds()
    ext = hi - lo

    # ---- 1) SCALE REPORT ----
    print("=" * 64)
    print("DRONE-CAMERA SCALE vs ROOM")
    print(f"  scene size (m):     {ext[0]:.1f} (W) x {ext[2]:.1f} (D) x {ext[1]:.1f} (H)")
    print(f"  drone collision dia: {2*args.drone_radius:.2f} m  (camera optical center = a point)")
    print(f"  drone vs room width: 1 : {ext[0]/(2*args.drone_radius):.0f}")
    print(f"  drone vs room height:1 : {ext[1]/(2*args.drone_radius):.0f}")
    print(f"  -> the drone is ~{ext[0]/(2*args.drone_radius):.0f}x smaller than the room is wide")

    # ---- 2) FLY to 3D waypoints spanning the whole volume (all heights/floors) ----
    rng = np.random.default_rng(0)
    targets, frames = [], []
    tries = 0
    while len(targets) < args.waypoints and tries < 4000:
        tries += 1
        p = lo + rng.random(3).astype(np.float32) * ext  # ANY 3D point, any height
        if clearance_3d(sim, p, args.drone_radius):
            targets.append(p)
    print(f"\nFLEW to {len(targets)} free 3D waypoints (full-volume, multi-floor):")
    for i, p in enumerate(targets):
        # look toward scene center so each shot is informative
        ctr = (lo + hi) / 2
        T = look_at(p, [ctr[0], p[1], ctr[2]])
        st = habitat_sim.AgentState()
        st.position = (T[:3, 3] * np.array([1, -1, -1])).astype(np.float32)  # OpenCV->habitat
        import quaternion
        Rcv = T[:3, :3] @ np.diag([1, -1, -1]).astype(np.float32)  # back to habitat basis
        st.rotation = quaternion.from_rotation_matrix(Rcv)
        sim.get_agent(0).set_state(st)
        obs = sim.get_sensor_observations()
        frames.append(obs["rgb"][..., :3])
        print(f"  wp{i}: pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})  height={p[1]-lo[1]:.1f}m above floor")

    # ---- contact sheet ----
    if frames:
        try:
            from PIL import Image
            cols = min(3, len(frames)); rows = (len(frames) + cols - 1) // cols
            r = frames[0].shape[0]
            sheet = np.zeros((rows * r, cols * r, 3), np.uint8)
            for i, f in enumerate(frames):
                y, x = divmod(i, cols)
                sheet[y*r:(y+1)*r, x*r:(x+1)*r] = f
            Image.fromarray(sheet).save(args.out)
            print(f"\nsaved {len(frames)}-view contact sheet -> {args.out}")
        except Exception as exc:  # noqa: BLE001
            print(f"(could not save image: {exc!r})")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
