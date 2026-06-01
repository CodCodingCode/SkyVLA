"""Fly a REAL physics drone (Bullet rigid body) and film its onboard camera.

This is NOT a camera on rails: a DronePhysics rigid body with mass + gravity +
thrust + drag flies through the scene under velocity commands, physically
colliding with walls (Bullet resolves contacts — it cannot pass through). The
RGB camera rides on the body. Demonstrates an actual drone-with-physics in-sim.

Control: a simple autopilot issues velocity commands toward a sequence of
navmesh waypoints (so it goes somewhere sensible); physics does the rest, incl.
momentum and collisions. Frames -> ffmpeg -> mp4.

Usage (habitat env):
  python -m indoor_uav.scripts.film_physics_drone --scene <X.basis.glb> --out /tmp/uav.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np


def _build_sim(scene, res, hfov):
    import habitat_sim
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = scene
    bk.enable_physics = True   # REQUIRED for rigid-body dynamics + collisions
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [res, res]; rgb.hfov = hfov
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb]
    return habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/tmp/uav_physics.mp4")
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--waypoints", type=int, default=6)
    ap.add_argument("--mass", type=float, default=0.5)
    ap.add_argument("--max_speed", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim
    import imageio.v2 as imageio
    # load drone_body directly (avoid indoor_uav.sim package __init__, which pulls
    # in torch via base/synthetic_room — not available in the habitat env).
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "drone_body",
        os.path.join(os.path.dirname(__file__), "..", "sim", "drone_body.py"))
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    DronePhysics = _mod.DronePhysics

    sim = _build_sim(args.scene, args.res, args.hfov)
    pf = sim.pathfinder
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.25; ns.agent_height = 0.3; ns.agent_max_climb = 2.0; ns.cell_size = 0.07
    sim.recompute_navmesh(pf, ns)
    rng = np.random.default_rng(args.seed)

    # dominant floor + waypoints on it (goals for the autopilot)
    ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(300)], np.float32)
    h, e = np.histogram(ys, 40); lo = e[int(h.argmax())]
    floor_y = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
    goals = []
    while len(goals) < args.waypoints and len(goals) < 200:
        p = np.array(pf.get_random_navigable_point(), np.float32)
        if abs(float(p[1]) - floor_y) < 0.5:
            p[1] = floor_y + 1.0
            goals.append(p)
    goals = np.array(goals, np.float32)

    drone = DronePhysics(sim, mass=args.mass, max_speed=args.max_speed)
    drone.reset(goals[0], yaw=0.0)

    n_frames = int(args.seconds * args.fps)
    dt = 1.0 / args.fps
    tmpdir = tempfile.mkdtemp(prefix="uav_")
    print(f"flying physics drone: mass={args.mass}kg max_speed={args.max_speed}m/s, "
          f"{len(goals)} goals, {n_frames} frames")

    gi = 1
    collisions = 0
    for i in range(n_frames):
        target = goals[min(gi, len(goals) - 1)]
        to = target - drone.position
        dist = float(np.linalg.norm(to))
        if dist < 0.5 and gi < len(goals) - 1:
            gi += 1
            target = goals[gi]; to = target - drone.position; dist = float(np.linalg.norm(to))
        # desired heading -> body-frame velocity command (autopilot)
        des_yaw = float(np.arctan2(to[0], to[2])) if dist > 1e-3 else drone.yaw
        yaw_err = (des_yaw - drone.yaw + np.pi) % (2 * np.pi) - np.pi
        fwd_speed = args.max_speed if abs(yaw_err) < 0.6 else 0.3  # slow while turning
        vy = np.clip(to[1], -args.max_speed, args.max_speed)
        tel = drone.step(vx_body=0.0, vz_body=fwd_speed, vy=vy,
                         yaw_rate=np.clip(yaw_err * 3.0, -2.0, 2.0), dt=dt)
        collisions += int(tel["collided"])
        # render from the onboard camera
        sim.get_agent(0).set_state(drone.camera_state(sensor_height=0.0))
        frame = sim.get_sensor_observations()["rgb"][..., :3]
        imageio.imwrite(os.path.join(tmpdir, f"f{i:05d}.png"), frame)
        if i % 60 == 0:
            print(f"  frame {i}/{n_frames} pos=({drone.position[0]:.1f},"
                  f"{drone.position[1]:.1f},{drone.position[2]:.1f}) "
                  f"speed={tel['speed']:.2f} collided={tel['collided']}", flush=True)
    sim.close()
    print(f"flight done: {collisions}/{n_frames} frames had wall contact "
          f"(physics bounced the drone off — it did NOT pass through)")

    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(tmpdir, "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg error:\n", r.stderr[-1200:]); return 1
    print(f"DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
