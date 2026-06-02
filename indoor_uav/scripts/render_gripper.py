"""Third-person stills of the drone + 2-DoF gripper, properly LIT.

The drone's FPV camera can't see its own gripper (it hangs below the forward
camera). This builds a dedicated, *lit* sim (Habitat test scenes ship "no
lights", so inserted Phong objects render black otherwise), drops the drone +
gripper + a graspable cube into an open spot, and shoots a close 3/4 view:

  (1) gripper raised + open   (2) lowered, jaws open over the cube   (3) holding it

  conda activate habitat; export CUDA_HOME=/usr; export PYTHONUTF8=1
  python -m indoor_uav.scripts.render_gripper --scene <glb> --out_dir videos/gripper
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np


def _lookat_quat(eye, target):
    import quaternion
    fwd = np.asarray(target, np.float32) - np.asarray(eye, np.float32)
    fwd /= (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, np.array([0, 1, 0], np.float32))
    right /= (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], axis=1)          # habitat cam looks down -Z
    return quaternion.from_rotation_matrix(R)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", default="/home/ubuntu/SkyVLA/videos/gripper")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--tries", type=int, default=14)
    args = ap.parse_args()

    import habitat_sim
    from habitat_sim.gfx import LightInfo, LightPositionModel, DEFAULT_LIGHTING_KEY
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    from indoor_uav.sim.drone_body import DronePhysics
    from indoor_uav.sim.gripper import DroneGripper

    os.makedirs(args.out_dir, exist_ok=True)

    # --- a LIT sim: force the stage + objects to use our light setup ---
    bk = habitat_sim.SimulatorConfiguration()
    bk.scene_id = args.scene; bk.enable_physics = True
    bk.override_scene_light_defaults = True
    bk.scene_light_setup = DEFAULT_LIGHTING_KEY
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "rgb"; cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [args.res, args.res]; cam.hfov = args.hfov
    cam.position = [0.0, 0.0, 0.0]   # default is [0,1.5,0] (eye height) — zero it so
                                     # the camera sits exactly at the agent position
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [cam]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))
    sim.set_light_setup([
        LightInfo(vector=[1.0, 1.0, 0.7, 0.0], color=[1.5, 1.5, 1.5], model=LightPositionModel.Global),
        LightInfo(vector=[-1.0, 0.8, -0.6, 0.0], color=[1.0, 1.0, 1.1], model=LightPositionModel.Global),
    ], DEFAULT_LIGHTING_KEY)

    pf = sim.pathfinder
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.25; ns.agent_height = 0.3; ns.agent_max_climb = 2.0; ns.cell_size = 0.07
    sim.recompute_navmesh(pf, ns)

    drone = DronePhysics(sim, mass=0.5)
    gripper = DroneGripper(sim, drone, load_urdf=True)
    print(f"[render] gripper URDF loaded: {gripper.has_urdf}")
    # a bigger, brightly-coloured graspable cube for visibility
    otm = sim.get_object_template_manager(); rom = sim.get_rigid_object_manager()
    base = otm.get_template_handles("cube")[0]
    t = otm.get_template_by_handle(base); t.scale = np.array([0.12, 0.12, 0.12], np.float32)
    otm.register_template(t, "show_cube")
    cube = rom.add_object_by_template_handle("show_cube")
    cube.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    # inserted objects default to a no-light key -> render black. Bind them to our
    # lit DEFAULT_LIGHTING_KEY so the drone, gripper, and cube are actually visible.
    for o in (drone.obj, cube):
        try:
            o.set_light_setup(DEFAULT_LIGHTING_KEY)
        except Exception as exc:
            print(f"[render] set_light_setup(rigid) failed: {exc!r}")
    if gripper._art is not None:
        try:
            gripper._art.set_light_setup(DEFAULT_LIGHTING_KEY)
        except Exception as exc:
            print(f"[render] set_light_setup(art) failed: {exc!r}")

    def clr(p, ang, max_r=6.0):
        d = np.array([math.cos(ang), 0.0, math.sin(ang)], np.float32)
        h = sim.cast_ray(habitat_sim.geo.Ray(np.asarray(p, np.float32), d), max_distance=max_r)
        return float(h.hits[0].ray_distance) if (h.has_hits and len(h.hits) > 0) else max_r

    # dominant floor + an OPEN spawn (clearance both sides -> camera sits open,
    # backdrop is the far room, not a near wall)
    ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(300)], np.float32)
    hh, e = np.histogram(ys, 40); lo = e[int(hh.argmax())]
    floor = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
    angs = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    best = None
    for _ in range(args.tries):
        p = np.array(pf.get_random_navigable_point(), np.float32)
        if abs(float(p[1]) - floor) > 0.5:
            continue
        p[1] = floor + 1.3
        ca = max(angs, key=lambda a: min(clr(p, a), clr(p, a + math.pi)))
        score = min(clr(p, ca), clr(p, ca + math.pi))
        if best is None or score > best[0]:
            best = (score, p.copy(), float(ca))
    _, drone_p, cam_ang = best
    print(f"[render] open spawn drone={drone_p} floor={floor:.2f} corridor={best[0]:.1f}m")

    r = 1.5
    eye = drone_p + np.array([r * math.cos(cam_ang), 0.55, r * math.sin(cam_ang)], np.float32)
    tgt = drone_p + np.array([0.0, -0.28, 0.0], np.float32)   # frame the whole drone + hanging gripper

    import quaternion
    f = 0.5 * args.res / math.tan(math.radians(args.hfov) / 2); cc = args.res / 2
    Rcw = quaternion.as_rotation_matrix(_lookat_quat(eye, tgt))

    def project(P):
        pc = Rcw.T @ (np.asarray(P, np.float32) - eye)
        if pc[2] >= -1e-3:                       # behind camera (looks down -Z)
            return None
        return (cc + f * (pc[0] / -pc[2]), cc - f * (pc[1] / -pc[2]))

    def shoot():
        st = habitat_sim.AgentState(); st.position = eye; st.rotation = _lookat_quat(eye, tgt)
        sim.get_agent(0).set_state(st)
        return sim.get_sensor_observations()["rgb"][..., :3].astype(np.uint8)

    def annotate(rgb, title, items):
        im = Image.fromarray(rgb.copy()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 20], fill=(0, 0, 0)); dr.text((5, 5), title, fill=(255, 255, 255))
        for label, P, anchor in items:
            uv = project(P)
            if uv is None:
                continue
            u, v = uv; ax, ay = anchor
            dr.line([ax, ay, u, v], fill=(255, 210, 0), width=2)
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 210, 0), width=2)
            dr.rectangle([ax - 2, ay - 8, ax + 6 * len(label) + 4, ay + 8], fill=(0, 0, 0))
            dr.text((ax + 1, ay - 6), label, fill=(255, 210, 0))
        return np.asarray(im)

    mount = drone_p + np.array([0, -0.05, 0], np.float32)

    def arm_tip(lower):
        tip = drone_p + np.array([0, -(0.05 + lower * gripper.max_drop), 0], np.float32)
        return (mount + tip) / 2, tip

    def cpos():
        return np.array(cube.translation, np.float32)

    shots = []
    # (1) raised + open; cube on the floor beside the drone
    drone.reset(drone_p, 0.0); gripper.lower = 0.0; gripper.grip = 0.0; gripper.held = None
    cube.translation = np.array([drone_p[0] + 0.18, floor + 0.06, drone_p[2]], np.float32)
    gripper._pose_urdf(); _, tp = arm_tip(0.0)
    shots.append(("1_raised_open", shoot(), [
        ("drone body", drone_p, (430, 95)),
        ("gripper RAISED (arm retracted)", tp, (300, 300)),
        ("cube on floor", cpos(), (40, 545)),
    ]))
    # (2) lowered + open, jaws straddling the cube
    gripper.lower = 1.0; gripper.grip = 0.0
    tip = gripper.tip_world(); cube.translation = np.array([tip[0], floor + 0.06, tip[2]], np.float32)
    gripper._pose_urdf(); am, tp = arm_tip(1.0)
    shots.append(("2_lowered_over_cube", shoot(), [
        ("drone body", drone_p, (430, 80)),
        ("lower arm = DoF 1 (raise/lower)", am, (250, 300)),
        ("jaws = DoF 2 (open/close)", tp, (330, 470)),
        ("cube", cpos(), (40, 545)),
    ]))
    # (3) holding the cube (jaws closed)
    gripper.lower = 0.4; gripper.grip = 1.0; gripper.held = cube
    cube.translation = gripper.tip_world(); gripper._pose_urdf(); am, tp = arm_tip(0.4)
    shots.append(("3_holding_cube", shoot(), [
        ("drone body", drone_p, (430, 80)),
        ("lower arm (DoF 1)", am, (430, 300)),
        ("jaws holding cube (DoF 2)", tp, (330, 470)),
    ]))

    titles = {"1_raised_open": "1. gripper RAISED + open",
              "2_lowered_over_cube": "2. arm LOWERED (DoF1), jaws open (DoF2)",
              "3_holding_cube": "3. holding the cube (jaws closed)"}
    out = []
    for name, rgb, items in shots:
        ann = annotate(rgb, titles[name], items)
        path = os.path.join(args.out_dir, f"gripper_{name}.png")
        Image.fromarray(ann).save(path); out.append(ann)
        print(f"[render] {path}  mean={float(rgb.mean()):.1f}")
    imageio.imwrite(os.path.join(args.out_dir, "gripper_contact_sheet.png"), np.concatenate(out, axis=1))
    print(f"[render] {os.path.join(args.out_dir, 'gripper_contact_sheet.png')}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
