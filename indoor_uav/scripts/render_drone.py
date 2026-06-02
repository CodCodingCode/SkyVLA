"""Render the FULL quadrotor + gripper URDF (not the placeholder cube), labeled.

Loads indoor_uav/assets/gripper/drone_with_gripper.urdf as one articulated
object in a lit Habitat sim and shoots a labeled 3/4 view. (Habitat loads URDF,
not USD — there is no USD runtime/Isaac on this machine.)

  conda activate habitat; export CUDA_HOME=/usr; export PYTHONUTF8=1
  python -m indoor_uav.scripts.render_drone --scene <glb> --out_dir videos/gripper
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

_URDF = os.path.join(os.path.dirname(__file__), "..", "assets", "gripper", "drone_with_gripper.urdf")


def _lookat_quat(eye, target):
    import quaternion
    fwd = np.asarray(target, np.float32) - np.asarray(eye, np.float32)
    fwd /= (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, np.array([0, 1, 0], np.float32)); right /= (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, fwd)
    return quaternion.from_rotation_matrix(np.stack([right, up, -fwd], axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", default="/home/ubuntu/SkyVLA/videos/gripper")
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--tries", type=int, default=14)
    args = ap.parse_args()

    import habitat_sim
    import quaternion
    from habitat_sim.gfx import LightInfo, LightPositionModel, DEFAULT_LIGHTING_KEY
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    os.makedirs(args.out_dir, exist_ok=True)

    bk = habitat_sim.SimulatorConfiguration()
    bk.scene_id = args.scene; bk.enable_physics = True
    bk.override_scene_light_defaults = True; bk.scene_light_setup = DEFAULT_LIGHTING_KEY
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "rgb"; cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [args.res, args.res]; cam.hfov = args.hfov
    cam.position = [0.0, 0.0, 0.0]                      # zero the default 1.5m eye offset
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [cam]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))
    sim.set_light_setup([
        LightInfo(vector=[1.0, 1.0, 0.7, 0.0], color=[1.6, 1.6, 1.6], model=LightPositionModel.Global),
        LightInfo(vector=[-1.0, 0.8, -0.6, 0.0], color=[1.1, 1.1, 1.2], model=LightPositionModel.Global),
    ], DEFAULT_LIGHTING_KEY)

    pf = sim.pathfinder
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.25; ns.agent_height = 0.3; ns.agent_max_climb = 2.0; ns.cell_size = 0.07
    sim.recompute_navmesh(pf, ns)

    aom = sim.get_articulated_object_manager()
    drone = aom.add_articulated_object_from_urdf(os.path.normpath(_URDF), fixed_base=True)
    drone.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    try:
        drone.set_light_setup(DEFAULT_LIGHTING_KEY)
    except Exception as exc:
        print(f"[render] art light bind failed: {exc!r}")
    njoints = len(drone.joint_positions)
    print(f"[render] quad URDF loaded; movable joints={njoints} (expect 3: lower, grip_l, grip_r)")

    otm = sim.get_object_template_manager(); rom = sim.get_rigid_object_manager()
    base = otm.get_template_handles("cube")[0]
    t = otm.get_template_by_handle(base); t.scale = np.array([0.1, 0.1, 0.1], np.float32)
    otm.register_template(t, "show_cube")
    cube = rom.add_object_by_template_handle("show_cube")
    cube.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    try:
        cube.set_light_setup(DEFAULT_LIGHTING_KEY)
    except Exception:
        pass

    def clr(p, ang, max_r=6.0):
        d = np.array([math.cos(ang), 0.0, math.sin(ang)], np.float32)
        h = sim.cast_ray(habitat_sim.geo.Ray(np.asarray(p, np.float32), d), max_distance=max_r)
        return float(h.hits[0].ray_distance) if (h.has_hits and len(h.hits) > 0) else max_r

    ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(300)], np.float32)
    hh, e = np.histogram(ys, 40); lo = e[int(hh.argmax())]
    floor = float(np.median(ys[(ys >= lo) & (ys < lo + (e[1] - e[0]))]))
    angs = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    best = None
    for _ in range(args.tries):
        p = np.array(pf.get_random_navigable_point(), np.float32)
        if abs(float(p[1]) - floor) > 0.5:
            continue
        p[1] = floor + 1.4
        ca = max(angs, key=lambda a: min(clr(p, a), clr(p, a + math.pi)))
        score = min(clr(p, ca), clr(p, ca + math.pi))
        if best is None or score > best[0]:
            best = (score, p.copy(), float(ca))
    _, drone_p, cam_ang = best
    print(f"[render] open spawn drone={drone_p} floor={floor:.2f} corridor={best[0]:.1f}m")

    r = 1.7
    eye = drone_p + np.array([r * math.cos(cam_ang), 0.5, r * math.sin(cam_ang)], np.float32)
    tgt = drone_p + np.array([0.0, -0.30, 0.0], np.float32)
    f = 0.5 * args.res / math.tan(math.radians(args.hfov) / 2); cc = args.res / 2
    Rcw = quaternion.as_rotation_matrix(_lookat_quat(eye, tgt))

    def project(P):
        pc = Rcw.T @ (np.asarray(P, np.float32) - eye)
        if pc[2] >= -1e-3:
            return None
        return (cc + f * (pc[0] / -pc[2]), cc - f * (pc[1] / -pc[2]))

    def pose(lower, grip):
        drone.translation = drone_p
        jaw = (1.0 - grip) * 0.04
        jp = list(drone.joint_positions)
        vals = [lower * 0.5, jaw, jaw]                 # [lower, grip_l, grip_r]
        for i in range(min(len(jp), 3)):
            jp[i] = vals[i]
        drone.joint_positions = jp

    def shoot():
        st = habitat_sim.AgentState(); st.position = eye; st.rotation = _lookat_quat(eye, tgt)
        sim.get_agent(0).set_state(st)
        return sim.get_sensor_observations()["rgb"][..., :3].astype(np.uint8)

    def tip(lower):
        return drone_p + np.array([0, -(0.03 + lower * 0.5 + 0.31), 0], np.float32)

    def annotate(rgb, title, items):
        im = Image.fromarray(rgb.copy()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 22], fill=(0, 0, 0)); dr.text((6, 6), title, fill=(255, 255, 255))
        for label, P, anchor in items:
            uv = project(P)
            if uv is None:
                continue
            u, v = uv; ax, ay = anchor
            dr.line([ax, ay, u, v], fill=(255, 210, 0), width=2)
            dr.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(255, 210, 0), width=2)
            dr.rectangle([ax - 2, ay - 9, ax + 6 * len(label) + 4, ay + 9], fill=(0, 0, 0))
            dr.text((ax + 1, ay - 7), label, fill=(255, 210, 0))
        return np.asarray(im)

    out = []
    # (1) raised + open
    pose(0.0, 0.0); cube.translation = np.array([drone_p[0] + 0.25, floor + 0.05, drone_p[2]], np.float32)
    out.append(("1_raised_open", annotate(shoot(), "1. quadrotor, gripper RAISED + open", [
        ("quadrotor body", drone_p, (490, 90)),
        ("rotor x4", drone_p + np.array([0.21, 0.03, 0], np.float32), (560, 250)),
        ("gripper (raised)", tip(0.0), (300, 360)),
        ("cube", np.array(cube.translation, np.float32), (60, 600)),
    ])))
    # (2) lowered + open
    pose(1.0, 0.0); tp = tip(1.0)
    cube.translation = np.array([tp[0], floor + 0.05, tp[2]], np.float32)
    out.append(("2_lowered_over_cube", annotate(shoot(), "2. arm LOWERED (DoF1), jaws open (DoF2)", [
        ("quadrotor body", drone_p, (490, 80)),
        ("lower arm = DoF 1 (raise/lower)", drone_p + np.array([0, -0.28, 0], np.float32), (250, 330)),
        ("jaws = DoF 2 (open/close)", tp, (360, 520)),
        ("cube", np.array(cube.translation, np.float32), (60, 620)),
    ])))
    # (3) holding
    pose(0.4, 1.0); tp = tip(0.4); cube.translation = tp
    out.append(("3_holding_cube", annotate(shoot(), "3. holding the cube (jaws closed)", [
        ("quadrotor body", drone_p, (490, 80)),
        ("lower arm (DoF 1)", drone_p + np.array([0, -0.20, 0], np.float32), (490, 320)),
        ("jaws holding cube (DoF 2)", tp, (360, 470)),
    ])))

    for name, img in out:
        Image.fromarray(img).save(os.path.join(args.out_dir, f"drone_{name}.png"))
        print(f"[render] drone_{name}.png  mean={float(img.mean()):.1f}")
    imageio.imwrite(os.path.join(args.out_dir, "drone_contact_sheet.png"),
                    np.concatenate([i for _, i in out], axis=1))
    print(f"[render] {os.path.join(args.out_dir, 'drone_contact_sheet.png')}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
