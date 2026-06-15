"""SNATCH grasp gate (scripted) — close-then-lift test for assets/drone_snatch.usd.

Minimal Isaac Lab scene (no RL env): spawn the converted SNATCH drone articulation
+ a 5 cm cube on the floor + a ground plane. Place the drone straddling the cube
at grasp height (fingertips around the cube), command the 4 cage jaws CLOSED, let
them clamp, then ascend the drone bodily and verify the cube is carried up.

Per step prints:  cube_z | base_z | grip | d_reach
  - cube_z   : cube world Z
  - base_z   : drone base world Z
  - grip     : commanded jaw closure (0 open .. 1 closed)
  - d_reach  : horizontal+vertical dist from the cage center to the cube

PASS: while ascending, cube_z tracks base_z UPWARD (cube lifts off the floor) and
d_reach stays small (~0.01 .. cube stays caged). Prints SNATCH_GRASP_PASS / FAIL.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1 \
    PYTHONPATH=$PWD python skyvla_isaac/scripts/snatch_grasp_test.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
sim_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import (  # noqa: E402
    Articulation, ArticulationCfg, RigidObject, RigidObjectCfg)
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(HERE, "assets", "configuration", "drone_snatch.usd")

# fingertip pads are centered at base_z - 0.33 (see URDF). Put that at the cube
# center (0.025) so the cube sits inside the cage walls when we close them.
GRASP_BASE_Z = 0.355
CUBE_Z0 = 0.025
DT = 1.0 / 120.0

# ---------------------------------------------------------------- sim + scene
sim = SimulationContext(SimulationCfg(
    dt=DT, device="cuda:0",
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0, dynamic_friction=1.8, friction_combine_mode="max")))

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1500.0))

robot_cfg = ArticulationCfg(
    prim_path="/World/Drone",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16,
            solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, GRASP_BASE_Z),
        joint_pos={"grip_xl": 0.0, "grip_xr": 0.0, "grip_yl": 0.0, "grip_yr": 0.0},
    ),
    actuators={
        # jaws: firm but not so hard they punt the free cube out before caging it.
        "grip": ImplicitActuatorCfg(joint_names_expr=["grip_.*"],
                                    effort_limit=80.0, velocity_limit=1.0,
                                    stiffness=2000.0, damping=10.0),
    },
)
robot = Articulation(robot_cfg)

cube_cfg = RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=1.6, friction_combine_mode="max"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CUBE_Z0)),
)
cube = RigidObject(cube_cfg)

sim.reset()
device = sim.device

grip_i, grip_names = robot.find_joints("grip_.*")
print(f"[grasp] grip joints: {grip_names}")
JAW_CLOSED = 0.02   # full closure target (URDF upper limit)


def cage_center():
    """World pos of the cage center (between the 4 fingers): base minus 0.33 Z."""
    root = robot.data.root_pos_w[:, :3].clone()
    root[:, 2] -= 0.33
    return root


def step_phys():
    robot.write_data_to_sim()
    cube.write_data_to_sim()
    sim.step()
    robot.update(DT)
    cube.update(DT)


def diag(label, grip_cmd):
    cz = float(cube.data.root_pos_w[0, 2])
    bz = float(robot.data.root_pos_w[0, 2])
    d_reach = float(torch.norm(cage_center()[0] - cube.data.root_pos_w[0, :3]))
    print(f"[{label}] cube_z={cz:+.3f} | base_z={bz:+.3f} | grip={grip_cmd:.2f} | d_reach={d_reach:.3f}")
    return cz, bz, d_reach


# ---------------------------------------------------------------- pin the base
# We pivot the test on the gripper, not the flight controller: hold the base at a
# commanded Z (settle, grasp, then ascend) by writing root state each step. This
# isolates the GRASP gate (does the cage hold the cube?) from flight tuning.
def set_base_z(z, vz=0.0):
    root = robot.data.default_root_state.clone()
    root[:, 0:3] = torch.tensor([[0.0, 0.0, z]], device=device)
    root[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    root[:, 7:13] = 0.0
    root[:, 9] = vz
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])


def set_jaws(closed_frac):
    tgt = torch.full((robot.num_instances, len(grip_i)),
                     closed_frac * JAW_CLOSED, device=device)
    robot.set_joint_position_target(tgt, joint_ids=grip_i)


# Phase 0: settle straddling the cube, jaws OPEN.
print("=== phase: settle (jaws open, straddling cube) ===")
set_jaws(0.0)
for _ in range(30):
    set_base_z(GRASP_BASE_Z)
    step_phys()
diag("settle", 0.0)

# Phase 1: CLOSE the jaws on the cube, hold position so they clamp.
print("=== phase: close jaws ===")
for k in range(60):
    set_base_z(GRASP_BASE_Z)
    set_jaws(1.0)
    step_phys()
    if k % 20 == 0:
        diag("close", 1.0)
diag("closed", 1.0)

# Phase 2: ASCEND bodily, jaws stay closed; cube should be carried up.
print("=== phase: ascend (jaws closed) ===")
ASCEND_VZ = 0.4
N_ASC = 80
cube_zs, base_zs, reaches = [], [], []
for k in range(N_ASC):
    target_z = GRASP_BASE_Z + ASCEND_VZ * (k + 1) * DT
    set_base_z(target_z, vz=ASCEND_VZ)
    set_jaws(1.0)
    step_phys()
    if k % 10 == 0 or k == N_ASC - 1:
        cz, bz, dr = diag("ascend", 1.0)
        cube_zs.append(cz); base_zs.append(bz); reaches.append(dr)

# ---------------------------------------------------------------- verdict
cube_lift = cube_zs[-1] - CUBE_Z0
base_lift = base_zs[-1] - GRASP_BASE_Z
max_reach = max(reaches)
tracking = cube_zs[-1] - base_zs[-1]   # should stay ~ -0.33 (cube under the cage)

print("\n=== SNATCH grasp gate verdict ===")
print(f"  cube lifted by   : {cube_lift:+.3f} m  (base lifted {base_lift:+.3f} m)")
print(f"  final cube_z      : {cube_zs[-1]:+.3f}   final base_z: {base_zs[-1]:+.3f}")
print(f"  max d_reach       : {max_reach:.3f} m (cube vs cage center)")
print(f"  cube-base offset  : {tracking:+.3f} m (expect ~ -0.33 if caged)")

# PASS: the cube came off the floor by most of the base's ascent (it tracks the
# drone up) AND it stayed inside the cage (small d_reach throughout).
PASS = (cube_lift > 0.10) and (cube_lift > 0.5 * base_lift) and (max_reach < 0.06)
print("SNATCH_GRASP_PASS" if PASS else "SNATCH_GRASP_FAIL")

sim_app.close()
