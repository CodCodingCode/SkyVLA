"""Convert the drone+gripper URDF into a USD PhysX articulation (Isaac importer).

The movable joints (lower, grip_l, grip_r) become real actuated PhysX DoFs, so
grasping is contact/friction-based — not the kinematic-attach hack from Habitat.
Fixed boom/rotor links are merged into the base. Free-floating base = a drone.

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES \
    .venv311/bin/python skyvla_isaac/scripts/convert_urdf.py
"""
import os

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(HERE, "assets", "drone_with_gripper.urdf")
USD = os.path.join(HERE, "assets", "drone_with_gripper.usd")

res, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = True          # fold booms/rotors into base; keep movable joints
cfg.fix_base = False                   # free-floating base -> a flying drone
cfg.make_default_prim = True
cfg.self_collision = False
cfg.import_inertia_tensor = True
cfg.distance_scale = 1.0
cfg.default_drive_strength = 1e4       # actuated joints
cfg.default_position_drive_damping = 1e3

res, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=URDF, import_config=cfg, dest_path=USD)
print(f"[convert] import status={res} prim_path={prim_path} usd_exists={os.path.exists(USD)}")

if os.path.exists(USD):
    stage = Usd.Stage.Open(USD)
    pris = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    rev = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    arts = [p.GetPath().pathString for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"[convert] articulation roots: {arts}")
    print(f"[convert] prismatic joints ({len(pris)}): {pris}")
    print(f"[convert] revolute joints ({len(rev)}): {rev}")
    print("[convert] CONVERT_OK")
else:
    print("[convert] CONVERT_FAILED: no USD produced")

sim_app.close()
