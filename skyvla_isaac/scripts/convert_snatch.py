"""Convert the SNATCH drone URDF into a USD PhysX articulation (Isaac importer).

Adapted from convert_urdf.py for assets/drone_snatch.urdf. After dropping the
`lower` joint, the ONLY movable joints are the 4 cage jaws
(grip_xl/grip_xr/grip_yl/grip_yr) -> they become real actuated PhysX DoFs and the
grasp is contact/friction-based. Fixed boom/rotor/camera links fold into the base.
Free-floating base = a drone.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
    python skyvla_isaac/scripts/convert_snatch.py
"""
import os
import sys

# Isaac's carb log capture buffers stdout; force line-buffered so the [convert]
# report (and CONVERT_OK) always flush even without `python -u`.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(HERE, "assets", "drone_snatch.urdf")
USD_DIR = os.path.join(HERE, "assets", "configuration")
os.makedirs(USD_DIR, exist_ok=True)
USD = os.path.join(USD_DIR, "drone_snatch.usd")

res, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = True          # fold booms/rotors/cameras into base; keep jaw joints
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
    print(f"[convert] USD written to: {USD}")
    print(f"[convert] articulation roots: {arts}")
    print(f"[convert] prismatic joints ({len(pris)}): {pris}")
    print(f"[convert] revolute joints ({len(rev)}): {rev}")
    if len(pris) == 4 and len(rev) == 0:
        print("[convert] joint check OK: exactly 4 grip joints, no lower DOF")
    else:
        print(f"[convert] WARNING: expected exactly 4 prismatic grip joints (got {len(pris)}) "
              f"and 0 revolute (got {len(rev)})")
    print("[convert] CONVERT_OK")
else:
    print("[convert] CONVERT_FAILED: no USD produced")

sim_app.close()
