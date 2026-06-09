"""Prove the Gaussian-splat environment runs in Isaac: capture RGB-D from an
Isaac Lab Camera around a small scene, fuse into the (ported) GaussianMap, and
render a novel view. Confirms the Habitat->Isaac GS port end to end.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES \
    python skyvla_isaac/scripts/gs_isaac_demo.py
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(_REPO, "videos/gs_isaac.png"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True          # required for Camera sensor RGB-D rendering
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from skyvla_isaac.gs import GaussianMap  # noqa: E402
from skyvla_isaac.gs.isaac_camera import frame_from_camera  # noqa: E402

sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cuda"))

# --- a small scene to reconstruct: ground + a ring of colored boxes ---
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))
colors = [(0.9, 0.2, 0.2), (0.2, 0.8, 0.3), (0.3, 0.4, 0.95), (0.95, 0.8, 0.2), (0.8, 0.3, 0.8)]
for i, c in enumerate(colors):
    a = 2 * math.pi * i / len(colors)
    p = (1.2 * math.cos(a), 1.2 * math.sin(a), 0.25)
    cfg = sim_utils.CuboidCfg(size=(0.4, 0.4, 0.5),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))
    cfg.func(f"/World/box{i}", cfg, translation=p)

cam_cfg = CameraCfg(
    prim_path="/World/cam", height=240, width=320, update_period=0.0,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
)
camera = Camera(cam_cfg)
sim.reset()

gmap = GaussianMap(device=torch.device("cuda"))
# orbit the camera around the scene, capturing posed RGB-D
eyes, tgts = [], []
for k in range(12):
    a = 2 * math.pi * k / 12
    eyes.append([3.0 * math.cos(a), 3.0 * math.sin(a), 1.6])
    tgts.append([0.0, 0.0, 0.3])
eyes_t = torch.tensor(eyes, device=sim.device)
tgts_t = torch.tensor(tgts, device=sim.device)

added = 0
for k in range(len(eyes)):
    camera.set_world_poses_from_view(eyes_t[k:k + 1], tgts_t[k:k + 1])
    for _ in range(2):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    rgb, depth, pose, K = frame_from_camera(camera, 0)
    added += gmap.add_from_rgbd(rgb, depth, pose, K, stride=2, max_depth=20.0)
print(f"[gs] fused {len(eyes)} Isaac frames -> {gmap.num_gaussians} gaussians (added {added})")

# render a NOVEL overhead-ish view that wasn't captured
camera.set_world_poses_from_view(torch.tensor([[0.2, 0.2, 4.0]], device=sim.device),
                                 torch.tensor([[0.0, 0.0, 0.2]], device=sim.device))
for _ in range(2):
    sim.step(); camera.update(dt=sim.get_physics_dt())
_, _, pose_novel, K = frame_from_camera(camera, 0)
rgb_out, alpha = gmap.render(pose_novel, K, 320, 240)
img = (rgb_out.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
imageio.imwrite(args.out, img)
print(f"[gs] rendered novel view -> {args.out}  (mean alpha {float(alpha.mean()):.2f})")
print("GS_ISAAC_OK")
sim_app.close()
