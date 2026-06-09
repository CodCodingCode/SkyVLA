"""SNATCH camera smoke test (Agent A2).

Spawns a trivial Isaac scene (ground + a couple cuboids + dome light), attaches
the two SNATCH depth cameras via `snatch.perception` cfgs, steps a few times, and
prints the captured depth tensor shapes + min/max for both cameras. Then runs the
two (separate) DepthEncoders on the captured depth and prints the latent shapes.

  conda activate isaac
  OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1 PYTHONPATH=$PWD \
    python scripts/snatch_cam_smoke.py

Prints CAM_SMOKE_OK on success.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True  # required for Camera sensor depth rendering
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402

from skyvla_isaac.snatch.perception import (  # noqa: E402
    top_camera_cfg,
    bottom_camera_cfg,
    DepthEncoder,
    build_obs_latents,
    freeze_backbone,
)

DEVICE = "cuda"
sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0, device=DEVICE))

# --- trivial scene: ground + dome light + a couple of cuboids ---
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=3000.0).func(
    "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
)
for i, (p, c) in enumerate([
    ((0.6, 0.0, 0.15), (0.9, 0.2, 0.2)),
    ((0.0, 0.5, 0.10), (0.2, 0.6, 0.95)),
]):
    cfg = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2 if i == 0 else 0.15),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c),
    )
    cfg.func(f"/World/box{i}", cfg, translation=p)

# --- the two SNATCH cameras (absolute prim paths for this standalone scene) ---
top_cfg = top_camera_cfg(prim_path="/World/top_cam")
bot_cfg = bottom_camera_cfg(prim_path="/World/bottom_cam")
top_cam = Camera(top_cfg)
bottom_cam = Camera(bot_cfg)

sim.reset()

# Pose the cameras over the scene: top looks forward across the boxes from a low
# vantage; bottom looks straight down at the boxes from above.
top_cam.set_world_poses_from_view(
    torch.tensor([[-0.8, 0.0, 0.4]], device=DEVICE),
    torch.tensor([[0.6, 0.0, 0.15]], device=DEVICE),
)
bottom_cam.set_world_poses_from_view(
    torch.tensor([[0.3, 0.2, 2.0]], device=DEVICE),
    torch.tensor([[0.3, 0.2, 0.0]], device=DEVICE),
)

# step a few times so the renderer produces depth
for _ in range(8):
    sim.step()
    top_cam.update(dt=sim.get_physics_dt())
    bottom_cam.update(dt=sim.get_physics_dt())

top_depth = top_cam.data.output["distance_to_image_plane"][..., 0]      # (N,H,W)
bottom_depth = bottom_cam.data.output["distance_to_image_plane"][..., 0]  # (N,H,W)


def _finite_minmax(t):
    f = t[torch.isfinite(t)]
    if f.numel() == 0:
        return float("nan"), float("nan")
    return float(f.min()), float(f.max())


t_min, t_max = _finite_minmax(top_depth)
b_min, b_max = _finite_minmax(bottom_depth)
print(f"[cam] top_depth   shape={tuple(top_depth.shape)}   "
      f"min={t_min:.3f} max={t_max:.3f}  (expect 480x848, FoV 87)")
print(f"[cam] bottom_depth shape={tuple(bottom_depth.shape)} "
      f"min={b_min:.3f} max={b_max:.3f}  (expect 480x640, FoV 120)")

# sanity on shapes
assert tuple(top_depth.shape[1:]) == (480, 848), top_depth.shape
assert tuple(bottom_depth.shape[1:]) == (480, 640), bottom_depth.shape
assert t_max > t_min and b_max > b_min, "degenerate depth range"

# --- encoders: SEPARATE instances, no shared weights ---
enc_top = DepthEncoder().to(DEVICE).eval()
enc_bottom = DepthEncoder().to(DEVICE).eval()
freeze_backbone(enc_top)
freeze_backbone(enc_bottom)
assert enc_top is not enc_bottom
assert enc_top.proj.weight.data_ptr() != enc_bottom.proj.weight.data_ptr(), \
    "encoders must not share weights"

with torch.no_grad():
    f_top = enc_top(top_depth)
    f_bot = enc_bottom(bottom_depth)
    latents = build_obs_latents(top_depth, bottom_depth, enc_top, enc_bottom)

print(f"[enc] top latent    shape={tuple(f_top.shape)}   (expect (1, 512))")
print(f"[enc] bottom latent shape={tuple(f_bot.shape)}   (expect (1, 512))")
print(f"[enc] obs latents   shape={tuple(latents.shape)} (expect (1, 1024))")

assert tuple(f_top.shape) == (1, 512)
assert tuple(f_bot.shape) == (1, 512)
assert tuple(latents.shape) == (1, 1024)

# confirm freeze: backbone params frozen, proj trainable
n_trainable = sum(p.requires_grad for p in enc_top.parameters())
n_bb_frozen = sum((not p.requires_grad) for p in enc_top.backbone.parameters())
print(f"[enc] frozen backbone params={n_bb_frozen}, trainable tensors={n_trainable} (proj only)")

print("CAM_SMOKE_OK")
sim_app.close()
