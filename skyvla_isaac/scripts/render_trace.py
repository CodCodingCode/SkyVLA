"""Render a RECORDED pipeline trajectory (run_pipeline.py --trace_out) to mp4.

The three-policy pipeline succeeds end-to-end ~0.6% of the time, so re-running it live
with a camera attached and hoping the followed env is one of the winners is impractical.
Instead this replays a known-good episode kinematically: the drone / cube / pad-B poses
recorded from the real Isaac PhysX rollout are written into the sim each frame and the
3rd-person camera renders them. The motion is exactly what the policies produced.

  .venv311/bin/python skyvla_isaac/scripts/render_trace.py \
      --ep best_ep.npz --out videos/delivery.mp4
"""
import argparse, os, subprocess, tempfile
import numpy as np
from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = argparse.ArgumentParser()
p.add_argument("--ep", required=True, help=".npz from the trace extractor (drone/cube/phase/platB)")
p.add_argument("--out", default=os.path.join(_REPO, "videos/delivery.mp4"))
p.add_argument("--stride", type=int, default=2)
p.add_argument("--fps", type=int, default=25)
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--supersample", type=float, default=1.5)
p.add_argument("--crf", type=int, default=18)
p.add_argument("--quality", choices=["fast", "high"], default="high")
p.add_argument("--framing", choices=["follow", "wide"], default="wide",
               help="wide = fixed shot framing BOTH pads (reads as a delivery); "
                    "follow = camera chases the drone")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import torch, imageio.v2 as imageio                                    # noqa: E402
import isaaclab.sim as sim_utils                                        # noqa: E402
from isaaclab.sim import RenderCfg                                      # noqa: E402
from skyvla_isaac.snatch.pick_place_env import (                        # noqa: E402
    DroneSnatchEnv, DroneSnatchEnvCfg)

ep = np.load(args.ep)
dr, cu, ph = ep["drone"], ep["cube"], ep["phase"]
bx, by = [float(v) for v in ep["platB"]]
n = len(dr)

rw, rh = int(args.width * args.supersample), int(args.height * args.supersample)
cfg = DroneSnatchEnvCfg()
cfg.scene.num_envs = 1
cfg.use_cameras = False
cfg.render_camera = True
cfg.render_cam_w, cfg.render_cam_h = rw, rh
cfg.two_platform = True
cfg.plat_sep = cfg.plat_sep_max = max(1.0, (bx**2 + by**2) ** 0.5)
cfg.grasp_latch = False
cfg.scene.env_spacing = 12.0
if args.quality == "high":
    cfg.sim.render = RenderCfg(enable_dl_denoiser=True, samples_per_pixel=2)
cfg.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.85, 0.12, 0.12), roughness=0.45, metallic=0.0)
cfg.platform.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.42, 0.30, 0.20), roughness=0.75, metallic=0.0)
cfg.plat_b.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.20, 0.38, 0.46), roughness=0.70, metallic=0.0)

env = DroneSnatchEnv(cfg)
env.reset()
dev = env.device
org = env.scene.env_origins[0]
print(f"[trace-render] {n} steps, pad B at ({bx:+.2f},{by:+.2f}), stride {args.stride}", flush=True)

# pad B is kinematic furniture -- pin it where the recording had it
bpose = torch.zeros(1, 7, device=dev)
bpose[0, :2] = torch.tensor([bx, by], device=dev) + org[:2]
bpose[0, 2] = 0.15
bpose[0, 3] = 1.0
env.plat_b.write_root_pose_to_sim(bpose, torch.tensor([0], device=dev))

cam = env._render_cam
eye = torch.tensor([2.3, 2.3, 0.35], device=dev)
tgt = torch.tensor([0.0, 0.0, -0.22], device=dev)
# Wide framing: a fixed camera on the A->B midpoint, pulled back far enough that both pads
# and the whole arc stay in shot. A drone-following camera loses the pads, which is exactly
# the information a delivery clip needs to show.
mid = torch.tensor([bx / 2, by / 2, 0.30], device=dev) + org
span = max(2.5, (bx**2 + by**2) ** 0.5)
wide_eye = mid + torch.tensor([span * 1.85, span * 1.55, span * 1.15], device=dev)
zero = torch.zeros(1, 6, device=dev)
idx = torch.tensor([0], device=dev)
tmp = tempfile.mkdtemp(prefix="trace_"); nf = 0

for k in range(0, n, args.stride):
    rp = torch.zeros(1, 7, device=dev)
    rp[0, :3] = torch.tensor(dr[k], device=dev, dtype=torch.float32) + org
    rp[0, 3] = 1.0
    cp = torch.zeros(1, 7, device=dev)
    cp[0, :3] = torch.tensor(cu[k], device=dev, dtype=torch.float32) + org
    cp[0, 3] = 1.0
    env.robot.write_root_pose_to_sim(rp, idx)
    env.robot.write_root_velocity_to_sim(zero, idx)
    env.object.write_root_pose_to_sim(cp, idx)
    env.object.write_root_velocity_to_sim(zero, idx)
    env.sim.step(render=True)
    if args.framing == "wide":
        cam.set_world_poses_from_view(wide_eye.unsqueeze(0), mid.unsqueeze(0))
    else:
        d = env.robot.data.root_pos_w[0]
        cam.set_world_poses_from_view((d + eye).unsqueeze(0), (d + tgt).unsqueeze(0))
    cam.update(env.sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(tmp, f"f{nf:05d}.png"), rgb); nf += 1
    if nf % 50 == 0:
        print(f"  {nf} frames ({['approach','transit','deposit'][int(ph[k])]})", flush=True)

print(f"[trace-render] captured {nf} frames", flush=True)
vf = f"scale={args.width}:{args.height}:flags=lanczos" if (rw, rh) != (args.width, args.height) else "null"
cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-profile:v", "high",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", str(args.crf), args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-800:])
if os.path.exists(args.out):
    print(f"[trace-render] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
print("TRACE_RENDER_DONE")
os._exit(0)   # app.close() hangs on this host
