"""Render a pick-and-place rollout with a GAUSSIAN-SPLAT indoor room as the backdrop.

Pipeline (the Habitat->Isaac GS port, applied as a rollout backdrop):
  1. Spawn a visual-only indoor room in Isaac.
  2. Orbit the render camera, fuse RGB-D into a GaussianMap -- but only ROOM pixels
     (drone/cube are masked out via semantic segmentation), so the splat is the
     empty room.
  3. Run the trained policy (physics on the flat floor). Each frame: render the GS
     room at the follow-camera pose, and composite the live drone+cube (segmentation
     mask) on top. Physics is NOT against the splat -- it's a cosmetic backdrop.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1 \
  python skyvla_isaac/scripts/render_rollout_gs.py \
     --checkpoint logs/isaac/drone_pick_place/model_5999.pt --out videos/isaac_pickplace_gs.mp4

NOTE: PYTHONUTF8=1 is REQUIRED -- gsplat 1.5.3 ships without prebuilt CUDA kernels
and JIT-compiles on first use; torch's JIT reads the .cu sources with the locale
codec, which crashes (UnicodeDecodeError) unless Python is forced to UTF-8.
"""
import argparse
import os
import subprocess
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default="/home/ubuntu/SkyVLA/videos/isaac_pickplace_gs.mp4")
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--cur_p", type=float, default=0.0)   # full fly-in-from-altitude task
# --- GS scene cache (skip the slow RGB-D room orbit on repeat runs) ---
parser.add_argument("--gs_cache", default=None,
                    help="scene-cache path (default skyvla_isaac/gs/cache/room_splat.pt). "
                         "Loaded if present so the room orbit is skipped.")
parser.add_argument("--rebuild_gs", action="store_true",
                    help="force a fresh room orbit + fusion even if the cache exists.")
# --- rollout cache (re-render the whole video later with NO Isaac) ---
parser.add_argument("--save_rollout", action="store_true",
                    help="also cache per-frame foreground+pose so render_gs_cache.py "
                         "can re-composite the video without booting Isaac.")
parser.add_argument("--rollout_cache", default=None,
                    help="rollout-cache path (default cache/rollout_<checkpoint>.npz).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import math  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
import subprocess as _sp  # noqa: E402  (kept for clarity)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.tasks.pick_place_env import DronePickPlaceEnv, DronePickPlaceEnvCfg  # noqa: E402
from skyvla_isaac.gs import GaussianMap  # noqa: E402
from skyvla_isaac.gs import cache as gs_cache  # noqa: E402
from skyvla_isaac.gs.isaac_camera import frame_from_camera  # noqa: E402

cfg = DronePickPlaceEnvCfg()
cfg.curriculum_p_start = cfg.curriculum_p_end = args.cur_p
cfg.scene.num_envs = 1
cfg.render_camera = True
cfg.indoor_room = True
env = DronePickPlaceEnv(cfg)
wenv = RslRlVecEnvWrapper(env)
dev = env.device

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50,
    experiment_name="drone_pick_place", logger="tensorboard", empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
                                  actor_hidden_dims=[256, 128, 64],
                                  critic_hidden_dims=[256, 128, 64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0),
)
runner = OnPolicyRunner(wenv, class_to_dict(agent_cfg), log_dir=None, device=dev)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=dev)
print(f"[gs-render] loaded {args.checkpoint}")

cam = env._render_cam
W, H = 720, 540


def seg_ids_for(classes):
    """Return the set of segmentation ids whose label matches any of `classes`."""
    info = cam.data.info[0]["semantic_segmentation"]
    id2lab = info["idToLabels"] if isinstance(info, dict) else info
    want = set()
    for k, v in id2lab.items():
        lab = str(v).lower()
        if any(c in lab for c in classes):
            want.add(int(k))
    return want


def seg_mask(classes):
    """(H,W) bool mask of pixels belonging to any of `classes`."""
    seg = cam.data.output["semantic_segmentation"][0]
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = seg.to(dev).long()
    ids = seg_ids_for(classes)
    if not ids:
        return torch.zeros(H, W, dtype=torch.bool, device=dev)
    m = torch.zeros_like(seg, dtype=torch.bool)
    for i in ids:
        m |= (seg == i)
    return m


def step_render(n=2):
    for _ in range(n):
        env.sim.step(render=True)
    env.scene.update(env.sim.get_physics_dt())
    cam.update(env.sim.get_physics_dt())


# ---------------------------------------------------------------- #
# 1) GET THE GAUSSIAN-SPLAT ROOM — load the cache, or fuse it once.
#    The room is static geometry (no policy/checkpoint dependence), so a
#    cached splat is reused verbatim and the slow RGB-D orbit is skipped.
# ---------------------------------------------------------------- #
env.reset()                                   # initialize sim before stepping it directly
scene_path = args.gs_cache or gs_cache.DEFAULT_SCENE

if os.path.exists(scene_path) and not args.rebuild_gs:
    gmap, K_fix, Wc, Hc, meta = gs_cache.load_scene(scene_path, device=dev)
    if (Wc, Hc) != (W, H):
        print(f"[gs-render] WARN cached render size {(Wc, Hc)} != current {(W, H)}; "
              "rebuild with --rebuild_gs if the room looks wrong.")
    print(f"[gs-render] loaded scene cache {scene_path} -> {gmap.num_gaussians} gaussians "
          f"(skipped the room orbit){' meta=' + str(meta) if meta else ''}")
else:
    gmap = GaussianMap(device=torch.device(dev))
    poses = []
    # rings around the center looking in -> capture far walls/floor/ceiling behind center
    for h in (0.9, 1.8, 2.6):
        for k in range(12):
            a = 2 * math.pi * k / 12
            poses.append(((2.6 * math.cos(a), 2.6 * math.sin(a), h), (0.0, 0.0, 1.2)))
    # from near-center looking OUT -> capture each wall up close
    for k in range(8):
        a = 2 * math.pi * k / 8
        poses.append(((0.3 * math.cos(a), 0.3 * math.sin(a), 1.5),
                      (3.5 * math.cos(a), 3.5 * math.sin(a), 1.3)))

    added = 0
    for eye, tgt in poses:
        cam.set_world_poses_from_view(torch.tensor([eye], device=dev),
                                      torch.tensor([tgt], device=dev))
        step_render(2)
        rgb, depth, pose, K = frame_from_camera(cam, 0)
        fg = seg_mask(["drone", "cube"])          # exclude moving foreground from the map
        depth = depth.clone()
        depth[fg] = 0.0                            # 0 depth -> skipped by add_from_rgbd
        added += gmap.add_from_rgbd(rgb, depth, pose, K, stride=2, max_depth=15.0, scale=0.04)
    print(f"[gs-render] fused {len(poses)} views -> {gmap.num_gaussians} gaussians (added {added})")

    # fixed intrinsics for the splat render (same camera model, constant K)
    _, _, _, K_fix = frame_from_camera(cam, 0)
    gs_cache.save_scene(scene_path, gmap, K_fix, W, H,
                        meta={"views": len(poses), "added": added, "cur_p": args.cur_p})
    print(f"[gs-render] saved scene cache -> {scene_path}")

room_fill = torch.tensor([0.72, 0.69, 0.63], device=dev)   # neutral wall color for any holes

# ---------------------------------------------------------------- #
# 2) ROLLOUT + COMPOSITE
# ---------------------------------------------------------------- #
obs, _ = wenv.get_observations()
off_eye = torch.tensor([1.3, 1.3, 0.55], device=dev)
off_tgt = torch.tensor([0.0, 0.0, -0.2], device=dev)


def follow():
    dp = env.robot.data.root_pos_w[0]
    cam.set_world_poses_from_view((dp + off_eye).unsqueeze(0), (dp + off_tgt).unsqueeze(0))


tmp = tempfile.mkdtemp(prefix="rollout_gs_")
nf = 0
roll_fg, roll_poses = [], []                                   # rollout cache buffers
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, _, _ = wenv.step(act)
    follow()
    cam.update(env.sim.get_physics_dt())                       # refresh outputs at new pose

    isaac_rgb = cam.data.output["rgb"][0, ..., :3].float() / 255.0   # (H,W,3) [0,1]
    fg = seg_mask(["drone", "cube"]).unsqueeze(-1).float()           # (H,W,1)

    _, _, pose, _ = frame_from_camera(cam, 0)
    gs_rgb, gs_alpha = gmap.render(pose, K_fix, W, H)               # (H,W,3),(H,W,1)
    gs_rgb = gs_rgb.clamp(0, 1)
    covered = (gs_alpha > 0.3).float()
    backdrop = covered * gs_rgb + (1 - covered) * room_fill         # fill holes

    out = fg * isaac_rgb + (1 - fg) * backdrop
    img = (out.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), img)
    nf += 1

    if args.save_rollout:
        # foreground RGB premultiplied by the mask (background -> 0 so it
        # compresses away), mask in alpha; + the camera pose for the GS render.
        fg_rgb = (fg * isaac_rgb).clamp(0, 1)
        rgba = torch.cat([fg_rgb, fg], dim=-1)                      # (H,W,4) [0,1]
        roll_fg.append((rgba * 255).to(torch.uint8).cpu())
        roll_poses.append(pose.detach().cpu())
print(f"[gs-render] captured {nf} composited frames")

if args.save_rollout:
    roll_path = args.rollout_cache or gs_cache.default_rollout_path(args.checkpoint)
    gs_cache.save_rollout(roll_path, torch.stack(roll_fg), torch.stack(roll_poses),
                          fps=args.fps, width=W, height=H,
                          meta={"checkpoint": os.path.basename(args.checkpoint),
                                "steps": nf, "cur_p": args.cur_p})
    print(f"[gs-render] saved rollout cache -> {roll_path}  "
          f"(re-render with: render_gs_cache.py --rollout {roll_path})")

cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-crf", "21", args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-800:])
print(f"[gs-render] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
print("RENDER_DONE")
sim_app.close()
