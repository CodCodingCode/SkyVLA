"""Render a SNATCH (state-based) pick-and-place rollout to MP4 with a 3rd-person cam.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1
  python skyvla_isaac/scripts/render_snatch.py \
     --checkpoint logs/isaac/drone_snatch_state/model_650.pt --out videos/snatch_pickplace.mp4

Quality profiles (--quality):
  fast   real-time RTX, no extra effects   -> quick iteration, looks like before
  high   RTX "quality" preset + GI/AO/reflections/soft-shadows + DL denoiser  (default)
  ptrace OptiX path tracing, accumulated over subframes -> film-grade, slow

AA: this host's A100 has no RT cores, so DLSS/DLAA are avoided. Crisp edges come
from supersampling (--supersample 1.5) rendered big then Lanczos-downscaled by ffmpeg.
"""
import argparse, os, subprocess, tempfile
from isaaclab.app import AppLauncher


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default=os.path.join(_REPO, "videos/snatch_pickplace.mp4"))
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--cur_p", type=float, default=0.0)   # from-altitude demo
parser.add_argument("--quality", choices=["fast", "high", "ptrace"], default="high")
parser.add_argument("--width", type=int, default=1920)    # output resolution
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--supersample", type=float, default=1.5,
                    help="render at this x output res, then downscale for free AA")
parser.add_argument("--pt_subframes", type=int, default=12,
                    help="ptrace only: accumulate this many render passes per frame")
parser.add_argument("--crf", type=int, default=16)        # lower = higher quality
parser.add_argument("--rc_diag", type=float, default=None,
                    help="diagnosis: spawn at reverse-curriculum difficulty X (e.g. 0.10) to watch the descent")
parser.add_argument("--rc_dist_diag", type=float, default=None,
                    help="diagnosis: distance-mode spawn at difficulty X (X*10m fly-in) to watch the approach")
parser.add_argument("--side_spawn", type=float, default=None,
                    help="diagnosis: randomized side-spawn radius (m) to watch navigate->hover->descend")
parser.add_argument("--no_latch", action="store_true",
                    help="REAL PHYSICS grasp (floored-scoop cage): film what the contact forces do")
parser.add_argument("--carry_demo", action="store_true",
                    help="spawn ALREADY CARRYING (shelf-seated cube, jaws nearly shut, airborne)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import torch, numpy as np, imageio.v2 as imageio  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.sim import RenderCfg  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

# render at supersample x output res, then Lanczos-downscale in ffmpeg (cheap, GPU-agnostic AA)
ss = max(1.0, args.supersample)
rw = (round(args.width * ss) // 2) * 2          # even dims for libx264
rh = (round(args.height * ss) // 2) * 2

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False          # state-based policy (model_650)
cfg.render_camera = True         # 3rd-person RGB cam + cinematic light rig for the video
cfg.render_cam_w, cfg.render_cam_h = rw, rh
cfg.curriculum_p_start = cfg.curriculum_p_end = args.cur_p
if args.rc_diag is not None:
    cfg.reverse_curriculum = True       # spawn at the failing frontier to watch the descent
    cfg.rc_start = args.rc_diag
if args.rc_dist_diag is not None:
    cfg.reverse_curriculum = True
    cfg.rc_distance_mode = True          # spawn at a fly-in distance to watch the approach
    cfg.rc_start = args.rc_dist_diag
    cfg.episode_length_s = 22.0
if args.side_spawn is not None:
    cfg.side_spawn_max = args.side_spawn  # same navigate->hover->descend spawns as training
    cfg.staged_curriculum = True          # soft table-touch: match training termination, or the
                                          # film kills every episode at the first table graze
    cfg.episode_length_s = max(cfg.episode_length_s, 10.0 + args.side_spawn / max(cfg.speed, 0.3))
if args.no_latch:
    cfg.grasp_latch = False
if args.carry_demo:
    cfg.carry_demo_p = 1.0
cfg.scene.num_envs = 1

# --- renderer quality profile ---------------------------------------------- #
# NOTE: explicit feature flags set carb /rtx/* vars directly -> no dependence on the
# (missing) "rendering_modes" preset extension in this IsaacLab install.
# This host's A100 (GA100) has NO RT cores, so global illumination + reflections fall
# back to software ray tracing and never finish the first frame (~25 min, 0 frames).
# `high` therefore keeps only the cheap, real-time, high-impact effects: directional
# soft shadows + ambient-occlusion contact shadows + the DL denoiser. The big visual
# wins (resolution, supersampled AA, the 2-light rig, PBR materials) are GPU-cheap.
# GI/reflections live in `ptrace` for anyone running on an RT-core GPU.
if args.quality == "high":
    cfg.sim.render = RenderCfg(
        enable_ambient_occlusion=True, enable_shadows=True,
        enable_dl_denoiser=True, samples_per_pixel=2)
elif args.quality == "ptrace":
    cfg.sim.render = RenderCfg(
        enable_dl_denoiser=True,
        carb_settings={
            "/rtx/rendermode": "PathTracing",
            "/rtx/pathtracing/spp": 8,
            "/rtx/pathtracing/totalSpp": 256,
            "/rtx/pathtracing/maxBounces": 6,
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 6,
            "/rtx/pathtracing/optixDenoiser/enabled": 1,
        })

# --- PBR materials: matte plastic cube, slightly rough wooden table -------- #
cfg.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.85, 0.12, 0.12), roughness=0.45, metallic=0.0)
cfg.platform.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.42, 0.30, 0.20), roughness=0.75, metallic=0.0)

env = DroneSnatchEnv(cfg); wenv = RslRlVecEnvWrapper(env)
print(f"[snatch-render] quality={args.quality} render={rw}x{rh} -> out={args.width}x{args.height}")

agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24, max_iterations=1, save_interval=50,
    experiment_name="drone_snatch", logger="tensorboard", empirical_normalization=True,
    policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, noise_std_type="log",
                                  actor_hidden_dims=[256, 128, 64],
                                  critic_hidden_dims=[256, 128, 64], activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True,
                                   clip_param=0.2, entropy_coef=0.01, num_learning_epochs=10,
                                   num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
                                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0))
runner = OnPolicyRunner(wenv, class_to_dict(agent_cfg), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
print(f"[snatch-render] loaded {args.checkpoint}")

obs, _ = wenv.get_observations()
cam = env._render_cam
off_eye = torch.tensor([1.9, 1.9, -0.18], device=env.device)   # side-on, slightly below drone
off_tgt = torch.tensor([0.0, 0.0, -0.20], device=env.device)   # so the cube hanging below is clear
def follow():
    dp = env.robot.data.root_pos_w[0]
    cam.set_world_poses_from_view((dp + off_eye).unsqueeze(0), (dp + off_tgt).unsqueeze(0))
follow()
pt_sub = args.pt_subframes if args.quality == "ptrace" else 1
tmp = tempfile.mkdtemp(prefix="snatch_"); nf = 0
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    obs, _, _, _ = wenv.step(act)
    follow()
    # path tracing converges over accumulated passes -> re-render the held frame
    for _ in range(pt_sub):
        cam.update(env.sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), rgb); nf += 1
print(f"[snatch-render] captured {nf} frames")
# Lanczos-downscale the supersampled frames to the output res for crisp anti-aliased edges.
vf = (f"scale={args.width}:{args.height}:flags=lanczos"
      if (rw, rh) != (args.width, args.height) else "null")
cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-profile:v", "high",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", str(args.crf), args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-800:])
print(f"[snatch-render] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
print("SNATCH_RENDER_DONE")
# app.close() reliably HANGS on this host (3 zombie renderers so far, each squatting
# ~5GB GPU + CPU for hours). The mp4 is already written and synced -- hard-exit instead.
os._exit(0)
