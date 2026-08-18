"""Film a PLACE-stage rollout: fly to pad B, set the cube down, open the jaws.

DOES NOT RUN ON THIS HOST. It needs Isaac's RTX Camera sensor, and this headless box has no
working Vulkan (`vkCreateInstance failed. Vulkan 1.1 is not supported`). Env construction
then HANGS forever at the Camera sensor -- observed 80 minutes with zero frames and no error.
Keep this for a machine with a working RTX/Vulkan stack; on this host film a take with
film_place_rollout.py (headless physics) + film_place_draw.py (matplotlib) instead.

Place-stage sibling of render_snatch.py (which films the single-platform grasp and has no
--two_platform/--place_only flags, so it cannot show the deposit at all).

  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1
  .venv311/bin/python skyvla_isaac/scripts/render_place.py \
     --checkpoint logs/isaac/drone_snatch_place_only_v4/model_20000.pt --out videos/place.mp4

The camera tracks the MIDPOINT of drone and pad B so the approach, the descent and the
cube's landing all stay in frame. The run prints the deposit step, contact speed and
landing error so the clip can be verified as a real success and not a near-miss.
"""
import argparse, os, subprocess, tempfile
from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default=os.path.join(_REPO, "videos/place.mp4"))
parser.add_argument("--steps", type=int, default=170)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--seed", type=int, default=0, help="pick a different take")
parser.add_argument("--spawn_r", type=float, default=0.55, help="drone spawn disc radius around B")
parser.add_argument("--spawn_h", type=float, nargs=2, default=[0.60, 0.75])
parser.add_argument("--quality", choices=["fast", "high", "ptrace"], default="high")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--supersample", type=float, default=1.5)
parser.add_argument("--pt_subframes", type=int, default=12)
parser.add_argument("--crf", type=int, default=16)
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

ss = max(1.0, args.supersample)
rw = (round(args.width * ss) // 2) * 2
rh = (round(args.height * ss) // 2) * 2

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False
cfg.render_camera = True
cfg.render_cam_w, cfg.render_cam_h = rw, rh
# the place stage exactly as trained: two platforms, deposit is success, real friction cage,
# every episode starts already holding the cube near B
cfg.place_only = cfg.two_platform = cfg.release_only = True
cfg.grasp_latch = False
cfg.carry_demo_p = 1.0
cfg.curriculum_p_start = cfg.curriculum_p_end = 0.0
cfg.place_curriculum = False          # fixed spawn spec, no easy rungs
cfg.place_spawn_r = args.spawn_r
cfg.place_spawn_h_lo, cfg.place_spawn_h_hi = args.spawn_h
cfg.plat_sep = cfg.plat_sep_max = 1.5
cfg.episode_length_s = 10.0
cfg.seed = args.seed
cfg.scene.env_spacing = max(cfg.scene.env_spacing, 2.0 * (cfg.plat_sep_max + cfg.place_spawn_r) + 2.0)

if args.quality == "high":
    cfg.sim.render = RenderCfg(enable_ambient_occlusion=True, enable_shadows=True,
                               enable_dl_denoiser=True, samples_per_pixel=2)
elif args.quality == "ptrace":
    cfg.sim.render = RenderCfg(enable_dl_denoiser=True, carb_settings={
        "/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 8,
        "/rtx/pathtracing/totalSpp": 256, "/rtx/pathtracing/maxBounces": 6,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 6,
        "/rtx/pathtracing/optixDenoiser/enabled": 1})

cfg.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.85, 0.12, 0.12), roughness=0.45, metallic=0.0)
cfg.platform.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.42, 0.30, 0.20), roughness=0.75, metallic=0.0)
# pad B in a distinct green so "where it is meant to go" reads instantly on screen
cfg.plat_b.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.18, 0.45, 0.24), roughness=0.70, metallic=0.0)
cfg.scene.num_envs = 1

env = DroneSnatchEnv(cfg); wenv = RslRlVecEnvWrapper(env)
print(f"[place-render] quality={args.quality} render={rw}x{rh} -> out={args.width}x{args.height}")

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
print(f"[place-render] loaded {args.checkpoint}")

_o = wenv.get_observations()
obs = _o[0] if isinstance(_o, tuple) else _o
cam = env._render_cam
off_eye = torch.tensor([1.55, 1.55, 0.62], device=env.device)   # 3/4 view, above the pad line
def follow():
    dp = env.robot.data.root_pos_w[0]
    b = env.plat_b.data.root_pos_w[0]
    tgt = 0.5 * (dp + b)                     # keep drone AND destination pad in frame
    cam.set_world_poses_from_view((tgt + off_eye).unsqueeze(0), tgt.unsqueeze(0))
follow()

pt_sub = args.pt_subframes if args.quality == "ptrace" else 1
tmp = tempfile.mkdtemp(prefix="place_"); nf = 0
dep_step = rel_step = -1
for i in range(args.steps):
    with torch.no_grad():
        act = policy(obs)
    _s = wenv.step(act); obs = _s[0]
    if rel_step < 0 and bool(env._released[0]):
        rel_step = i
    if dep_step < 0 and bool(env._deposited[0]):
        dep_step = i
    follow()
    for _ in range(pt_sub):
        cam.update(env.sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), rgb); nf += 1

pad = env.cfg.surface_z
print(f"[place-render] captured {nf} frames")
print(f"[place-render] jaws opened at step  {rel_step}"
      f"   ({rel_step / 50.0:.2f}s of sim time)" if rel_step >= 0 else
      "[place-render] jaws never opened")
if dep_step >= 0:
    print(f"[place-render] DEPOSITED at step   {dep_step}  ({dep_step / 50.0:.2f}s)")
    print(f"[place-render] contact speed       {float(env._impact_v[0]):.3f} m/s   [8mm-fall floor ~0.40]")
    print(f"[place-render] landing error       {float(env._d_plat_b[0]) * 100:.1f} cm  "
          f"[gate <{env.cfg.place_radius * 100:.0f}]")
    print("[place-render] TAKE_OK")
else:
    print("[place-render] NO DEPOSIT in this take -- rerun with a different --seed")

vf = (f"scale={args.width}:{args.height}:flags=lanczos"
      if (rw, rh) != (args.width, args.height) else "null")
cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-profile:v", "high",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", str(args.crf), args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-800:])
print(f"[place-render] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
print("PLACE_RENDER_DONE")
# app.close() reliably hangs on this host (see render_snatch.py) -- hard-exit instead.
os._exit(0)
