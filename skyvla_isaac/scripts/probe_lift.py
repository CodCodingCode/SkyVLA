"""MECHANICAL PROBE: can the (wide) floored-scoop cage physically lift the cube?

No policy. Spawn straddling the cube (cur_p=1), scripted control:
  phase A (1s): hover in place, gripper OPEN  -> settle
  phase B (1s): gripper CLOSED, hold altitude -> press
  phase C (3s): gripper CLOSED, full ascent   -> does the cube come?

Reports cube height vs drone climb at several horizontal offsets (cube is centred
at spawn; we also probe with a deliberate sideways nudge to test off-centre grip).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app = AppLauncher(args).app

import torch  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import DroneSnatchEnv, DroneSnatchEnvCfg  # noqa: E402

cfg = DroneSnatchEnvCfg()
cfg.use_cameras = False
cfg.grasp_latch = False                  # REAL PHYSICS
cfg.curriculum_p_start = cfg.curriculum_p_end = 1.0   # all straddle spawns
cfg.carry_demo_p = 1.0                   # ALL envs spawn shelf-seated airborne (demo test)
cfg.scene.num_envs = args.num_envs
env = DroneSnatchEnv(cfg, render_mode=None)
env.reset()

N = env.num_envs
dev = env.device
act = torch.zeros(N, 5, device=dev)

def step(n, vz, grip):
    act[:, :] = 0.0
    act[:, 2] = vz
    act[:, 4] = grip
    for _ in range(n):
        env.step(act)

def cube_h():
    rest = cfg.surface_z + 0.5 * cfg.cube_size
    return (env._obj_p[:, 2] - rest).clamp(min=0)

def report(tag):
    h = cube_h()
    d = env._base_p[:, 2]
    print(f"[probe] {tag}: cube_lift mean={h.mean()*100:.2f}cm max={h.max()*100:.2f}cm "
          f"frac>2cm={(h > 0.02).float().mean():.2f} drone_z mean={d.mean():.3f}m", flush=True)

def variant(tag, close_fn, vz):
    env.reset()
    step(50, 0.0, -1.0)
    close_fn()
    step(25, 0.0, 1.0)                   # settle the press
    step(50, vz, 1.0)
    h = cube_h()
    print(f"[probe] {tag}: carry_frac={(h > 0.02).float().mean():.2f} "
          f"lift mean={h.mean()*100:.1f}cm drone_z={env._base_p[:,2].mean():.2f}m", flush=True)

def snap():            # close at full speed (baseline)
    step(25, 0.0, 1.0)

def ramp():            # close over ~1.5s
    for g in torch.linspace(-1, 1, 75):
        step(1, 0.0, float(g))

def demo_retention(tag):
    env.reset()                          # all envs: airborne, shelf-seated, jaws 0.026
    for k, n_steps in (("0.5s", 25), ("1.5s", 50), ("4.5s", 150)):
        step(n_steps, 0.0, 1.0)          # hover, squeeze
        h = cube_h()
        held = (h > 0.05).float().mean()
        print(f"[probe] {tag} @{k}: held_frac={held:.2f} cube_lift mean={h.mean()*100:.1f}cm "
              f"d_reach={env._d_reach.mean()*100:.1f}cm", flush=True)

demo_retention("DEMO shelf-seated hover")
print("PROBE_DONE", flush=True)
