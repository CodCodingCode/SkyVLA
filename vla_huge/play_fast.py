"""Offline playback in the HUGE-Bench scaffolded scene — no RTX, no Vulkan.

Runs the Stage-2 frozen waypoint controller against a HUGE-style target
inside ``VLAHugeDroneEnv`` with ``physics_only=True`` so no RGB cameras
are spawned and Isaac never touches Vulkan. Drone position / target /
HUGE prompt are logged every step, then stitched into a matplotlib MP4
(top-down + altitude side panel).

Until the HUGE Isaac Sim toolchain is released, every scene resolves to
``warehouse_full.usd``; the prompt overlay still says HUGE so the video
faithfully shows what the Stage-7 smoke path looks like end-to-end.

Launch:
    cd /home/ubuntu/IsaacLab
    xvfb-run -a ./isaaclab.sh -p /home/ubuntu/drone_project/vla_huge/play_fast.py \\
        --scene_id 1_office --duration_s 5 \\
        --output /home/ubuntu/drone_project/videos/vla_huge_smoke.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("OMNI_LOG_LEVEL", "ERROR")

from isaaclab.app import AppLauncher

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(description="Fast offline playback in vla_huge env.")
parser.add_argument("--scene_id", type=str, default="1_office",
                    choices=["1_office", "2_park", "3_campus", "4_lake"])
parser.add_argument("--duration_s", type=float, default=5.0,
                    help="Wallclock seconds of footage to render.")
parser.add_argument("--target_forward_m", type=float, default=12.0,
                    help="Target offset along +X body of spawn pose (m).")
parser.add_argument("--target_up_m", type=float, default=4.0,
                    help="Target offset along +Z world of spawn pose (m).")
parser.add_argument("--target_right_m", type=float, default=3.0,
                    help="Target offset along +Y body of spawn pose (m).")
parser.add_argument("--target_range", type=float, default=3.0,
                    help="Tanh cap for body-frame target (must match the HierarchicalVLAActor).")
parser.add_argument("--output", type=str,
                    default=os.path.join(_DRONE_PROJECT, "videos", "vla_huge_smoke.mp4"))
parser.add_argument("--checkpoint", type=str,
                    default=os.path.join(_DRONE_PROJECT, "checkpoints", "stage2_waypoint.pt"))
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force physics-only; matches the rendering capability of this host.
args_cli.headless = True
if getattr(args_cli, "enable_cameras", None):
    args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on AppLauncher being live.
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla_huge  # noqa: F401, E402  registers Isaac-VLADrone-Huge-v0
from vla_huge.scenes import get_scene  # noqa: E402
from vla_huge.vla_huge_env import VLAHugeDroneEnvCfg  # noqa: E402


# ---------------------------------------------------------------------------
# Stage-2 waypoint inference (lightweight, no RSL-RL runner needed)
# ---------------------------------------------------------------------------

class Stage2Waypoint:
    """Frozen Stage-2 waypoint MLP loaded from a .pt checkpoint."""

    def __init__(self, ckpt_path: str, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["actor_state_dict"]
        self.w0, self.b0 = sd["mlp.0.weight"].to(device), sd["mlp.0.bias"].to(device)
        self.w1, self.b1 = sd["mlp.2.weight"].to(device), sd["mlp.2.bias"].to(device)
        self.w2, self.b2 = sd["mlp.4.weight"].to(device), sd["mlp.4.bias"].to(device)
        self.mean = sd["obs_normalizer._mean"].to(device)
        self.std = sd["obs_normalizer._std"].to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, flight_state9: torch.Tensor, target_body3: torch.Tensor,
                 pos_error_w3: torch.Tensor) -> torch.Tensor:
        obs15 = torch.cat([flight_state9, target_body3, pos_error_w3], dim=-1)
        x = (obs15 - self.mean) / (self.std + 1e-8)
        x = F.elu(F.linear(x, self.w0, self.b0))
        x = F.elu(F.linear(x, self.w1, self.b1))
        return F.linear(x, self.w2, self.b2)


# ---------------------------------------------------------------------------
# Quaternion helpers (Isaac uses w,x,y,z)
# ---------------------------------------------------------------------------

def _quat_rotate(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply q to v (body -> world for an Isaac root_quat_w)."""
    w = q_wxyz[..., 0:1]
    xyz = q_wxyz[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_rotate_inverse(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply q^-1 to v (world -> body for an Isaac root_quat_w)."""
    w = q_wxyz[..., 0:1]
    xyz = q_wxyz[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

def _make_video(drone_xyz: np.ndarray, target_xyz: np.ndarray,
                instruction: str, scene_id: str, fps: int, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(11, 5.5), facecolor="#0d0d0d")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    for ax in (ax_top, ax_side):
        ax.set_facecolor("#181818")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    pad = 1.5
    xmin = float(min(drone_xyz[:, 0].min(), target_xyz[:, 0].min()) - pad)
    xmax = float(max(drone_xyz[:, 0].max(), target_xyz[:, 0].max()) + pad)
    ymin = float(min(drone_xyz[:, 1].min(), target_xyz[:, 1].min()) - pad)
    ymax = float(max(drone_xyz[:, 1].max(), target_xyz[:, 1].max()) + pad)
    zmin = float(min(drone_xyz[:, 2].min(), target_xyz[:, 2].min()) - pad)
    zmax = float(max(drone_xyz[:, 2].max(), target_xyz[:, 2].max()) + pad)

    ax_top.set_xlim(xmin, xmax)
    ax_top.set_ylim(ymin, ymax)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x (m)", color="#aaaaaa")
    ax_top.set_ylabel("y (m)", color="#aaaaaa")
    ax_top.set_title(
        f"vla_huge :: {scene_id}  (top-down)",
        color="#dddddd", fontsize=11,
    )
    ax_top.grid(True, color="#2a2a2a", lw=0.5)

    ax_side.set_xlim(0, drone_xyz.shape[0] / fps)
    ax_side.set_ylim(zmin, zmax)
    ax_side.set_xlabel("t (s)", color="#aaaaaa")
    ax_side.set_ylabel("altitude (m)", color="#aaaaaa")
    ax_side.set_title("Altitude", color="#dddddd", fontsize=11)
    ax_side.grid(True, color="#2a2a2a", lw=0.5)
    ax_side.axhline(target_xyz[0, 2], color="#ffb300", lw=1.0, alpha=0.8, label="target z")
    ax_side.legend(loc="lower right", facecolor="#181818", labelcolor="#dddddd",
                   edgecolor="#444444")

    (trail,) = ax_top.plot([], [], "-", color="#4ea3ff", lw=2.0, alpha=0.85)
    (drone_dot,) = ax_top.plot([], [], "o", color="#4ea3ff", ms=10,
                                markeredgecolor="#cce6ff", markeredgewidth=1.0)
    (target_dot,) = ax_top.plot([], [], "*", color="#ffb300", ms=14,
                                 markeredgecolor="#fff0b3", markeredgewidth=1.0)
    (alt_line,) = ax_side.plot([], [], "-", color="#4ea3ff", lw=1.6)

    fig.text(0.012, 0.965, f'"{instruction}"',
             color="#f0f0f0", fontsize=11, fontstyle="italic",
             bbox=dict(facecolor="#1f1f1f", edgecolor="#444444", pad=4))
    fig.text(0.012, 0.04,
             "Stage-7 smoke: warehouse_full placeholder USD until HUGE sim ships.\n"
             "Stage-2 waypoint policy steered by hardcoded HUGE-style body-frame target.",
             color="#999999", fontsize=8.5)

    times = np.arange(drone_xyz.shape[0]) / fps

    def update(frame: int):
        trail.set_data(drone_xyz[: frame + 1, 0], drone_xyz[: frame + 1, 1])
        drone_dot.set_data([drone_xyz[frame, 0]], [drone_xyz[frame, 1]])
        target_dot.set_data([target_xyz[frame, 0]], [target_xyz[frame, 1]])
        alt_line.set_data(times[: frame + 1], drone_xyz[: frame + 1, 2])
        return trail, drone_dot, target_dot, alt_line

    anim = FuncAnimation(fig, update, frames=drone_xyz.shape[0],
                         interval=1000.0 / fps, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(out_path, writer=writer, savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"[INFO] video saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] scene_id   = {args_cli.scene_id}")
    print(f"[INFO] checkpoint = {args_cli.checkpoint}")
    print(f"[INFO] output     = {args_cli.output}")

    huge_scene = get_scene(args_cli.scene_id)
    instruction = huge_scene.prompts[0]
    print(f"[INFO] instruction= {instruction!r}")

    env_cfg = VLAHugeDroneEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.clone_in_fabric = False
    env_cfg.scene_id = args_cli.scene_id
    env_cfg.physics_only = True
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    env = gym.make("Isaac-VLADrone-Huge-v0", cfg=env_cfg)
    env_impl = env.unwrapped
    device = torch.device(env_cfg.sim.device)

    # Warm a few zero-action steps so the drone settles after spawn.
    zero_action = torch.zeros((1, 4), device=device)
    for _ in range(10):
        env.step(zero_action)

    spawn_pos = env_impl._robot.data.root_pos_w[0].clone()  # (3,)
    spawn_quat = env_impl._robot.data.root_quat_w[0].clone()  # (4,) wxyz

    # Build a HUGE-style target: forward + right + up offsets in body frame
    # at spawn, projected back into world. Capped to target_range so the
    # waypoint MLP receives the same scale the high-level head emits.
    target_body = torch.tensor(
        [args_cli.target_forward_m, args_cli.target_right_m, args_cli.target_up_m],
        device=device, dtype=torch.float32,
    )
    target_body_capped = target_body.clone()
    norm = float(target_body_capped.norm())
    if norm > args_cli.target_range:
        target_body_capped = target_body_capped * (args_cli.target_range / norm)

    target_world = spawn_pos + _quat_rotate(spawn_quat.unsqueeze(0),
                                            target_body.unsqueeze(0)).squeeze(0)
    print(f"[INFO] spawn_pos    = {spawn_pos.cpu().numpy()}")
    print(f"[INFO] target_world = {target_world.cpu().numpy()}")
    print(f"[INFO] target_body  = {target_body.cpu().numpy()}  (capped -> {target_body_capped.cpu().numpy()})")

    waypoint = Stage2Waypoint(args_cli.checkpoint, device=device)

    # The env runs at sim_dt * decimation. For VLA the default is 50 Hz env step.
    env_step_hz = 1.0 / float(env_impl.cfg.sim.dt * env_impl.cfg.decimation)
    num_steps = int(round(args_cli.duration_s * env_step_hz))
    video_fps = int(round(env_step_hz))
    print(f"[INFO] env_step_hz={env_step_hz:.1f}  num_steps={num_steps}  video_fps={video_fps}")

    drone_xyz = np.zeros((num_steps, 3), dtype=np.float32)
    target_xyz = np.zeros((num_steps, 3), dtype=np.float32)
    target_xyz[:] = target_world.cpu().numpy()

    for step in range(num_steps):
        robot = env_impl._robot.data
        flight_state = torch.cat([
            robot.root_lin_vel_b[0],
            robot.root_ang_vel_b[0],
            robot.projected_gravity_b[0],
        ], dim=-1)  # (9,)

        cur_pos = robot.root_pos_w[0]
        cur_quat = robot.root_quat_w[0]
        pos_err_w = target_world - cur_pos
        pos_err_b = _quat_rotate_inverse(
            cur_quat.unsqueeze(0), pos_err_w.unsqueeze(0)
        ).squeeze(0)
        target_body_step = pos_err_b.clamp(-args_cli.target_range, args_cli.target_range)

        action = waypoint(
            flight_state.unsqueeze(0),
            target_body_step.unsqueeze(0),
            pos_err_w.unsqueeze(0),
        )
        env.step(action.clamp(-1.0, 1.0))

        drone_xyz[step] = cur_pos.cpu().numpy()

        if step % 25 == 0:
            d = float(pos_err_w.norm())
            print(f"[step {step:4d}] pos=({cur_pos[0]:+.2f},{cur_pos[1]:+.2f},{cur_pos[2]:+.2f}) "
                  f"dist_to_target={d:.2f}m")

    print("[INFO] rendering matplotlib video...")
    _make_video(drone_xyz, target_xyz, instruction=instruction,
                scene_id=args_cli.scene_id, fps=video_fps, out_path=args_cli.output)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
