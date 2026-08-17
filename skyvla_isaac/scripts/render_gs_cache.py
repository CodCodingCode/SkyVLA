"""Render the cached Gaussian-splat scene FAST — with NO Isaac Sim.

``render_rollout_gs.py`` is the heavy producer: it boots Isaac, builds the splat
room, and (optionally) caches the room + the rollout. This script is the light
consumer: it loads those caches and renders straight from the splat, so iterating
on the camera path / render look / compositing takes *seconds* instead of an Isaac
boot + 44-view orbit.

Two modes
---------
* ORBIT (default) — fly a ring around the room and render the bare splat. Great
  for sanity-checking the cached scene and tuning render params.

      # any interpreter with torch + gsplat
      .venv/bin/python skyvla_isaac/scripts/render_gs_cache.py --out videos/gs_orbit.mp4

* REPLAY (``--rollout``) — re-composite a captured rollout (drone+cube foreground
  saved by ``render_rollout_gs.py --save_rollout``) over the splat backdrop, with
  no physics and no Isaac. Change the fill colour / coverage threshold / fps and
  re-render in seconds.

      .venv/bin/python skyvla_isaac/scripts/render_gs_cache.py \
          --rollout skyvla_isaac/gs/cache/rollout_model_5999.npz \
          --out videos/replay.mp4

NOTE: gsplat JIT-compiles its CUDA kernels on first use and torch reads the .cu
sources with the locale codec, which crashes unless Python is UTF-8. If you hit a
UnicodeDecodeError, prefix the command with ``PYTHONUTF8=1``.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile

import imageio.v2 as imageio
import torch
import torch.nn.functional as F

from skyvla_isaac.gs import cache as gs_cache

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def look_at_c2w(eye, target, up=(0.0, 0.0, 1.0), device="cuda") -> torch.Tensor:
    """OpenCV camera-to-world (x right, y down, z forward) looking from eye->target."""
    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)
    f = target - eye
    f = f / f.norm().clamp_min(1e-8)            # forward (+z)
    s = torch.cross(f, up, dim=-1)
    s = s / s.norm().clamp_min(1e-8)            # right (+x)
    d = torch.cross(f, s, dim=-1)               # down (+y)  (= -up_cam)
    pose = torch.eye(4, device=device)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = s, d, f
    pose[:3, 3] = eye
    return pose


def encode(frame_dir: str, n: int, fps: int, out: str) -> None:
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(frame_dir, "f%05d.png"),
           "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-crf", "21", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg error:\n" + r.stderr[-800:])
    else:
        print(f"[gs-cache] DONE -> {out} ({os.path.getsize(out) / 1e6:.1f} MB, {n} frames)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None,
                   help="scene-cache path (default skyvla_isaac/gs/cache/room_splat.pt).")
    p.add_argument("--rollout", default=None,
                   help="rollout-cache .npz to replay + composite (else render an orbit).")
    p.add_argument("--out", default=os.path.join(_REPO, "videos/gs_cache.mp4"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=30)
    # orbit-mode camera path
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--radius", type=float, default=2.6)
    p.add_argument("--height", type=float, default=1.6)
    p.add_argument("--look_z", type=float, default=1.2)
    # compositing knobs (shared)
    p.add_argument("--cover_thresh", type=float, default=0.3)
    p.add_argument("--fill", type=float, nargs=3, default=[0.72, 0.69, 0.63])
    args = p.parse_args()

    dev = torch.device(args.device)
    scene_path = args.scene or gs_cache.DEFAULT_SCENE
    if not os.path.exists(scene_path):
        raise SystemExit(
            f"no scene cache at {scene_path} — build one first:\n"
            "  .venv/bin/python skyvla_isaac/scripts/render_rollout_gs.py --checkpoint <ckpt> --save_rollout"
        )
    gmap, K, W, H, meta = gs_cache.load_scene(scene_path, device=dev)
    print(f"[gs-cache] scene {scene_path}: {gmap.num_gaussians} gaussians, "
          f"render {W}x{H}{' meta=' + str(meta) if meta else ''}")
    fill = torch.tensor(args.fill, device=dev)

    tmp = tempfile.mkdtemp(prefix="gs_cache_")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.rollout:
        fg_rgba, poses, fps, Wr, Hr, rmeta = gs_cache.load_rollout(args.rollout, device=dev)
        resize_fg = (Wr, Hr) != (W, H)
        if resize_fg:
            print(f"[gs-cache] WARN rollout size {(Wr, Hr)} != scene {(W, H)} — "
                  "resizing foreground to scene size (may misalign if aspect differs).")
        n = fg_rgba.shape[0]
        print(f"[gs-cache] replay {n} frames from {args.rollout}"
              f"{' meta=' + str(rmeta) if rmeta else ''}")
        for i in range(n):
            rgba = fg_rgba[i].float() / 255.0          # (Hr,Wr,4)
            if resize_fg:                               # -> (H,W,4) to match the GS render
                rgba = F.interpolate(rgba.permute(2, 0, 1)[None], size=(H, W),
                                     mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            fg_rgb, fg = rgba[..., :3], rgba[..., 3:4]
            gs_rgb, gs_alpha = gmap.render(poses[i], K, W, H)
            gs_rgb = gs_rgb.clamp(0, 1)
            covered = (gs_alpha > args.cover_thresh).float()
            backdrop = covered * gs_rgb + (1 - covered) * fill
            out = fg * fg_rgb + (1 - fg) * backdrop
            img = (out.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), img)
        encode(tmp, n, fps, args.out)
    else:
        n = args.steps
        print(f"[gs-cache] orbit {n} frames (r={args.radius}, h={args.height})")
        for i in range(n):
            a = 2 * math.pi * i / n
            eye = (args.radius * math.cos(a), args.radius * math.sin(a), args.height)
            pose = look_at_c2w(eye, (0.0, 0.0, args.look_z), device=dev)
            gs_rgb, gs_alpha = gmap.render(pose, K, W, H)
            covered = (gs_alpha > args.cover_thresh).float()
            backdrop = covered * gs_rgb.clamp(0, 1) + (1 - covered) * fill
            img = (backdrop.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            imageio.imwrite(os.path.join(tmp, f"f{i:05d}.png"), img)
        encode(tmp, n, args.fps, args.out)

    shutil.rmtree(tmp, ignore_errors=True)              # don't leak rendered frames on /tmp
    print("RENDER_DONE")


if __name__ == "__main__":
    main()
