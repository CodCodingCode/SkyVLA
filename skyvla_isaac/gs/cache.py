"""Disk cache for the Gaussian-splat *scene* (and an optional rendered rollout).

Why this exists
---------------
Rebuilding the GS room in ``render_rollout_gs.py`` is the slowest, most-repeated
step of testing: a ~44-view RGB-D orbit (88 ``sim.step(render=True)`` calls) plus
fusion, on *every* run. Yet the room is STATIC geometry — identical every run,
independent of the policy/checkpoint. So we build it once and reload it in
milliseconds.

Two artifacts
-------------
* **scene cache** (``.pt``)  — the ``GaussianMap`` (five tensors) + intrinsics
  ``K`` + the render width/height it was captured at. Built once *with* Isaac;
  reused by every later render. This is "the rendering scene."
* **rollout cache** (``.npz``) — per-frame foreground (drone+cube) RGBA + the
  follow-camera pose, for one checkpoint. Lets ``render_gs_cache.py`` re-composite
  and re-render the *whole video with NO Isaac at all*, so iterating on the
  camera path / render look takes seconds. Foreground pixels outside the
  drone+cube mask are zeroed, so ``np.savez_compressed`` crushes the file
  (mostly zeros) despite the nominal RGBA size.

Files live under ``skyvla_isaac/gs/cache/`` (gitignored). Writes are atomic
(tmp + ``os.replace``) so an interrupted run never leaves a half-written cache.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .gaussian_map import GaussianMap

# Bump when the on-disk layout changes so stale caches are rejected, not
# silently mis-read.
SCENE_FORMAT = 1
ROLLOUT_FORMAT = 1

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DEFAULT_SCENE = os.path.join(CACHE_DIR, "room_splat.pt")


def default_rollout_path(checkpoint: str) -> str:
    """Cache path for a rollout, keyed on the checkpoint file name."""
    stem = os.path.splitext(os.path.basename(checkpoint))[0] or "rollout"
    return os.path.join(CACHE_DIR, f"rollout_{stem}.npz")


def _atomic(path: str, write_fn) -> str:
    """Write via ``path + '.tmp'`` then atomically rename into place."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# Scene cache: the splat + intrinsics + render resolution.
# --------------------------------------------------------------------------- #
def save_scene(path: str, gmap: GaussianMap, K: torch.Tensor,
               width: int, height: int, meta: dict | None = None) -> str:
    """Persist a GaussianMap scene (splat + ``K`` + ``width``/``height``)."""
    blob = {
        "format": SCENE_FORMAT,
        "splat": gmap.state_dict(),
        "K": K.detach().cpu().float(),
        "width": int(width),
        "height": int(height),
        "num_gaussians": int(gmap.num_gaussians),
        "meta": dict(meta or {}),
    }
    return _atomic(path, lambda f: torch.save(blob, f))


def load_scene(path: str, device="cuda"):
    """Load a scene cache -> ``(gmap, K, width, height, meta)``.

    Raises ``ValueError`` on a format mismatch (stale cache) so the caller can
    rebuild rather than render garbage.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    fmt = blob.get("format")
    if fmt != SCENE_FORMAT:
        raise ValueError(
            f"scene cache {path} is format {fmt}, expected {SCENE_FORMAT} — "
            "rebuild it (e.g. --rebuild_gs)."
        )
    dev = torch.device(device)
    gmap = GaussianMap.from_state_dict(blob["splat"], dev)
    K = blob["K"].to(dev).float()
    return gmap, K, int(blob["width"]), int(blob["height"]), blob.get("meta", {})


# --------------------------------------------------------------------------- #
# Rollout cache: foreground RGBA + follow-cam pose per frame, for one checkpoint.
# --------------------------------------------------------------------------- #
def save_rollout(path: str, fg_rgba, poses, fps: int,
                 width: int, height: int, meta: dict | None = None) -> str:
    """Persist a captured rollout for Isaac-free re-rendering.

    ``fg_rgba``: (N,H,W,4) uint8 — drone+cube RGB with the foreground mask in
    alpha; background pixels zeroed (compresses to near nothing).
    ``poses``: (N,4,4) OpenCV camera-to-world per frame.
    """
    fg = fg_rgba.detach().cpu().numpy() if torch.is_tensor(fg_rgba) else np.asarray(fg_rgba)
    ps = poses.detach().cpu().numpy() if torch.is_tensor(poses) else np.asarray(poses)
    meta_keys = np.array(list((meta or {}).keys()))
    meta_vals = np.array([str(v) for v in (meta or {}).values()])

    def _w(f):
        np.savez_compressed(
            f,
            format=np.int64(ROLLOUT_FORMAT),
            fg_rgba=fg.astype(np.uint8),
            poses=ps.astype(np.float32),
            fps=np.int64(fps),
            width=np.int64(width),
            height=np.int64(height),
            meta_keys=meta_keys,
            meta_vals=meta_vals,
        )

    return _atomic(path, _w)


def load_rollout(path: str, device="cuda"):
    """Load a rollout cache -> ``(fg_rgba, poses, fps, width, height, meta)``."""
    z = np.load(path, allow_pickle=False)
    fmt = int(z["format"]) if "format" in z else None
    if fmt != ROLLOUT_FORMAT:
        raise ValueError(
            f"rollout cache {path} is format {fmt}, expected {ROLLOUT_FORMAT} — "
            "recapture it (drop --reuse_rollout)."
        )
    dev = torch.device(device)
    fg = torch.from_numpy(z["fg_rgba"]).to(dev)
    poses = torch.from_numpy(z["poses"]).to(dev).float()
    meta = dict(zip(z["meta_keys"].tolist(), z["meta_vals"].tolist())) \
        if "meta_keys" in z else {}
    return fg, poses, int(z["fps"]), int(z["width"]), int(z["height"]), meta
