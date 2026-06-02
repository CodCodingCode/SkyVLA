"""Lift a 2D bounding box to a 3D world goal using metric depth + camera pose.

This is the geometric core of "VLM box -> fly there". It mirrors the
back-projection in ``indoor_uav/gs/gaussian_map.py:add_from_rgbd`` and assumes
the SAME conventions:

* ``pose_c2w`` is an OpenCV camera-to-world (+x right, +y down, +z forward),
  the corrected one (det = +1) — see the diag(1,-1,-1) flip fix.
* ``K`` is a 3x3 pinhole intrinsic in pixels.

Two outcomes:
* **point** — the box has valid depth: we get an exact 3D location and a goal
  ``standoff`` metres short of it (so the drone stops in front, not inside it).
* **bearing** — the box is beyond depth range (e.g. a building 80 m away): we
  only know the *direction*. Fly toward it and re-detect as you close in.
"""
from __future__ import annotations

import numpy as np


def _depth_in_box(depth: np.ndarray, box, max_depth: float):
    """Robust depth for a box: the 30th percentile of valid in-range pixels.
    Objects tend to be the NEARER cluster inside their box, so a low percentile
    avoids latching onto background seen past the object's edges."""
    H, W = depth.shape
    x0, y0, x1, y1 = box
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W, int(round(x1))); y1 = min(H, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = depth[y0:y1, x0:x1]
    v = crop[(crop > 1e-3) & (crop < max_depth)]
    if v.size < 8:
        return None
    return float(np.percentile(v, 30))


def bbox_to_world(box, depth, pose_c2w, K, *, max_depth: float = 10.0,
                  standoff: float = 1.5) -> dict:
    """Map a pixel box (x0,y0,x1,y1) to a world goal. See module docstring."""
    depth = np.asarray(depth, np.float32)
    H, W = depth.shape
    x0, y0, x1, y1 = box
    u = 0.5 * (x0 + x1); v = 0.5 * (y0 + y1)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pose = np.asarray(pose_c2w, np.float32)

    # viewing ray for the box centre, in world frame (always available)
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], np.float32)
    ray_world = pose[:3, :3] @ ray_cam
    ray_world /= (np.linalg.norm(ray_world) + 1e-8)

    d = _depth_in_box(depth, (x0, y0, x1, y1), max_depth)
    if d is None:
        return {"kind": "bearing", "dir": ray_world, "uv": (u, v)}

    p_cam = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d, 1.0], np.float32)
    p_world = (pose @ p_cam)[:3]
    goal = p_world - ray_world * standoff           # stop `standoff` m in front
    return {"kind": "point", "point": p_world, "goal": goal,
            "depth": d, "dir": ray_world, "uv": (u, v)}
