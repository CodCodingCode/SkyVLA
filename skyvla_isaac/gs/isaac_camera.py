"""Adapter: Isaac Lab Camera sensor -> (rgb, depth, pose_c2w, K) for GaussianMap.

The GaussianMap reconstruction is sim-agnostic; this is the entire "port" from
Habitat to Isaac — feed it Isaac camera output instead of habitat_sim output.

Isaac Lab's `Camera` / `TiledCamera` sensor exposes, per env:
  * data.output["rgb"]            (N, H, W, 3) uint8
  * data.output["distance_to_image_plane"]  (N, H, W, 1) metric depth (planar)
  * data.pos_w, data.quat_w_world  camera pose in world (ROS/USD convention)
  * data.intrinsic_matrices        (N, 3, 3)

We convert the camera pose to an OpenCV camera-to-world (the convention
GaussianMap.add_from_rgbd / render expect: +x right, +y down, +z forward).
"""
from __future__ import annotations

import torch


def _quat_to_R(q):  # q = (w, x, y, z), Isaac convention
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


# USD/ROS camera (x right, y up, -z forward) -> OpenCV (x right, y down, z fwd):
# flip Y and Z. Same idea as the Habitat _HAB2CV fix.
_USD2CV = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])


def frame_from_camera(camera, env_index: int = 0):
    """Return (rgb (H,W,3) float[0,1], depth (H,W), pose_c2w (4,4), K (3,3)).
    `camera` is an Isaac Lab Camera sensor; pass the per-env index."""
    dev = camera.data.pos_w.device
    rgb = camera.data.output["rgb"][env_index][..., :3].float() / 255.0
    depth = camera.data.output["distance_to_image_plane"][env_index, ..., 0].float()
    K = camera.data.intrinsic_matrices[env_index].to(dev).float()

    R_usd = _quat_to_R(camera.data.quat_w_world[env_index].to(dev).float())
    R_cv = R_usd @ _USD2CV.to(dev)
    pose = torch.eye(4, device=dev)
    pose[:3, :3] = R_cv
    pose[:3, 3] = camera.data.pos_w[env_index].to(dev).float()
    return rgb, depth, pose, K
