"""Gaussian-splat reconstruction, ported into the Isaac project.

GaussianMap is sim-agnostic — it consumes (rgb, depth, pose_c2w, K). The only
'port' needed is an adapter from an Isaac Lab Camera sensor to those tensors;
see isaac_camera.frame_from_camera. (Requires gsplat in the `isaac` env:
`pip install gsplat` — builds against torch 2.12+cu13.)
"""
from .gaussian_map import GaussianMap

__all__ = ["GaussianMap"]
