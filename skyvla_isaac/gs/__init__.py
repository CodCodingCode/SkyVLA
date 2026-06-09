"""Gaussian-splat reconstruction, ported into the Isaac project.

GaussianMap is sim-agnostic — it consumes (rgb, depth, pose_c2w, K). The only
'port' needed is an adapter from an Isaac Lab Camera sensor to those tensors;
see isaac_camera.frame_from_camera. (Requires gsplat in the `isaac` env:
`pip install gsplat` — builds against torch 2.12+cu13.)
"""
from .gaussian_map import GaussianMap
from . import cache
from .cache import (
    DEFAULT_SCENE,
    default_rollout_path,
    load_rollout,
    load_scene,
    save_rollout,
    save_scene,
)

__all__ = [
    "GaussianMap",
    "cache",
    "DEFAULT_SCENE",
    "default_rollout_path",
    "save_scene",
    "load_scene",
    "save_rollout",
    "load_rollout",
]
