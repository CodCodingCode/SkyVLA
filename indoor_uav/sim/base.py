"""Sim abstraction for indoor UAV navigation.

Tasks and the GS reward depend ONLY on this interface, never on a concrete
engine — so the renderer (Habitat-Sim headless, or the synthetic ray-cast room)
is swappable. A backend supplies, for any 6-DOF drone pose: an RGB image, a
metric depth image, the intrinsics K, scene bounds, and a free-space query.
Pose convention: OpenCV camera-to-world (+x right, +y down, +z forward).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class Frame:
    """One rendered observation."""
    rgb: torch.Tensor       # (H,W,3) float [0,1]
    depth: torch.Tensor     # (H,W) metric depth, 0=invalid
    pose_c2w: torch.Tensor  # (4,4) camera-to-world, OpenCV convention
    K: torch.Tensor         # (3,3) intrinsics


class IndoorSim(ABC):
    """Minimal indoor-navigation sim contract."""

    width: int
    height: int
    device: torch.device

    @abstractmethod
    def intrinsics(self) -> torch.Tensor: ...

    @abstractmethod
    def render(self, pose_c2w: torch.Tensor) -> Frame: ...

    @abstractmethod
    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(min_xyz (3,), max_xyz (3,)) navigable scene extent."""

    @abstractmethod
    def is_free(self, xyz: torch.Tensor) -> bool:
        """True if a point is collision-free (flyable)."""

    def reset(self) -> None:
        return None
