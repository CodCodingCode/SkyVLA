"""Incremental 3D Gaussian map for indoor UAV navigation/reconstruction.

A thin, training-friendly wrapper around ``gsplat`` rasterization. The map is
an explicit set of 3D Gaussians (mean, rotation quaternion, scale, opacity,
RGB). Two operations matter for navigation, and both are cheap (a forward
rasterize — milliseconds on an A100):

  * ``render(pose, K)`` — the photometric view from a camera (RGB + alpha).
  * coverage / visibility — "what does this camera see," from the per-pixel
    accumulated alpha and the Gaussians that contribute.

The expensive part of Gaussian Splatting is *fitting* (gradient descent over
many images). We deliberately avoid fitting in any inner loop: the map grows by
*splatting in* the Gaussians implied by a posed RGB-D frame (``add_from_rgbd``),
a closed-form back-projection. This is what makes a GS-derived navigation
reward affordable.

Conventions
-----------
* Poses are camera-to-world 4x4 (OpenCV: +x right, +y down, +z forward).
  gsplat consumes world-to-camera, so we invert internally.
* Intrinsics ``K`` are 3x3. All tensors live on ``device`` in float32.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

try:
    import gsplat
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "indoor_uav.gs requires gsplat (`pip install gsplat`)."
    ) from exc


def _quat_identity(n: int, device: torch.device) -> torch.Tensor:
    """(n,4) identity quaternions in gsplat's (w,x,y,z) order."""
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q


@dataclass
class GaussianMap:
    """A growable set of 3D Gaussians, renderable via gsplat."""

    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    means: torch.Tensor = field(init=False)
    quats: torch.Tensor = field(init=False)
    scales: torch.Tensor = field(init=False)
    opacities: torch.Tensor = field(init=False)
    colors: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        d = self.device
        self.means = torch.empty(0, 3, device=d)
        self.quats = torch.empty(0, 4, device=d)
        self.scales = torch.empty(0, 3, device=d)
        self.opacities = torch.empty(0, device=d)
        self.colors = torch.empty(0, 3, device=d)

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    # ------------------------------------------------------------------ #
    # Serialization. The map is just five tensors; persisting them lets a
    # caller skip the (expensive) RGB-D fusion that built it and reload the
    # finished splat in milliseconds. See gs/cache.py for the scene bundle
    # (these tensors + intrinsics) used by the render scripts.
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        """A CPU snapshot of the splat — the five Gaussian tensors."""
        return {
            "means": self.means.detach().cpu(),
            "quats": self.quats.detach().cpu(),
            "scales": self.scales.detach().cpu(),
            "opacities": self.opacities.detach().cpu(),
            "colors": self.colors.detach().cpu(),
        }

    def load_state_dict(self, sd: dict) -> "GaussianMap":
        """Replace this map's Gaussians from a ``state_dict`` (in place)."""
        d = self.device
        self.means = sd["means"].to(d).float()
        self.quats = sd["quats"].to(d).float()
        self.scales = sd["scales"].to(d).float()
        self.opacities = sd["opacities"].to(d).float()
        self.colors = sd["colors"].to(d).float()
        return self

    @classmethod
    def from_state_dict(cls, sd: dict, device) -> "GaussianMap":
        """Build a map directly from a ``state_dict`` on ``device``."""
        return cls(device=torch.device(device)).load_state_dict(sd)

    def add(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        *,
        scale: float = 0.03,
        opacity: float = 0.9,
    ) -> None:
        """Append isotropic Gaussians at ``means`` (N,3) coloured by ``colors`` (N,3)."""
        n = means.shape[0]
        if n == 0:
            return
        d = self.device
        means = means.to(d).float()
        colors = colors.to(d).float()
        self.means = torch.cat([self.means, means], dim=0)
        self.quats = torch.cat([self.quats, _quat_identity(n, d)], dim=0)
        self.scales = torch.cat(
            [self.scales, torch.full((n, 3), float(scale), device=d)], dim=0
        )
        self.opacities = torch.cat(
            [self.opacities, torch.full((n,), float(opacity), device=d)], dim=0
        )
        self.colors = torch.cat([self.colors, colors], dim=0)

    @torch.no_grad()
    def add_from_rgbd(
        self,
        rgb: torch.Tensor,        # (H,W,3) float [0,1] or uint8
        depth: torch.Tensor,      # (H,W) metric depth, 0 = invalid
        pose_c2w: torch.Tensor,   # (4,4) camera-to-world
        K: torch.Tensor,          # (3,3) intrinsics
        *,
        stride: int = 4,
        max_depth: float = 10.0,
        scale: float = 0.03,
    ) -> int:
        """Back-project a posed RGB-D frame into world-space Gaussians.

        Closed-form (no optimization): each valid, subsampled pixel becomes one
        Gaussian at its back-projected 3D point, coloured by the pixel. Returns
        the number of Gaussians added.
        """
        d = self.device
        rgb = rgb.to(d).float()
        if rgb.max() > 1.5:  # uint8-ish
            rgb = rgb / 255.0
        depth = depth.to(d).float()
        H, W = depth.shape
        ys = torch.arange(0, H, stride, device=d)
        xs = torch.arange(0, W, stride, device=d)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        z = depth[gy, gx]
        valid = (z > 1e-3) & (z < max_depth)
        if int(valid.sum().item()) == 0:
            return 0
        gxv, gyv, zv = gx[valid].float(), gy[valid].float(), z[valid]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (gxv - cx) / fx * zv
        y = (gyv - cy) / fy * zv
        pts_cam = torch.stack([x, y, zv, torch.ones_like(zv)], dim=-1)  # (M,4)
        pts_world = (pose_c2w.to(d).float() @ pts_cam.T).T[:, :3]
        cols = rgb[gyv.long(), gxv.long()]
        self.add(pts_world, cols, scale=scale)
        return int(valid.sum().item())

    @staticmethod
    def _viewmat(pose_c2w: torch.Tensor) -> torch.Tensor:
        """camera-to-world -> world-to-camera (gsplat wants w2c)."""
        return torch.linalg.inv(pose_c2w)

    @torch.no_grad()
    def render(
        self,
        pose_c2w: torch.Tensor,   # (4,4)
        K: torch.Tensor,          # (3,3)
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize the map. Returns (rgb (H,W,3), alpha (H,W,1)).

        ``alpha`` is the accumulated opacity per pixel — how much of the view is
        already "covered" by the map. Empty map -> zeros.
        """
        d = self.device
        if self.num_gaussians == 0:
            return (
                torch.zeros(height, width, 3, device=d),
                torch.zeros(height, width, 1, device=d),
            )
        viewmat = self._viewmat(pose_c2w.to(d).float())[None]
        out, alpha, _ = gsplat.rasterization(
            self.means, self.quats, self.scales, self.opacities, self.colors,
            viewmat, K.to(d).float()[None], width, height,
        )
        return out[0], alpha[0]
