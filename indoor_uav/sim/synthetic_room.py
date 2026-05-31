"""Dependency-free synthetic indoor room — real ray-cast RGB-D on the GPU.

A procedural room (floor, ceiling, four walls) + a few box "furniture" obstacles,
all solid axis-aligned boxes. RGB-D is produced by vectorized ray-AABB casting
from any 6-DOF pose, so it feeds the GS map and coverage reward exactly like a
real sim — no external assets, no display. This is the CI / bring-up backend;
HabitatRoom is the photorealistic drop-in with the same IndoorSim contract.
Faces are shaded by normal + checker so the GS map has texture to reconstruct.
"""

from __future__ import annotations

import torch

from .base import Frame, IndoorSim


class SyntheticRoom(IndoorSim):
    def __init__(
        self,
        size: tuple[float, float, float] = (8.0, 3.0, 6.0),  # Lx, Ly(height), Lz
        width: int = 256,
        height: int = 256,
        hfov_deg: float = 90.0,
        n_obstacles: int = 4,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width, self.height = width, height
        self.size = torch.tensor(size, device=self.device, dtype=torch.float32)
        f = 0.5 * width / torch.tan(torch.tensor(hfov_deg * 3.14159265 / 360.0))
        self._K = torch.tensor(
            [[float(f), 0, width / 2], [0, float(f), height / 2], [0, 0, 1]],
            device=self.device, dtype=torch.float32,
        )
        self._boxes = self._build_boxes(n_obstacles, seed)  # (B,2,3) min,max

    def _build_boxes(self, n_obstacles: int, seed: int) -> torch.Tensor:
        d = self.device
        Lx, Ly, Lz = [float(v) for v in self.size]
        t = 0.1
        boxes = [
            [[0, -t, 0], [Lx, 0, Lz]],          # floor
            [[0, Ly, 0], [Lx, Ly + t, Lz]],     # ceiling
            [[-t, 0, 0], [0, Ly, Lz]],          # wall -x
            [[Lx, 0, 0], [Lx + t, Ly, Lz]],     # wall +x
            [[0, 0, -t], [Lx, Ly, 0]],          # wall -z
            [[0, 0, Lz], [Lx, Ly, Lz + t]],     # wall +z
        ]
        g = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(n_obstacles):
            cx = (0.15 + 0.7 * torch.rand(1, generator=g).item()) * Lx
            cz = (0.15 + 0.7 * torch.rand(1, generator=g).item()) * Lz
            w = 0.4 + 0.6 * torch.rand(1, generator=g).item()
            h = 0.5 + 1.2 * torch.rand(1, generator=g).item()
            dd = 0.4 + 0.6 * torch.rand(1, generator=g).item()
            boxes.append([[cx - w, 0, cz - dd], [cx + w, h, cz + dd]])
        return torch.tensor(boxes, device=d, dtype=torch.float32)

    def intrinsics(self) -> torch.Tensor:
        return self._K

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor([0.3, 0.3, 0.3], device=self.device), self.size - 0.3)

    def is_free(self, xyz: torch.Tensor) -> bool:
        p = xyz.to(self.device).float().reshape(3)
        lo, hi = self.bounds()
        if bool((p < lo).any()) or bool((p > hi).any()):
            return False
        for b in self._boxes[6:]:  # obstacles only
            if bool((p >= b[0]).all()) and bool((p <= b[1]).all()):
                return False
        return True

    @torch.no_grad()
    def render(self, pose_c2w: torch.Tensor) -> Frame:
        d = self.device
        H, W = self.height, self.width
        pose = pose_c2w.to(d).float()
        K = self._K
        ys, xs = torch.meshgrid(
            torch.arange(H, device=d), torch.arange(W, device=d), indexing="ij"
        )
        x = (xs.float() - K[0, 2]) / K[0, 0]
        y = (ys.float() - K[1, 2]) / K[1, 1]
        dirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        R = pose[:3, :3]
        o = pose[:3, 3]
        rd = (dirs.reshape(-1, 3) @ R.T)
        ro = o.reshape(1, 3).expand_as(rd)
        depth, normal = self._ray_boxes(ro, rd)
        depth = depth.reshape(H, W)
        normal = normal.reshape(H, W, 3)
        hit_xyz = ro.reshape(H, W, 3) + rd.reshape(H, W, 3) * depth.unsqueeze(-1)
        rgb = self._shade(hit_xyz, normal, depth)
        return Frame(rgb=rgb, depth=depth, pose_c2w=pose, K=K)

    @torch.no_grad()
    def _ray_boxes(self, ro: torch.Tensor, rd: torch.Tensor):
        d = self.device
        P = ro.shape[0]
        eps = 1e-8
        inv = 1.0 / torch.where(rd.abs() < eps, torch.full_like(rd, eps), rd)
        best_t = torch.full((P,), float("inf"), device=d)
        best_n = torch.zeros(P, 3, device=d)
        arangeP = torch.arange(P, device=d)
        for b in self._boxes:
            t0 = (b[0] - ro) * inv
            t1 = (b[1] - ro) * inv
            tmin = torch.minimum(t0, t1)
            tmax = torch.maximum(t0, t1)
            t_near = tmin.max(dim=-1).values
            t_far = tmax.min(dim=-1).values
            hit = (t_far >= t_near.clamp(min=0)) & (t_near > 1e-3) & (t_near < best_t)
            if bool(hit.any()):
                best_t = torch.where(hit, t_near, best_t)
                axis = tmin.argmax(dim=-1)
                n = torch.zeros(P, 3, device=d)
                n[arangeP, axis] = -torch.sign(rd[arangeP, axis])
                best_n = torch.where(hit.unsqueeze(-1), n, best_n)
        depth = torch.where(torch.isfinite(best_t), best_t, torch.zeros_like(best_t))
        return depth, best_n

    @torch.no_grad()
    def _shade(self, hit_xyz, normal, depth):
        base = 0.5 + 0.5 * normal
        checker = ((hit_xyz * 1.5).floor().sum(dim=-1) % 2.0)
        tint = 0.75 + 0.25 * checker.unsqueeze(-1)
        rgb = (base * tint).clamp(0, 1)
        rgb[depth <= 0] = 0.0
        return rgb
