#!/usr/bin/env python
"""End-to-end smoke: indoor sim -> GS map -> coverage reward -> NBV (synthetic).

Proves the full indoor-UAV-with-gsplat loop on one GPU, no external assets:
  1. SyntheticRoom renders RGB-D from a drone pose.
  2. Frames are splatted into a GaussianMap (incremental, no fitting).
  3. As the drone patrols, the map grows and coverage accumulates.
  4. exploration_gain is higher at a less-seen pose than a well-seen one.
  5. best_next_view (analytic NBV) runs over candidates.
"""
from __future__ import annotations

import sys

import torch

from indoor_uav.sim import SyntheticRoom
from indoor_uav.gs import GaussianMap, view_coverage, exploration_gain, best_next_view


def _yaw_pose(x, z, yaw, y, device):
    """Camera-to-world for a drone at (x,y,z) looking along yaw about the up axis."""
    c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device).float()
    flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float()
    T = torch.eye(4, device=device)
    T[:3, :3] = R @ flip
    T[:3, 3] = torch.tensor([x, y, z], device=device).float()
    return T


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = SyntheticRoom(width=200, height=200, device=dev, seed=1)
    K = sim.intrinsics()
    gmap = GaussianMap(device=dev)
    res = {}

    Lx, Ly, Lz = [float(v) for v in sim.size]
    cy = Ly * 0.5
    poses = [_yaw_pose(Lx * 0.5, Lz * 0.5, yaw, cy, dev)
             for yaw in (0.0, 1.57, 3.14, 4.71)]

    f0 = sim.render(poses[0])
    res["(1) sim RGB-D valid"] = (
        tuple(f0.rgb.shape) == (200, 200, 3)
        and (f0.depth > 0).float().mean().item() > 0.5
        and bool(torch.isfinite(f0.depth).all())
    )

    covs = []
    for p in poses:
        fr = sim.render(p)
        gmap.add_from_rgbd(fr.rgb, fr.depth, fr.pose_c2w, K, stride=3, max_depth=20.0)
        covs.append(view_coverage(gmap, p, K, sim.width, sim.height))
    res["(2) map grows"] = gmap.num_gaussians > 1000
    cov_first_after = view_coverage(gmap, poses[0], K, sim.width, sim.height)
    res["(3) coverage accumulates"] = cov_first_after > 0.1

    unvisited = _yaw_pose(Lx * 0.9, Lz * 0.9, 3.9, cy, dev)
    g_unvisited = exploration_gain(gmap, unvisited, K, sim.width, sim.height)
    g_seen = exploration_gain(gmap, poses[0], K, sim.width, sim.height)
    res["(4) unvisited gain > seen gain"] = g_unvisited > g_seen

    cands = torch.stack([poses[0], unvisited], dim=0)
    best, scores = best_next_view(gmap, cands, K, sim.width, sim.height, metric="exploration")
    res["(5) NBV runs"] = len(scores) == 2 and best in (0, 1)

    print("=" * 62)
    print(f"smoke_pipeline (device={dev})  room={tuple(float(v) for v in sim.size)}")
    print(f"  gaussians={gmap.num_gaussians}")
    print(f"  coverage per step: {[round(c, 3) for c in covs]}")
    print(f"  gain unvisited={g_unvisited:.3f}  seen={g_seen:.3f}")
    print(f"  NBV scores={[round(s, 3) for s in scores]} -> pick {best}")
    print("-" * 62)
    ok = True
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = ok and v
    print("=" * 62)
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
