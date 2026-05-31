#!/usr/bin/env python
"""Smoke test for indoor_uav.gs — GPU, no sim needed.

Builds a synthetic Gaussian map and checks the analytic navigation signals:
  (a) render shapes/finite;
  (b) coverage facing the map > facing away;
  (c) exploration_gain empty-view > covered-view;
  (d) add_from_rgbd back-projects a posed depth frame;
  (e) best_next_view runs and returns scores.
Exit code nonzero iff any check FAILs.
"""
from __future__ import annotations

import sys

import torch

from indoor_uav.gs import GaussianMap, view_coverage, exploration_gain, best_next_view


def _pose_looking_at(eye, target, device):
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    fwd = target - eye; fwd = fwd / fwd.norm()
    up = torch.tensor([0.0, -1.0, 0.0], device=device)  # OpenCV +y down
    right = torch.linalg.cross(fwd, up); right = right / right.norm()
    up = torch.linalg.cross(fwd, right)
    R = torch.stack([right, up, fwd], dim=1)
    T = torch.eye(4, device=device)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = H = 256
    f = 128.0
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev).float()
    res = {}

    gmap = GaussianMap(device=dev)
    pts = torch.rand(2000, 3, device=dev) + torch.tensor([-0.5, -0.5, 2.5], device=dev)
    gmap.add(pts, torch.rand(2000, 3, device=dev), scale=0.04, opacity=0.9)

    pose_at = _pose_looking_at([0, 0, 0], [0, 0, 3], dev)
    pose_away = _pose_looking_at([0, 0, 0], [0, 0, -3], dev)

    rgb, alpha = gmap.render(pose_at, K, W, H)
    res["(a) render shapes/finite"] = (
        tuple(rgb.shape) == (H, W, 3) and tuple(alpha.shape) == (H, W, 1)
        and bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(alpha).all())
    )

    cov_at = view_coverage(gmap, pose_at, K, W, H)
    cov_away = view_coverage(gmap, pose_away, K, W, H)
    res["(b) coverage facing > away"] = cov_at > cov_away and cov_at > 0.01

    empty = GaussianMap(device=dev)
    gain_empty = exploration_gain(empty, pose_at, K, W, H)
    gain_covered = exploration_gain(gmap, pose_at, K, W, H)
    res["(c) gain empty > covered"] = gain_empty > gain_covered

    depth = torch.full((H, W), 3.0, device=dev)
    m2 = GaussianMap(device=dev)
    n_added = m2.add_from_rgbd(torch.rand(H, W, 3, device=dev), depth, pose_at, K,
                              stride=8, max_depth=10.0)
    res["(d) add_from_rgbd"] = n_added > 0 and m2.num_gaussians == n_added

    cands = torch.stack([pose_away, pose_at], dim=0)
    best, scores = best_next_view(gmap, cands, K, W, H, metric="exploration")
    res["(e) best_next_view"] = best in (0, 1) and len(scores) == 2

    print("=" * 60)
    print(f"smoke_gs (device={dev})  Gaussians={gmap.num_gaussians}")
    print(f"  cov_facing={cov_at:.4f} cov_away={cov_away:.4f}")
    print(f"  gain_empty={gain_empty:.4f} gain_covered={gain_covered:.4f}  rgbd_added={n_added}")
    print("-" * 60)
    ok = True
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = ok and v
    print("=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
