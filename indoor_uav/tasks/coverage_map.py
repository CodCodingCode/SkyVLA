"""Top-down coverage/occupancy memory + frontier signal for exploration.

This is the agent's MEMORY of what it has and hasn't seen — the thing that makes
"don't get stuck in one room / find the upstairs" learnable. A purely reactive
RGB policy can't know unexplored space exists; this map tells it.

Three channels over a fixed-resolution top-down grid (world XZ plane):
  * navigable : 1 where the scene floor is reachable (ground truth, from the
                habitat navmesh top-down view) — "where rooms exist at all".
  * visited   : 1 where the drone has already observed (accumulated each step).
  * frontier  : navigable AND not-visited AND adjacent to visited — the boundary
                of the known region = "go here to discover new space".

Coverage fraction = visited∩navigable / navigable. Frontier cells give both a
dense input channel and a "distance to nearest frontier" signal that directly
counters local-optimum sticking.
"""

from __future__ import annotations

import numpy as np


class CoverageMap:
    def __init__(self, lo, hi, res: int = 96, visit_radius_m: float = 1.5):
        """lo/hi: world (x,y,z) bounds. Grid uses habitat's NATIVE top-down
        resolution (rows=Z, cols=X) at ~``res`` cells on the long side, so cell
        size is isotropic and world->cell alignment matches the navmesh exactly.
        The observation is later resampled to a fixed square for the policy."""
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)
        self.res = int(res)
        self.visit_radius_m = float(visit_radius_m)
        ext = self.hi - self.lo
        # isotropic meters-per-cell sized so the LONGER of X/Z spans ~res cells
        self.mpp = max(max(ext[0], ext[2]), 1e-3) / self.res
        self._mx = self._mz = self.mpp
        self.navigable = np.zeros((1, 1), np.float32)  # set from pathfinder
        self.visited = np.zeros((1, 1), np.float32)

    # ------------------------------------------------------------------ #
    def set_navigable_from_pathfinder(self, pf, height: float):
        """Use the navmesh top-down view AS the grid (native rows=Z, cols=X)."""
        td = np.asarray(pf.get_topdown_view(meters_per_pixel=self.mpp, height=float(height)))
        self.navigable = td.astype(np.float32)
        self.visited = np.zeros_like(self.navigable)

    def world_to_cell(self, p):
        # topdown grid: row = (z-lo_z)/mpp, col = (x-lo_x)/mpp  (verified mapping A)
        h, w = self.navigable.shape
        cz = int(np.clip((p[2] - self.lo[2]) / self.mpp, 0, h - 1))
        cx = int(np.clip((p[0] - self.lo[0]) / self.mpp, 0, w - 1))
        return cz, cx

    def mark_visited(self, p):
        """Stamp a disk of radius visit_radius_m around the drone as observed."""
        cz, cx = self.world_to_cell(p)
        rz = max(1, int(self.visit_radius_m / self._mz))
        rx = max(1, int(self.visit_radius_m / self._mx))
        z0, z1 = max(0, cz - rz), min(self.res, cz + rz + 1)
        x0, x1 = max(0, cx - rx), min(self.res, cx + rx + 1)
        self.visited[z0:z1, x0:x1] = 1.0

    # ------------------------------------------------------------------ #
    def frontier(self) -> np.ndarray:
        """navigable ∧ ¬visited ∧ adjacent-to-visited -> the explore boundary."""
        nav_unseen = self.navigable * (1.0 - self.visited)
        v = self.visited
        adj = np.zeros_like(v)
        adj[1:, :] += v[:-1, :]; adj[:-1, :] += v[1:, :]
        adj[:, 1:] += v[:, :-1]; adj[:, :-1] += v[:, 1:]
        return ((nav_unseen > 0) & (adj > 0)).astype(np.float32)

    def coverage_fraction(self) -> float:
        nav = float(self.navigable.sum())
        if nav < 1:
            return 0.0
        return float((self.visited * self.navigable).sum() / nav)

    def dist_to_frontier(self, p) -> float:
        """Normalised distance (cells) from the drone to the nearest frontier
        cell; 1.0 if no frontier remains (fully explored / trapped)."""
        fr = self.frontier()
        idx = np.argwhere(fr > 0)
        if len(idx) == 0:
            return 1.0
        cz, cx = self.world_to_cell(p)
        d = np.sqrt(((idx[:, 0] - cz) ** 2 + (idx[:, 1] - cx) ** 2)).min()
        return float(min(d / max(self.navigable.shape), 1.0))

    def observation(self, out_res: int) -> np.ndarray:
        """(3, out_res, out_res) float32 — navigable/visited/frontier resampled
        from the native (possibly non-square) grid to a fixed square for the net."""
        chans = [self.navigable, self.visited, self.frontier()]
        h, w = self.navigable.shape
        rs = np.linspace(0, h - 1, out_res).astype(int)
        cs = np.linspace(0, w - 1, out_res).astype(int)
        return np.stack([c[np.ix_(rs, cs)] for c in chans], axis=0).astype(np.float32)
