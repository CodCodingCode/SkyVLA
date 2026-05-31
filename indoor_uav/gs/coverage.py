"""GS-derived navigation signals — the cheap analytic reward.

All quantities are computed by a *forward rasterize* of an existing
:class:`~indoor_uav.gs.gaussian_map.GaussianMap` — no GS fitting, no
optimization. This is the crux of the project: a Gaussian-Splatting-aware
exploration reward that costs milliseconds, usable inside an RL loop or to
drive a greedy next-best-view planner.

Signals
-------
* ``view_coverage`` — fraction of a view already reconstructed (mean alpha, [0,1]).
* ``exploration_gain`` — expected *new* coverage of a candidate view = mean
  unreconstructed alpha, optionally masked by ground-truth geometry.
* ``fisher_view_info`` — FisherRF-style (Jiang et al. 2024) closed-form proxy
  for a viewpoint's information gain: visible Gaussians weighted by inverse
  observation count (rarely-seen = uncertain).
* ``best_next_view`` — greedy NBV over candidate poses (the learning-free baseline).
"""

from __future__ import annotations

import torch

from .gaussian_map import GaussianMap


@torch.no_grad()
def view_coverage(gmap: GaussianMap, pose_c2w, K, width, height) -> float:
    """Mean accumulated alpha over the view -> fraction reconstructed in [0,1]."""
    _, alpha = gmap.render(pose_c2w, K, width, height)
    return float(alpha.mean().item())


@torch.no_grad()
def exploration_gain(
    gmap: GaussianMap, pose_c2w, K, width, height, *, geometry_mask=None
) -> float:
    """Expected *new* coverage of a candidate view = mean(1 - alpha) in [0,1].

    If ``geometry_mask`` (H,W bool) is given (e.g. sim depth>0 at this pose),
    only count pixels that have surface, so empty space isn't rewarded.
    """
    _, alpha = gmap.render(pose_c2w, K, width, height)
    unreconstructed = (1.0 - alpha.squeeze(-1)).clamp(min=0.0)
    if geometry_mask is not None:
        m = geometry_mask.to(unreconstructed.device).bool()
        if int(m.sum().item()) == 0:
            return 0.0
        return float(unreconstructed[m].mean().item())
    return float(unreconstructed.mean().item())


@torch.no_grad()
def fisher_view_info(gmap: GaussianMap, pose_c2w, K, width, height, obs_count=None) -> float:
    """FisherRF-style information gain of a viewpoint (cheap proxy).

    info(view) ≈ Σ_{g visible} w_g / (1 + n_g), with w_g the Gaussian's
    visible footprint*opacity and n_g its prior observation count. Rarely-seen,
    strongly-visible Gaussians dominate — the high-uncertainty directions Fisher
    info points to. Returns a non-negative scalar (only relative values matter).
    """
    n = gmap.num_gaussians
    if n == 0:
        return 0.0
    import gsplat

    d = gmap.device
    viewmat = torch.linalg.inv(pose_c2w.to(d).float())[None]
    _, _, meta = gsplat.rasterization(
        gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
        viewmat, K.to(d).float()[None], width, height,
    )
    radii = meta.get("radii", None)
    if radii is None:
        return view_coverage(gmap, pose_c2w, K, width, height) * n
    radii = radii.reshape(-1).float()
    if radii.shape[0] != n:  # some versions pack (B,N,2)
        radii = radii.reshape(n, -1).max(dim=-1).values
    visible = radii > 0
    w = (radii * gmap.opacities).clamp(min=0.0)
    inv_unc = torch.ones(n, device=d) if obs_count is None else 1.0 / (1.0 + obs_count.to(d).float())
    return float((w * inv_unc * visible.float()).sum().item())


@torch.no_grad()
def best_next_view(
    gmap: GaussianMap, candidate_poses, K, width, height, *,
    metric: str = "exploration", obs_count=None,
) -> tuple[int, list[float]]:
    """Greedy next-best-view: score each candidate pose, return (argmax, scores).

    ``metric`` in {"exploration", "fisher"}. This is the analytic NBV baseline a
    learned policy must beat — same reward, no learning.
    """
    scores: list[float] = []
    for v in range(candidate_poses.shape[0]):
        pose = candidate_poses[v]
        if metric == "fisher":
            scores.append(fisher_view_info(gmap, pose, K, width, height, obs_count))
        else:
            scores.append(exploration_gain(gmap, pose, K, width, height))
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    return best, scores
