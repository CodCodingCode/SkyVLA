"""Gaussian-Splatting map + analytic coverage/info-gain navigation signals."""
from .gaussian_map import GaussianMap
from .coverage import view_coverage, exploration_gain, fisher_view_info, best_next_view

__all__ = [
    "GaussianMap",
    "view_coverage",
    "exploration_gain",
    "fisher_view_info",
    "best_next_view",
]
