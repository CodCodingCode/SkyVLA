"""Outdoor aerial VLN via the OpenFly platform (AirSim / UE / 3DGS).

Public surface for the eval harness, dataset, action utilities, and
policy adapters. The trainable PaliGemma model lives under
``openfly.models.paligemma_vln``.
"""

from openfly.actions import (
    ACTION_NAMES,
    ACTION_VECTORS,
    action_id_to_vector,
    apply_action,
    distance3d,
    goal_heuristic_action,
    success_within,
    target_body_to_openfly,
    vector_to_action_id,
)
from openfly.episodes import group_by_env, load_episodes

__all__ = [
    "ACTION_NAMES",
    "ACTION_VECTORS",
    "action_id_to_vector",
    "apply_action",
    "distance3d",
    "goal_heuristic_action",
    "group_by_env",
    "load_episodes",
    "success_within",
    "target_body_to_openfly",
    "vector_to_action_id",
]
