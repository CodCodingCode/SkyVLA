"""OpenFly discrete UAV action space (matches train/eval.py)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# 0=stop, 1=forward 3m, 2=turn left 30°, 3=turn right 30°, 4=up, 5=down,
# 6=strafe left, 7=strafe right, 8=forward 6m, 9=forward 9m
ACTION_NAMES: dict[int, str] = {
    0: "stop",
    1: "forward_3m",
    2: "turn_left_30",
    3: "turn_right_30",
    4: "up_3m",
    5: "down_3m",
    6: "strafe_left_3m",
    7: "strafe_right_3m",
    8: "forward_6m",
    9: "forward_9m",
}

STEP_SIZE = 3.0


def distance3d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2
    )


def success_within(
    pose: Sequence[float],
    goal: Sequence[float],
    threshold: float = 20.0,
) -> bool:
    """OpenFly eval uses 20 m success radius (planar distance in practice)."""
    return distance3d(pose[:3], goal[:3]) < threshold


def apply_action(pose: Sequence[float], action: int) -> list[float]:
    """Return new [x, y, z, yaw] after discrete action."""
    x, y, z, yaw = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])

    if action == 0:
        pass
    elif action == 1:
        x += STEP_SIZE * math.cos(yaw)
        y += STEP_SIZE * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += STEP_SIZE
    elif action == 5:
        z -= STEP_SIZE
    elif action == 6:
        x -= STEP_SIZE * math.sin(yaw)
        y += STEP_SIZE * math.cos(yaw)
    elif action == 7:
        x += STEP_SIZE * math.sin(yaw)
        y -= STEP_SIZE * math.cos(yaw)
    elif action == 8:
        x += STEP_SIZE * math.cos(yaw) * 2
        y += STEP_SIZE * math.sin(yaw) * 2
    elif action == 9:
        x += STEP_SIZE * math.cos(yaw) * 3
        y += STEP_SIZE * math.sin(yaw) * 3
    else:
        raise ValueError(f"Unknown OpenFly action id {action}")

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return [x, y, z, yaw]


def goal_heuristic_action(
    pose: Sequence[float],
    goal: Sequence[float],
    *,
    success_dist: float = 20.0,
    yaw_tol_deg: float = 25.0,
) -> int:
    """Oracle-ish mapper: turn toward goal, then forward, then stop."""
    if success_within(pose, goal, success_dist):
        return 0
    dx = goal[0] - pose[0]
    dy = goal[1] - pose[1]
    bearing = math.atan2(dy, dx)
    yaw_err = (bearing - pose[3] + math.pi) % (2 * math.pi) - math.pi
    if abs(math.degrees(yaw_err)) > yaw_tol_deg:
        return 2 if yaw_err > 0 else 3
    return 1


# 8-dim training vectors used by upstream OpenFly (RLDS / vln_norm).
# Only one slot non-zero per step; magnitudes match upstream eval.py
# `convert_to_action_id` exact-match table.
ACTION_VECTORS: dict[int, list[float]] = {
    0: [1, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 3, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 15, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 15, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 2, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 2, 0, 0],
    6: [0, 0, 0, 0, 0, 0, 5, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 5],
    8: [0, 6, 0, 0, 0, 0, 0, 0],
    9: [0, 9, 0, 0, 0, 0, 0, 0],
}


def action_id_to_vector(action_id: int) -> np.ndarray:
    """Return the 8-d float32 vector OpenFly's training pipeline uses."""
    if action_id not in ACTION_VECTORS:
        raise ValueError(f"Unknown OpenFly action id {action_id}")
    return np.asarray(ACTION_VECTORS[action_id], dtype=np.float32)


def vector_to_action_id(vec: Sequence[float]) -> int:
    """Round a continuous 8-d output and exact-match to a discrete action id.

    Falls back to action 0 (stop) on no match — same default as upstream
    `train/eval.py:convert_to_action_id`.
    """
    arr = np.asarray(vec, dtype=np.float32).round().astype(np.int32)
    for aid, ref in ACTION_VECTORS.items():
        if np.array_equal(arr, np.asarray(ref, dtype=np.int32)):
            return aid
    return 0


def target_body_to_openfly(
    pose: Sequence[float],
    goal: Sequence[float],
    *,
    success_dist: float = 20.0,
    yaw_tol_deg: float = 25.0,
    altitude_tol: float = 2.0,
    long_forward_dist: float = 18.0,
    medium_forward_dist: float = 9.0,
) -> int:
    """Map a 3-D body-frame goal to one of the 10 OpenFly macros.

    Generalises ``goal_heuristic_action`` with altitude correction and
    long-forward selection, useful as a target-body discretiser for
    continuous-output policies.
    """
    if success_within(pose, goal, success_dist):
        return 0

    dz = goal[2] - pose[2]
    if dz > altitude_tol:
        return 4
    if dz < -altitude_tol:
        return 5

    dx = goal[0] - pose[0]
    dy = goal[1] - pose[1]
    bearing = math.atan2(dy, dx)
    yaw_err = (bearing - pose[3] + math.pi) % (2 * math.pi) - math.pi
    if abs(math.degrees(yaw_err)) > yaw_tol_deg:
        return 2 if yaw_err > 0 else 3

    planar = math.hypot(dx, dy)
    if planar > long_forward_dist:
        return 9
    if planar > medium_forward_dist:
        return 8
    return 1
