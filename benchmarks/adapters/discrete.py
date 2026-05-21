"""Map continuous body-frame targets to discrete aerial VLN actions.

Used for CityNav (6 actions) and AirNav (4 actions) when plugging our
hierarchical / waypoint policies into their simulators.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Sequence

import numpy as np


class AirNavAction(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_RIGHT = 2
    TURN_LEFT = 3


class CityNavAction(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_RIGHT = 2
    TURN_LEFT = 3
    GO_UP = 4
    GO_DOWN = 5


def target_body_to_airnav(
    target_body: Sequence[float],
    *,
    stop_dist: float = 0.35,
    yaw_thresh: float = 0.35,
) -> int:
    """Body-frame target (x forward, y left, z up) → AirNav action id."""
    x, y, z = float(target_body[0]), float(target_body[1]), float(target_body[2])
    horiz = math.hypot(x, y)
    if horiz < stop_dist and abs(z) < 0.5:
        return int(AirNavAction.STOP)
    yaw_err = math.atan2(y, x) if horiz > 1e-3 else 0.0
    if abs(yaw_err) > yaw_thresh:
        return int(AirNavAction.TURN_LEFT if yaw_err > 0 else AirNavAction.TURN_RIGHT)
    return int(AirNavAction.MOVE_FORWARD)


def target_body_to_citynav(
    target_body: Sequence[float],
    *,
    stop_dist: float = 0.35,
    yaw_thresh: float = 0.35,
    z_thresh: float = 1.0,
) -> int:
    """Body-frame target → CityNav DiscreteAction index."""
    x, y, z = float(target_body[0]), float(target_body[1]), float(target_body[2])
    horiz = math.hypot(x, y)
    if horiz < stop_dist and abs(z) < 0.5:
        return int(CityNavAction.STOP)
    if z > z_thresh:
        return int(CityNavAction.GO_UP)
    if z < -z_thresh:
        return int(CityNavAction.GO_DOWN)
    yaw_err = math.atan2(y, x) if horiz > 1e-3 else 0.0
    if abs(yaw_err) > yaw_thresh:
        return int(CityNavAction.TURN_LEFT if yaw_err > 0 else CityNavAction.TURN_RIGHT)
    return int(CityNavAction.MOVE_FORWARD)


def state_to_target_body(
    state_xyz_yaw: Sequence[float],
    goal_xyz: Sequence[float],
) -> np.ndarray:
    """World-frame goal relative to drone pose → body-frame direction (unnormalized)."""
    x, y, z, yaw = [float(v) for v in state_xyz_yaw]
    gx, gy, gz = [float(v) for v in goal_xyz]
    dx, dy, dz = gx - x, gy - y, gz - z
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rotate world delta into body frame (yaw about z)
    bx = cy * dx + sy * dy
    by = -sy * dx + cy * dy
    bz = dz
    return np.array([bx, by, bz], dtype=np.float32)
