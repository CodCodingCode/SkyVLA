"""Small numpy helpers shared by navigate + tests (no AppLauncher side effects)."""

from __future__ import annotations

import numpy as np


def quat_rotate_inverse_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of unit quaternion (w, x, y, z)."""
    w, x, y, z = q_wxyz
    q_inv = np.array([w, -x, -y, -z], dtype=np.float32)
    qxyz = q_inv[1:]
    t = 2.0 * np.cross(qxyz, v)
    return v + q_inv[0] * t + np.cross(qxyz, t)
