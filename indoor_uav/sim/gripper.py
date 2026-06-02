"""Drone + 2-DoF underslung gripper, for aerial pick-and-place in Habitat.

Wraps ``DronePhysics`` and adds a gripper with two CONTROLLED DoF:

  * ``lower`` in [0,1]  -> how far the gripper is extended down (0..max_drop m)
  * ``grip``  in [0,1]  -> jaw closure (0 = open, 1 = closed)

Grasping uses a *kinematic-attach* abstraction (standard for aerial-manip RL):
when the jaws are closed and the gripper tip is within ``grasp_radius`` of a
graspable object, the object is attached and rides at the tip; opening releases
it (it then settles to the surface below). This avoids brittle contact-rich
grasp dynamics while still requiring the policy to *position, descend, and
close* correctly — the actual hard part of the task.

The URDF (assets/gripper/drone_gripper.urdf) is loaded as a KINEMATIC
articulated object purely for visualization + joint posing; the grasp itself is
geometric (tip position), so training works headless without it.

Frame: habitat world, +Y up, so "down" = -Y. The tip hangs straight below the
drone body by ``mount_offset + lower * max_drop``.
"""
from __future__ import annotations

import os

import numpy as np

_URDF = os.path.join(os.path.dirname(__file__), "..", "assets", "gripper", "drone_gripper.urdf")


class DroneGripper:
    def __init__(self, sim, drone, *, max_drop: float = 0.5, mount_offset: float = 0.05,
                 grasp_radius: float = 0.32, load_urdf: bool = True):
        self.sim = sim
        self.drone = drone
        self.max_drop = float(max_drop)
        self.mount_offset = float(mount_offset)
        self.grasp_radius = float(grasp_radius)

        self.lower = 0.0          # [0,1]
        self.grip = 0.0           # [0,1]; >0.5 == closed
        self.held = None          # ManagedRigidObject currently grasped, or None

        self._art = None
        self._jidx = {}
        if load_urdf:
            self._load_urdf()

    # ------------------------------------------------------------------ #
    def _load_urdf(self):
        import habitat_sim
        aom = self.sim.get_articulated_object_manager()
        path = os.path.normpath(_URDF)
        try:
            self._art = aom.add_articulated_object_from_urdf(path, fixed_base=False)
            self._art.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            # map joint name -> position index (best-effort across habitat versions)
            try:
                names = self._art.get_link_joint_names() if hasattr(self._art, "get_link_joint_names") else []
                self._jidx = {n: i for i, n in enumerate(names)}
            except Exception:
                self._jidx = {}
        except Exception as exc:  # visualization is optional; never block training
            print(f"[DroneGripper] URDF load failed (continuing headless): {exc!r}")
            self._art = None

    @property
    def has_urdf(self) -> bool:
        return self._art is not None

    # ------------------------------------------------------------------ #
    def tip_world(self) -> np.ndarray:
        """World position of the gripper tip (between the jaws)."""
        p = np.asarray(self.drone.position, np.float32).copy()
        p[1] -= (self.mount_offset + self.lower * self.max_drop)   # +Y up -> go down
        return p

    @property
    def closed(self) -> bool:
        return self.grip > 0.5

    # ------------------------------------------------------------------ #
    def reset(self, position, yaw: float = 0.0):
        self.drone.reset(position, yaw)
        self.lower = 0.0; self.grip = 0.0; self.held = None
        self._pose_urdf()

    def step(self, vx_body, vz_body, vy, yaw_rate, lower_cmd, grip_cmd,
             dt: float = 1.0 / 30.0, graspables=None):
        """Drive the drone + gripper one tick. ``graspables`` is a list of
        ManagedRigidObjects the gripper may pick up. Returns drone telemetry."""
        self.lower = float(np.clip(lower_cmd, 0.0, 1.0))
        self.grip = float(np.clip(grip_cmd, 0.0, 1.0))
        tel = self.drone.step(vx_body, vz_body, vy, yaw_rate, dt=dt)
        self._update_grasp(graspables or [])
        self._pose_urdf()
        tel.update({"lower": self.lower, "grip": self.grip, "holding": self.held is not None})
        return tel

    # ------------------------------------------------------------------ #
    def _update_grasp(self, graspables):
        tip = self.tip_world()
        if self.held is not None:
            if not self.closed:                       # opened -> release
                self.held = None
            else:                                     # carry: object rides at tip
                self.held.translation = tip
        elif self.closed:                             # try to grab the nearest in range
            best, bestd = None, self.grasp_radius
            for obj in graspables:
                d = float(np.linalg.norm(np.asarray(obj.translation, np.float32) - tip))
                if d < bestd:
                    best, bestd = obj, d
            if best is not None:
                import habitat_sim
                best.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                best.translation = tip
                self.held = best

    def _pose_urdf(self):
        if self._art is None:
            return
        import habitat_sim
        # pin the gripper base under the drone, aligned with its yaw
        st = self.drone.camera_state()
        self._art.translation = np.asarray(self.drone.position, np.float32) \
            + np.array([0, -self.mount_offset, 0], np.float32)
        try:
            self._art.rotation = st.rotation
        except Exception:
            pass
        # drive joints from (lower, grip)
        try:
            pos = list(self._art.joint_positions)
            drop = self.lower * self.max_drop
            jaw = (1.0 - self.grip) * 0.035          # open=0.035, closed=0
            order = ["lower", "grip_l", "grip_r"]
            vals = {"lower": drop, "grip_l": jaw, "grip_r": jaw}
            if self._jidx:
                for n, i in self._jidx.items():
                    if n in vals and i < len(pos):
                        pos[i] = vals[n]
            else:                                     # fall back to declared order
                for k, n in enumerate(order):
                    if k < len(pos):
                        pos[k] = vals[n]
            self._art.joint_positions = pos
        except Exception:
            pass
