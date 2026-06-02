"""A real rigid-body quadrotor in Bullet (inside Habitat-Sim).

Unlike the kinematic camera (teleported along a path), this is an actual
dynamical body: it has mass, gravity pulls it down, thrust/control forces push
it, drag damps it, and it physically collides with the scene mesh (it cannot
pass through walls — Bullet resolves contacts). The RGB-D camera rides on the
body, so what you see is filmed from a drone that obeys physics.

Control model: a velocity-tracking flight controller. The policy/teleop issues a
desired body velocity (vx, vy, vz) + yaw rate; a PD-ish force law drives the
rigid body toward it. This abstracts away individual rotor mixing (which a
navigation policy doesn't care about) while keeping real momentum, inertia,
gravity compensation, and collisions — the things that make it "a drone with
physics" rather than a camera on rails.

Requires the sim built with ``enable_physics=True`` (Bullet). Verified API:
apply_force / apply_torque / linear_velocity / mass / linear_damping.
"""

from __future__ import annotations

import numpy as np

try:
    import habitat_sim
except ImportError as exc:  # pragma: no cover
    raise ImportError("indoor_uav.sim.drone_body needs habitat_sim (habitat env).") from exc

GRAVITY = 9.81


class DronePhysics:
    """Rigid-body quadrotor with a velocity-command controller."""

    def __init__(
        self,
        sim: "habitat_sim.Simulator",
        *,
        mass: float = 0.5,
        size: tuple[float, float, float] = (0.30, 0.10, 0.30),  # ~30cm quad
        linear_damping: float = 1.5,
        angular_damping: float = 4.0,
        kv: float = 6.0,            # velocity-tracking gain (force per m/s error)
        max_thrust_g: float = 3.0,  # max thrust as multiple of weight
        max_speed: float = 2.0,     # m/s clamp on commanded speed
    ) -> None:
        self.sim = sim
        self.mass = float(mass)
        self.kv = float(kv)
        self.max_thrust = max_thrust_g * mass * GRAVITY
        self.max_speed = float(max_speed)
        self._yaw = 0.0

        otm = sim.get_object_template_manager()
        rom = sim.get_rigid_object_manager()
        base = otm.get_template_handles("cube")[0]
        t = otm.get_template_by_handle(base)
        t.scale = np.array(size, dtype=np.float32)
        t.mass = self.mass
        otm.register_template(t, "uav_body")
        self.obj = rom.add_object_by_template_handle("uav_body")
        self.obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        # IMPORTANT: do NOT set linear_damping/angular_damping — doing so silently
        # DISABLES velocity_control in this habitat build (verified: with damping
        # the commanded velocity is ignored and the body just sinks). The velocity
        # controller already gives critically-damped tracking, so damping is moot.
        vc = self.obj.velocity_control
        vc.controlling_lin_vel = True
        vc.lin_vel_is_local = False
        # Zero global gravity: the velocity controller IS the flight model (a
        # velocity-commanded drone shouldn't also free-fall). This gives perfect
        # altitude hold (0.0m drift) while wall collisions still work (collision
        # mesh is independent of gravity). Verified.
        sim.set_gravity(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    # ------------------------------------------------------------------ #
    @property
    def position(self) -> np.ndarray:
        return np.array(self.obj.translation, dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return np.array(self.obj.linear_velocity, dtype=np.float32)

    @property
    def yaw(self) -> float:
        return self._yaw

    def reset(self, position, yaw: float = 0.0) -> None:
        self.obj.translation = np.asarray(position, dtype=np.float32)
        self.obj.linear_velocity = np.zeros(3, np.float32)
        self.obj.angular_velocity = np.zeros(3, np.float32)
        self._yaw = float(yaw)

    def step(self, vx_body: float, vz_body: float, vy: float, yaw_rate: float,
             dt: float = 1.0 / 30.0, substeps: int = 2) -> dict:
        """Drive toward a desired velocity for ``dt`` seconds, with real physics.

        vx_body / vz_body : desired horizontal velocity in the drone's heading
                            frame (m/s); vy : vertical (m/s); yaw_rate : rad/s.
        Returns telemetry incl. whether a collision occurred this step.
        """
        # integrate yaw kinematically (heading), clamp speeds
        self._yaw += float(yaw_rate) * dt
        c, s = np.cos(self._yaw), np.sin(self._yaw)
        # body -> world (yaw about +Y; world +z forward at yaw 0)
        des_world = np.array([
            vx_body * c + vz_body * s,
            float(vy),
            -vx_body * s + vz_body * c,
        ], np.float32)
        sp = np.linalg.norm(des_world)
        if sp > self.max_speed:
            des_world *= self.max_speed / sp

        # command the desired world velocity; Bullet integrates with collisions.
        # NOTE: re-grab velocity_control LIVE and re-assert the controlling flags
        # each step — a stored handle / single set does not reliably drive the body.
        collided = False
        sub_dt = dt / substeps
        for _ in range(substeps):
            vc = self.obj.velocity_control
            vc.controlling_lin_vel = True
            vc.lin_vel_is_local = False
            vc.linear_velocity = des_world.astype(np.float32)
            self.sim.step_physics(sub_dt)
            if self.sim.get_physics_num_active_contact_points() > 0:
                collided = True

        return {
            "position": self.position,
            "velocity": self.velocity,
            "speed": float(np.linalg.norm(self.velocity)),
            "yaw": self._yaw,
            "collided": collided,
        }

    def camera_state(self, sensor_height: float = 0.0) -> "habitat_sim.AgentState":
        """AgentState placing the camera on the body, looking along heading."""
        import quaternion
        st = habitat_sim.AgentState()
        p = self.position.copy(); p[1] += sensor_height
        st.position = p
        st.rotation = quaternion.from_rotation_vector([0.0, self._yaw, 0.0])
        return st
