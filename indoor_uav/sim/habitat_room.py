"""Habitat-Sim backend — photorealistic indoor scenes for UAV navigation.

Implements the same :class:`IndoorSim` contract as :class:`SyntheticRoom`, so
tasks and the GS reward are unchanged when swapping to real scenery (HM3D /
Gibson / MP3D, or the lightweight Habitat test scenes).

Habitat-Sim runs in a DEDICATED conda env ``habitat`` (see
indoor_uav/scripts/setup_habitat.sh); two usage modes:

  * In-process (run from the ``habitat`` env): construct ``HabitatRoom`` and
    call ``render(pose)`` — returns torch tensors on ``device``.
  * Out-of-process (training env): use indoor_uav/scripts/render_habitat.py to
    dump RGB-D frames to disk, then feed them to the GS map. Keeps the fragile
    sim env isolated from the cu130 training stack.

Pose convention: OpenCV camera-to-world (+x right, +y down, +z forward), matched
to SyntheticRoom and the GS map. Habitat's native frame (+x right, +y up, -z
forward) is converted internally.
"""

from __future__ import annotations

import torch

from .base import Frame, IndoorSim

# Habitat <-> OpenCV camera basis change: flip Y and Z.
_HAB2CV = torch.tensor(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32
)


class HabitatRoom(IndoorSim):
    def __init__(
        self,
        scene_path: str,
        width: int = 256,
        height: int = 256,
        hfov_deg: float = 90.0,
        sensor_height: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        try:
            import habitat_sim  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HabitatRoom requires habitat_sim. Run from the dedicated "
                "'habitat' conda env (indoor_uav/scripts/setup_habitat.sh), or "
                "use render_habitat.py to dump frames and feed them offline."
            ) from exc
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width, self.height = width, height
        self._sim = self._make_sim(scene_path, width, height, hfov_deg, sensor_height)
        f = 0.5 * width / torch.tan(torch.tensor(hfov_deg * 3.14159265 / 360.0))
        self._K = torch.tensor(
            [[float(f), 0, width / 2], [0, float(f), height / 2], [0, 0, 1]],
            device=self.device, dtype=torch.float32,
        )

    def _make_sim(self, scene_path, width, height, hfov, sensor_height):
        import habitat_sim

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene_path
        backend.enable_physics = False

        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [height, width]; rgb.hfov = hfov
        rgb.position = [0.0, sensor_height, 0.0]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"; depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [height, width]; depth.hfov = hfov
        depth.position = [0.0, sensor_height, 0.0]

        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [rgb, depth]
        return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))

    def intrinsics(self) -> torch.Tensor:
        return self._K

    def bounds(self):
        lo, hi = self._sim.pathfinder.get_bounds()
        d = self.device
        return (torch.tensor(lo, device=d).float(), torch.tensor(hi, device=d).float())

    def is_free(self, xyz: torch.Tensor) -> bool:
        p = xyz.detach().cpu().float().reshape(3).tolist()
        hab = [p[0], -p[1], -p[2]]  # OpenCV world -> habitat
        return bool(self._sim.pathfinder.is_navigable(hab))

    @torch.no_grad()
    def render(self, pose_c2w: torch.Tensor) -> Frame:
        import numpy as np
        import habitat_sim
        import quaternion  # provided by habitat

        d = self.device
        hab_c2w = pose_c2w.detach().cpu().float() @ _HAB2CV
        pos = hab_c2w[:3, 3].numpy().astype("float32")
        quat = quaternion.from_rotation_matrix(hab_c2w[:3, :3].numpy().astype("float32"))

        state = habitat_sim.AgentState()
        state.position = pos
        state.rotation = quat
        self._sim.get_agent(0).set_state(state)
        obs = self._sim.get_sensor_observations()
        rgb = torch.from_numpy(np.ascontiguousarray(obs["rgb"][..., :3])).to(d).float() / 255.0
        depth = torch.from_numpy(np.ascontiguousarray(obs["depth"])).to(d).float()
        return Frame(rgb=rgb, depth=depth, pose_c2w=pose_c2w.to(d).float(), K=self._K)

    def close(self):
        self._sim.close()
