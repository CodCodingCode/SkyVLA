"""Waypoint env with RTX chase camera for play_sim.py recording."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_mul

from .waypoint_nav_env import WaypointNavEnv


def _chase_camera_offset_quat(
    body_offset: tuple[float, float, float] = (-2.0, 0.0, 1.4),
    pitch_down_deg: float = 28.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chase cam behind drone, pitched down toward the arena."""
    from scipy.spatial.transform import Rotation as scipy_R

    cam_offset = np.array(body_offset, dtype=np.float64)
    forward = -cam_offset / np.linalg.norm(cam_offset)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    rot_mat = np.column_stack([right, cam_up, -forward])
    pitch = scipy_R.from_euler("y", -np.deg2rad(pitch_down_deg))
    rot_mat = rot_mat @ pitch.as_matrix()
    scipy_quat = scipy_R.from_matrix(rot_mat).as_quat()
    body_t = torch.tensor(body_offset, dtype=torch.float32)
    quat_t = torch.tensor(
        [scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]],
        dtype=torch.float32,
    )
    return body_t, quat_t


class WaypointNavEnvWithChaseCam(WaypointNavEnv):
    """Third-person chase camera via RTX TiledCamera."""

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        observer_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/CamChase",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=100.0,
                horizontal_aperture=24.0,
                clipping_range=(0.1, 50.0),
            ),
            width=1280,
            height=720,
        )
        self._chase_camera = TiledCamera(observer_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        for pos, color, name in [
            ((2.0, 0.0, 0.15), (0.8, 0.1, 0.1), "marker_red"),
            ((-2.0, 0.0, 0.15), (0.1, 0.1, 0.8), "marker_blue"),
            ((0.0, 2.0, 0.15), (0.1, 0.8, 0.1), "marker_green"),
            ((0.0, -2.0, 0.15), (0.8, 0.8, 0.1), "marker_yellow"),
        ]:
            m_cfg = sim_utils.CylinderCfg(
                radius=0.08,
                height=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            m_cfg.func(f"/World/envs/env_.*/{name}", m_cfg, translation=pos)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["chase_camera"] = self._chase_camera

        light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)
        dist_light = sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.85))
        dist_light.func("/World/SunLight", dist_light)

        self._wp_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/Visuals/waypoint_markers",
                markers={
                    "waypoint": sim_utils.SphereCfg(
                        radius=0.2,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
        )

        self._chase_body_offset, self._chase_cam_quat = _chase_camera_offset_quat()

    def _update_chase_camera_pose(self):
        drone_pos = self._robot.data.root_pos_w
        drone_quat = self._robot.data.root_quat_w
        w, x, y, z = drone_quat[:, 0], drone_quat[:, 1], drone_quat[:, 2], drone_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        half = yaw * 0.5
        yaw_quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        world_offset = quat_apply(
            yaw_quat, self._chase_body_offset.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        )
        cam_pos = drone_pos + world_offset
        cam_offset_quat = self._chase_cam_quat.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        cam_quat = quat_mul(yaw_quat, cam_offset_quat)
        self._chase_camera._view.set_world_poses(cam_pos, cam_quat)

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        self._update_chase_camera_pose()

    def get_chase_frame_bgr(self) -> np.ndarray | None:
        import cv2

        rgb = self._chase_camera.data.output["rgb"][0, ..., :3]
        frame = rgb.detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        if frame.size == 0 or frame.max() == 0:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


gym.register(
    id="Isaac-WaypointNav-Direct-ChaseCam-v0",
    entry_point=f"{__name__}:WaypointNavEnvWithChaseCam",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "waypoint_nav.waypoint_nav_env:WaypointNavEnvCfg",
        "rsl_rl_cfg_entry_point": "waypoint_nav.agents.rsl_rl_ppo_cfg:WaypointNavPPORunnerCfg",
    },
)
