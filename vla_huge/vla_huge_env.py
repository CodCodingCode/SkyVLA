"""HUGE-Bench Isaac Sim environment stub.

While the official HUGE-Bench Isaac Sim toolchain is unreleased, this env
mirrors the VLAWarehouseDroneEnv contract so the Stage 6 / Stage 7
``--resume_path`` code path is testable today. The two intentional
deviations from the warehouse env are:

  1. The instruction bank is HUGE-style ("Fly to N meters above ..."),
     loaded from ``vla_huge.scenes`` and aligned with HUGE-Bench task0
     scene ids.
  2. ``cfg.scene_id`` selects a HUGE scene; the matching USD is loaded
     into every cloned env. Until the digital-twin USDs ship, every
     scene resolves to ``warehouse_full`` so Stage 7 is a smoke test.

When HUGE releases the sim, replace ``HugeScene.usd_path`` in
``scenes.py`` and (optionally) wire scene-specific POI banks.
"""

from __future__ import annotations

import random

import torch

from isaaclab.utils import configclass

from vla_warehouse import pois as _pois_mod
from vla_warehouse.vla_warehouse_env import (
    VLAWarehouseDroneEnv as _BaseEnv,
    VLAWarehouseDroneEnvCfg as _BaseCfg,
)
from vla.vla_drone_env import _NUM_IMAGE_TOKENS

from vla_huge import scenes as _scenes


@configclass
class VLAHugeDroneEnvCfg(_BaseCfg):
    """Config for the HUGE-Bench scaffolded env.

    ``scene_id`` selects a HUGE digital-twin scene; per-step instructions
    are sampled from that scene's prompt bank in ``vla_huge.scenes``.

    ``scene_name`` (warehouse-style key) is forced to ``warehouse_full``
    until HUGE sim drops — so Stage 7 smoke tests reuse the same Nucleus
    asset path that Stage 6 already validated.
    """

    scene_id: str = "1_office"
    scene_name: str = "warehouse_full"
    episode_length_s = 45.0  # outdoor scale: longer episodes to reach distant goals
    distance_tanh_scale = 15.0
    success_threshold = 2.0
    proximity_radius = 6.0
    hover_at_target_radius = 3.0
    hover_radius_start = 3.0
    hover_radius_end = 3.0
    hover_max_speed = 1.5
    spawn_altitude = 5.0
    spawn_altitude_jitter = 1.0
    spawn_xy_radius = 6.0
    altitude_warning_low = 1.0
    altitude_warning_high = 25.0
    terminate_altitude_low = 0.5
    terminate_altitude_high = 35.0


class VLAHugeDroneEnv(_BaseEnv):
    """HUGE-Bench scaffolded env.

    Inherits the warehouse env and only overrides the instruction bank
    plus a scene-id-aware spawn altitude. Reward / termination /
    observation pipeline are unchanged from the warehouse parent so
    weights from Stage 5 / Stage 6 transfer with shape-tolerant load.
    """

    cfg: VLAHugeDroneEnvCfg

    def __init__(self, cfg: VLAHugeDroneEnvCfg, render_mode: str | None = None, **kwargs):
        try:
            self._huge_scene = _scenes.get_scene(cfg.scene_id)
        except KeyError as exc:
            raise SystemExit(
                f"[vla_huge] {exc}. Edit vla_huge/scenes.py to add the scene "
                "or pick one of the existing entries."
            ) from None
        cfg.spawn_altitude = self._huge_scene.spawn_altitude
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Override only the instruction-sampling path; reuse all other reset
    # logic (drone spawn, POI placement, marker writes) from the warehouse
    # parent.
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if not hasattr(self, "_active_poi_idx"):
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        extras: dict = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Curriculum/nav_multiplier"] = self._get_nav_multiplier()
        extras["Curriculum/precision_scale"] = self._get_precision_scale()
        self.extras.setdefault("log", {}).update(extras)

        self._actions[env_ids] = 0.0

        n = len(env_ids)
        env_origins = self._terrain.env_origins[env_ids]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        theta = torch.rand(n, device=self.device) * (2 * torch.pi)
        radius = torch.sqrt(torch.rand(n, device=self.device)) * self.cfg.spawn_xy_radius
        spawn_xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        spawn_z = self.cfg.spawn_altitude + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2 * self.cfg.spawn_altitude_jitter

        default_root_state[:, 0] = env_origins[:, 0] + spawn_xy[:, 0]
        default_root_state[:, 1] = env_origins[:, 1] + spawn_xy[:, 1]
        default_root_state[:, 2] = env_origins[:, 2] + spawn_z
        default_root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # POI selection — same machinery as warehouse for now.
        num_pois = len(self._poi_bank)
        active = self.cfg.num_active_pois
        if num_pois < active:
            raise RuntimeError(
                f"Scene fallback '{self.cfg.scene_name}' has {num_pois} POIs but "
                f"num_active_pois={active}. Add entries to vla_warehouse/pois.py."
            )
        picks = torch.stack([
            torch.randperm(num_pois, device=self.device)[:active] for _ in range(n)
        ], dim=0)
        self._active_poi_idx[env_ids] = picks

        poi_local = self._poi_local[picks]
        poi_world = env_origins.unsqueeze(1) + poi_local

        env_ids_list = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        self._cube_view.set_local_poses(translations=poi_local[:, 0], indices=env_ids_list)
        self._sphere_view.set_local_poses(translations=poi_local[:, 1], indices=env_ids_list)
        self._cylinder_view.set_local_poses(translations=poi_local[:, 2], indices=env_ids_list)

        self._obj_pos_w[env_ids, 0] = poi_world[:, 0]
        self._obj_pos_w[env_ids, 1] = poi_world[:, 1]
        self._obj_pos_w[env_ids, 2] = poi_world[:, 2]

        target_slot = torch.randint(0, active, (n,), device=self.device)
        self._target_obj_idx[env_ids] = target_slot

        # HUGE-style instructions: pull from this scene's prompt bank.
        commands = [random.choice(self._huge_scene.prompts) for _ in range(n)]

        # Tokenize only when PaliGemma is loaded; physics-only smoke runs
        # skip the processor entirely and leave _text_tokens / _text_mask
        # at their reset zero values (the VLA actor isn't run anyway).
        if not getattr(self.cfg, "physics_only", False):
            prefixed_commands = ["\n" + cmd for cmd in commands]
            tokenized = self._processor.tokenizer(
                prefixed_commands,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_text_length - _NUM_IMAGE_TOKENS,
            )
            batch_size = len(commands)
            img_tokens = torch.full(
                (batch_size, _NUM_IMAGE_TOKENS), self._image_token_id, dtype=torch.long
            )
            img_mask = torch.ones(batch_size, _NUM_IMAGE_TOKENS, dtype=torch.long)
            full_ids = torch.cat([img_tokens, tokenized["input_ids"]], dim=1)
            full_mask = torch.cat([img_mask, tokenized["attention_mask"]], dim=1)
            self._text_tokens[env_ids] = full_ids.to(self.device)
            self._text_mask[env_ids] = full_mask.to(self.device)

        for i, eid in enumerate(env_ids):
            self._current_commands[int(eid)] = commands[i]

        self._steps_since_capture[env_ids] = self.cfg.camera_every_n
        self._hover_dwell[env_ids] = 0.0


# sys.modules-patch aliases for the train.py shim.
VLADroneEnvCfg = VLAHugeDroneEnvCfg
VLADroneEnv = VLAHugeDroneEnv
