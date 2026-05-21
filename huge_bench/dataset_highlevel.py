"""HUGE-Bench dataset variant that yields body-frame future-waypoint labels.

Used by ``huge_bench/train_vla_highlevel.py`` for Stage 6 hierarchical SFT.

Re-uses :class:`huge_bench.dataset.HugeTask0` for parquet/episode loading,
then adds a per-frame label by looking ``K`` frames ahead inside the same
episode and projecting the world-frame displacement into the drone's body
frame (using the recorded yaw at the current frame).

Body-frame convention (matches `vla/vla_drone_env.py`):
  x forward, y left, z up
  rotation about z by yaw  ->  body_x = cos(yaw)*dx + sin(yaw)*dy
                               body_y = -sin(yaw)*dx + cos(yaw)*dy
                               body_z = dz
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from huge_bench.dataset import HugeTask0, IMG_SIZE


class HugeTask0WithFutureWaypoint(Dataset):
    """One sample per (episode, frame); label = body-frame future waypoint.

    Args:
        split: "train" / "test_seen" / "test_unseen"
        lookahead_frames: K frames to look ahead in the same episode (default 25,
            i.e. ~5s at 5 Hz). Clamped to last frame at episode boundary.
        max_target_norm: Cap |target_body| at this many meters (matches
            HierarchicalVLAActor.target_range default of 3.0). Targets beyond
            this are rescaled to the cap; lets the policy's tanh head saturate
            on long-distance goals without exploding gradients.
        rgb_in_minus_one_to_one: If True (default), images are kept in [-1, 1]
            (SigLIP normalization) as the parent dataset emits them. If False,
            convert to [0, 1] for HierarchicalVLAActor's ``preprocess_images``.
    """

    def __init__(
        self,
        split: str = "train",
        lookahead_frames: int = 25,
        max_target_norm: float = 3.0,
        rgb_in_minus_one_to_one: bool = True,
    ):
        self.base = HugeTask0(split=split, normalize_actions=True)
        self.lookahead = int(lookahead_frames)
        self.max_target_norm = float(max_target_norm)
        self.rgb_in_minus_one_to_one = bool(rgb_in_minus_one_to_one)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.base)

    def num_episodes(self) -> int:
        return self.base.num_episodes()

    @property
    def episodes(self):
        return self.base.episodes

    # ------------------------------------------------------------------
    @staticmethod
    def _decode(cell) -> np.ndarray:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        img = (
            Image.open(io.BytesIO(b))
            .convert("RGB")
            .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
        return np.asarray(img, dtype=np.float32) / 127.5 - 1.0

    def _future_position(self, df: pd.DataFrame, frame_pos: int, ep_length: int) -> np.ndarray:
        target_frame = min(frame_pos + self.lookahead, ep_length - 1)
        return np.asarray(df.iloc[target_frame]["state"], dtype=np.float32)

    def _world_to_body(self, dx: float, dy: float, dz: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        return np.asarray([bx, by, dz], dtype=np.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        ep_pos, frame_pos = self.base._flat[idx]
        ep = self.base.episodes[ep_pos]
        df = self.base._load_episode(ep_pos)
        row = df.iloc[frame_pos]

        image = self._decode(row["image"])
        first_image = self._decode(row["first_image"])
        if not self.rgb_in_minus_one_to_one:
            image = (image + 1.0) * 0.5
            first_image = (first_image + 1.0) * 0.5

        cur_state = np.asarray(row["state"], dtype=np.float32)  # (x, y, z, yaw_rad)
        fut_state = self._future_position(df, frame_pos, ep.length)
        dx = fut_state[0] - cur_state[0]
        dy = fut_state[1] - cur_state[1]
        dz = fut_state[2] - cur_state[2]
        target_body = self._world_to_body(dx, dy, dz, cur_state[3])

        norm = float(np.linalg.norm(target_body))
        if norm > self.max_target_norm and norm > 1e-6:
            target_body = target_body * (self.max_target_norm / norm)

        return {
            "image": torch.from_numpy(image),                       # (224,224,3)
            "first_image": torch.from_numpy(first_image),           # (224,224,3)
            "instruction": ep.task,                                 # str
            "target_body": torch.from_numpy(target_body),           # (3,)
            "raw_target_body": torch.from_numpy(np.asarray([dx, dy, dz], dtype=np.float32)),
            "yaw": torch.tensor(float(cur_state[3]), dtype=torch.float32),
            "task_index": int(row["task_index"]),
            "env_id": ep.env_id,
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
        }


def collate_highlevel(batch: list[dict]) -> dict:
    out: dict = {}
    tensor_keys = ("image", "first_image", "target_body", "raw_target_body", "yaw")
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["instruction"] = [b["instruction"] for b in batch]
    out["task_index"] = torch.tensor([b["task_index"] for b in batch], dtype=torch.long)
    out["episode_index"] = torch.tensor([b["episode_index"] for b in batch], dtype=torch.long)
    out["frame_index"] = torch.tensor([b["frame_index"] for b in batch], dtype=torch.long)
    out["env_id"] = [b["env_id"] for b in batch]
    return out
