"""PyTorch dataset over OpenFly VLN trajectories (`train.json` / `seen.json`).

Each annotation entry describes one trajectory:

  {
    "image_path": "env_airsim_16/astar_data/medium_average/2025-1-9_...",
    "gpt_instruction": "Proceed in a straight line ...",
    "action": [9, 9, 9, 3, 1, 0],          # discrete ids 0-9
    "index_list": ["20250109_131812_2", ...],
    "pos": [[x, y, z], ...],
    "yaw": [1.57, ...]
  }

We unroll each entry into per-step training samples
``(rgb, history_rgb, instruction, action_id, pose, goal)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from openfly.episodes import load_episodes


def default_image_root() -> Path:
    """Where extracted trajectory image folders live.

    Override with ``OPENFLY_IMAGE_ROOT``. Falls back to upstream OpenFly
    convention ``OpenFly-Platform/uav_vln_data`` when present.
    """
    env = os.environ.get("OPENFLY_IMAGE_ROOT")
    if env:
        return Path(env).expanduser()
    platform = Path(os.environ.get("OPENFLY_ROOT", Path.home() / "OpenFly-Platform"))
    candidate = platform / "uav_vln_data"
    if candidate.is_dir():
        return candidate
    return Path.home() / "assets" / "OpenFly" / "images"


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _resolve_frame(image_root: Path, image_path: str, index: str) -> Path | None:
    base = image_root / image_path
    for ext in _IMG_EXTS:
        candidate = base / f"{index}{ext}"
        if candidate.is_file():
            return candidate
    # Some upstream dumps store frames as a flat list under image_path
    for ext in _IMG_EXTS:
        candidate = image_root / f"{image_path}_{index}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _load_rgb(path: Path | None, image_size: int = 224) -> np.ndarray:
    if path is None:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


@dataclass
class OpenFlySample:
    """One training step from a trajectory."""

    rgb: np.ndarray             # (H, W, 3) uint8 — current frame
    history: np.ndarray         # (n_history, H, W, 3) uint8
    instruction: str
    action_id: int              # ground-truth discrete action 0..9
    pose: np.ndarray            # (4,) float32  [x, y, z, yaw]
    goal: np.ndarray            # (3,) float32  episode goal xyz


class OpenFlyDataset(Dataset):
    """Per-step samples from OpenFly trajectory annotations.

    Args:
        split:        ``train`` / ``seen`` / ``unseen`` / ``eval_test``.
        image_root:   Override default image root (see :func:`default_image_root`).
        history_frames: Past frames concatenated to the current observation.
        image_size:   Square crop size; matches PaliGemma's 224.
        env_filter:   Substring filter on episode env name.
        max_episodes: Cap loaded episodes (debug).
        max_samples:  Cap unrolled steps (debug).
        require_images: If True, raise when frames cannot be resolved on disk.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        image_root: Path | str | None = None,
        history_frames: int = 2,
        image_size: int = 224,
        env_filter: str | None = None,
        max_episodes: int = 0,
        max_samples: int = 0,
        require_images: bool = False,
    ) -> None:
        self.image_root = Path(image_root) if image_root else default_image_root()
        self.history_frames = max(0, int(history_frames))
        self.image_size = int(image_size)
        self.require_images = require_images

        episodes = load_episodes(
            split,
            max_episodes=max_episodes,
            env_filter=env_filter,
        )

        # Flatten to per-step records: (episode_idx, step_idx)
        index: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(episodes):
            n = min(len(ep.get("action", [])), len(ep.get("index_list", [])))
            for s in range(n):
                index.append((ep_i, s))
        if max_samples > 0:
            index = index[:max_samples]

        self._episodes = episodes
        self._index = index
        self._missing_warned = False

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> OpenFlySample:
        ep_i, step = self._index[i]
        ep = self._episodes[ep_i]
        instruction: str = ep.get("gpt_instruction", "")
        actions: list[int] = ep["action"]
        indices: list[str] = ep["index_list"]
        positions: list[list[float]] = ep["pos"]
        yaws: list[float] = ep["yaw"]

        action_id = int(actions[step])
        pose_xyz = positions[step] if step < len(positions) else positions[-1]
        yaw = float(yaws[step]) if step < len(yaws) else float(yaws[-1])
        pose = np.asarray([pose_xyz[0], pose_xyz[1], pose_xyz[2], yaw], dtype=np.float32)
        goal = np.asarray(positions[-1], dtype=np.float32)

        cur_path = _resolve_frame(self.image_root, ep["image_path"], indices[step])
        if cur_path is None and self.require_images:
            raise FileNotFoundError(
                f"Frame missing for episode {ep['image_path']} index {indices[step]}; "
                f"set OPENFLY_IMAGE_ROOT or download the trajectory dump."
            )
        if cur_path is None and not self._missing_warned:
            print(
                f"[openfly.dataset] WARN frames not found under {self.image_root} — "
                f"using zero RGB. Set OPENFLY_IMAGE_ROOT to a directory containing "
                f"trajectory image folders."
            )
            self._missing_warned = True
        rgb = _load_rgb(cur_path, self.image_size)

        history_imgs: list[np.ndarray] = []
        for k in range(self.history_frames, 0, -1):
            j = step - k
            if j >= 0:
                hp = _resolve_frame(self.image_root, ep["image_path"], indices[j])
            else:
                hp = cur_path  # repeat current frame for padding
            history_imgs.append(_load_rgb(hp, self.image_size))
        if not history_imgs:
            history = np.zeros((0, self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            history = np.stack(history_imgs, axis=0)

        return OpenFlySample(
            rgb=rgb,
            history=history,
            instruction=instruction,
            action_id=action_id,
            pose=pose,
            goal=goal,
        )


def _to_tensor_uint8_chw(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


def collate(samples: Sequence[OpenFlySample]) -> dict[str, Any]:
    """Default collate: stacks tensors, keeps instructions as a list of str."""
    rgb = torch.stack([_to_tensor_uint8_chw(s.rgb) for s in samples], dim=0)
    if samples[0].history.size > 0:
        history = torch.stack(
            [_to_tensor_uint8_chw(s.history) for s in samples], dim=0
        )
    else:
        history = torch.empty(
            (len(samples), 0, samples[0].rgb.shape[0], samples[0].rgb.shape[1], 3),
            dtype=torch.uint8,
        )
    actions = torch.tensor([s.action_id for s in samples], dtype=torch.long)
    poses = torch.from_numpy(np.stack([s.pose for s in samples], axis=0))
    goals = torch.from_numpy(np.stack([s.goal for s in samples], axis=0))
    instructions = [s.instruction for s in samples]
    return {
        "rgb": rgb,
        "history": history,
        "instruction": instructions,
        "action_id": actions,
        "pose": poses,
        "goal": goals,
    }
