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

from openfly.actions import (
    ACTION_NAMES,
    NUM_TRAINABLE_ACTIONS,
    TRAINABLE_ACTION_IDS,
    action_id_to_logit_index,
)
from openfly.episodes import load_episodes


def _build_sub_instruction(actions: Sequence[int], step: int) -> str:
    """Template sub-instruction over the maximal same-action run starting at ``step``.

    OpenFly trajectories alternate long runs of one primitive (e.g. seven
    consecutive ``forward_3m``) and single transition actions (turn, ascend).
    Each run corresponds to one "sub-trajectory" in the OpenFly data-gen
    pipeline. Until we have the original VLM-generated sub-instructions on
    disk, we surface a deterministic textual summary of the upcoming run
    so the model has *something* to condition on at the sub-segment level.

    Mid-run steps see only the *remainder* of the run, which doubles as a
    cheap within-segment progress signal ("3 meters of forward left").
    """
    if step >= len(actions):
        return ""
    cur = int(actions[step])
    run = 1
    for j in range(step + 1, len(actions)):
        if int(actions[j]) == cur:
            run += 1
        else:
            break
    if cur == 0:
        return "stop here"
    if cur == 1:
        return f"move forward {3 * run} meters"
    if cur == 2:
        return f"turn left {30 * run} degrees"
    if cur == 3:
        return f"turn right {30 * run} degrees"
    if cur == 4:
        return f"ascend {3 * run} meters"
    if cur == 5:
        return f"descend {3 * run} meters"
    if cur == 8:
        return f"move forward {6 * run} meters"
    if cur == 9:
        return f"move forward {9 * run} meters"
    return ACTION_NAMES.get(cur, str(cur))

# Number of classes the model's action head supervises. Strafe ids 6/7 are
# never emitted by OpenFly's A* planner, so they're excluded — see
# ``openfly.actions.TRAINABLE_ACTION_IDS``. The constant name is preserved
# for backward compatibility with imports across the repo; semantically it
# is now the trainable head dim, not the size of OpenFly's raw vocab.
NUM_OPENFLY_ACTIONS = NUM_TRAINABLE_ACTIONS  # = 8


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
    """One training step from a trajectory.

    Fields:
        rgb:          current-frame RGB image, (H, W, 3) uint8.
        history:     stack of past keyframe RGBs, (n_history, H, W, 3) uint8.
                     Keyframes are action-transition frames (plus step 0);
                     padded on the left by repeating the oldest pick.
        instruction:  natural-language navigation instruction.
        action_id:    ground-truth expert action as a **logit index** in
                      ``[0, NUM_OPENFLY_ACTIONS)``. See
                      ``openfly.actions.TRAINABLE_ACTION_IDS`` for the
                      logit-index → raw-OpenFly-id mapping.
        pose:         (4,) float32 [x, y, z, yaw] at ``step``.
        goal:         (3,) float32 episode terminal-pose xyz.
        last_action:  expert action at ``step - 1`` as a logit index, or
                      ``NUM_OPENFLY_ACTIONS`` (= 8) as a START token when
                      ``step == 0``. Used as a recurrence proxy by the policy.
        next_pose:    (4,) float32 [x, y, z, yaw] at ``step + 1``; at the
                      terminal step this duplicates ``pose``. Drives the
                      next-pose auxiliary loss.
        progress:     float in [0, 1] = ``step / max(1, traj_len - 1)``.
                      Fed into the action head as a small embedding;
                      targets VLN's stop-when-close failure mode (large
                      OSR–SR gap) by giving the model an explicit "how
                      far along am I" signal.
        sub_instruction: text summary of the upcoming same-action run
                      (see :func:`_build_sub_instruction`). Concatenated
                      into the PaliGemma prompt at training time. Empty
                      string is allowed (inference fallback when no
                      high-level policy is plumbed in yet).
        subgoal_rgb:  RGB frame at the next action-transition step (end of
                      the current same-action run). When no future
                      transition exists (terminal run), this duplicates
                      ``rgb`` and ``subgoal_valid`` is ``False``. Drives the
                      world-model / subgoal-DiT supervision: the DiT learns
                      ``(curr_siglip, instruction, sub_instruction) ->
                      subgoal_siglip`` in feature space.
        subgoal_horizon: integer number of trajectory steps between
                      ``step`` and the subgoal step. ``0`` for invalid
                      (terminal) subgoals; ``>= 1`` otherwise. Useful as a
                      conditioning signal and as a sanity check.
        subgoal_valid: ``True`` when a future action-transition was found.
                      ``False`` for terminal-run steps; the DiT trainer
                      skips invalid samples.
        subgoal_pose: (4,) float32 [x, y, z, yaw] at the subgoal step.
                      Used as an auxiliary pose-delta conditioning signal
                      for the DiT (the env teleports kinematically so the
                      delta is deterministic given the action sequence; we
                      feed it explicitly to keep the DiT focused on visual
                      content rather than reinventing forward kinematics).
    """

    rgb: np.ndarray             # (H, W, 3) uint8 — current frame
    history: np.ndarray         # (n_history, H, W, 3) uint8
    instruction: str
    action_id: int              # ground-truth action as a logit index 0..7
    pose: np.ndarray            # (4,) float32  [x, y, z, yaw]
    goal: np.ndarray            # (3,) float32  episode goal xyz
    last_action: int            # 0..7 logit index at step-1, or 8 (START)
    next_pose: np.ndarray       # (4,) float32  [x, y, z, yaw] at step+1
    progress: float             # [0, 1] fraction of trajectory completed
    sub_instruction: str        # upcoming same-action run summary
    subgoal_rgb: np.ndarray     # (H, W, 3) uint8 — next-transition frame
    subgoal_horizon: int        # # steps between ``step`` and subgoal (0 if invalid)
    subgoal_valid: bool         # False at terminal-run steps
    subgoal_pose: np.ndarray    # (4,) float32 pose at subgoal step


class OpenFlyDataset(Dataset):
    """Per-step samples from OpenFly trajectory annotations.

    Args:
        split:        ``train`` / ``seen`` / ``unseen`` / ``eval_test``.
        image_root:   Override default image root (see :func:`default_image_root`).
        history_frames: Past keyframes returned in :attr:`OpenFlySample.history`.
                        Keyframes are action-transition frames (plus step 0); left
                        padded by repeating the oldest pick. Falls back to uniform
                        ``step-k`` indexing for early steps without transitions.
        image_size:   Square crop size; matches PaliGemma's 224.
        env_filter:   Substring filter on episode env name.
        max_episodes: Cap loaded episodes (debug).
        max_samples:  Cap unrolled steps (debug). Applied before stop oversampling.
        require_images: If True, raise when frames cannot be resolved on disk.
        oversample_stop: Per-step duplication factor for ``action == 0`` (stop)
                         samples. ``2.0`` means each stop step appears ~twice in
                         ``_index`` (one extra copy). Set to ``<= 1.0`` to disable.
                         Default ``2.0`` — existing callers automatically get
                         stop-class oversampling; pass ``oversample_stop=1.0`` for
                         the legacy behaviour.
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
        oversample_stop: float = 2.0,
    ) -> None:
        self.image_root = Path(image_root) if image_root else default_image_root()
        self.history_frames = max(0, int(history_frames))
        self.image_size = int(image_size)
        self.require_images = require_images
        self.oversample_stop = float(oversample_stop)

        episodes = load_episodes(
            split,
            max_episodes=max_episodes,
            env_filter=env_filter,
        )

        # Flatten to per-step records: (episode_idx, step_idx)
        index: list[tuple[int, int]] = []
        skipped_steps = 0
        for ep_i, ep in enumerate(episodes):
            n = min(len(ep.get("action", [])), len(ep.get("index_list", [])))
            indices = ep.get("index_list", [])
            for s in range(n):
                # Restrict to actions the head supervises (excludes strafes,
                # which never appear in train.json, and pre-trim -1 / -2).
                if int(ep["action"][s]) not in TRAINABLE_ACTION_IDS:
                    continue
                if require_images:
                    if _resolve_frame(
                        self.image_root, ep["image_path"], indices[s]
                    ) is None:
                        skipped_steps += 1
                        continue
                index.append((ep_i, s))
        if require_images and skipped_steps:
            print(
                f"[openfly.dataset] require_images: indexed {len(index)} steps "
                f"(skipped {skipped_steps} without frames under {self.image_root})"
            )
        if max_samples > 0:
            index = index[:max_samples]

        if self.oversample_stop > 1.0:
            extra_copies = int(round(self.oversample_stop)) - 1
            if extra_copies > 0:
                stop_entries = [
                    (ep_i, s) for (ep_i, s) in index
                    if int(episodes[ep_i]["action"][s]) == 0
                ]
                pre = len(index)
                # Deterministic order: original index, then duplicated stops.
                # DataLoader handles shuffling at training time.
                index.extend(stop_entries * extra_copies)
                print(
                    f"[openfly.dataset] oversample_stop={self.oversample_stop}: "
                    f"{pre} -> {len(index)} samples "
                    f"(stop steps={len(stop_entries)}, "
                    f"added {len(stop_entries) * extra_copies} copies)"
                )

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

        # Remap the raw OpenFly action id (0..9) to a compact logit index
        # in [0, NUM_OPENFLY_ACTIONS). The index filter above guarantees
        # ``actions[step]`` is a supervised id, so the lookup cannot fail.
        action_id = action_id_to_logit_index(int(actions[step]))
        pose_xyz = positions[step] if step < len(positions) else positions[-1]
        yaw = float(yaws[step]) if step < len(yaws) else float(yaws[-1])
        pose = np.asarray([pose_xyz[0], pose_xyz[1], pose_xyz[2], yaw], dtype=np.float32)
        goal_xyz = positions[-1]
        goal = np.asarray(goal_xyz[:3], dtype=np.float32)

        # Last expert action (recurrence proxy). Stored as a logit index;
        # START token == NUM_OPENFLY_ACTIONS (= 8). train.json contains
        # stray -1 / -2 values and the occasional un-supervised id, so
        # anything outside ``TRAINABLE_ACTION_IDS`` falls back to START
        # rather than poisoning the embedding lookup.
        if step > 0:
            prev = int(actions[step - 1])
            if prev in TRAINABLE_ACTION_IDS:
                last_action = action_id_to_logit_index(prev)
            else:
                last_action = NUM_OPENFLY_ACTIONS  # START sentinel
        else:
            last_action = NUM_OPENFLY_ACTIONS

        # Next-step pose for the auxiliary "next pose" prediction head. At the
        # terminal step we duplicate the current pose so the delta is zero.
        np_step = step + 1
        if np_step < len(positions):
            np_xyz = positions[np_step]
            np_yaw = float(yaws[np_step]) if np_step < len(yaws) else yaw
            next_pose = np.asarray(
                [np_xyz[0], np_xyz[1], np_xyz[2], np_yaw], dtype=np.float32
            )
        else:
            next_pose = pose.copy()

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
        if self.history_frames > 0:
            # Keyframes: step 0 plus any action-transition frame strictly before
            # ``step``. Falls back to uniform ``step-k`` indexing for early steps
            # that have no transitions yet.
            cands = [
                j for j in range(step)
                if j == 0 or int(actions[j]) != int(actions[j - 1])
            ]
            if cands:
                picked = cands[-self.history_frames :]
                if len(picked) < self.history_frames:
                    # Left-pad by repeating the oldest pick.
                    picked = [picked[0]] * (self.history_frames - len(picked)) + picked
                for j in picked:
                    hp = _resolve_frame(self.image_root, ep["image_path"], indices[j])
                    history_imgs.append(_load_rgb(hp, self.image_size))
            else:
                # Early-step fallback: uniform step-k stack with current-frame
                # padding when j < 0 (matches the legacy behaviour).
                for k in range(self.history_frames, 0, -1):
                    j = step - k
                    if j >= 0:
                        hp = _resolve_frame(
                            self.image_root, ep["image_path"], indices[j]
                        )
                    else:
                        hp = cur_path  # repeat current frame for padding
                    history_imgs.append(_load_rgb(hp, self.image_size))
        if not history_imgs:
            history = np.zeros((0, self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            history = np.stack(history_imgs, axis=0)

        traj_len = min(len(actions), len(indices))
        progress = float(step) / float(max(1, traj_len - 1))
        sub_instruction = _build_sub_instruction(actions, step)

        # Subgoal frame: the trajectory step where the current same-action
        # run ends (next action-transition). When no future transition is
        # found (terminal run, e.g. the final ``stop``) we duplicate the
        # current frame and mark the sample invalid so the DiT trainer
        # can skip it.
        cur_action = int(actions[step])
        sg_step = step
        for j in range(step + 1, min(len(actions), len(indices))):
            if int(actions[j]) != cur_action:
                sg_step = j
                break
        subgoal_valid = sg_step > step
        subgoal_horizon = max(0, sg_step - step)

        if subgoal_valid:
            sg_path = _resolve_frame(self.image_root, ep["image_path"], indices[sg_step])
            subgoal_rgb = _load_rgb(sg_path, self.image_size)
            sg_xyz = positions[sg_step] if sg_step < len(positions) else positions[-1]
            sg_yaw = float(yaws[sg_step]) if sg_step < len(yaws) else yaw
            subgoal_pose = np.asarray(
                [sg_xyz[0], sg_xyz[1], sg_xyz[2], sg_yaw], dtype=np.float32
            )
        else:
            subgoal_rgb = rgb.copy()
            subgoal_pose = pose.copy()

        return OpenFlySample(
            rgb=rgb,
            history=history,
            instruction=instruction,
            action_id=action_id,
            pose=pose,
            goal=goal,
            last_action=last_action,
            next_pose=next_pose,
            progress=progress,
            sub_instruction=sub_instruction,
            subgoal_rgb=subgoal_rgb,
            subgoal_horizon=subgoal_horizon,
            subgoal_valid=subgoal_valid,
            subgoal_pose=subgoal_pose,
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
    last_actions = torch.tensor(
        [s.last_action for s in samples], dtype=torch.long
    )
    next_poses = torch.from_numpy(
        np.stack([s.next_pose for s in samples], axis=0)
    )
    instructions = [s.instruction for s in samples]
    sub_instructions = [s.sub_instruction for s in samples]
    progress = torch.tensor(
        [float(s.progress) for s in samples], dtype=torch.float32
    )
    subgoal_rgb = torch.stack(
        [_to_tensor_uint8_chw(s.subgoal_rgb) for s in samples], dim=0
    )
    subgoal_horizon = torch.tensor(
        [int(s.subgoal_horizon) for s in samples], dtype=torch.long
    )
    subgoal_valid = torch.tensor(
        [bool(s.subgoal_valid) for s in samples], dtype=torch.bool
    )
    subgoal_pose = torch.from_numpy(
        np.stack([s.subgoal_pose for s in samples], axis=0)
    )
    return {
        "rgb": rgb,
        "history": history,
        "instruction": instructions,
        "sub_instruction": sub_instructions,
        "action_id": actions,
        "pose": poses,
        "goal": goals,
        "last_action": last_actions,
        "next_pose": next_poses,
        "progress": progress,
        "subgoal_rgb": subgoal_rgb,
        "subgoal_horizon": subgoal_horizon,
        "subgoal_valid": subgoal_valid,
        "subgoal_pose": subgoal_pose,
    }
