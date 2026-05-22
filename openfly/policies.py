"""Policy adapters for OpenFly VLN evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from openfly.actions import goal_heuristic_action


class OpenFlyPolicy(ABC):
    @abstractmethod
    def reset(self, instruction: str, goal: Sequence[float]) -> None:
        ...

    @abstractmethod
    def act(
        self,
        rgb: np.ndarray,
        pose: Sequence[float],
        step: int,
        history: list[int],
    ) -> int:
        ...


class GoalHeuristicPolicy(OpenFlyPolicy):
    """Oracle baseline: fly toward episode goal (tests sim + metrics)."""

    def __init__(self, success_dist: float = 20.0):
        self.success_dist = success_dist
        self.goal: list[float] = [0.0, 0.0, 0.0]
        self.instruction = ""

    def reset(self, instruction: str, goal: Sequence[float]) -> None:
        self.instruction = instruction
        self.goal = list(goal[:3])

    def act(
        self,
        rgb: np.ndarray,
        pose: Sequence[float],
        step: int,
        history: list[int],
    ) -> int:
        del rgb, step, history
        return goal_heuristic_action(pose, self.goal, success_dist=self.success_dist)


class OpenFlyAgentPolicy(OpenFlyPolicy):
    """Official OpenFly-Agent (OpenVLA 7B) from HuggingFace."""

    def __init__(
        self,
        model_id: str = "IPEC-COMMUNITY/openfly-agent-7b",
        device: str = "cuda:0",
        history_steps: int = 2,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from openfly.platform import load_eval_module

        self._eval = load_eval_module()
        self.history_steps = history_steps
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.instruction = ""
        self.goal: list[float] = []

    def reset(self, instruction: str, goal: Sequence[float]) -> None:
        self.instruction = instruction
        self.goal = list(goal[:3])

    def act(
        self,
        rgb: np.ndarray,
        pose: Sequence[float],
        step: int,
        history: list[int],
    ) -> int:
        del pose, step
        image_list = [rgb]  # upstream stacks history internally via get_images
        action_id = self._eval.get_action(
            self.model,
            self.processor,
            image_list,
            self.instruction,
            history,
            if_his=True,
            his_step=self.history_steps,
        )
        return int(action_id)


class PaliGemmaOpenFlyPolicy(OpenFlyPolicy):
    """Custom PaliGemma BC policy fine-tuned on OpenFly via `train_paligemma.py`.

    Loads a checkpoint produced by ``openfly.train_paligemma`` and predicts
    discrete OpenFly action ids (0..9) from the current RGB frame plus a
    rolling history.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        history_frames: int = 2,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        paligemma_model: str = "google/paligemma-3b-pt-224",
        image_size: int = 224,
    ):
        import torch
        from transformers import AutoProcessor

        from openfly.models.paligemma_vln import PaliGemmaVLNPolicy

        self._torch = torch
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.history_frames = int(history_frames)

        self.model = PaliGemmaVLNPolicy(
            history_frames=history_frames,
            paligemma_model_name=paligemma_model,
        ).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[paligemma] missing keys: {len(missing)} (likely PaliGemma frozen weights)")
        if unexpected:
            print(f"[paligemma] unexpected keys: {len(unexpected)}")
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(paligemma_model)
        self.instruction = ""
        self._history: list[np.ndarray] = []

    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            from PIL import Image as _Image

            img = _Image.fromarray(rgb).resize(
                (self.image_size, self.image_size), _Image.BILINEAR
            )
            return np.asarray(img, dtype=np.uint8)
        return rgb.astype(np.uint8)

    def reset(self, instruction: str, goal: Sequence[float]) -> None:
        del goal  # not used at inference time
        self.instruction = instruction
        self._history = []

    def act(
        self,
        rgb: np.ndarray,
        pose: Sequence[float],
        step: int,
        history: list[int],
    ) -> int:
        del step, history
        torch = self._torch

        cur = self._resize_rgb(rgb)
        if self.history_frames > 0:
            while len(self._history) < self.history_frames:
                self._history.append(cur)
            past = np.stack(self._history[-self.history_frames :], axis=0)
        else:
            past = np.zeros((0, self.image_size, self.image_size, 3), dtype=np.uint8)

        rgb_t = torch.from_numpy(cur).unsqueeze(0).to(self.device)
        history_t = torch.from_numpy(past).unsqueeze(0).to(self.device)
        pose_t = torch.tensor(
            [[float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])]],
            device=self.device,
            dtype=torch.float32,
        )

        proc = self.processor(
            text=[f"<image>\n{self.instruction}"],
            images=[cur],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=256,
        )
        input_ids = proc["input_ids"].to(self.device)
        attention_mask = proc["attention_mask"].to(self.device)

        action = self.model.predict_action(
            instruction_input_ids=input_ids,
            instruction_attention_mask=attention_mask,
            rgb_current=rgb_t,
            rgb_history=history_t,
            pose=pose_t,
        )

        # Slide history window after acting.
        if self.history_frames > 0:
            self._history.append(cur)
            if len(self._history) > self.history_frames:
                self._history = self._history[-self.history_frames :]
        return int(action)


def build_policy(name: str, **kwargs: Any) -> OpenFlyPolicy:
    name = name.lower()
    if name in ("heuristic", "oracle", "goal"):
        return GoalHeuristicPolicy(**kwargs)
    if name in ("openfly", "openfly-agent", "agent"):
        return OpenFlyAgentPolicy(**kwargs)
    if name in ("paligemma", "vla"):
        if "checkpoint" not in kwargs:
            raise ValueError(
                "paligemma policy requires --paligemma_ckpt (checkpoint=...) "
                "produced by openfly.train_paligemma"
            )
        return PaliGemmaOpenFlyPolicy(**kwargs)
    raise ValueError(f"Unknown policy {name!r}. Use: heuristic, openfly-agent, paligemma")
