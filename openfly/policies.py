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
    rolling history. Mirrors the training-time forward signature
    (``last_action`` + ``next_pose``); at inference we use ``next_pose ==
    pose`` because we don't know the next state ahead of time — the model's
    ``goal_pred`` auxiliary output is ignored.
    """

    # START token id used during training when step == 0.
    _START_ACTION_ID: int = 10

    def __init__(
        self,
        checkpoint: str,
        *,
        history_frames: int = 2,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        paligemma_model: str = "google/paligemma-3b-pt-224",
        image_size: int = 224,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
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
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
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
        self._last_action: int = self._START_ACTION_ID

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
        self._last_action = self._START_ACTION_ID

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
        pose_vec = [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(pose[3]),
        ]
        pose_t = torch.tensor(
            [pose_vec], device=self.device, dtype=torch.float32
        )
        # At inference we don't know the next pose; pass current pose. The
        # downstream goal_pred output is unused for action selection.
        next_pose_t = pose_t.clone()
        last_action_t = torch.tensor(
            [int(self._last_action)], device=self.device, dtype=torch.long
        )

        proc = self.processor(
            text=[f"<image>\n{self.instruction}"],
            images=[cur],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        )
        input_ids = proc["input_ids"].to(self.device)
        attention_mask = proc["attention_mask"].to(self.device)

        action = self.model.predict_action(
            instruction_input_ids=input_ids,
            instruction_attention_mask=attention_mask,
            rgb_current=rgb_t,
            rgb_history=history_t,
            pose=pose_t,
            last_action=last_action_t,
            next_pose=next_pose_t,
        )

        # Slide history window after acting.
        if self.history_frames > 0:
            self._history.append(cur)
            if len(self._history) > self.history_frames:
                self._history = self._history[-self.history_frames :]
        self._last_action = int(action)
        return int(action)


class OpenFlyAgentRLPolicy(OpenFlyPolicy):
    """OpenFly-Agent 7B with PPO LoRA + value head from Phase 5.

    Wraps :class:`openfly.models.openfly_agent_rl.OpenFlyAgentRL` and
    loads the trainable ``lora_state`` produced by
    ``train_ppo_openfly_agent.py``. The backbone stays at the upstream
    HF weights so the file on disk is small.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        model_id: str = "IPEC-COMMUNITY/openfly-agent-7b",
        device: str = "cuda:0",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        history_steps: int = 2,
        do_sample: bool = False,
    ):
        import torch

        from openfly.models.openfly_agent_rl import OpenFlyAgentRL

        self.model = OpenFlyAgentRL(
            model_id=model_id,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=device,
            history_steps=history_steps,
        )
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("lora_state", ckpt)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(
                f"[ppo-agent-policy] loaded {checkpoint}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )
        self.do_sample = bool(do_sample)
        self.instruction = ""

    def reset(self, instruction: str, goal: Sequence[float]) -> None:
        del goal
        self.instruction = instruction

    def act(
        self,
        rgb: np.ndarray,
        pose: Sequence[float],
        step: int,
        history: list[int],
    ) -> int:
        del pose, step
        return int(
            self.model.act(rgb, self.instruction, history, do_sample=self.do_sample)
        )


def build_policy(name: str, **kwargs: Any) -> OpenFlyPolicy:
    name = name.lower()
    if name in ("heuristic", "oracle", "goal"):
        return GoalHeuristicPolicy(**kwargs)
    if name in ("openfly", "openfly-agent", "agent"):
        return OpenFlyAgentPolicy(**kwargs)
    if name in ("paligemma", "vla", "grpo", "dagger"):
        # GRPO and DAgger ckpts share the PaliGemmaVLNPolicy state dict, so
        # they are loaded by the same adapter — the alias just signals which
        # training stage produced the weights.
        if "checkpoint" not in kwargs:
            raise ValueError(
                f"{name} policy requires --paligemma_ckpt (checkpoint=...) "
                "produced by openfly.train_paligemma / train_dagger / train_grpo_paligemma"
            )
        return PaliGemmaOpenFlyPolicy(**kwargs)
    if name in ("ppo", "ppo-agent", "openfly-agent-rl"):
        if "checkpoint" not in kwargs:
            raise ValueError(
                "ppo policy requires --ppo_ckpt (checkpoint=...) "
                "produced by openfly.train_ppo_openfly_agent"
            )
        return OpenFlyAgentRLPolicy(**kwargs)
    raise ValueError(
        f"Unknown policy {name!r}. Use: heuristic, openfly-agent, paligemma, dagger, grpo, ppo"
    )
