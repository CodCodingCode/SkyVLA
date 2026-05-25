"""RL wrappers for ``IPEC-COMMUNITY/openfly-agent-7b``.

OpenFly-Agent is an OpenVLA-style VLM that emits a fixed-length action
token sequence (8 tokens → 8-dim continuous action vector). To do PPO
we need three things the stock HF model does not give us:

1. **Logprob access** over the sampled action tokens (the HF
   ``predict_action`` returns the rounded action only).
2. A **value head** on the last hidden state for the GAE baseline.
3. A small **LoRA adapter** on the LLM's attention projections so we
   keep the 7B backbone frozen and only optimise a few million params.

This module provides:

* :class:`OpenFlyAgentRL` — wraps the HF model, applies LoRA to
  ``q_proj``/``v_proj`` of the language tower, adds a value head, and
  exposes ``act_with_logprob`` and ``evaluate_actions`` for PPO.

The action quantisation step (``convert_to_action_id``) mirrors
upstream ``train/eval.py`` so the policy still emits one of the 10
OpenFly macro action ids at the env interface.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from openfly.actions import NUM_TRAINABLE_ACTIONS, vector_to_action_id

# Number of classes the auxiliary BC anchor head supervises — matches
# ``openfly.dataset.NUM_OPENFLY_ACTIONS``. Strafe ids 6 / 7 are absent
# from OpenFly's training data, so the head excludes them; targets must
# be remapped to logit-index space via ``action_id_to_logit_index``.
NUM_OPENFLY_ACTIONS = NUM_TRAINABLE_ACTIONS  # = 8


def _apply_lora_to_named_module(
    model: nn.Module,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Wrap targeted ``nn.Linear`` layers with manual LoRA adapters.

    Reuses the ``LoRALinear`` implementation from :mod:`vla.vla_policy`
    rather than introducing a PEFT dependency. Returns the number of
    layers wrapped.
    """
    try:
        from vla.vla_policy import LoRALinear
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vla.vla_policy.LoRALinear is required for LoRA wrapping") from exc

    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(t in name for t in targets):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parts = name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        parent = dict(model.named_modules())[parts[0]]
        setattr(parent, parts[1], LoRALinear(module, rank=rank, alpha=alpha))
        replaced += 1
    print(f"[openfly-agent-rl] LoRA wrapped {replaced} linear layers (rank={rank})")
    return replaced


class ValueHead(nn.Module):
    """Two-layer MLP value head reading the last hidden state.

    The hidden size is auto-detected from the wrapped model; we read
    the very last token's hidden state at generation time.
    """

    def __init__(self, hidden_dim: int, mid_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


class OpenFlyAgentRL(nn.Module):
    """OpenFly-Agent 7B + LoRA + value head, ready for PPO rollouts.

    Args:
        model_id: HF repo or local path of the OpenVLA-style policy.
        lora_rank, lora_alpha: LoRA hyperparameters; default rank 8.
        lora_targets: Name substrings of the linear layers to wrap.
        freeze_vision: If True, leave the vision tower untouched (you
            typically want this — visual features generalise well, the
            language head is what needs adaptation).
        device, dtype: Where/how the underlying VLM is loaded.

    The class implements a minimal :meth:`act` for env stepping plus
    :meth:`act_with_logprob` and :meth:`evaluate_actions` for PPO.
    """

    def __init__(
        self,
        *,
        model_id: str = "IPEC-COMMUNITY/openfly-agent-7b",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        freeze_vision: bool = True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        history_steps: int = 2,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from openfly.platform import load_eval_module

        self._eval = load_eval_module()
        self.history_steps = int(history_steps)
        self.device = torch.device(device)
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        try:
            self.policy = AutoModelForVision2Seq.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
        except Exception as exc:
            print(f"[openfly-agent-rl] flash-attn failed ({exc}); falling back to eager")
            self.policy = AutoModelForVision2Seq.from_pretrained(
                model_id,
                attn_implementation="eager",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)

        for p in self.policy.parameters():
            p.requires_grad = False

        # LoRA on the language model attention projections.
        _apply_lora_to_named_module(
            self.policy, targets=lora_targets, rank=lora_rank, alpha=lora_alpha
        )

        # Detect hidden dim. OpenVLA-7B is Llama-2-7B based → 4096.
        hidden_dim = getattr(getattr(self.policy.config, "text_config", self.policy.config), "hidden_size", 4096)
        self.value_head = ValueHead(hidden_dim).to(self.device, dtype=dtype)
        # Auxiliary BC-anchor head: 8-class classifier over OpenFly's
        # supervised action ids (strafes excluded — see
        # ``openfly.actions.TRAINABLE_ACTION_IDS``). Fed from the prefix's
        # last hidden state and trained with CE against expert (oracle /
        # DAgger) action ids — remapped to logit indices by the trainer —
        # to prevent the LoRA-updated representation from forgetting the
        # expert distribution during PPO. Active only when
        # ``--bc_coef > 0`` in train_ppo_openfly_agent.py.
        self.bc_head = nn.Linear(hidden_dim, NUM_OPENFLY_ACTIONS).to(self.device, dtype=dtype)

        if freeze_vision:
            for name, p in self.policy.named_parameters():
                if "vision" in name and "lora_" not in name:
                    p.requires_grad = False

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[openfly-agent-rl] trainable {n_train:,} / total {n_total:,} "
            f"({100 * n_train / max(n_total, 1):.3f}%)"
        )

    # ------------------------------------------------------------------
    # action helpers
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        image_list: list[np.ndarray],
        instruction: str,
        history_actions: list[int],
    ) -> dict[str, torch.Tensor]:
        from PIL import Image

        images = self._eval.get_images(image_list, True, self.history_steps) if image_list else None
        if isinstance(images, np.ndarray):
            img = Image.fromarray(images)
            images = [img, img, img]
        elif images is not None:
            images = [Image.fromarray(img) for img in images]
        inputs = self.processor(instruction, images).to(self.device, dtype=self.dtype)
        return inputs

    @torch.no_grad()
    def act(
        self,
        rgb: np.ndarray,
        instruction: str,
        history: list[int],
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> int:
        inputs = self._prepare_inputs([rgb], instruction, history)
        action = self.policy.predict_action(
            **inputs,
            unnorm_key="vlnv1",
            do_sample=do_sample,
            temperature=temperature,
        )
        action_id = int(vector_to_action_id(np.asarray(action).round()))
        return action_id

    @torch.no_grad()
    def act_with_logprob(
        self,
        rgb: np.ndarray,
        instruction: str,
        history: list[int],
        *,
        temperature: float = 1.0,
    ) -> tuple[int, float, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action_id, logprob, action_token_ids, value).

        We score the sampled action tokens against the policy's logits
        to recover the per-step logprob. The value head consumes the
        last hidden state of the prefix (image + instruction).
        """
        inputs = self._prepare_inputs([rgb], instruction, history)

        prefix_out = self.policy.model(  # type: ignore[attr-defined]
            **inputs,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        last_hidden = prefix_out.hidden_states[-1][:, -1, :]
        value = self.value_head(last_hidden).squeeze().float()

        # OpenVLA emits 8 action tokens; generate them autoregressively
        # with explicit sampling so we can recover logprobs.
        n_action_tokens = 8
        gen_inputs = self.processor(instruction, None).to(self.device, dtype=self.dtype)
        try:
            gen = self.policy.generate(
                **inputs,
                max_new_tokens=n_action_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )
        except Exception as exc:  # pragma: no cover — generate API differs across HF
            raise RuntimeError(
                f"OpenFly-Agent generate() failed ({exc}); fall back to act() until "
                "the upstream HF wrapper exposes a compatible signature."
            ) from exc

        scores = torch.stack(gen.scores, dim=1)  # (1, n_tokens, vocab)
        new_tokens = gen.sequences[:, -n_action_tokens:]
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        token_lp = log_probs.gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
        seq_logprob = float(token_lp.sum().item())

        # Decode tokens → action vector → action id (mirrors upstream eval).
        action_vec = self.policy.tokens_to_action(new_tokens, unnorm_key="vlnv1")  # type: ignore[attr-defined]
        action_id = int(vector_to_action_id(np.asarray(action_vec).round()))
        return action_id, seq_logprob, new_tokens.detach(), value.detach()

    def evaluate_actions(
        self,
        rgb: np.ndarray,
        instruction: str,
        history: list[int],
        action_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score saved ``action_tokens`` under the current policy.

        Returns ``(logprob, entropy, value)`` for the PPO update.
        """
        inputs = self._prepare_inputs([rgb], instruction, history)
        prefix_out = self.policy.model(  # type: ignore[attr-defined]
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = prefix_out.hidden_states[-1][:, -1, :]
        value = self.value_head(last_hidden).squeeze().float()

        # Teacher-force the action tokens through the policy to get logits.
        token_ids = action_tokens.to(self.device)
        # Concatenate prefix + actions for a single forward pass.
        prefix_ids = inputs["input_ids"]
        full_ids = torch.cat([prefix_ids, token_ids], dim=1)
        full_attn = torch.ones_like(full_ids)
        out = self.policy(  # type: ignore[misc]
            input_ids=full_ids,
            attention_mask=full_attn,
            pixel_values=inputs.get("pixel_values"),
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits[:, prefix_ids.shape[1] - 1 : -1, :]  # predict each action token
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_lp = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        seq_lp = token_lp.sum(dim=-1)

        with torch.no_grad():
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        return seq_lp, entropy, value

    def forward_bc_logits(
        self,
        rgb: np.ndarray,
        instruction: str,
        history: list[int],
    ) -> torch.Tensor:
        """Return ``(NUM_OPENFLY_ACTIONS,)`` logits for the BC anchor head.

        Reads the same prefix-last-hidden-state as the value head and
        runs it through ``self.bc_head``. The caller computes CE against
        the expert action id.
        """
        inputs = self._prepare_inputs([rgb], instruction, history)
        prefix_out = self.policy.model(  # type: ignore[attr-defined]
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = prefix_out.hidden_states[-1][:, -1, :]
        return self.bc_head(last_hidden).squeeze(0).float()

    def trainable_param_groups(
        self,
        *,
        lora_lr: float = 1e-5,
        value_lr: float = 3e-4,
        bc_lr: float | None = None,
    ) -> list[dict[str, Any]]:
        lora_params: list[nn.Parameter] = []
        value_params: list[nn.Parameter] = []
        bc_params: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "value_head" in name:
                value_params.append(p)
            elif "bc_head" in name:
                bc_params.append(p)
            else:
                lora_params.append(p)
        groups: list[dict[str, Any]] = [
            {"params": lora_params, "lr": lora_lr, "name": "lora"},
            {"params": value_params, "lr": value_lr, "name": "value"},
        ]
        if bc_params:
            groups.append(
                {
                    "params": bc_params,
                    "lr": bc_lr if bc_lr is not None else value_lr,
                    "name": "bc",
                }
            )
        return groups


__all__ = ["OpenFlyAgentRL", "ValueHead"]
