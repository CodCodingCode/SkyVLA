"""PaliGemma BC policy for OpenFly's 10-class discrete action space.

Architecture (post-accuracy-fix; LSTM removed):

* Single RGB stream: current frame keeps all 256 SigLIP tokens, each history
  frame is pooled to one ``[CLS]`` token via cross-attention. This keeps the
  scene context for the model's current decision while collapsing the
  long-horizon temporal signal to a few tokens.
* Cross-attention from text tokens into the image-token stack produces a
  scene pool; we add a Gemma "last-text-token" summary (projected from the
  PaliGemma hidden state at the final non-pad text position of the *current*
  frame pass) so both spatial (SigLIP) and lingual (Gemma) context flow into
  the action head.
* The previous LSTM is gone — we condition the action MLP on
  ``[scene_summary, pose_feat, last_action_emb]`` directly. ``last_action``
  is an integer in ``[0, NUM_OPENFLY_ACTIONS]`` (``NUM_OPENFLY_ACTIONS`` is
  the START sentinel used at ``step == 0``).
* Optional auxiliary head regresses a 3-d next-step body-frame delta from
  the same fused vector. The model only produces ``goal_pred``; the trainer
  derives the target from ``next_pose``.

Trainable components: PaliGemma LoRA adapters (q/k/v/o projections, rank 16
by default) + image/text/gemma projections + per-frame [CLS] pool +
cross-attention + pose encoder + last-action embedding + action head.

This file intentionally breaks state-dict compatibility with previous
checkpoints (the LSTM is gone and new layers are added); train from scratch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vla.vla_policy import PaliGemmaFeatureExtractor


NUM_OPENFLY_ACTIONS = 10
# Sentinel value used for ``last_action`` at the first step of an episode.
# Embedding table therefore has ``NUM_OPENFLY_ACTIONS + 1`` entries.
LAST_ACTION_START_TOKEN = NUM_OPENFLY_ACTIONS


class PaliGemmaVLNPolicy(nn.Module):
    """PaliGemma + LoRA → token-pool fusion → 10-class action head.

    Args:
        history_frames: Past RGB frames pooled (1 [CLS] token each) into the
            image-token stack alongside the current frame's 256 tokens.
        embed_dim: Cross-attention working dimension.
        paligemma_model_name: HF id of the PaliGemma backbone.
        lora_rank / lora_alpha: LoRA hyper-parameters (defaults are tuned for
            BC training, see plan ``paligemma_accuracy_fixes``).
        lora_targets: PaliGemma module-name substrings to LoRA-adapt.
            Defaults cover all attention projections.
        aux_goal_head: If True, also regress a 3-d next-step body-frame
            delta from the fused head input. The model returns ``goal_pred``;
            the trainer computes the target from ``next_pose``.
    """

    def __init__(
        self,
        history_frames: int = 2,
        embed_dim: int = 256,
        paligemma_model_name: str = "google/paligemma-3b-pt-224",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        aux_goal_head: bool = True,
    ) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.embed_dim = int(embed_dim)
        self.aux_goal_head = bool(aux_goal_head)

        self.paligemma = PaliGemmaFeatureExtractor(
            model_name=paligemma_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
        )
        for p in self.paligemma.parameters():
            p.requires_grad = False
        for n, p in self.paligemma.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        feat = PaliGemmaFeatureExtractor.FEATURE_DIM
        self.image_proj = nn.Linear(feat, embed_dim)
        self.text_proj = nn.Linear(feat, embed_dim)
        # Projects Gemma's last-text-token hidden state (lingual summary of
        # the current-frame fused forward) into the cross-attn embed space.
        self.gemma_proj = nn.Linear(feat, embed_dim)

        # Per-frame embedding so the model can tell current vs history tokens.
        self.frame_embed = nn.Embedding(self.history_frames + 1, embed_dim)

        # Cross-attention pool: text tokens query the image-token stack.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )

        # Per-history-frame [CLS] pool: collapses 256 SigLIP tokens → 1 token
        # per history frame so we don't carry 768+ tokens into cross-attention.
        self.frame_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.frame_pool = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )

        # Pose features: 4-d pose (x, y, z, yaw).
        self.pose_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.ELU(),
            nn.Linear(32, 32),
        )

        # Last-action embedding: discrete action id at step-1, with
        # LAST_ACTION_START_TOKEN as the sentinel for step 0.
        self.last_action_embed = nn.Embedding(NUM_OPENFLY_ACTIONS + 1, 32)

        head_in_dim = embed_dim + 32 + 32
        self.action_head = nn.Sequential(
            nn.Linear(head_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, NUM_OPENFLY_ACTIONS),
        )

        if self.aux_goal_head:
            # Predicts next-step body-frame delta (3-d). Trainer supplies the
            # target derived from ``next_pose``; the model is target-agnostic.
            self.goal_head = nn.Sequential(
                nn.Linear(head_in_dim, 64),
                nn.ELU(),
                nn.Linear(64, 3),
            )

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[paligemma_vln] trainable {n_train:,} / total {n_total:,} "
            f"({100 * n_train / n_total:.3f}%)"
        )

    @staticmethod
    def _rgb_uint8_to_float(rgb_uint8: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 → (B, H, W, 3) float in [0, 1]."""
        return rgb_uint8.to(torch.float32) / 255.0

    def _tokenize(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        rgb_current: torch.Tensor,
        rgb_history: torch.Tensor,
        with_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PaliGemma over each frame and assemble the token stack.

        Returns
        -------
        image_tokens   : (B, K_history + 256, embed_dim)
            History frames pooled to one [CLS] token each, current frame at
            full 256-token SigLIP resolution. Frame embedding already added.
        text_tokens    : (B, L, embed_dim)
            Gemma text-token hidden states from the *current* frame pass
            (image-token positions stripped).
        text_mask      : (B, L)
            Matching attention mask for ``text_tokens``.
        gemma_summary  : (B, embed_dim)
            Gemma hidden state at the final non-pad text position of the
            current-frame pass, projected to ``embed_dim``. Zero vector for
            samples whose text mask is empty (sim rollouts).
        """
        B = rgb_current.shape[0]

        rgb_current_f = self._rgb_uint8_to_float(rgb_current)
        if rgb_history.numel() > 0:
            rgb_history_f = self._rgb_uint8_to_float(rgb_history)
            frames = [rgb_history_f[:, k] for k in range(rgb_history_f.shape[1])]
            frames.append(rgb_current_f)
        else:
            frames = [rgb_current_f]

        token_per_frame = 256  # PaliGemma image tokens

        history_pooled: list[torch.Tensor] = []
        current_tokens: torch.Tensor | None = None
        gemma_last: torch.Tensor | None = None
        for f_idx, rgb in enumerate(frames):
            pixel_values = self.paligemma.preprocess_images(rgb)
            if with_grad:
                gemma_feats, siglip_feats = self.paligemma.forward_tokens_with_grad(
                    pixel_values, instruction_ids, instruction_mask
                )
            else:
                gemma_feats, siglip_feats = self.paligemma.forward_tokens(
                    pixel_values, instruction_ids, instruction_mask
                )
            self.paligemma.clear_cache()

            siglip_emb = self.image_proj(siglip_feats)  # (B, 256, embed_dim)
            siglip_emb = siglip_emb + self.frame_embed.weight[f_idx]

            is_current = f_idx == len(frames) - 1
            if is_current:
                current_tokens = siglip_emb
            else:
                cls_query = self.frame_cls.expand(B, 1, self.embed_dim).to(siglip_emb.dtype)
                pooled, _ = self.frame_pool(
                    query=cls_query, key=siglip_emb, value=siglip_emb
                )
                history_pooled.append(pooled)  # (B, 1, embed_dim)
            gemma_last = gemma_feats

        assert current_tokens is not None
        if history_pooled:
            image_tokens = torch.cat(history_pooled + [current_tokens], dim=1)
        else:
            image_tokens = current_tokens

        # PaliGemma prepends ``token_per_frame`` image tokens to the Gemma
        # sequence; strip them so text features align with ``instruction_mask``.
        assert gemma_last is not None
        if gemma_last.shape[1] > token_per_frame:
            text_feats_raw = gemma_last[:, token_per_frame:]
        else:
            text_feats_raw = gemma_last
        text_tokens = self.text_proj(text_feats_raw)

        if instruction_mask.shape[1] > token_per_frame:
            text_mask = instruction_mask[:, token_per_frame:]
        else:
            text_mask = instruction_mask

        if text_mask.shape[1] > 0 and int(text_mask.sum().item()) > 0:
            # Clamp guards against rows whose mask is all-zero in a mixed batch.
            seq_len = text_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
            b_idx = torch.arange(B, device=text_feats_raw.device)
            gemma_summary_raw = text_feats_raw[b_idx, seq_len]  # (B, FEATURE_DIM)
            gemma_summary = self.gemma_proj(gemma_summary_raw)  # (B, embed_dim)
        else:
            gemma_summary = torch.zeros(
                B,
                self.embed_dim,
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )

        return image_tokens, text_tokens, text_mask, gemma_summary

    def forward(
        self,
        *,
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        rgb_current: torch.Tensor,
        rgb_history: torch.Tensor,
        pose: torch.Tensor,
        last_action: torch.Tensor,
        next_pose: torch.Tensor,  # noqa: ARG002 — signature parity; target derived by trainer
        with_grad: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute action logits (and optional goal regression) for one step.

        Args:
            instruction_input_ids: (B, L)
            instruction_attention_mask: (B, L)
            rgb_current: (B, H, W, 3) uint8
            rgb_history: (B, T, H, W, 3) uint8 — empty tensor allowed
            pose: (B, 4) float32 — [x, y, z, yaw]
            last_action: (B,) long — expert action at step-1, or
                ``LAST_ACTION_START_TOKEN`` (10) for step 0.
            next_pose: (B, 4) float32 — [x, y, z, yaw] of the next step
                (or current pose at terminal step). Accepted for inference
                signature parity; the trainer uses it to build the
                ``goal_pred`` regression target.
            with_grad: enable LoRA gradients during training; False in eval.
        """
        image_tokens, text_tokens, text_mask, gemma_summary = self._tokenize(
            instruction_input_ids,
            instruction_attention_mask,
            rgb_current,
            rgb_history,
            with_grad=with_grad,
        )

        # Sim rollouts pass real RGB through the processor; Gemma may leave
        # no text tokens after stripping image placeholders — fall back to
        # mean-pooling image tokens only.
        if text_tokens.shape[1] == 0 or int(text_mask.sum().item()) == 0:
            scene_summary = image_tokens.mean(dim=1)
        else:
            fused, _ = self.cross_attn(
                query=text_tokens,
                key=image_tokens,
                value=image_tokens,
                average_attn_weights=True,
            )
            mask = text_mask.unsqueeze(-1).to(fused.dtype)
            cross_attn_pool = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1.0
            )
            scene_summary = cross_attn_pool + gemma_summary

        pose_feat = self.pose_proj(pose.to(scene_summary.dtype))
        last_action_emb = self.last_action_embed(last_action.long())
        head_in = torch.cat([scene_summary, pose_feat, last_action_emb], dim=-1)

        action_logits = self.action_head(head_in)
        out: dict[str, torch.Tensor] = {"action_logits": action_logits}
        if self.aux_goal_head:
            out["goal_pred"] = self.goal_head(head_in)
        return out

    @torch.no_grad()
    def predict_action(
        self,
        *,
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        rgb_current: torch.Tensor,
        rgb_history: torch.Tensor,
        pose: torch.Tensor,
        last_action: torch.Tensor,
        next_pose: torch.Tensor,
    ) -> int:
        out = self.forward(
            instruction_input_ids=instruction_input_ids,
            instruction_attention_mask=instruction_attention_mask,
            rgb_current=rgb_current,
            rgb_history=rgb_history,
            pose=pose,
            last_action=last_action,
            next_pose=next_pose,
            with_grad=False,
        )
        return int(out["action_logits"].argmax(dim=-1).item())


def lora_and_head_param_groups(
    model: PaliGemmaVLNPolicy,
    *,
    lora_lr: float = 1e-6,
    head_lr: float = 3e-4,
) -> list[dict[str, Any]]:
    """Optimizer parameter groups: tiny LR on LoRA, normal LR on the head."""
    lora_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "paligemma" in name and "lora_" in name:
            lora_params.append(p)
        else:
            head_params.append(p)
    return [
        {"params": lora_params, "lr": lora_lr, "name": "lora"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]
