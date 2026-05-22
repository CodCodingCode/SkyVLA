"""PaliGemma BC policy for OpenFly's 10-class discrete action space.

Builds on ``vla.vla_policy.PaliGemmaFeatureExtractor``. Key differences from
the earlier multi-camera Isaac variant:

* Single RGB input (no 4-camera rig, no depth maps, no waypoint policy).
* History of past frames is folded into the SigLIP token stream by
  concatenating tokens across frames.
* Output head produces 10-class action logits (cross-entropy with the
  expert macros from ``train.json``). An optional auxiliary head
  regresses the next-step body-frame goal vector.

Trainable components: PaliGemma LoRA adapters + cross-attention head
+ LSTM + action head (~ a few million params).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.vla_policy import PaliGemmaFeatureExtractor


NUM_OPENFLY_ACTIONS = 10


class PaliGemmaVLNPolicy(nn.Module):
    """PaliGemma + LoRA → cross-attn → LSTM → 10-class action head.

    Args:
        history_frames: Past RGB frames stacked alongside the current one in
            the SigLIP token stream.
        embed_dim: Cross-attention working dimension.
        lstm_hidden: LSTM hidden size.
        paligemma_model_name: HF id of the PaliGemma backbone.
        lora_rank / lora_alpha: LoRA hyper-parameters.
        aux_goal_head: If True, also regress the body-frame goal vector
            (3-d) from the LSTM hidden state, returning it for an L1 loss.
    """

    def __init__(
        self,
        history_frames: int = 2,
        embed_dim: int = 256,
        lstm_hidden: int = 256,
        paligemma_model_name: str = "google/paligemma-3b-pt-224",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        aux_goal_head: bool = True,
    ) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.embed_dim = int(embed_dim)
        self.lstm_hidden = int(lstm_hidden)
        self.aux_goal_head = bool(aux_goal_head)

        self.paligemma = PaliGemmaFeatureExtractor(
            model_name=paligemma_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        for p in self.paligemma.parameters():
            p.requires_grad = False
        for n, p in self.paligemma.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        feat = PaliGemmaFeatureExtractor.FEATURE_DIM
        self.image_proj = nn.Linear(feat, embed_dim)
        self.text_proj = nn.Linear(feat, embed_dim)

        # Per-frame embedding so the model can tell current vs history tokens.
        self.frame_embed = nn.Embedding(self.history_frames + 1, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )

        # Pose features: 4-d pose (x, y, z, yaw) → encoded into the LSTM input.
        self.pose_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.ELU(),
            nn.Linear(32, 32),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim + 32,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ELU(),
            nn.Linear(128, NUM_OPENFLY_ACTIONS),
        )

        if self.aux_goal_head:
            self.goal_head = nn.Sequential(
                nn.Linear(lstm_hidden, 64),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PaliGemma over each frame.

        Returns
        -------
        image_tokens : (B, T*256, embed_dim)  with frame embedding added
        text_tokens : (B, L, embed_dim)
        text_mask   : (B, L)
        """
        B = rgb_current.shape[0]
        n_img = self.paligemma._img_size  # noqa: SLF001 — informational only

        rgb_current_f = self._rgb_uint8_to_float(rgb_current)
        if rgb_history.numel() > 0:
            rgb_history_f = self._rgb_uint8_to_float(rgb_history)
            frames = [rgb_history_f[:, k] for k in range(rgb_history_f.shape[1])]
            frames.append(rgb_current_f)
        else:
            frames = [rgb_current_f]

        token_per_frame = 256  # PaliGemma image tokens
        gemma_text_only_mask = instruction_mask  # (B, L)

        siglip_chunks = []
        gemma_last = None
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
            siglip_chunks.append(siglip_emb)
            gemma_last = gemma_feats

        image_tokens = torch.cat(siglip_chunks, dim=1)  # (B, T*256, embed_dim)

        # Text tokens come from the last forward pass; PaliGemma prepends
        # ``token_per_frame`` image tokens to the sequence, so strip them.
        assert gemma_last is not None
        text_feats = gemma_last[:, token_per_frame:]
        text_tokens = self.text_proj(text_feats)
        text_mask = gemma_text_only_mask[:, token_per_frame:] if gemma_text_only_mask.shape[1] > token_per_frame else gemma_text_only_mask
        return image_tokens, text_tokens, text_mask

    def forward(
        self,
        *,
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        rgb_current: torch.Tensor,
        rgb_history: torch.Tensor,
        pose: torch.Tensor,
        with_grad: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute action logits (and optional goal regression) for one step.

        Args:
            instruction_input_ids:  (B, L)
            instruction_attention_mask: (B, L)
            rgb_current: (B, H, W, 3) uint8
            rgb_history: (B, T, H, W, 3) uint8 — empty tensor allowed
            pose: (B, 4) float32 — [x, y, z, yaw]
            with_grad: enable LoRA gradients during training; False in eval.
        """
        image_tokens, text_tokens, text_mask = self._tokenize(
            instruction_input_ids,
            instruction_attention_mask,
            rgb_current,
            rgb_history,
            with_grad=with_grad,
        )

        fused, _ = self.cross_attn(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            average_attn_weights=True,
        )
        mask = text_mask.unsqueeze(-1).to(fused.dtype)
        scene_summary = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        pose_feat = self.pose_proj(pose.to(fused.dtype))
        lstm_in = torch.cat([scene_summary, pose_feat], dim=-1).unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        memory = lstm_out.squeeze(1)

        action_logits = self.action_head(memory)
        out: dict[str, torch.Tensor] = {"action_logits": action_logits}
        if self.aux_goal_head:
            out["goal_pred"] = self.goal_head(memory)
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
    ) -> int:
        out = self.forward(
            instruction_input_ids=instruction_input_ids,
            instruction_attention_mask=instruction_attention_mask,
            rgb_current=rgb_current,
            rgb_history=rgb_history,
            pose=pose,
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
