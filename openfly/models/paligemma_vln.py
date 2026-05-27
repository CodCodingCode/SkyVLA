"""PaliGemma BC policy for OpenFly's 8-class supervised action space.

The head outputs logits over the eight action ids OpenFly's A* planner
actually emits (``openfly.actions.TRAINABLE_ACTION_IDS``). Strafe actions
(raw ids 6 / 7) never appear in ``train.json``, so giving them dedicated
logits only burns capacity. The simulator still speaks raw OpenFly ids;
policy adapters remap argmax → raw id at the env boundary.

Architecture (post-leakage-fix):

* Single RGB stream: current frame keeps all 256 SigLIP tokens, each history
  frame is pooled to one ``[CLS]`` token via cross-attention. History frames
  are dropped out entirely with probability ``history_dropout_p`` during
  training and replaced with a learned ``frame_drop_embed`` mask token —
  forces the action head to derive context from the current frame +
  subgoal pathway rather than leaning on temporal continuity.
* Cross-attention from text tokens into the image-token stack produces a
  scene pool; we add a Gemma "last-text-token" summary (projected from the
  PaliGemma hidden state at the final non-pad text position of the *current*
  frame pass) so both spatial (SigLIP) and lingual (Gemma) context flow into
  the action head.
* Action head conditions on ``[scene_summary, pose_feat]`` only. The
  previous shortcut features — ``progress`` scalar and ``last_action``
  embedding — were removed: they saturated the floor metric and starved
  the subgoal cross-attention pathway of gradient. ``last_action`` is
  still accepted as a forward kwarg because the frozen subgoal DiT
  consumes it as conditioning; it does not reach the action head.
  ``progress`` is accepted but ignored.
* Optional auxiliary head regresses a 3-d next-step body-frame delta from
  the fused vector. A second aux head predicts trajectory progress from
  the *same* head input (no input-side leakage) — the supervision pushes
  scene+pose features to encode trajectory phase implicitly.

Trainable components: PaliGemma LoRA adapters (q/k/v/o projections, rank 16
by default) + image/text/gemma projections + per-frame [CLS] pool +
frame-drop mask token + cross-attention + pose encoder + action head.

This file intentionally breaks state-dict compatibility with previous
checkpoints (last-action embedding and progress projection removed,
frame-drop embed added); train from scratch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from openfly.models.subgoal_dit import SubgoalDiT
from vla.vla_policy import PaliGemmaFeatureExtractor


# Number of supervised action classes — see ``openfly.actions.TRAINABLE_ACTION_IDS``.
NUM_OPENFLY_ACTIONS = 8


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
        aux_progress_head: bool = True,
        subgoal_dit: SubgoalDiT | None = None,
        subgoal_sample_steps: int = 4,
        history_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.embed_dim = int(embed_dim)
        self.aux_goal_head = bool(aux_goal_head)
        self.aux_progress_head = bool(aux_progress_head)
        self.history_dropout_p = float(history_dropout_p)
        # Optional generative subgoal world model. ``None`` disables the
        # whole pathway — the policy behaves identically to the BC
        # baseline. When set, its predicted next-keyframe SigLIP tokens
        # are projected through ``image_proj`` and appended to the image
        # token stack the cross-attention consumes, with a dedicated
        # frame embedding slot (last index) so the model can distinguish
        # "predicted future" from "actual past/present" tokens.
        self.subgoal_dit = subgoal_dit
        self.subgoal_sample_steps = int(subgoal_sample_steps)
        # Freeze the DiT — it's pretrained in phase P2 (see
        # ``openfly.train_subgoal_dit``). Phase P3 only trains the
        # downstream consumers.
        if self.subgoal_dit is not None:
            for p in self.subgoal_dit.parameters():
                p.requires_grad = False
            self.subgoal_dit.eval()

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
        # Slot layout: [history_0, ..., history_{H-1}, current, predicted_subgoal].
        # The last slot is only used when ``subgoal_dit`` is set; allocating
        # it unconditionally keeps state_dict shape stable across the
        # ablation (BC vs BC+subgoals) without needing two model classes.
        self.frame_embed = nn.Embedding(self.history_frames + 2, embed_dim)
        # Convenience index for the predicted-subgoal slot.
        self._subgoal_frame_slot = self.history_frames + 1

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

        # Learned "history frame dropped" mask token. During training a
        # Bernoulli draw (prob ``history_dropout_p``) per history-frame
        # per-sample replaces the pooled CLS with this token, forcing the
        # action head to justify each prediction from the current frame +
        # subgoal channel rather than temporal continuity from the past.
        self.frame_drop_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Pose features: 4-d pose (x, y, z, yaw) — current robot state.
        # This is the *only* proprioceptive feature reaching the action
        # head. ``last_action`` (previously embedded here) was removed:
        # it correlated trivially with the next action in the
        # forward_9m-dominated action distribution and saturated the loss
        # before the subgoal cross-attention pathway could earn its keep.
        self.pose_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.ELU(),
            nn.Linear(32, 32),
        )

        head_in_dim = embed_dim + 32
        self.action_head = nn.Sequential(
            nn.Linear(head_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, NUM_OPENFLY_ACTIONS),
        )

        if self.aux_progress_head:
            # Predicts a scalar in [0, 1] = fraction of trajectory traversed
            # at the current step. Trained against the dataset's
            # ``progress`` field with smooth-L1. Supervision pushes
            # scene+pose features to encode trajectory phase implicitly,
            # without admitting progress as an input-side shortcut.
            self.progress_head = nn.Sequential(
                nn.Linear(head_in_dim, 64),
                nn.ELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        raw_features  : dict with ``curr_siglip`` (B, 256, FEATURE_DIM) and
            ``text_embed`` (B, FEATURE_DIM) — the un-projected versions of
            the current frame's SigLIP tokens and the Gemma last-text
            token. Used by the optional subgoal DiT, which operates in
            PaliGemma's native 2048-d space.
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
        raw_curr_siglip: torch.Tensor | None = None
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
                # Stash raw SigLIP features for the DiT (still in 2048-d).
                raw_curr_siglip = siglip_feats
            else:
                cls_query = self.frame_cls.expand(B, 1, self.embed_dim).to(siglip_emb.dtype)
                pooled, _ = self.frame_pool(
                    query=cls_query, key=siglip_emb, value=siglip_emb
                )
                # History-frame dropout: with prob ``history_dropout_p`` per
                # sample, replace the pooled CLS with the learned drop
                # token so the model can't rely on temporal continuity
                # from this slot. Only active in training mode.
                if self.training and self.history_dropout_p > 0.0:
                    drop_mask = (
                        torch.rand(B, 1, 1, device=pooled.device)
                        < self.history_dropout_p
                    )
                    drop_token = self.frame_drop_embed.to(pooled.dtype).expand(
                        B, 1, self.embed_dim
                    )
                    pooled = torch.where(drop_mask, drop_token, pooled)
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
            gemma_summary_raw = torch.zeros(
                B,
                PaliGemmaFeatureExtractor.FEATURE_DIM,
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )
            gemma_summary = torch.zeros(
                B,
                self.embed_dim,
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )

        assert raw_curr_siglip is not None
        raw_features = {
            "curr_siglip": raw_curr_siglip,
            "text_embed": gemma_summary_raw,
        }
        return image_tokens, text_tokens, text_mask, gemma_summary, raw_features

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
        progress: torch.Tensor | None = None,
        with_grad: bool = True,
        subgoal_pose_delta: torch.Tensor | None = None,
        subgoal_horizon: torch.Tensor | None = None,
        subgoal_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute action logits (and optional goal regression) for one step.

        Args:
            instruction_input_ids: (B, L)
            instruction_attention_mask: (B, L)
            rgb_current: (B, H, W, 3) uint8
            rgb_history: (B, T, H, W, 3) uint8 — empty tensor allowed
            pose: (B, 4) float32 — [x, y, z, yaw]
            last_action: (B,) long — expert action at step-1, or
                ``NUM_OPENFLY_ACTIONS`` (= 8) as the START sentinel for
                step 0. Not read by the action head; passed through to the
                frozen subgoal DiT, whose embedding table sizes for
                ``NUM_OPENFLY_ACTIONS + 1`` to accept this value.
            next_pose: (B, 4) float32 — [x, y, z, yaw] of the next step
                (or current pose at terminal step). Accepted for inference
                signature parity; the trainer uses it to build the
                ``goal_pred`` regression target.
            progress: (B,) float32 in [0, 1] — fraction of expected
                trajectory completed. Defaults to zeros when omitted
                (legacy callers and the first step of a rollout).
            with_grad: enable LoRA gradients during training; False in eval.
            subgoal_pose_delta: (B, 4) optional body-frame pose delta to
                the expected subgoal. Only consumed when ``subgoal_dit``
                is set. ``None`` falls back to zeros — the DiT must
                handle absent pose conditioning (trained with dropout
                in P2 once added).
            subgoal_horizon: (B,) optional integer "steps to subgoal"
                hint passed to the DiT's horizon embedding. Defaults to
                a constant ``4`` when ``None``.
        """
        image_tokens, text_tokens, text_mask, gemma_summary, raw_features = self._tokenize(
            instruction_input_ids,
            instruction_attention_mask,
            rgb_current,
            rgb_history,
            with_grad=with_grad,
        )

        # ---- subgoal-token pathway ---------------------------------
        # Three modes (priority order):
        #   1. ``subgoal_tokens`` directly supplied → just project + append.
        #      Used by P3 BC + subgoals (oracle path) and P3.5 joint refine.
        #   2. ``subgoal_dit`` set on the policy → run DDIM internally to
        #      generate subgoal tokens. Used at inference / rollout.
        #   3. Neither → no subgoal pathway at all (identical to BC baseline).
        # In all cases the cross-attn block downstream is unchanged; it just
        # sees an extra 256 tokens with a dedicated frame-embedding slot.
        pred_subgoal_raw: torch.Tensor | None = None
        if subgoal_tokens is not None:
            # Caller supplied tokens directly (already in 2048-d SigLIP space)
            pred_subgoal_raw = subgoal_tokens
        elif self.subgoal_dit is not None:
            B = image_tokens.shape[0]
            device = image_tokens.device
            if subgoal_pose_delta is None:
                pose_delta = torch.zeros(B, 4, device=device, dtype=torch.float32)
            else:
                pose_delta = subgoal_pose_delta.to(device).float()
            if subgoal_horizon is None:
                horizon = torch.full((B,), 4, device=device, dtype=torch.long)
            else:
                horizon = subgoal_horizon.to(device).long()

            with torch.no_grad():
                pred_subgoal_raw = self.subgoal_dit.ddim_sample(
                    curr_tokens=raw_features["curr_siglip"],
                    text_embed=raw_features["text_embed"],
                    pose_delta=pose_delta,
                    last_action=last_action.long(),
                    horizon=horizon,
                    num_steps=self.subgoal_sample_steps,
                )

        if pred_subgoal_raw is not None:
            # Project into embed_dim and attach the dedicated frame embedding.
            pred_subgoal_emb = self.image_proj(pred_subgoal_raw.to(image_tokens.dtype))
            pred_subgoal_emb = (
                pred_subgoal_emb + self.frame_embed.weight[self._subgoal_frame_slot]
            )
            image_tokens = torch.cat([image_tokens, pred_subgoal_emb], dim=1)

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
        # ``progress`` and ``last_action`` kwargs are accepted for caller
        # compatibility but do not reach the action head — both were
        # removed as input features after diagnosing shortcut leakage
        # (progress directly encoded "time to stop"; last_action ≈
        # next_action in the forward_9m-dominated distribution).
        # ``last_action`` is still consumed by the frozen subgoal DiT
        # above as its trained conditioning input.
        del progress  # explicitly drop; supervision target lives in the trainer
        head_in = torch.cat([scene_summary, pose_feat], dim=-1)

        action_logits = self.action_head(head_in)
        out: dict[str, torch.Tensor] = {"action_logits": action_logits}
        if self.aux_goal_head:
            out["goal_pred"] = self.goal_head(head_in)
        if self.aux_progress_head:
            out["progress_pred"] = self.progress_head(head_in).squeeze(-1)
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
        progress: torch.Tensor | None = None,
        subgoal_pose_delta: torch.Tensor | None = None,
        subgoal_horizon: torch.Tensor | None = None,
        subgoal_tokens: torch.Tensor | None = None,
    ) -> int:
        out = self.forward(
            instruction_input_ids=instruction_input_ids,
            instruction_attention_mask=instruction_attention_mask,
            rgb_current=rgb_current,
            rgb_history=rgb_history,
            pose=pose,
            last_action=last_action,
            next_pose=next_pose,
            progress=progress,
            with_grad=False,
            subgoal_pose_delta=subgoal_pose_delta,
            subgoal_horizon=subgoal_horizon,
            subgoal_tokens=subgoal_tokens,
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
