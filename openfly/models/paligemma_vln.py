"""PaliGemma BC policy for OpenFly's 8-class supervised action space.

The head outputs logits over the eight action ids OpenFly's A* planner
actually emits (``openfly.actions.TRAINABLE_ACTION_IDS``). Strafe actions
(raw ids 6 / 7) never appear in ``train.json``, so giving them dedicated
logits only burns capacity. The simulator still speaks raw OpenFly ids;
policy adapters remap argmax → raw id at the env boundary.

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
checkpoints (the LSTM is gone, the action head shrank 10→8, and the
last-action embedding shrank 11→9); train from scratch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from openfly.models.subgoal_dit import SubgoalDiT
from vla.vla_policy import PaliGemmaFeatureExtractor


# Number of supervised action classes — see ``openfly.actions.TRAINABLE_ACTION_IDS``.
NUM_OPENFLY_ACTIONS = 8
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
        aux_progress_head: bool = True,
        subgoal_dit: SubgoalDiT | None = None,
        subgoal_sample_steps: int = 4,
    ) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.embed_dim = int(embed_dim)
        self.aux_goal_head = bool(aux_goal_head)
        self.aux_progress_head = bool(aux_progress_head)
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

        # Pose features: 4-d pose (x, y, z, yaw).
        self.pose_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.ELU(),
            nn.Linear(32, 32),
        )

        # Last-action embedding: discrete action id at step-1, with
        # LAST_ACTION_START_TOKEN as the sentinel for step 0.
        self.last_action_embed = nn.Embedding(NUM_OPENFLY_ACTIONS + 1, 32)

        # Progress encoder: scalar in [0, 1] → 16-d feature. We pass the
        # raw scalar through a small MLP; this gives the action head an
        # explicit "how far through the trajectory am I" signal so the
        # stop-class can be triggered more decisively as progress → 1.
        # Targets the OSR–SR gap (flies near goal, never stops).
        self.progress_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU(),
            nn.Linear(16, 16),
        )

        head_in_dim = embed_dim + 32 + 32 + 16
        self.action_head = nn.Sequential(
            nn.Linear(head_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, NUM_OPENFLY_ACTIONS),
        )

        # Width of the head input WITHOUT the progress feature. The aux
        # progress head reads from this slice so the supervision target
        # (the ground-truth progress scalar) can never leak through its
        # own input feature — the model is forced to predict trajectory
        # phase from scene + pose + last_action alone.
        self._head_in_dim_no_progress = embed_dim + 32 + 32

        if self.aux_progress_head:
            # Predicts a scalar in [0, 1] = fraction of trajectory traversed
            # at the current step. Trained against the dataset's
            # ``progress`` field with smooth-L1; output passed through a
            # sigmoid so the prediction stays in [0, 1] even before the
            # auxiliary loss converges. Reads from the no-progress slice
            # to prevent input leakage. At inference time, this is also a
            # cleaner "where am I in the trajectory" signal than the
            # ``step_idx / max_steps`` proxy used by the rollout closure.
            self.progress_head = nn.Sequential(
                nn.Linear(self._head_in_dim_no_progress, 64),
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

        # ---- optional subgoal-DiT pathway ---------------------------------
        # Generate a predicted subgoal in PaliGemma's 2048-d feature space,
        # project it into the cross-attention embed space with the dedicated
        # "predicted subgoal" frame slot, and append to ``image_tokens``.
        # The cross-attn block downstream is unchanged — it just sees an
        # extra 256 tokens. With ``subgoal_dit=None`` this whole branch is
        # a no-op and the policy is identical to the BC baseline.
        if self.subgoal_dit is not None:
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
        last_action_emb = self.last_action_embed(last_action.long())
        if progress is None:
            progress_scalar = torch.zeros(
                pose.shape[0], 1, device=pose.device, dtype=scene_summary.dtype
            )
        else:
            progress_scalar = progress.to(scene_summary.dtype).reshape(-1, 1).clamp(0.0, 1.0)
        progress_feat = self.progress_proj(progress_scalar)
        # ``head_in_no_progress`` is the slice the aux progress head reads
        # from so the supervision target can never leak through its own
        # input. Action / goal heads see the full ``head_in`` including
        # progress because that's the conditioning signal we want them
        # to use.
        head_in_no_progress = torch.cat(
            [scene_summary, pose_feat, last_action_emb], dim=-1
        )
        head_in = torch.cat([head_in_no_progress, progress_feat], dim=-1)

        action_logits = self.action_head(head_in)
        out: dict[str, torch.Tensor] = {"action_logits": action_logits}
        if self.aux_goal_head:
            out["goal_pred"] = self.goal_head(head_in)
        if self.aux_progress_head:
            out["progress_pred"] = self.progress_head(head_in_no_progress).squeeze(-1)
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
