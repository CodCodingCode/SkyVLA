"""PixArt-Σ pretrained backbone for SubgoalDiT.

Wraps :class:`diffusers.PixArtTransformer2DModel` — DiT-XL/2 (~610M
params) pretrained on 33M curated image-text pairs by Huawei — and
replaces the VAE-latent I/O heads with SigLIP-shaped ones so the same
denoising objective from :mod:`openfly.models.subgoal_dit` benefits
from web-scale visual priors. This is the π0.7 move: don't train the
world model from scratch on aerial data alone, start it from a model
that already knows what scenes look like.

What we reuse vs replace:

  Reused (~600M params, web-pretrained)
    * 28 ``BasicTransformerBlock`` layers (self-attn + cross-attn-to-text + MLP,
      with AdaLN-Single modulation from timestep)
    * Timestep modulation (``adaln_single``)
    * Final norm and modulation (``norm_out`` + ``scale_shift_table``)
    * Text projection (``caption_projection`` — 4096 → 1152)

  Replaced (~20M params, SigLIP-shaped)
    * Input projection: SigLIP 2048 → PixArt hidden 1152 (was VAE 4 → 1152)
    * Output projection: PixArt hidden 1152 → SigLIP 2048 (was 1152 → 4·patch²)
    * Position embeddings for a 512-token sequence (256 curr + 256 subgoal)
    * Role embedding (curr / subgoal) — PixArt has no analog

  Added (SkyVLA-specific conditioning)
    * Gemma text summary 2048 → PixArt caption_channels 4096 (adapter only)
    * Pose-delta, last-action, horizon → folded into the embedded timestep
      so they flow into the final output modulation alongside ``t``.

Architecturally this is closer to π0.7's BAGEL-init recipe than the
from-scratch DiT was. The bet is that the 28 transformer blocks already
encode "what aerial / outdoor scenes look like," and the small adapter
heads just need to translate between SigLIP's feature space and the
backbone's hidden dim. ~150M from-scratch parameters → ~620M
web-pretrained parameters, ~10x parameter scale with no extra data.

Training cost vs the from-scratch DiT (depth=12, hidden=1024):
  * Per-step cost: ~2.5–3× slower (28 deeper layers, bigger hidden)
  * Per-step VRAM: ~2× more activations
  * Convergence: should reach val_cos ≥ 0.65 much faster (the priors do
    the heavy lifting), so total wall-clock to "useful" should be lower.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from openfly.models.subgoal_dit import cosine_alpha_bar


class PixArtSubgoalDiT(nn.Module):
    """SubgoalDiT with a pretrained PixArt-Σ transformer backbone.

    Drop-in replacement for :class:`openfly.models.subgoal_dit.SubgoalDiT`
    with the same ``forward``, ``q_sample``, ``ddim_sample`` interface —
    so the trainer doesn't need to know which backbone it has.
    """

    HIDDEN_DIM = 1152  # PixArt inner dim (num_attention_heads * attention_head_dim = 16 * 72)

    def __init__(
        self,
        pretrained_path: str,
        token_dim: int = 2048,
        text_dim: int = 2048,
        pose_delta_dim: int = 4,
        num_last_actions: int = 9,
        num_timesteps: int = 1000,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_timesteps = int(num_timesteps)

        from diffusers import PixArtTransformer2DModel
        backbone = PixArtTransformer2DModel.from_pretrained(
            pretrained_path, subfolder="transformer", torch_dtype=torch.float32
        )
        # We bypass PixArt's spatial patch embedding and its VAE-shaped
        # output projection. Deleting them frees ~20M params and makes
        # state-dict inspection less confusing.
        del backbone.pos_embed
        del backbone.proj_out
        self.backbone = backbone

        # We will assert the config matches our assumptions so weight
        # shapes downstream don't silently mismatch.
        cfg = backbone.config
        assert cfg.num_attention_heads * cfg.attention_head_dim == self.HIDDEN_DIM, (
            f"PixArt inner_dim {cfg.num_attention_heads * cfg.attention_head_dim} "
            f"!= expected {self.HIDDEN_DIM}; the wrapper assumes hidden=1152"
        )
        # caption_channels is the dim PixArt's caption_projection consumes (4096 for PixArt-Σ).
        self.caption_channels = int(cfg.caption_channels)

        # ---- I/O adapters in SigLIP space -----------------------------
        self.token_in = nn.Linear(token_dim, self.HIDDEN_DIM)
        self.token_out = nn.Linear(self.HIDDEN_DIM, token_dim)
        # Small-std init on the output projection: keeps initial ε-predictions
        # near zero (so the first few steps don't dominate the loss landscape)
        # while still allowing gradients to flow back through the backbone.
        # DON'T zero-init token_out.weight — that would kill gradient flow
        # into the entire backbone via the chain rule.
        nn.init.normal_(self.token_out.weight, std=0.02)
        nn.init.zeros_(self.token_out.bias)

        # ---- Sequence-level embeddings -------------------------------
        # Role: which half is this token from? 0 = current frame, 1 = noisy subgoal.
        self.role_embed = nn.Embedding(2, self.HIDDEN_DIM)
        nn.init.normal_(self.role_embed.weight, std=0.02)
        # Learnable 1D positional embeddings over the 512-token sequence.
        # Initialized small; the backbone's own attention will dominate at start.
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, self.HIDDEN_DIM))
        nn.init.normal_(self.pos_embed, std=0.02)

        # ---- Text condition adapter (Gemma → PixArt caption_channels) ----
        # PixArt-Σ was trained with T5-XXL (4096-d); we feed a single text
        # token assembled from the Gemma summary projected to 4096.
        self.text_to_caption = nn.Linear(text_dim, self.caption_channels)

        # ---- Extra conditioning (pose, last_action, horizon) -----------
        # These get added into PixArt's ``embedded_timestep`` so they
        # influence the final output modulation in the same channel
        # the model is already trained to use for global conditioning.
        self.pose_proj = nn.Linear(pose_delta_dim, self.HIDDEN_DIM)
        self.last_action_emb = nn.Embedding(num_last_actions, self.HIDDEN_DIM)
        self.horizon_embed = nn.Embedding(33, self.HIDDEN_DIM)

        # ---- Diffusion schedule (buffer follows .to(device)) ---------
        alpha_bar = cosine_alpha_bar(self.num_timesteps)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

        # ---- Optional freezing ---------------------------------------
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Sanity / accounting
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[pixart_subgoal_dit] backbone={'frozen' if freeze_backbone else 'trainable'} "
            f"params trainable={n_train:,} total={n_total:,}"
        )

    # ---- Diffusion utilities --------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t.long()].to(x0.dtype).view(-1, 1, 1)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return x_t, noise

    # ---- Forward --------------------------------------------------

    def forward(
        self,
        curr_tokens: torch.Tensor,      # (B, 256, token_dim)
        noisy_subgoal: torch.Tensor,    # (B, 256, token_dim)
        t: torch.Tensor,                # (B,) long timesteps
        text_embed: torch.Tensor,       # (B, text_dim) Gemma summary
        pose_delta: torch.Tensor,       # (B, pose_delta_dim)
        last_action: torch.Tensor,      # (B,) long
        horizon: torch.Tensor,          # (B,) long
    ) -> torch.Tensor:
        """Return ε-prediction on the noisy subgoal half: (B, 256, token_dim)."""
        B, S, _ = curr_tokens.shape
        assert noisy_subgoal.shape == curr_tokens.shape
        device = curr_tokens.device

        # 1) Project to backbone hidden dim, add role + position embeddings.
        x_curr = self.token_in(curr_tokens) + self.role_embed.weight[0]
        x_sub = self.token_in(noisy_subgoal) + self.role_embed.weight[1]
        hidden = torch.cat([x_curr, x_sub], dim=1) + self.pos_embed[:, : 2 * S]

        # Cast to the backbone's expected dtype.
        backbone_dtype = next(self.backbone.parameters()).dtype
        hidden = hidden.to(backbone_dtype)

        # 2) PixArt timestep modulation: (t_mod, embedded_t).
        #    PixArt-Σ doesn't use additional conditions (size/aspect-ratio),
        #    so added_cond_kwargs=None is fine.
        t_mod, embedded_t = self.backbone.adaln_single(
            t, None, batch_size=B, hidden_dtype=hidden.dtype
        )

        # Fold our extra conditioning into embedded_t (used for final modulation).
        extra = (
            self.pose_proj(pose_delta.to(embedded_t.dtype))
            + self.last_action_emb(last_action.long())
            + self.horizon_embed(horizon.clamp(min=0, max=32).long())
        )
        embedded_t = embedded_t + extra

        # 3) Text -> PixArt caption embedding (single text-token).
        caption_raw = self.text_to_caption(text_embed.to(hidden.dtype))  # (B, 4096)
        caption_raw = caption_raw.unsqueeze(1)                            # (B, 1, 4096)
        caption = self.backbone.caption_projection(caption_raw)           # (B, 1, 1152)

        # PixArt's blocks expect a 2D-broadcastable mask formed the same way
        # ``PixArtTransformer2DModel.forward`` builds it.
        ones = torch.ones(B, 1, dtype=hidden.dtype, device=device)
        encoder_attention_mask = (1 - ones) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)      # (B, 1, 1)

        # 4) Run through the 28 pretrained transformer blocks.
        for block in self.backbone.transformer_blocks:
            hidden = block(
                hidden,
                attention_mask=None,
                encoder_hidden_states=caption,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_mod,
                cross_attention_kwargs=None,
                class_labels=None,
            )

        # 5) Final PixArt-style modulation (shift / scale from scale_shift_table + t).
        shift, scale = (
            self.backbone.scale_shift_table[None] + embedded_t[:, None]
        ).chunk(2, dim=1)
        hidden = self.backbone.norm_out(hidden)
        hidden = hidden * (1 + scale) + shift

        # Keep only the subgoal half and project back to SigLIP space.
        subgoal_hidden = hidden[:, S:]                                     # (B, 256, 1152)
        eps_pred = self.token_out(subgoal_hidden.to(self.token_out.weight.dtype))
        return eps_pred

    # ---- DDIM sampling (inference) --------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        curr_tokens: torch.Tensor,
        text_embed: torch.Tensor,
        pose_delta: torch.Tensor,
        last_action: torch.Tensor,
        horizon: torch.Tensor,
        num_steps: int = 20,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = curr_tokens.device
        B, S, D = curr_tokens.shape
        ts = (
            torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device)
            .round()
            .long()
        )
        x = torch.randn(
            (B, S, D), device=device, dtype=curr_tokens.dtype, generator=generator
        )
        for i in range(num_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            ab_cur = self.alpha_bar[t_cur].to(x.dtype)
            ab_next = self.alpha_bar[t_next].to(x.dtype)
            eps = self.forward(
                curr_tokens=curr_tokens,
                noisy_subgoal=x,
                t=t_cur.expand(B),
                text_embed=text_embed,
                pose_delta=pose_delta,
                last_action=last_action,
                horizon=horizon,
            )
            x0_pred = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt().clamp(min=1e-6)
            if eta > 0.0 and i < num_steps - 1:
                sigma = eta * ((1 - ab_next) / (1 - ab_cur)).sqrt() * (
                    1 - ab_cur / ab_next
                ).sqrt()
                noise = torch.randn_like(x)
            else:
                sigma = torch.zeros((), device=device, dtype=x.dtype)
                noise = torch.zeros_like(x)
            x = (
                ab_next.sqrt() * x0_pred
                + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps
                + sigma * noise
            )
        return x
