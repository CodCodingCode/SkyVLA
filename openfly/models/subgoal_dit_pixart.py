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

        # ---- Text condition adapter -------------------------------------
        # FIX (vs the earlier broken wrapper): we used to project Gemma's
        # 2048-d summary to 4096-d and then run it through PixArt's
        # pretrained ``caption_projection`` (a 4096 → 1152 Linear trained
        # for T5-XXL embeddings). That fed the cross-attention layers
        # encoder_hidden_states with statistics nothing like what they
        # learned to handle, and the model collapsed to a "predict zero"
        # plateau (loss ~0.99 indefinitely). The pretrained
        # ``caption_projection`` is now bypassed: we learn a fresh
        # Gemma 2048 → 1152 projection that directly produces the
        # encoder_hidden_states the transformer blocks expect. This is
        # what diffusers calls "swapping the text encoder."
        self.text_to_caption = nn.Linear(text_dim, self.HIDDEN_DIM)

        # ---- Extra conditioning (pose, last_action, horizon) -----------
        # FIX: we used to fold these directly into PixArt's
        # ``embedded_timestep`` so they participated in the final
        # ``scale_shift_table`` modulation. That blew up the pretrained
        # modulation's expected distribution. They now feed a separate
        # zero-init modulation branch that adds an extra shift/scale to
        # the final hidden, leaving PixArt's pretrained timestep path
        # untouched and learning the new conditioning gradually.
        self.pose_proj = nn.Linear(pose_delta_dim, self.HIDDEN_DIM)
        self.last_action_emb = nn.Embedding(num_last_actions, self.HIDDEN_DIM)
        self.horizon_embed = nn.Embedding(33, self.HIDDEN_DIM)
        # Zero-init the extra-modulation projection so the wrapper starts
        # as a pure PixArt forward (extras contribute nothing) and learns
        # to use them only if training data warrants it. Mirrors DiT-Zero
        # gate-zero init philosophy.
        self.extra_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.HIDDEN_DIM, 2 * self.HIDDEN_DIM),
        )
        nn.init.zeros_(self.extra_modulation[-1].weight)
        nn.init.zeros_(self.extra_modulation[-1].bias)

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

    # ---- Optimizer parameter groups -------------------------------

    def param_groups(
        self,
        *,
        backbone_lr: float = 1e-6,
        adapter_lr: float = 1e-4,
    ) -> list[dict]:
        """Optimizer parameter groups with separate LRs for backbone vs adapters.

        Pretrained ``backbone.*`` params want gentle updates (default 1e-6)
        so the priors don't wash out. The randomly-initialised adapters
        (``token_in``, ``token_out``, ``role_embed``, ``pos_embed``,
        ``text_to_caption``, ``pose_proj``, ``last_action_emb``,
        ``horizon_embed``, ``extra_modulation``) need to learn from
        scratch and want ~100× higher LR (default 1e-4).

        Using a single global LR for all 624M params was the underlying
        reason v1–v4 plateaued at loss ~1.0: adapters never got enough
        gradient signal to escape near-zero predictions while the
        backbone-friendly LR was being applied. This helper fixes that.
        """
        backbone_params = []
        adapter_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                adapter_params.append(p)
        return [
            {"params": backbone_params, "lr": float(backbone_lr), "name": "backbone"},
            {"params": adapter_params,  "lr": float(adapter_lr),  "name": "adapter"},
        ]

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

        # 2) PixArt timestep modulation. Keep ``embedded_t`` PURE here —
        #    the pretrained ``scale_shift_table`` expects this distribution.
        #    Our extras enter through a separate zero-init branch below.
        t_mod, embedded_t = self.backbone.adaln_single(
            t, None, batch_size=B, hidden_dtype=hidden.dtype
        )

        # Build the extra conditioning vector (pose + last_action + horizon).
        # Will only contribute non-trivially once ``extra_modulation`` learns.
        extra_cond = (
            self.pose_proj(pose_delta.to(hidden.dtype))
            + self.last_action_emb(last_action.long())
            + self.horizon_embed(horizon.clamp(min=0, max=32).long())
        )

        # 3) Text → encoder_hidden_states. FIX: bypass the pretrained
        #    ``caption_projection`` (trained for T5-XXL) and feed our
        #    Gemma-projected vector straight into the transformer-block
        #    cross-attention. The pretrained cross-attn KV projections
        #    will be re-learned as a byproduct of the finetune; that's a
        #    lot more stable than feeding them garbage and expecting them
        #    to compensate.
        caption = self.text_to_caption(text_embed.to(hidden.dtype))      # (B, HIDDEN_DIM)
        caption = caption.unsqueeze(1)                                    # (B, 1, HIDDEN_DIM)

        # PixArt-style cross-attention mask (same shape construction as
        # ``PixArtTransformer2DModel.forward`` produces internally).
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

        # 5) Final PixArt-style modulation. ``shift``/``scale`` come from
        #    the pretrained ``scale_shift_table`` + the PURE
        #    ``embedded_t`` (timestep only). Our extras add an extra
        #    shift/scale through a zero-init projection — starts as a no-op,
        #    grows in only if useful.
        shift, scale = (
            self.backbone.scale_shift_table[None] + embedded_t[:, None]
        ).chunk(2, dim=1)
        extra_shift, extra_scale = self.extra_modulation(extra_cond).chunk(2, dim=-1)
        # broadcast extras to (B, 1, HIDDEN_DIM) so they apply uniformly across tokens
        extra_shift = extra_shift.unsqueeze(1)
        extra_scale = extra_scale.unsqueeze(1)

        hidden = self.backbone.norm_out(hidden)
        hidden = hidden * (1 + scale + extra_scale) + shift + extra_shift

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
