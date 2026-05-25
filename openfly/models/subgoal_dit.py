"""Feature-space Diffusion Transformer for next-keyframe subgoals.

Predicts the **SigLIP image-token features** of the next action-transition
frame, given the SigLIP tokens of the current frame plus a text + pose
conditioning vector. Operates entirely in the 2048-d PaliGemma feature
space — no pixels, no VAE — so inference is fast (4-step CM target:
~40 ms / sample on an A100).

The choice of "next-transition frame" as the subgoal is intentional:
OpenFly's A* planner alternates *runs* of one primitive (e.g. seven
``forward_3m`` in a row) and single transition actions (turn, ascend),
and the templated ``sub_instruction`` already describes the current run.
That makes ``(sub_instruction text, end-of-run frame)`` a natural paired
training signal — see ``openfly.dataset.OpenFlySample.subgoal_rgb``.

Design choices that the docstrings call out:

* **Why feature-space, not pixel-space.** PaliGemma already eats SigLIP
  tokens; predicting pixels just to re-encode them with SigLIP would be
  wasted compute. The cost is interpretability — to *see* a predicted
  subgoal as an image you'd need a separate decoder (deferred).
* **Why diffusion, not a deterministic regressor.** The next-transition
  frame is multimodal: at an intersection the trajectory might turn left
  or right. An MSE regressor would average them; diffusion samples from
  the distribution.
* **Why text + pose + last-action conditioning, not just sub-instruction.**
  OpenFly's templated sub-instructions ("move forward 6 meters") are
  information-poor by construction. Conditioning on the full instruction
  *plus* the templated sub-instruction *plus* the body-frame pose delta
  to the subgoal gives the DiT a richer signal and forces it to predict
  visual content (not just re-emit the action label).
* **Why an explicit pose-delta condition.** The env is kinematic — the
  pose at the subgoal step is deterministic given the action sequence.
  Feeding it explicitly lets the DiT focus on *what the scene looks like
  there* rather than reinventing forward kinematics inside its blocks.

Architecture is a standard DiT with AdaLN-Zero modulation:

* Two streams of tokens — 256 current + 256 noisy subgoal — concatenated
  along the sequence axis with a learned role embedding.
* Conditioning vector ``c = time_embed(t) + text_proj(text_embed) +
  pose_proj(pose_delta) + last_action_emb(last_action)``.
* ``N`` DiT blocks: AdaLN-scale/shift on attention + MLP, both pre-norm,
  residual gating via AdaLN ``alpha`` (initialised to zero so the model
  starts as identity on the noisy subgoal half).
* Output projection reads only the subgoal half of the token stream.

Parameter budget at the defaults (depth=12, hidden=1024, heads=16):
* Token I/O projections: ~8 M
* 12 DiT blocks × ~12 M = ~145 M
* Total trainable: ~155 M

That's a one-A100 model. Bumping depth=24 or hidden=1280 takes you to
~400 M if you want more capacity later.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (DDPM-style)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer sinusoidal positional embedding repurposed for ``t``."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: (B,) integer or float timesteps. Returns (B, dim)."""
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# AdaLN-Zero modulated DiT block
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: ``x * (1 + scale) + shift``. ``shift/scale``: (B, D)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-norm transformer block with AdaLN-Zero conditional modulation.

    Each block takes a conditioning vector ``c`` (B, D) and produces six
    modulation vectors (shift_msa, scale_msa, gate_msa, shift_mlp,
    scale_mlp, gate_mlp). ``gate`` is initialised to zero (identity
    residual) so the model starts as a no-op on the noisy half.
    """

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 6 * hidden),
        )
        # Zero-init the gate path (DiT-zero) so the residual contribution
        # starts at zero — training is stable from step 0.
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(query=h, key=h, value=h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """Final AdaLN + linear: hidden → token_dim."""

    def __init__(self, hidden: int, token_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, token_dim)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(c).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Cosine noise schedule (DDPM)
# ---------------------------------------------------------------------------

def cosine_alpha_bar(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Improved-DDPM cosine schedule. Returns ``alpha_bar`` of shape (T+1,).

    ``alpha_bar[t] = cos²((t/T + s) / (1 + s) * pi / 2)`` clamped for
    numerical stability. Used to build ``alpha_bar[t]`` and noise the
    target features as
    ``x_t = sqrt(alpha_bar[t]) x_0 + sqrt(1 - alpha_bar[t]) eps``.
    """
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    f = torch.cos(((x / num_steps + s) / (1 + s)) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(min=1e-5, max=0.9999).float()


# ---------------------------------------------------------------------------
# SubgoalDiT
# ---------------------------------------------------------------------------

class SubgoalDiT(nn.Module):
    """Feature-space DiT for next-keyframe subgoal prediction.

    Args:
        token_dim: PaliGemma SigLIP feature width (2048).
        hidden:    DiT working dimension (1024 at the defaults).
        depth:     Number of DiT blocks.
        num_heads: Self-attention heads per block.
        mlp_ratio: MLP hidden = ``hidden * mlp_ratio``.
        text_dim:  Width of the text conditioning vector (2048 — matches
                   PaliGemma's hidden state at the last text token).
        pose_delta_dim: Width of the body-frame pose-delta input (4 by
                   default — [dx_body, dy_body, dz, dyaw]).
        num_last_actions: Vocab size for the optional last-action
                   conditioning embedding (``NUM_OPENFLY_ACTIONS + 1``
                   including the START sentinel — 9 by default).
        num_timesteps: Diffusion schedule length used at training time.
                   Inference can use fewer steps (DDIM / consistency).
    """

    def __init__(
        self,
        token_dim: int = 2048,
        hidden: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        text_dim: int = 2048,
        pose_delta_dim: int = 4,
        num_last_actions: int = 9,
        num_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden = int(hidden)
        self.num_timesteps = int(num_timesteps)

        # Token I/O projections (shared between current and noisy subgoal halves)
        self.token_in = nn.Linear(token_dim, hidden)
        self.final = FinalLayer(hidden, token_dim)

        # Role embedding: 0 = current SigLIP, 1 = noisy subgoal SigLIP.
        self.role_embed = nn.Embedding(2, hidden)

        # Conditioning: timestep + text + pose-delta + last-action.
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_delta_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.last_action_embed = nn.Embedding(num_last_actions, hidden)
        # Horizon embedding — # steps to subgoal, capped at 32.
        self.horizon_embed = nn.Embedding(33, hidden)

        # DiT stack
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # Noise schedule (registered as a buffer so it follows ``.to(device)``)
        alpha_bar = cosine_alpha_bar(self.num_timesteps)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

        # Parameter accounting (printed at construction for visibility)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[subgoal_dit] depth={depth} hidden={hidden} heads={num_heads} "
            f"params trainable={n_train:,} total={n_total:,}"
        )

    # ---- conditioning -----------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        pose_delta: torch.Tensor,
        last_action: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        """Build the (B, hidden) conditioning vector for AdaLN modulation."""
        c = self.time_embed(t)
        c = c + self.text_proj(text_embed.to(c.dtype))
        c = c + self.pose_proj(pose_delta.to(c.dtype))
        c = c + self.last_action_embed(last_action.long())
        c = c + self.horizon_embed(horizon.clamp(min=0, max=32).long())
        return c

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        curr_tokens: torch.Tensor,         # (B, 256, token_dim)
        noisy_subgoal: torch.Tensor,       # (B, 256, token_dim)
        t: torch.Tensor,                   # (B,) integer timesteps
        text_embed: torch.Tensor,          # (B, text_dim)
        pose_delta: torch.Tensor,          # (B, pose_delta_dim)
        last_action: torch.Tensor,         # (B,) long
        horizon: torch.Tensor,             # (B,) long
    ) -> torch.Tensor:
        """Predict noise on the subgoal half of the token stream.

        Returns: (B, 256, token_dim) — predicted ``eps`` on the
        ``noisy_subgoal`` half. Following DDPM-eps parameterisation.
        """
        B, S, _ = curr_tokens.shape
        assert noisy_subgoal.shape == curr_tokens.shape, (
            f"shape mismatch: curr {curr_tokens.shape} vs noisy {noisy_subgoal.shape}"
        )

        x_curr = self.token_in(curr_tokens)
        x_sub = self.token_in(noisy_subgoal)
        # Add role embeddings (broadcast over the sequence axis)
        x_curr = x_curr + self.role_embed.weight[0]
        x_sub = x_sub + self.role_embed.weight[1]
        x = torch.cat([x_curr, x_sub], dim=1)  # (B, 2S, hidden)

        c = self._build_cond(t, text_embed, pose_delta, last_action, horizon)
        for blk in self.blocks:
            x = blk(x, c)
        x_sub_out = x[:, S:]  # only the subgoal half
        eps_pred = self.final(x_sub_out, c)
        return eps_pred

    # ---- diffusion utilities ---------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-diffuse target tokens to timestep ``t``.

        ``x0``: (B, S, D) clean target SigLIP tokens.
        ``t``:  (B,) integer timesteps in ``[0, num_timesteps)``.

        Returns ``(x_t, noise)`` — both (B, S, D).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t.long()].to(x0.dtype)  # (B,)
        ab = ab.view(-1, 1, 1)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return x_t, noise

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
        """Deterministic-by-default DDIM sampler.

        Inference path used by the policy. ``num_steps=20`` is a sane
        default; once a consistency-model distillation is added this
        will drop to 4.
        """
        device = curr_tokens.device
        B, S, D = curr_tokens.shape

        # Subsample a strictly-decreasing schedule of integer timesteps.
        ts = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1, device=device
        ).round().long()

        # Start from Gaussian noise (same dtype as the inputs).
        x = torch.randn(
            (B, S, D), device=device, dtype=curr_tokens.dtype, generator=generator
        )

        for i in range(num_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            ab_cur = self.alpha_bar[t_cur].to(x.dtype)
            ab_next = self.alpha_bar[t_next].to(x.dtype)

            t_batch = t_cur.expand(B)
            eps_pred = self.forward(
                curr_tokens=curr_tokens,
                noisy_subgoal=x,
                t=t_batch,
                text_embed=text_embed,
                pose_delta=pose_delta,
                last_action=last_action,
                horizon=horizon,
            )
            x0_pred = (x - (1 - ab_cur).sqrt() * eps_pred) / ab_cur.sqrt().clamp(min=1e-6)

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
                + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_pred
                + sigma * noise
            )
        return x
