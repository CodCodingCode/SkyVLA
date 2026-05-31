#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU-friendly smoke test for :mod:`openfly.dit_ddpo`.

Exercises every public symbol against a tiny **stub** DiT (a few-parameter
``nn.Module`` exposing the exact ``.forward`` / ``.alpha_bar`` / ``.q_sample`` /
``.subgoal_len`` / ``.token_dim`` surface the real ``PixArtSubgoalDiT`` provides)
so the DDPO/GRPO math runs in milliseconds without the 600M backbone.

Checks (each prints PASS/FAIL):
  (a) sample_with_logprob runs and returns correctly-shaped buffers.
  (b) trajectory_logprob on the SAME weights reproduces logprob_old
      (max|delta| < 1e-3 and ratio ~= 1).
  (c) kl_to_reference vs an exact deep-copy reference == 0.
  (d) group_advantages yields ~zero-mean / unit-std advantages per group.
  (e) ppo_step_loss is finite and carries a gradient.

Run:  python -m openfly.scripts.smoke_dit_ddpo
Exit code is non-zero iff any check FAILs.
"""
from __future__ import annotations

import copy
import math
import sys

import torch
import torch.nn as nn

from openfly.dit_ddpo import (
    DDPOConfig,
    DiTConds,
    gaussian_logprob,
    group_advantages,
    kl_to_reference,
    ppo_step_loss,
    sample_with_logprob,
    trajectory_logprob,
)

# Small dims keep the test instant on CPU while still being > 1 token-dim so the
# event-dim summation in gaussian_logprob is non-trivially exercised.
TOKEN_DIM = 16
NUM_TIMESTEPS = 1000
SEED = 0


class StubDiT(nn.Module):
    """Minimal stand-in for ``PixArtSubgoalDiT`` (epsilon-prediction interface).

    Replicates exactly the attributes/methods ``openfly.dit_ddpo`` reads:
      * ``token_dim`` (so ``dit_token_dim`` resolves D without the 2048 fallback)
      * ``subgoal_len``
      * ``alpha_bar`` buffer, shape (NUM_TIMESTEPS + 1,), cosine schedule clamped
        to [1e-5, 0.9999] -- byte-identical construction to the real model.
      * ``q_sample(x0, t, noise=None)`` -> sqrt(ab)*x0 + sqrt(1-ab)*noise
      * ``forward(curr_tokens, noisy_subgoal, t, text_embed, pose_delta,
        last_action, horizon) -> eps`` of shape (B, S, D)

    The forward is a small, fully-differentiable, deterministic function of the
    conditioning + noisy state + (continuous) timestep, so:
      - two forwards with the SAME weights are identical (check b), and
      - a deep-copied module gives identical means (check c, KL == 0).
    """

    def __init__(self, token_dim: int = TOKEN_DIM, subgoal_len: int = 1,
                 num_timesteps: int = NUM_TIMESTEPS):
        super().__init__()
        self.token_dim = token_dim
        self.subgoal_len = subgoal_len
        self.num_timesteps = num_timesteps
        self.register_buffer("alpha_bar", self._cosine_alpha_bar(num_timesteps))
        # A couple of tiny learnable layers so the loss has real parameters to
        # backprop into; init small so eps stays well-scaled.
        self.state_proj = nn.Linear(token_dim, token_dim)
        self.cond_proj = nn.Linear(token_dim, token_dim)
        for m in (self.state_proj, self.cond_proj):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _cosine_alpha_bar(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps) / num_timesteps
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0].clone()
        return ab.clamp(1e-5, 0.9999)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def forward(self, curr_tokens, noisy_subgoal, t, text_embed, pose_delta,
                last_action, horizon):
        # Pool the (B, 256, D) frame tokens + (B, D) text into a (B, 1, D) bias.
        cond = curr_tokens.mean(dim=1) + text_embed  # (B, D)
        cond = self.cond_proj(cond).unsqueeze(1)  # (B, 1, D)
        # Continuous timestep embedding keeps forward a smooth fn of t.
        t_scale = (t.float() / float(self.num_timesteps)).view(-1, 1, 1)
        eps = self.state_proj(noisy_subgoal) + cond + t_scale
        return eps


def _make_conds(batch: int, device: torch.device, subgoal_len: int) -> DiTConds:
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return DiTConds(
        curr_tokens=torch.randn(batch, 256, TOKEN_DIM, generator=g).to(device),
        text_embed=torch.randn(batch, TOKEN_DIM, generator=g).to(device),
        pose_delta=torch.randn(batch, 4, generator=g).to(device),
        last_action=torch.randint(0, 8, (batch,), generator=g).to(device),
        horizon=torch.randint(1, 16, (batch,), generator=g).to(device),
        subgoal_len=subgoal_len,
    )


def main() -> int:
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    cfg = DDPOConfig()  # K=4, eta=1.0, sigma_min=1e-3, group_size=8, etc.

    n_contexts = 2
    group_size = cfg.group_size  # 8
    subgoal_len = 1  # pooled subgoal path (S=1)

    dit = StubDiT(subgoal_len=subgoal_len).to(device).eval()
    ref_dit = copy.deepcopy(dit).eval()  # exact reference -> KL must be 0

    # Replicate each context group_size times (contiguous groups) -> N = B*G.
    base_conds = _make_conds(n_contexts, device, subgoal_len)
    conds = base_conds.repeat(group_size)
    N = conds.batch_size
    assert N == n_contexts * group_size

    results: dict[str, bool] = {}

    # ----------------------------------------------------------------- (a) ---
    gen = torch.Generator(device=device).manual_seed(SEED + 7)
    traj = sample_with_logprob(dit, conds, cfg, generator=gen)
    K = cfg.num_sample_steps
    D = dit.token_dim
    state_shape = (N, subgoal_len, D)
    shapes_ok = (
        traj.num_steps == K
        and len(traj.x_in) == K
        and len(traj.x_out) == K
        and traj.logprob_old.shape == (N, K)
        and traj.final_subgoal.shape == state_shape
        and all(z.shape == state_shape for z in traj.x_in)
        and all(z.shape == state_shape for z in traj.x_out)
        and torch.isfinite(traj.logprob_old).all().item()
    )
    results["(a) sample_with_logprob shapes/finite"] = bool(shapes_ok)

    # ----------------------------------------------------------------- (b) ---
    # Recompute log-probs under the SAME weights -> must match the stored old
    # log-probs (the rollout used these exact params), so ratio ~= 1.
    logp_new = trajectory_logprob(dit, traj, conds, cfg)
    max_delta = (logp_new - traj.logprob_old).abs().max().item()
    ratio = (logp_new - traj.logprob_old).clamp(-20, 20).exp()
    ratio_ok = torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)
    results["(b) trajectory_logprob == logprob_old (max|d|<1e-3, ratio~=1)"] = bool(
        max_delta < 1e-3 and ratio_ok
    )

    # ----------------------------------------------------------------- (c) ---
    kl = kl_to_reference(dit, ref_dit, traj, conds, cfg)
    kl_ok = kl.shape == (N,) and torch.isfinite(kl).all().item() and kl.abs().max().item() < 1e-6
    results["(c) kl_to_reference == 0 vs exact copy"] = bool(kl_ok)

    # ----------------------------------------------------------------- (d) ---
    # Distinct per-trajectory rewards within each group so std > 0.
    rewards = torch.randn(N, generator=torch.Generator().manual_seed(SEED + 3))
    adv = group_advantages(rewards, group_size)
    adv_g = adv.view(n_contexts, group_size)
    mean_ok = adv_g.mean(dim=1).abs().max().item() < 1e-5
    # Sample std (unbiased, matching torch.std default) of normalised advantages.
    std_ok = (adv_g.std(dim=1) - 1.0).abs().max().item() < 1e-4
    results["(d) group_advantages ~zero-mean & unit-std per group"] = bool(
        adv.shape == (N,) and mean_ok and std_ok
    )

    # ----------------------------------------------------------------- (e) ---
    # PPO loss must be finite and produce a non-None gradient on dit params.
    dit.zero_grad(set_to_none=True)
    logp_new_grad = trajectory_logprob(dit, traj, conds, cfg)
    loss, stats = ppo_step_loss(logp_new_grad, traj.logprob_old, adv, cfg.clip_ratio)
    finite_loss = torch.isfinite(loss).item()
    loss.backward()
    grad = dit.state_proj.weight.grad
    has_grad = grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0
    results["(e) ppo_step_loss finite & has grad"] = bool(finite_loss and has_grad)

    # --------------------------------------------------------------- report ---
    print("=" * 68)
    print("smoke_dit_ddpo  (stub DiT, CPU)")
    print("-" * 68)
    print(f"  N(=contexts*G)={N}  K={K}  S={subgoal_len}  D={D}")
    print(f"  (b) max|logp_new - logp_old| = {max_delta:.3e}")
    print(f"  (c) max|KL|                  = {kl.abs().max().item():.3e}")
    print(f"  (d) max|group mean|          = {adv_g.mean(dim=1).abs().max().item():.3e}")
    print(f"      max|group std - 1|       = {(adv_g.std(dim=1) - 1.0).abs().max().item():.3e}")
    print(f"  (e) loss={loss.item():.4f}  pg_loss={stats['pg_loss']:.4f}  "
          f"ratio_mean={stats['ratio_mean']:.4f}  clip_frac={stats['clip_frac']:.3f}")
    print("-" * 68)
    all_ok = True
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("=" * 68)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
