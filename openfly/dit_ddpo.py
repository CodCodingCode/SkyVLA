"""DDPO / GRPO reinforcement-learning core for the subgoal world model.

This module turns the K-step stochastic DDIM reverse chain of
:class:`openfly.models.subgoal_dit_pixart.PixArtSubgoalDiT` into a K-step
Markov decision process and provides everything needed to fine-tune the
denoiser with GRPO (group-relative PPO, no value head), as in DDPO
(Black et al. 2023, arXiv 2305.13301) and GRPO (Shao et al. 2024,
arXiv 2402.03300).

Mental model
------------
Each DDIM step ``i`` (going from timestep ``t_cur`` to ``t_next``) is a
diagonal-Gaussian policy::

    eps     = dit.forward(curr_tokens, x, t_cur, <conds>)
    x0_pred = (x - sqrt(1-ab_cur)*eps) / sqrt(ab_cur)
    sigma   = eta * sqrt((1-ab_next)/(1-ab_cur)) * sqrt(1 - ab_cur/ab_next)
    mean    = sqrt(ab_next)*x0_pred + sqrt(1 - ab_next - sigma^2)*eps
    x_next  = mean + sigma * noise            (the sampled "action")
    logprob = sum_event[ -0.5*((x_next-mean)/sigma)^2 - log(sigma)
                         - 0.5*log(2*pi) ]

The DDIM mean/sigma algebra is byte-identical to the deterministic
sampler ``PixArtSubgoalDiT.ddim_sample`` (subgoal_dit_pixart.py:504-531);
the only differences are (a) ``sigma`` is clamped to ``sigma_min`` so the
log-prob is finite on every step (including the last, where the
deterministic sampler forces sigma=0), and (b) noise is always injected.

The whole chain is differentiable through ``mean`` (hence through the
network's ``eps``); ``sigma`` depends only on the fixed noise schedule, so
it carries no gradient. We therefore:

  * SAMPLE a rollout under ``no_grad`` (storing per-step inputs/outputs and
    the detached old log-probs), then
  * RECOMPUTE log-probs / KL under the current (and reference) parameters
    WITH grad from the stored states, exactly as DDPO does.

Everything probabilistic (log-prob, KL, entropy) is computed in float32
regardless of the network's compute dtype, for numerical stability.

Public API (consumed by the RL harness — names are load-bearing):

    DDPOConfig, DiTConds, SampleTrajectory
    ddim_mean_sigma, gaussian_logprob
    sample_with_logprob, trajectory_logprob
    group_advantages, ppo_step_loss, kl_to_reference
    RewardModel, PolicyLikelihoodReward, RolloutReward
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

__all__ = [
    "DDPOConfig",
    "DiTConds",
    "SampleTrajectory",
    "ddim_mean_sigma",
    "gaussian_logprob",
    "sample_with_logprob",
    "trajectory_logprob",
    "group_advantages",
    "ppo_step_loss",
    "kl_to_reference",
    "RewardModel",
    "PolicyLikelihoodReward",
    "RolloutReward",
]

_LOG_2PI = math.log(2.0 * math.pi)


# ======================================================================
# Config & data containers
# ======================================================================


@dataclass
class DDPOConfig:
    """Hyper-parameters for the DDPO/GRPO fine-tune.

    Attributes
    ----------
    eta:
        DDIM stochasticity. ``1.0`` => full DDPM-equivalent noise (needed
        for a meaningful policy gradient — at eta=0 the chain is
        deterministic and there is no exploration).
    num_sample_steps:
        K, the number of DDIM reverse steps = MDP horizon. Default 4 to
        match deployment (``PaliGemmaVLNPolicy.subgoal_sample_steps``).
    sigma_min:
        Floor on the per-step std so the Gaussian log-prob stays finite on
        every step, including the final step where the schedule sigma is 0.
    clip_ratio:
        PPO clipping epsilon.
    kl_coef:
        Weight on the closed-form KL-to-reference regulariser.
    group_size:
        G — trajectories drawn per conditioning context for the GRPO
        group-relative advantage baseline.
    ppo_epochs:
        Inner optimisation passes over the SAME rollout buffer.
    entropy_coef:
        Optional entropy bonus weight (subtracted from the loss).
    max_grad_norm:
        Gradient-norm clip used by the harness (carried here for a single
        source of truth).
    """

    eta: float = 1.0
    num_sample_steps: int = 4
    sigma_min: float = 1e-3
    clip_ratio: float = 0.2
    kl_coef: float = 0.1
    group_size: int = 8
    ppo_epochs: int = 4
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class DiTConds:
    """The fixed conditioning passed to ``PixArtSubgoalDiT.forward`` per step.

    Mirrors the non-``noisy_subgoal``/``t`` arguments of
    ``PixArtSubgoalDiT.forward`` (subgoal_dit_pixart.py:318):
    ``curr_tokens (B,256,D)``, ``text_embed (B,text_dim)``,
    ``pose_delta (B,pose_dim)``, ``last_action (B,) long``,
    ``horizon (B,) long``. ``subgoal_len`` is ``dit.subgoal_len`` (S; 1
    when the model was built with ``subgoal_pool='mean'``).
    """

    curr_tokens: torch.Tensor      # (B, 256, D)
    text_embed: torch.Tensor       # (B, text_dim)
    pose_delta: torch.Tensor       # (B, pose_dim)
    last_action: torch.Tensor      # (B,) long
    horizon: torch.Tensor          # (B,) long
    subgoal_len: int = 1

    @property
    def batch_size(self) -> int:
        return int(self.curr_tokens.shape[0])

    @property
    def device(self) -> torch.device:
        return self.curr_tokens.device

    def repeat(self, g: int) -> "DiTConds":
        """Replicate each conditioning context ``g`` times along dim 0.

        Used to draw the GRPO group: a (B,...) context becomes (B*g,...).
        Repetition is INTERLEAVED (``repeat_interleave``) so trajectories
        ``[k, k+1, ..., k+g-1]`` of the flattened batch all share context
        ``k``. ``group_advantages`` assumes this exact contiguous-group
        layout (with ``group_size == g``).
        """
        if g <= 1:
            return self

        def _rep(x: torch.Tensor) -> torch.Tensor:
            return x.repeat_interleave(g, dim=0)

        return DiTConds(
            curr_tokens=_rep(self.curr_tokens),
            text_embed=_rep(self.text_embed),
            pose_delta=_rep(self.pose_delta),
            last_action=_rep(self.last_action),
            horizon=_rep(self.horizon),
            subgoal_len=self.subgoal_len,
        )


@dataclass
class SampleTrajectory:
    """A stored stochastic-DDIM rollout (the PPO replay buffer for one batch).

    All per-step lists have length K = ``cfg.num_sample_steps``.

    x_in[i]   : (B, S, D)  state entering step i (the denoiser input)
    x_out[i]  : (B, S, D)  sampled next state (the "action" taken)
    t_cur[i]  : (B,) long  source timestep for step i
    t_next[i] : (B,) long  target timestep for step i
    logprob_old : (B, K)   detached log-prob of each action under the
                            sampling-time params (the PPO "old" policy)
    final_subgoal : (B, S, D)  x after the last step — the generated subgoal
    """

    x_in: list[torch.Tensor]
    x_out: list[torch.Tensor]
    t_cur: list[torch.Tensor]
    t_next: list[torch.Tensor]
    logprob_old: torch.Tensor
    final_subgoal: torch.Tensor

    @property
    def num_steps(self) -> int:
        return len(self.x_in)

    @property
    def batch_size(self) -> int:
        return int(self.final_subgoal.shape[0])


# ======================================================================
# Shared DDIM step math  (the ONE source of truth — keep byte-identical
# to PixArtSubgoalDiT.ddim_sample, subgoal_dit_pixart.py:504-531)
# ======================================================================


def ddim_mean_sigma(
    eps: torch.Tensor,        # (B, S, D)  network ε-prediction at t_cur
    x: torch.Tensor,          # (B, S, D)  current state
    ab_cur: torch.Tensor,     # scalar tensor  alpha_bar[t_cur]
    ab_next: torch.Tensor,    # scalar tensor  alpha_bar[t_next]
    eta: float,
    sigma_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One stochastic-DDIM step -> (mean, sigma).

    MATH (matches ddim_sample exactly, then clamps sigma)::

        x0_pred = (x - sqrt(1-ab_cur)*eps) / sqrt(ab_cur).clamp(min=1e-6)
        sigma   = eta * sqrt((1-ab_next)/(1-ab_cur)) * sqrt(1 - ab_cur/ab_next)
        sigma   = sigma.clamp(min=sigma_min)
        mean    = sqrt(ab_next)*x0_pred + sqrt(1 - ab_next - sigma^2).clamp(0)*eps

    ``x0_pred`` and the ``sqrt`` of the direction term use the IDENTICAL
    clamps as the deterministic sampler. ``sigma`` is a 0-dim tensor
    broadcast over the event dims; it depends only on the noise schedule
    (no gradient flows through it). Returns ``mean`` (differentiable
    through ``eps``) and ``sigma`` (schedule-only). The reverse step is
    then ``x_next = mean + sigma * noise``.
    """
    x0_pred = (x - (1.0 - ab_cur).sqrt() * eps) / ab_cur.sqrt().clamp(min=1e-6)
    # sigma can be ill-defined as ab_next -> 0 (last step, t_next=0 =>
    # ab_next=0.9999 here so it's fine, but clamp the radicands defensively).
    sigma = (
        eta
        * ((1.0 - ab_next) / (1.0 - ab_cur)).clamp(min=0.0).sqrt()
        * (1.0 - ab_cur / ab_next).clamp(min=0.0).sqrt()
    )
    sigma = sigma.clamp(min=sigma_min)
    dir_coef = (1.0 - ab_next - sigma**2).clamp(min=0.0).sqrt()
    mean = ab_next.sqrt() * x0_pred + dir_coef * eps
    return mean, sigma


def gaussian_logprob(
    x: torch.Tensor,        # (B, S, D)  the sampled value
    mean: torch.Tensor,     # (B, S, D)  distribution mean
    sigma: torch.Tensor,    # scalar or broadcastable std (> 0)
) -> torch.Tensor:
    """Diagonal-Gaussian log-density summed over the event dims -> (B,).

    MATH (per element, then summed over S*D)::

        log N(x; mean, sigma) =
            -0.5*((x-mean)/sigma)^2 - log(sigma) - 0.5*log(2*pi)

    Computed in float32 for stability even when ``x``/``mean`` are bf16.
    ``sigma`` is the per-step scalar std; if it is a 0-dim tensor the
    ``log(sigma)`` term is multiplied by the number of event elements.
    """
    x32 = x.float()
    mean32 = mean.float()
    sigma32 = sigma.float()
    var = sigma32 * sigma32
    # event dims = everything except batch
    event_dims = tuple(range(1, x32.dim()))
    sq = ((x32 - mean32) ** 2) / var
    quad = -0.5 * sq.sum(dim=event_dims)                         # (B,)
    n_event = 1
    for d in event_dims:
        n_event *= x32.shape[d]
    # sigma may be scalar (broadcast) -> log term scales with n_event.
    log_sigma_term = torch.log(sigma32)
    if log_sigma_term.dim() == 0:
        log_sigma_sum = log_sigma_term * n_event
    else:
        log_sigma_sum = log_sigma_term.expand_as(x32).sum(dim=event_dims)
    const = -0.5 * _LOG_2PI * n_event
    return quad - log_sigma_sum + const


# ======================================================================
# Rollout (sampling) and differentiable recomputation
# ======================================================================


def _alpha_bar(dit: Any, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Scalar ``alpha_bar[t]`` (0-dim) in the requested compute dtype."""
    return dit.alpha_bar[t].to(dtype)


def _make_timesteps(dit: Any, k: int, device: torch.device) -> torch.Tensor:
    """``linspace(num_timesteps-1, 0, K+1).round().long()`` — same as sampler."""
    return (
        torch.linspace(dit.num_timesteps - 1, 0, k + 1, device=device)
        .round()
        .long()
    )


@torch.no_grad()
def sample_with_logprob(
    dit: Any,
    conds: DiTConds,
    cfg: DDPOConfig,
    generator: torch.Generator | None = None,
) -> SampleTrajectory:
    """Roll out the K-step stochastic-DDIM chain, recording everything PPO needs.

    Runs under ``no_grad``; ``dit`` should be in eval mode (the caller is
    responsible — training-mode CFG dropout would inject extra randomness).
    For each step it stores the input state ``x_in[i]``, the sampled output
    ``x_out[i]``, the (scalar) source/target timesteps, and the DETACHED
    log-prob of the sampled action (these become the PPO "old" log-probs).

    Returns a :class:`SampleTrajectory`. ``final_subgoal`` is ``x`` after
    the last step, ready to be scored by a :class:`RewardModel`.
    """
    device = conds.device
    b = conds.batch_size
    d = int(conds.curr_tokens.shape[-1])
    s = int(conds.subgoal_len)
    # Sample the initial latent in the same dtype as the model's inputs so
    # the forward pass dtype-casting matches the deterministic sampler.
    init_dtype = conds.curr_tokens.dtype
    x = torch.randn((b, s, d), device=device, dtype=init_dtype, generator=generator)

    ts = _make_timesteps(dit, cfg.num_sample_steps, device)

    x_in: list[torch.Tensor] = []
    x_out: list[torch.Tensor] = []
    t_cur_list: list[torch.Tensor] = []
    t_next_list: list[torch.Tensor] = []
    logp_steps: list[torch.Tensor] = []

    for i in range(cfg.num_sample_steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        ab_cur = _alpha_bar(dit, t_cur, x.dtype)
        ab_next = _alpha_bar(dit, t_next, x.dtype)

        eps = dit.forward(
            curr_tokens=conds.curr_tokens,
            noisy_subgoal=x,
            t=t_cur.expand(b),
            text_embed=conds.text_embed,
            pose_delta=conds.pose_delta,
            last_action=conds.last_action,
            horizon=conds.horizon,
        )
        mean, sigma = ddim_mean_sigma(
            eps, x, ab_cur, ab_next, cfg.eta, cfg.sigma_min
        )
        noise = torch.randn(
            x.shape, device=device, dtype=x.dtype, generator=generator
        )
        x_next = mean + sigma * noise
        logp = gaussian_logprob(x_next, mean, sigma)            # (B,)

        x_in.append(x.detach())
        x_out.append(x_next.detach())
        t_cur_list.append(t_cur.detach())
        t_next_list.append(t_next.detach())
        logp_steps.append(logp.detach())

        x = x_next

    logprob_old = torch.stack(logp_steps, dim=1)                 # (B, K)
    return SampleTrajectory(
        x_in=x_in,
        x_out=x_out,
        t_cur=t_cur_list,
        t_next=t_next_list,
        logprob_old=logprob_old,
        final_subgoal=x.detach(),
    )


def _step_mean_sigma(
    dit: Any,
    conds: DiTConds,
    x_in: torch.Tensor,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    cfg: DDPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute (mean, sigma) for one stored step under the CURRENT params.

    ``eps`` is differentiable; ``sigma`` is schedule-only. Identical math
    to the sampling path, so old/new log-probs are directly comparable.
    """
    b = x_in.shape[0]
    ab_cur = _alpha_bar(dit, t_cur, x_in.dtype)
    ab_next = _alpha_bar(dit, t_next, x_in.dtype)
    eps = dit.forward(
        curr_tokens=conds.curr_tokens,
        noisy_subgoal=x_in,
        t=t_cur.expand(b),
        text_embed=conds.text_embed,
        pose_delta=conds.pose_delta,
        last_action=conds.last_action,
        horizon=conds.horizon,
    )
    return ddim_mean_sigma(eps, x_in, ab_cur, ab_next, cfg.eta, cfg.sigma_min)


def trajectory_logprob(
    dit: Any,
    traj: SampleTrajectory,
    conds: DiTConds,
    cfg: DDPOConfig,
) -> torch.Tensor:
    """Recompute per-step log-probs of the STORED actions under current params.

    WITH grad: re-runs ``dit.forward`` on each stored ``x_in[i]`` to get a
    fresh ``mean`` (and the schedule-only ``sigma``), then evaluates
    ``gaussian_logprob(x_out[i], mean, sigma)``. Gradients flow through
    ``mean`` (the network). Returns ``(B, K)`` — the PPO "new" log-probs.
    """
    logps: list[torch.Tensor] = []
    for i in range(traj.num_steps):
        mean, sigma = _step_mean_sigma(
            dit, conds, traj.x_in[i], traj.t_cur[i], traj.t_next[i], cfg
        )
        logps.append(gaussian_logprob(traj.x_out[i], mean, sigma))
    return torch.stack(logps, dim=1)                            # (B, K)


# ======================================================================
# GRPO advantages, PPO loss, KL-to-reference
# ======================================================================


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """GRPO group-relative advantages -> ``(N,)``.

    Rewards are laid out in contiguous groups of ``group_size`` (matching
    ``DiTConds.repeat`` / ``repeat_interleave``): all trajectories of one
    conditioning context are adjacent. Within each group::

        adv = (R - mean_group(R)) / (std_group(R) + 1e-8)

    No value head — the group mean is the baseline (GRPO). Returns a flat
    ``(N,)`` advantage, one per trajectory, in float32.
    """
    r = rewards.float().reshape(-1)
    n = r.shape[0]
    if group_size <= 1:
        # Degenerate: a single-member group has no relative signal.
        return torch.zeros_like(r)
    if n % group_size != 0:
        raise ValueError(
            f"rewards length {n} not divisible by group_size {group_size}"
        )
    groups = r.view(-1, group_size)                             # (n_ctx, G)
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, keepdim=True)
    adv = (groups - mean) / (std + 1e-8)
    return adv.reshape(-1)


def ppo_step_loss(
    logp_new: torch.Tensor,   # (B, K)
    logp_old: torch.Tensor,   # (B, K)
    adv: torch.Tensor,        # (B,)
    clip_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped surrogate policy-gradient loss -> (loss, stats).

    MATH (per (trajectory, step); advantage broadcast across the K steps)::

        ratio = exp(logp_new - logp_old)          # logratio clamped first
        surr1 = ratio * adv
        surr2 = clamp(ratio, 1-clip, 1+clip) * adv
        pg    = -mean( min(surr1, surr2) )

    ``logp_old`` is treated as a constant (detached). The log-ratio is
    clamped to ``[-20, 20]`` before ``exp`` so a stale/diverged step can't
    produce an inf ratio. ``stats`` carries float scalars for logging
    (policy loss, mean ratio, clip fraction, approx-KL, mean advantage).
    """
    logp_old = logp_old.detach()
    adv = adv.detach().to(logp_new.dtype)
    adv_b = adv.unsqueeze(1)                                    # (B,1) -> broadcast over K

    logratio = (logp_new - logp_old).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(logratio)
    surr1 = ratio * adv_b
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
    pg_loss = -torch.min(surr1, surr2).mean()

    with torch.no_grad():
        clip_frac = (
            (torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()
        )
        # standard PPO approx-KL estimator (Schulman): mean(ratio-1 - logratio)
        approx_kl = (ratio - 1.0 - logratio).mean().item()
        stats = {
            "pg_loss": float(pg_loss.detach().item()),
            "ratio_mean": float(ratio.mean().item()),
            "clip_frac": float(clip_frac),
            "approx_kl": float(approx_kl),
            "adv_mean": float(adv.mean().item()),
            "adv_std": float(adv.std(unbiased=False).item()),
        }
    return pg_loss, stats


def kl_to_reference(
    dit: Any,
    ref_dit: Any,
    traj: SampleTrajectory,
    conds: DiTConds,
    cfg: DDPOConfig,
) -> torch.Tensor:
    """Closed-form KL( pi_theta || pi_ref ) summed over the chain -> ``(B,)``.

    Both policies are diagonal Gaussians with the SAME (schedule-only)
    ``sigma`` and different means, so per step::

        KL = sum_event[ (mean_theta - mean_ref)^2 / (2*sigma^2) ]

    summed over the K steps. ``mean_theta`` is differentiable (current
    params); ``mean_ref`` is computed under ``no_grad`` from the frozen
    reference copy. Done in float32. Returns ``(B,)``.
    """
    kl_total: torch.Tensor | None = None
    for i in range(traj.num_steps):
        mean_theta, sigma = _step_mean_sigma(
            dit, conds, traj.x_in[i], traj.t_cur[i], traj.t_next[i], cfg
        )
        with torch.no_grad():
            mean_ref, _ = _step_mean_sigma(
                ref_dit, conds, traj.x_in[i], traj.t_cur[i], traj.t_next[i], cfg
            )
        diff = mean_theta.float() - mean_ref.float()
        var = (sigma.float() ** 2)
        event_dims = tuple(range(1, diff.dim()))
        kl_step = (diff * diff).sum(dim=event_dims) / (2.0 * var)   # (B,)
        kl_total = kl_step if kl_total is None else kl_total + kl_step
    assert kl_total is not None  # num_steps >= 1
    return kl_total


# ======================================================================
# Reward models
# ======================================================================


class RewardModel(ABC):
    """Scores a generated subgoal against a dataset batch -> reward ``(B,)``.

    Implementations must be side-effect-free and return a 1-D float tensor
    of length B (higher = better). Scoring runs OUTSIDE the policy-gradient
    graph (rewards are constants), so implementations should use
    ``no_grad`` internally.
    """

    @abstractmethod
    def score(
        self, final_subgoal: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        """Return ``(B,)`` float reward for ``final_subgoal`` (B, S, D)."""
        raise NotImplementedError


class PolicyLikelihoodReward(RewardModel):
    """DEFAULT, simulator-free reward: how well does the generated subgoal
    help a FROZEN policy predict the ground-truth action?

    Feeds ``final_subgoal`` into the frozen ``PaliGemmaVLNPolicy`` via its
    ``subgoal_tokens`` path and reads back the log-probability the policy
    assigns to the ground-truth action (equivalently ``-cross_entropy``).
    Higher likelihood => the subgoal is a more useful prediction => higher
    reward. No environment / AirSim involved.

    The exact policy call site lives behind ``score_fn`` so this library
    stays decoupled from the policy's precise forward signature (which the
    harness, with the live policy object, supplies). If ``score_fn`` is not
    given, a best-effort default is used that:

      1. expects ``batch['gt_action']`` (or ``batch['action']``) long ``(B,)``,
      2. calls ``policy(subgoal_tokens=final_subgoal, **policy_kwargs)`` and
         expects either action ``logits``/``action_logits`` ``(B, A)`` back
         (a tensor or a dict/obj exposing one of those attributes),
      3. returns the per-sample log-prob of the GT action (``-CE``).

    Either path runs under ``no_grad`` and returns float32 ``(B,)`` on the
    subgoal's device.
    """

    def __init__(
        self,
        policy: Any,
        *,
        score_fn: Callable[[Any, torch.Tensor, dict[str, Any]], torch.Tensor]
        | None = None,
        action_key: str = "gt_action",
        policy_kwargs: dict[str, Any] | None = None,
        negative_ce: bool = True,
    ) -> None:
        self.policy = policy
        self.score_fn = score_fn
        self.action_key = action_key
        self.policy_kwargs = policy_kwargs or {}
        self.negative_ce = negative_ce
        # Freeze the policy defensively (it should already be eval/frozen).
        if hasattr(policy, "eval"):
            policy.eval()
        if hasattr(policy, "parameters"):
            for p in policy.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def score(
        self, final_subgoal: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        if self.score_fn is not None:
            r = self.score_fn(self.policy, final_subgoal, batch)
            return r.float().reshape(-1).to(final_subgoal.device)
        return self._default_score(final_subgoal, batch)

    @torch.no_grad()
    def _default_score(
        self, final_subgoal: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        gt = batch.get(self.action_key, batch.get("action", None))
        if gt is None:
            raise KeyError(
                f"PolicyLikelihoodReward: batch has neither {self.action_key!r} "
                "nor 'action' (ground-truth action label required). Provide "
                "score_fn for a custom policy call site."
            )
        gt = gt.to(final_subgoal.device).long().reshape(-1)            # (B,)

        out = self.policy(subgoal_tokens=final_subgoal, **self.policy_kwargs)
        logits = self._extract_logits(out)                            # (B, A)
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)                 # (B, A)
        # per-sample log-prob of the GT action == -cross_entropy
        nll = -log_probs.gather(-1, gt.unsqueeze(-1)).squeeze(-1)     # (B,)
        reward = -nll if self.negative_ce else (-nll)  # log-likelihood == -CE
        return reward.float().reshape(-1)

    @staticmethod
    def _extract_logits(out: Any) -> torch.Tensor:
        """Pull action logits out of a tensor / dict / object policy return."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            for k in ("logits", "action_logits", "action_logit"):
                if k in out:
                    return out[k]
            raise KeyError(
                "PolicyLikelihoodReward: policy dict output has no "
                "'logits'/'action_logits'; provide score_fn."
            )
        for attr in ("logits", "action_logits", "action_logit"):
            if hasattr(out, attr):
                return getattr(out, attr)
        raise TypeError(
            "PolicyLikelihoodReward: could not extract action logits from "
            f"policy output of type {type(out)!r}; provide score_fn."
        )


class RolloutReward(RewardModel):
    """HOOK for a true closed-loop reward (AirSim episode reward).

    Wraps a ``reward_fn(final_subgoal, batch) -> (B,)`` that drives the
    simulator (``compute_episode_reward`` / ``collect_episode``). Because
    the simulator is NOT confirmed to run in this environment, the default
    ``reward_fn`` raises unless one is supplied — keeping
    :class:`PolicyLikelihoodReward` the safe default. Pass a working
    ``reward_fn`` (and use ``--reward rollout`` in the harness) only when a
    contract/probe has confirmed AirSim is available.
    """

    def __init__(
        self,
        reward_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]
        | None = None,
    ) -> None:
        self.reward_fn = reward_fn

    @torch.no_grad()
    def score(
        self, final_subgoal: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        if self.reward_fn is None:
            raise RuntimeError(
                "RolloutReward requires a working simulator reward_fn. AirSim "
                "is not confirmed available in this environment; use the "
                "default PolicyLikelihoodReward (sim-free) unless --reward "
                "rollout is explicitly forced with a real reward_fn."
            )
        r = self.reward_fn(final_subgoal, batch)
        return r.float().reshape(-1).to(final_subgoal.device)
