"""Continuous (Gaussian) MLP actor-critic for drone pick-and-place.

State-based PPO policy: a small MLP maps the 19-D state to a diagonal-Gaussian
over the 6-D continuous action (drone vel + yaw rate + lower + grip) plus a
value. Tanh-squashing is handled by the env (it clips actions to [-1,1]); here
we keep a plain Gaussian with a learned, state-independent log-std, which is the
standard, stable choice for on-policy PPO on low-dim control.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 19, act_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        nn.init.orthogonal_(self.mu.weight, gain=0.01); nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0); nn.init.zeros_(self.v.bias)

    def _dist(self, h):
        std = self.log_std.clamp(-5.0, 2.0).exp()
        return torch.distributions.Normal(self.mu(h), std)

    def forward(self, state):
        h = self.trunk(state)
        return self.mu(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, state):
        h = self.trunk(state)
        dist = self._dist(h)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a, logp, self.v(h).squeeze(-1)

    @torch.no_grad()
    def act_mean(self, state):
        """Deterministic action (eval/demo)."""
        return torch.tanh(self.mu(self.trunk(state)))  # squashed to valid range

    def evaluate(self, state, actions):
        h = self.trunk(state)
        dist = self._dist(h)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, self.v(h).squeeze(-1)
