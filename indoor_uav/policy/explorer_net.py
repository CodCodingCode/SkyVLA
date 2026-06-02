"""Actor-critic for PhysicsCoverageEnv: RGB + coverage-map + proprio -> action.

Two conv encoders (onboard RGB, top-down nav/visited/frontier map) + the proprio
state, fused into a shared trunk with a categorical policy head and a value head.
The MAP branch is what lets the policy reason about where it hasn't been (find
the upstairs / leave a finished room) — a pure-RGB net structurally can't.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_stack(in_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 5, stride=2), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )


class ExplorerNet(nn.Module):
    def __init__(self, n_actions: int = 6, state_dim: int = 8):
        super().__init__()
        self.rgb_cnn = _conv_stack(3)     # onboard view
        self.map_cnn = _conv_stack(3)     # navigable / visited / frontier
        self.trunk = nn.Sequential(
            nn.Linear(64 + 64 + state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
        for h in (self.actor, self.critic):
            nn.init.orthogonal_(h.weight, gain=0.01); nn.init.zeros_(h.bias)

    def _feat(self, rgb_u8, mp, state):
        rgb = rgb_u8.float().div(255.0).permute(0, 3, 1, 2)
        return self.trunk(torch.cat([self.rgb_cnn(rgb), self.map_cnn(mp), state], dim=-1))

    def forward(self, rgb_u8, mp, state):
        h = self._feat(rgb_u8, mp, state)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, rgb_u8, mp, state):
        logits, v = self.forward(rgb_u8, mp, state)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v

    def evaluate(self, rgb_u8, mp, state, actions):
        logits, v = self.forward(rgb_u8, mp, state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), v
