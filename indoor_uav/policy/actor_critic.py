"""Small CNN actor-critic for the GS coverage task (RGB + proprioceptive state).

Deliberately compact (a few-M-param ConvNet, not a foundation model): the task
is geometric exploration, the reward is cheap, and we want fast RL iteration on
one GPU. Consumes the Dict obs from GSCoverageEnv (rgb (B,H,W,3) uint8 +
state (B,6) float) and outputs a categorical policy over the 6 moves plus a
scalar value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_res: int = 64, n_actions: int = 6, state_dim: int = 6):
        super().__init__()
        # RGB encoder: 64 -> 31 -> 14 -> 6, then GAP.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 + state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
        # small-init the heads for stable early training
        for head in (self.actor, self.critic):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def _features(self, rgb_u8: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = rgb_u8.float().div(255.0).permute(0, 3, 1, 2)  # (B,3,H,W)
        return self.trunk(torch.cat([self.cnn(x), state], dim=-1))

    def forward(self, rgb_u8: torch.Tensor, state: torch.Tensor):
        h = self._features(rgb_u8, state)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, rgb_u8, state):
        logits, value = self.forward(rgb_u8, state)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value

    def evaluate(self, rgb_u8, state, actions):
        logits, value = self.forward(rgb_u8, state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
