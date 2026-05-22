#!/usr/bin/env python3
"""Sanity-check :class:`AirSimVLNEnv` with the heuristic policy.

Drives a few episodes through the Gymnasium wrapper, prints reward
components, and writes a JSONL of trajectories under
``logs/openfly/rollouts/``. This is the validation gate G2 in the plan.

Usage:
  source ~/drone_project/openfly/activate.sh
  python -m openfly.scripts.smoke_rl_env --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_DRONE_ROOT = Path(__file__).resolve().parents[2]
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import goal_heuristic_action
from openfly.envs import AirSimVLNEnv, AirSimVLNEnvConfig
from openfly.rollout import (
    aggregate_metrics,
    collect_episode,
    save_trajectories_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--split", type=str, default="seen")
    p.add_argument("--env_filter", type=str, default="env_airsim_16")
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default=str(
            _DRONE_ROOT
            / "logs"
            / "openfly"
            / "rollouts"
            / "smoke_rl_env.jsonl"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = AirSimVLNEnvConfig(
        split=args.split,
        env_filter=args.env_filter or None,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    env = AirSimVLNEnv(cfg)

    rng = np.random.default_rng(args.seed)

    def heuristic_policy(obs, info):
        goal = obs["goal"].tolist()
        pose = obs["pose"].tolist()
        action_id = goal_heuristic_action(pose, goal)
        return action_id, {"source": "heuristic"}

    def random_policy(obs, info):
        return int(rng.integers(0, env.action_space.n)), {"source": "random"}

    policy_fn = heuristic_policy if args.policy == "heuristic" else random_policy

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    trajectories = []
    for i in range(args.episodes):
        traj = collect_episode(env, policy_fn, capture_obs=False)
        trajectories.append(traj)
        last_info = traj.step_infos[-1] if traj.step_infos else {}
        print(
            f"[smoke] ep {i:02d} steps={traj.steps} reward={traj.total_reward:+.3f} "
            f"SR={int(traj.success)} OSR={int(traj.osr)} NE={traj.ne_m:.1f} m "
            f"SPL={traj.spl:.3f} | success_term={last_info.get('success_term', 0):.2f} "
            f"spl_term={last_info.get('spl_term', 0):.2f} "
            f"ne_term={last_info.get('ne_term', 0):.2f}",
            flush=True,
        )

    save_trajectories_jsonl(trajectories, out_path)
    summary = aggregate_metrics(trajectories)
    print(f"\n[smoke] summary {summary}\n[smoke] wrote {out_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
