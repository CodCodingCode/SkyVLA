"""CityNav eval with oracle goal positions + waypoint→discrete mapper.

This measures how well our Stage-2 flight heuristic navigates when the
**goal XYZ is known** (not language grounding). Compare against CityNav
paper baselines (CMA/Seq2Seq) which must infer goals from vision+language.

Requires CITYNAV_ROOT with downloaded trajectories + rasterized maps + image cache.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_DRONE_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--citynav_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val_seen")
    p.add_argument("--difficulty", type=str, default="all")
    p.add_argument("--max_episodes", type=int, default=20)
    p.add_argument("--success_dist", type=float, default=20.0)
    p.add_argument("--eval_max_timestep", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.citynav_root).resolve()
    sys.path.insert(0, str(root))
    import os
    os.chdir(root)  # CityNav paths are relative to repo root

    from vlnce.actions import DiscreteAction
    from vlnce.dataset.generate import generate_episodes_from_mturk_trajectories
    from vlnce.dataset.mturk_trajectory import load_mturk_trajectories
    from vlnce.cityreferobject import get_city_refer_objects
    from vlnce.space import Pose4D, modulo_radians

    from benchmarks.adapters.discrete import state_to_target_body, target_body_to_citynav

    def _moved_pose(pose: Pose4D, action: DiscreteAction) -> Pose4D:
        x, y, z, yaw = pose
        d_forward, d_yaw, dz = action.value
        yaw = modulo_radians(yaw + d_yaw)
        dx = d_forward * np.cos(yaw)
        dy = d_forward * np.sin(yaw)
        return Pose4D(x + dx, y + dy, z + dz, yaw)
    objects = get_city_refer_objects()
    trajectories = load_mturk_trajectories(args.split, args.difficulty)
    episodes = generate_episodes_from_mturk_trajectories(objects, trajectories)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    action_logs = defaultdict(list)
    trajectory_logs = defaultdict(list)

    for episode in tqdm(episodes, desc="citynav-oracle"):
        pose = episode.start_pose
        goal = episode.target_position
        for t in range(args.eval_max_timestep + 1):
            action_logs[episode.id].append(0)
            trajectory_logs[episode.id].append(pose)
            dist = pose.xyz.dist_to(goal)
            if dist < args.success_dist:
                break
            state = np.array([pose.x, pose.y, pose.z, pose.yaw], dtype=np.float32)
            target_body = state_to_target_body(state, [goal.x, goal.y, goal.z])
            aid = target_body_to_citynav(target_body)
            action = DiscreteAction.from_index(aid)
            if action == DiscreteAction.STOP:
                break
            pose = _moved_pose(pose, action)

    navigation_errors = np.array([
        trajectory_logs[eps.id][-1].xyz.dist_to(eps.target_position) for eps in episodes
    ])
    oracle_dists = np.array([
        min(p.xyz.dist_to(eps.target_position) for p in trajectory_logs[eps.id])
        for eps in episodes
    ])
    ne = float(navigation_errors.mean())
    sr = float((navigation_errors < args.success_dist).mean())
    osr = float((oracle_dists <= args.success_dist).mean())
    print(f"\n=== CityNav oracle-waypoint ({args.split}, n={len(episodes)}) ===")
    print(f"  NE:  {ne:.1f} m")
    print(f"  SR:  {sr * 100:.2f}%")
    print(f"  OSR: {osr * 100:.2f}%")
    import json
    out = Path(_DRONE_ROOT) / "logs" / "benchmarks" / f"citynav_oracle_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "CityNav-oracle-waypoint",
        "split": args.split,
        "n_episodes": len(episodes),
        "navigation_error_m": ne,
        "success_rate": sr,
        "oracle_success_rate": osr,
        "success_dist_m": args.success_dist,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
