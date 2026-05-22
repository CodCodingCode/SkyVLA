#!/usr/bin/env python3
"""OpenFly outdoor aerial VLN benchmark evaluation.

Runs official OpenFly episodes (seen / unseen / eval_test) in AirSim, UE, or
3DGS simulators via the upstream OpenFly-Platform bridges.

Example:
  source ~/drone_project/openfly/activate.sh
  python -m openfly.eval_benchmark --split unseen --policy heuristic --max_episodes 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import apply_action, distance3d, success_within
from openfly.episodes import group_by_env, load_episodes
from openfly.platform import load_eval_module, make_bridge, openfly_root
from openfly.policies import build_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenFly VLN benchmark eval")
    p.add_argument("--split", type=str, default="unseen", help="seen|unseen|eval_test|train")
    p.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        help="heuristic|openfly-agent|paligemma|dagger|grpo|ppo",
    )
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all in split")
    p.add_argument("--env_filter", type=str, default="", help="Substring filter on env name")
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--success_dist", type=float, default=20.0)
    p.add_argument("--output", type=str, default="", help="JSON results path")
    p.add_argument("--model_id", type=str, default="IPEC-COMMUNITY/openfly-agent-7b")
    p.add_argument(
        "--paligemma_ckpt",
        type=str,
        default="",
        help="PaliGemma checkpoint (train_paligemma / train_dagger / train_grpo_paligemma).",
    )
    p.add_argument(
        "--ppo_ckpt",
        type=str,
        default="",
        help="OpenFly-Agent LoRA+value-head checkpoint (train_ppo_openfly_agent).",
    )
    return p.parse_args()


def _pitch_for_episode(image_path: str) -> float:
    return -45.0 if "high" in image_path else 0.0


def _set_pose(bridge, pose, pos_ratio: float, pitch: float) -> None:
    bridge.set_camera_pose(
        pose[0] / pos_ratio,
        pose[1] / pos_ratio,
        pose[2] / pos_ratio,
        pitch,
        math.degrees(pose[3]),
        0,
    )


def _cleanup_env(eval_mod, env_name: str) -> None:
    keywords = ["AirVLN", "guangzhou", "shanghai", "CitySample", "CrashReport"]
    if "airsim" in env_name:
        keywords.append(env_name)
    for kw in keywords:
        eval_mod.kill_env_process(kw)


def run(args: argparse.Namespace) -> dict:
    os.chdir(openfly_root() / "train")
    eval_mod = load_eval_module()

    env_filter = args.env_filter or None
    episodes = load_episodes(
        args.split,
        max_episodes=args.max_episodes,
        env_filter=env_filter,
    )
    groups = group_by_env(episodes)

    policy_kwargs: dict = {}
    pol = args.policy.lower()
    if pol in ("openfly", "openfly-agent", "agent"):
        policy_kwargs["model_id"] = args.model_id
    elif pol in ("paligemma", "vla", "dagger", "grpo"):
        if not args.paligemma_ckpt:
            raise SystemExit(f"--policy {pol} requires --paligemma_ckpt PATH")
        policy_kwargs["checkpoint"] = args.paligemma_ckpt
    elif pol in ("ppo", "ppo-agent", "openfly-agent-rl"):
        if not args.ppo_ckpt:
            raise SystemExit(f"--policy {pol} requires --ppo_ckpt PATH")
        policy_kwargs["checkpoint"] = args.ppo_ckpt
        policy_kwargs["model_id"] = args.model_id
    policy = build_policy(args.policy, **policy_kwargs)

    results: list[dict] = []
    totals = {"n": 0, "success": 0, "osr": 0, "spl_sum": 0.0, "ne_sum": 0.0}

    for env_name, env_eps in groups.items():
        print(f"[openfly] env={env_name} episodes={len(env_eps)}", flush=True)
        time.sleep(3)
        bridge, pos_ratio = make_bridge(env_name, eval_mod)

        for idx, item in enumerate(env_eps):
            instruction = item["gpt_instruction"]
            pos_list = item["pos"]
            start = pos_list[0]
            goal = pos_list[-1]
            yaw0 = item["yaw"][0]
            pose = [start[0], start[1], start[2], yaw0]
            pitch = _pitch_for_episode(item["image_path"])
            history: list[int] = []

            policy.reset(instruction, goal)
            _set_pose(bridge, pose, pos_ratio, pitch)
            time.sleep(0.2)

            pass_len = 1e-3
            old_pose = list(pose)
            flag_osr = 0
            image_error = False

            for step in range(args.max_steps):
                try:
                    rgb = bridge.get_camera_data()
                    action = policy.act(rgb, pose, step, history)
                    history.append(action)
                    new_pose = apply_action(pose, action)
                    _set_pose(bridge, new_pose, pos_ratio, pitch)
                    pass_len += distance3d(old_pose, new_pose)
                    if distance3d(goal, new_pose) < args.success_dist and flag_osr != 2:
                        flag_osr = 2
                    old_pose = new_pose
                    pose = new_pose
                    if action == 0:
                        break
                except Exception as exc:
                    print(f"[openfly] step error ep={idx}: {exc}", flush=True)
                    image_error = True
                    break

            ne = distance3d(goal, pose)
            traj_len = max(distance3d(start, goal), 1e-3)
            success = int(success_within(pose, goal, args.success_dist))
            spl = (traj_len / pass_len) if success else 0.0
            osr = 1 if flag_osr == 2 else 0

            ep_result = {
                "env": env_name,
                "episode_idx": idx,
                "image_path": item["image_path"],
                "instruction": instruction[:120],
                "success": bool(success),
                "osr": bool(osr),
                "ne_m": ne,
                "spl": spl,
                "steps": len(history),
                "image_error": image_error,
            }
            results.append(ep_result)
            totals["n"] += 1
            totals["success"] += success
            totals["osr"] += osr
            totals["spl_sum"] += spl
            totals["ne_sum"] += ne
            print(
                f"[openfly] ep={idx} SR={success} OSR={osr} NE={ne:.1f}m SPL={spl:.3f}",
                flush=True,
            )

        _cleanup_env(eval_mod, env_name)
        del bridge

    n = max(totals["n"], 1)
    summary = {
        "benchmark": "OpenFly-VLN",
        "split": args.split,
        "policy": args.policy,
        "success_rate": totals["success"] / n,
        "osr": totals["osr"] / n,
        "mean_ne_m": totals["ne_sum"] / n,
        "mean_spl": totals["spl_sum"] / n,
        "n_episodes": totals["n"],
        "episodes": results,
    }
    return summary


def main() -> int:
    args = parse_args()
    t0 = time.time()
    summary = run(args)
    summary["elapsed_s"] = time.time() - t0

    out = args.output
    if not out:
        out_dir = _DRONE_ROOT / "logs" / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = str(out_dir / f"openfly_{args.split}_{args.policy}_{ts}.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[openfly] done SR={summary['success_rate']:.3f} "
        f"OSR={summary['osr']:.3f} mean_NE={summary['mean_ne_m']:.1f}m "
        f"→ {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
