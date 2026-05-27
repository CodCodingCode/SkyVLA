"""Fast sim SR evaluator with per-class action metrics + stop diagnostics.

Why this exists (vs ``openfly.eval_benchmark``):

* ``eval_benchmark`` reports SR / OSR aggregates only. It does not capture
  per-step model actions, expert actions, or stop-emission timing — those
  are the metrics that diagnose *why* a policy is failing (e.g. flies
  forward past goal vs stops too early vs never deviates from forward).
* We need a script that fits inside a ``few-minute`` budget for periodic
  spot-checks during training: default 20 episodes, ~3-5 min on AirSim.

Reuses ``openfly.platform.make_bridge`` and the same kinematic pose-update
loop ``eval_benchmark`` uses, so SR numbers stay comparable.

CLI:
    python -m openfly.scripts.fast_sr_eval \\
        --checkpoint /path/best.pt --split unseen --n_episodes 20

For heuristic smoke tests (no checkpoint needed):
    python -m openfly.scripts.fast_sr_eval --policy heuristic --n_episodes 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_DRONE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import (
    ACTION_NAMES,
    TRAINABLE_ACTION_IDS,
    apply_action,
    distance3d,
    success_within,
)
from openfly.episodes import group_by_env, load_episodes
from openfly.platform import load_eval_module, make_bridge, openfly_root
from openfly.policies import build_policy


def _make_bridge_with_overrides(
    env_name: str,
    eval_mod,
    *,
    ue_port: int,
    ue_ip: str = "127.0.0.1",
):
    """``platform.make_bridge`` hardcodes UE port 9000; let callers override.

    Falls through to the upstream bridge factory for airsim / gs envs (and
    for UE when the port matches the default), so we only touch behaviour
    on the UE path that actually needs it.
    """
    if "ue" in env_name and ue_port != 9000:
        return (
            eval_mod.UEBridge(ue_ip=ue_ip, ue_port=str(ue_port), env_name=env_name),
            1.0,
        )
    return make_bridge(env_name, eval_mod)


# ---- small replicas of eval_benchmark private helpers -----------------

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


# ---- episode-level rollout --------------------------------------------

def _rollout_episode(
    bridge,
    pos_ratio: float,
    policy,
    episode: dict[str, Any],
    *,
    max_steps: int,
    success_dist: float,
) -> dict[str, Any]:
    """Run one episode. Returns a per-episode record with action traces."""
    instruction = episode["gpt_instruction"]
    pos_list = episode["pos"]
    start = pos_list[0]
    goal = pos_list[-1]
    yaw0 = float(episode["yaw"][0])
    pose = [start[0], start[1], start[2], yaw0]
    pitch = _pitch_for_episode(episode["image_path"])
    expert_actions: list[int] = list(episode.get("action", []))

    policy.reset(instruction, goal)
    _set_pose(bridge, pose, pos_ratio, pitch)
    time.sleep(0.2)

    pass_len = 1e-3
    old_pose = list(pose)
    flag_osr = False
    image_error = False
    model_actions: list[int] = []
    history: list[int] = []
    min_dist_to_goal = distance3d(goal, pose)
    stop_step: int | None = None
    stop_dist: float | None = None

    t0 = time.time()
    for step in range(max_steps):
        try:
            rgb = bridge.get_camera_data()
            action = int(policy.act(rgb, pose, step, history))
            history.append(action)
            model_actions.append(action)

            if action == 0:
                stop_step = step
                stop_dist = distance3d(goal, pose)
                break

            new_pose = apply_action(pose, action)
            _set_pose(bridge, new_pose, pos_ratio, pitch)
            pass_len += distance3d(old_pose, new_pose)
            d_goal = distance3d(goal, new_pose)
            min_dist_to_goal = min(min_dist_to_goal, d_goal)
            if d_goal < success_dist:
                flag_osr = True
            old_pose = new_pose
            pose = new_pose
        except Exception as exc:
            print(f"[fast_sr_eval] step error: {exc}", flush=True)
            image_error = True
            break

    elapsed = time.time() - t0
    final_dist = distance3d(goal, pose)
    success = success_within(pose, goal, success_dist)
    osr = bool(flag_osr or success)
    optimal = max(distance3d(start, goal), 1e-3)
    spl = (optimal / max(pass_len, 1e-3)) if success else 0.0

    return {
        "image_path": episode["image_path"],
        "instruction": instruction[:200],
        "env_name": episode["image_path"].split("/")[0],
        "start": list(start),
        "goal": list(goal),
        "optimal_len": optimal,
        "pass_len": pass_len,
        "success": success,
        "osr": osr,
        "ne_m": final_dist,
        "spl": spl,
        "min_dist_to_goal": min_dist_to_goal,
        "n_model_steps": len(model_actions),
        "n_expert_steps": len(expert_actions),
        "model_actions": model_actions,
        "expert_actions": expert_actions,
        "stop_step": stop_step,
        "stop_dist_to_goal": stop_dist,
        "image_error": image_error,
        "elapsed_s": elapsed,
    }


# ---- aggregate metrics ------------------------------------------------

def _aggregate(records: list[dict[str, Any]], success_dist: float) -> dict[str, Any]:
    n = len(records) or 1

    success_rate = sum(int(r["success"]) for r in records) / n
    osr = sum(int(r["osr"]) for r in records) / n
    mean_ne = sum(r["ne_m"] for r in records) / n
    mean_spl = sum(r["spl"] for r in records) / n
    mean_steps = sum(r["n_model_steps"] for r in records) / n
    mean_elapsed = sum(r["elapsed_s"] for r in records) / n

    # Stop emission stats
    stopped = [r for r in records if r["stop_step"] is not None]
    n_stopped = len(stopped)
    n_stopped_in_success = sum(
        1 for r in stopped if (r["stop_dist_to_goal"] or 1e9) < success_dist
    )
    stop_precision = (
        (n_stopped_in_success / n_stopped) if n_stopped > 0 else None
    )
    n_entered_success = sum(int(r["osr"]) for r in records)
    stop_recall_in_success = (
        (n_stopped_in_success / n_entered_success)
        if n_entered_success > 0 else None
    )

    # Per-class action distribution (over all model steps across all episodes).
    # OpenFly train.json action arrays sometimes contain -1 / -2 sentinels at
    # episode boundaries; ``dataset.py`` trims them but we read raw json here,
    # so filter them out.
    model_class_counts: Counter = Counter()
    expert_class_counts: Counter = Counter()
    for r in records:
        for a in r["model_actions"]:
            ai = int(a)
            if ai in ACTION_NAMES:
                model_class_counts[ai] += 1
        for a in r["expert_actions"]:
            ai = int(a)
            if ai in ACTION_NAMES:
                expert_class_counts[ai] += 1

    # Per-class precision / recall vs expert, comparing position-by-position
    # up to min(n_model, n_expert). This degrades after the first divergence
    # (expert path != model path) but is a reasonable diagnostic averaged
    # across many episodes.
    class_correct: Counter = Counter()
    class_predicted: Counter = Counter()
    class_expert: Counter = Counter()
    n_pairs = 0
    n_correct = 0
    non_fwd_pairs = 0
    non_fwd_correct = 0
    for r in records:
        ma, ea = r["model_actions"], r["expert_actions"]
        for k in range(min(len(ma), len(ea))):
            m, e = int(ma[k]), int(ea[k])
            if e not in ACTION_NAMES or m not in ACTION_NAMES:
                continue  # skip sentinel rows
            class_predicted[m] += 1
            class_expert[e] += 1
            n_pairs += 1
            if m == e:
                class_correct[e] += 1
                n_correct += 1
            # Non-forward conditional accuracy: expert ≠ forward_9m
            if e != 9:
                non_fwd_pairs += 1
                if m == e:
                    non_fwd_correct += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for aid in TRAINABLE_ACTION_IDS:
        pred = class_predicted.get(aid, 0)
        gt = class_expert.get(aid, 0)
        corr = class_correct.get(aid, 0)
        precision = (corr / pred) if pred > 0 else None
        recall = (corr / gt) if gt > 0 else None
        per_class[ACTION_NAMES[aid]] = {
            "raw_id": aid,
            "n_predicted": pred,
            "n_expert": gt,
            "n_correct": corr,
            "precision": precision,
            "recall": recall,
        }

    overall_acc = (n_correct / n_pairs) if n_pairs > 0 else None
    non_fwd_acc = (non_fwd_correct / non_fwd_pairs) if non_fwd_pairs > 0 else None

    # Per-env breakdown
    per_env: dict[str, dict[str, Any]] = {}
    env_groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        env_groups.setdefault(r["env_name"], []).append(r)
    for env_name, eps in sorted(env_groups.items()):
        ne = len(eps)
        per_env[env_name] = {
            "n_episodes": ne,
            "success_rate": sum(int(e["success"]) for e in eps) / ne,
            "osr": sum(int(e["osr"]) for e in eps) / ne,
            "mean_ne_m": sum(e["ne_m"] for e in eps) / ne,
            "mean_spl": sum(e["spl"] for e in eps) / ne,
        }

    return {
        "n_episodes": len(records),
        "success_rate": success_rate,
        "osr": osr,
        "osr_minus_sr_gap": osr - success_rate,
        "mean_ne_m": mean_ne,
        "mean_spl": mean_spl,
        "mean_steps": mean_steps,
        "mean_elapsed_s": mean_elapsed,
        "stop_emission": {
            "n_stopped": n_stopped,
            "n_entered_success_region": n_entered_success,
            "n_stopped_in_success_region": n_stopped_in_success,
            "stop_precision": stop_precision,        # of stops, fraction in goal region
            "stop_recall_in_success": stop_recall_in_success,  # of episodes that entered region, fraction that stopped
        },
        "action_accuracy": {
            "n_pairs_compared": n_pairs,
            "overall_accuracy": overall_acc,
            "non_forward_conditional_accuracy": non_fwd_acc,
            "n_non_forward_pairs": non_fwd_pairs,
            "per_class": per_class,
        },
        "model_action_distribution": {
            ACTION_NAMES[a]: int(model_class_counts.get(a, 0))
            for a in sorted(model_class_counts)
        },
        "expert_action_distribution": {
            ACTION_NAMES[a]: int(expert_class_counts.get(a, 0))
            for a in sorted(expert_class_counts)
        },
        "per_env": per_env,
    }


# ---- pretty printer ---------------------------------------------------

def _print_summary(agg: dict[str, Any], success_dist: float) -> None:
    print("\n=== fast SR eval summary ===")
    print(f"  n_episodes:    {agg['n_episodes']}")
    print(f"  SR:            {agg['success_rate']:.3f}")
    print(f"  OSR:           {agg['osr']:.3f}")
    print(f"  OSR-SR gap:    {agg['osr_minus_sr_gap']:.3f}  (how often the drone reaches the goal region but doesn't stop)")
    print(f"  mean NE:       {agg['mean_ne_m']:.2f} m  (final distance to goal)")
    print(f"  mean SPL:      {agg['mean_spl']:.3f}")
    print(f"  mean steps:    {agg['mean_steps']:.1f}")
    print(f"  mean episode:  {agg['mean_elapsed_s']:.2f} s")

    se = agg["stop_emission"]
    print("\n  stop emission:")
    print(f"    n_stopped:                        {se['n_stopped']}")
    print(f"    n_entered_success_region (OSR):   {se['n_entered_success_region']}")
    print(f"    n_stopped_in_success_region (SR): {se['n_stopped_in_success_region']}")
    if se["stop_precision"] is not None:
        print(f"    stop precision (in-region | stopped): {se['stop_precision']:.3f}")
    if se["stop_recall_in_success"] is not None:
        print(f"    stop recall  (stopped | in-region):   {se['stop_recall_in_success']:.3f}")

    aa = agg["action_accuracy"]
    print("\n  action accuracy vs expert (positional, degrades after divergence):")
    if aa["overall_accuracy"] is not None:
        print(f"    overall:                       {aa['overall_accuracy']:.3f}")
    if aa["non_forward_conditional_accuracy"] is not None:
        print(
            f"    non-forward conditional:       {aa['non_forward_conditional_accuracy']:.3f}  "
            f"(n={aa['n_non_forward_pairs']})"
        )
    print(f"    {'action':<18} {'pred':>7} {'expert':>7} {'correct':>8} {'prec':>7} {'recall':>7}")
    print("    " + "-" * 60)
    for name, stats in aa["per_class"].items():
        prec_s = f"{stats['precision']:.3f}" if stats["precision"] is not None else "  -  "
        rec_s = f"{stats['recall']:.3f}" if stats["recall"] is not None else "  -  "
        print(
            f"    {name:<18} {stats['n_predicted']:>7} {stats['n_expert']:>7} "
            f"{stats['n_correct']:>8} {prec_s:>7} {rec_s:>7}"
        )

    print("\n  per-env breakdown:")
    print(f"    {'env':<28} {'n':>5} {'SR':>7} {'OSR':>7} {'NE_m':>8} {'SPL':>7}")
    print("    " + "-" * 60)
    for env_name, env_data in agg["per_env"].items():
        print(
            f"    {env_name:<28} {env_data['n_episodes']:>5} "
            f"{env_data['success_rate']:>7.3f} {env_data['osr']:>7.3f} "
            f"{env_data['mean_ne_m']:>8.2f} {env_data['mean_spl']:>7.3f}"
        )

    print(f"\n  (success_dist = {success_dist:.1f} m)")


# ---- main -------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast OpenFly SR eval (~minutes)")
    p.add_argument("--split", default="unseen", help="seen|unseen|eval_test|train")
    p.add_argument("--n_episodes", type=int, default=20, help="Episodes per run (across all envs in the split)")
    p.add_argument("--env_filter", default="", help="Substring filter on env name (e.g. 'gtav')")
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--success_dist", type=float, default=20.0)
    p.add_argument(
        "--policy",
        default="paligemma",
        help="heuristic|openfly-agent|paligemma|grpo|ppo",
    )
    p.add_argument("--checkpoint", default="", help="Path to model checkpoint (for paligemma/grpo/ppo)")
    p.add_argument("--out", default="", help="JSON output path (default: logs/openfly/fast_eval/<timestamp>.json)")
    # Mirror eval_benchmark's PaliGemma flags so the same checkpoint+flags
    # produces directly-comparable numbers.
    p.add_argument("--ue_port", type=int, default=9000, help="UEBridge RPC port (env_ue_* envs). Default 9000.")
    p.add_argument("--paligemma_use_progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paligemma_use_sub_instruction", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--paligemma_use_learned_progress", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # eval.py (upstream) uses relative paths like
    # ``envs/airsim/env_airsim_16/...`` and
    # ``envs/ue/env_ue_smallcity/.../unrealcv.ini`` — both resolved from
    # ``OPENFLY_ROOT/train``. ``eval_benchmark.py`` does the same chdir.
    import os as _os
    _os.chdir(openfly_root() / "train")

    episodes = load_episodes(
        args.split,
        max_episodes=args.n_episodes,
        env_filter=args.env_filter or None,
    )
    groups = group_by_env(episodes)
    total_eps = sum(len(g) for g in groups.values())
    print(
        f"[fast_sr_eval] split={args.split}  n_episodes={total_eps}  "
        f"envs={list(groups.keys())}",
        flush=True,
    )

    policy_kwargs: dict[str, Any] = {}
    pol = args.policy.lower()
    if pol in ("paligemma", "vla", "grpo"):
        if not args.checkpoint:
            raise SystemExit(f"--policy {pol} requires --checkpoint PATH")
        policy_kwargs["checkpoint"] = args.checkpoint
        policy_kwargs["max_steps"] = args.max_steps
        policy_kwargs["use_progress"] = args.paligemma_use_progress
        policy_kwargs["use_sub_instruction"] = args.paligemma_use_sub_instruction
        policy_kwargs["use_learned_progress"] = args.paligemma_use_learned_progress
    elif pol in ("ppo", "ppo-agent", "openfly-agent-rl"):
        if not args.checkpoint:
            raise SystemExit(f"--policy {pol} requires --checkpoint PATH")
        policy_kwargs["checkpoint"] = args.checkpoint
    policy = build_policy(args.policy, **policy_kwargs)

    eval_mod = load_eval_module()
    records: list[dict[str, Any]] = []
    t_total = time.time()

    for env_name, env_eps in groups.items():
        print(f"[fast_sr_eval] env={env_name}  n={len(env_eps)}", flush=True)
        bridge, pos_ratio = _make_bridge_with_overrides(
            env_name, eval_mod, ue_port=args.ue_port,
        )
        time.sleep(3.0)
        try:
            for i, ep in enumerate(env_eps):
                t_ep = time.time()
                rec = _rollout_episode(
                    bridge, pos_ratio, policy, ep,
                    max_steps=args.max_steps,
                    success_dist=args.success_dist,
                )
                records.append(rec)
                print(
                    f"  ep {i + 1}/{len(env_eps)}  "
                    f"success={int(rec['success'])} osr={int(rec['osr'])} "
                    f"ne={rec['ne_m']:.1f}m steps={rec['n_model_steps']}  "
                    f"({time.time() - t_ep:.1f}s)",
                    flush=True,
                )
        finally:
            _cleanup_env(eval_mod, env_name)

    agg = _aggregate(records, args.success_dist)
    agg["total_elapsed_s"] = time.time() - t_total
    _print_summary(agg, args.success_dist)

    # Write JSON
    out_path = (
        Path(args.out)
        if args.out
        else Path("logs/openfly/fast_eval")
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.split}_{pol}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "split": args.split,
        "n_episodes": len(records),
        "success_dist": args.success_dist,
        "aggregate": agg,
        "episodes": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
