"""One-shot diagnostic: action distribution + episode-length stats per split.

Reads raw OpenFly Annotation JSON directly (no image loads, no dataset
construction) and prints per-class action counts, episode-length quantiles,
and optional per-env breakdown. Saves a JSON snapshot under
``logs/openfly/dataset_stats_{timestamp}.json``.

Why this exists: action-class distribution and stop-class share are the
two numbers that determine whether class weighting / stop oversampling is
needed, and whether action accuracy is a meaningful metric at all. Both
were unsurfaced in training logs until now.

CLI:
    python -m openfly.scripts.dataset_stats --splits train seen unseen
    python -m openfly.scripts.dataset_stats --splits unseen --per_env
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from openfly.actions import (
    ACTION_NAMES,
    TRAINABLE_ACTION_IDS,
)
from openfly.episodes import episode_env_name, load_episodes


ALL_ACTION_IDS = tuple(sorted(ACTION_NAMES.keys()))  # 0..9


def _percentile(xs: list[int], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "n": 0}
    return {
        "n": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "median": float(median(lengths)),
        "p90": _percentile(lengths, 0.90),
    }


def _count_actions(episodes: list[dict[str, Any]]) -> tuple[Counter, list[int]]:
    """Return (per-raw-id counts, episode-length list)."""
    counts: Counter = Counter()
    lengths: list[int] = []
    for ep in episodes:
        actions = ep.get("action", [])
        lengths.append(len(actions))
        for a in actions:
            counts[int(a)] += 1
    return counts, lengths


def _format_action_table(counts: Counter, total: int) -> str:
    lines = [f"  {'id':>3}  {'name':<18} {'count':>10}  {'pct':>7}  trainable"]
    lines.append("  " + "-" * 56)
    for aid in ALL_ACTION_IDS:
        n = counts.get(aid, 0)
        pct = (100.0 * n / total) if total else 0.0
        trainable = "yes" if aid in TRAINABLE_ACTION_IDS else "NO (strafe)"
        lines.append(
            f"  {aid:>3}  {ACTION_NAMES[aid]:<18} {n:>10}  {pct:>6.2f}%  {trainable}"
        )
    return "\n".join(lines)


def _format_length_table(stats: dict[str, float]) -> str:
    return (
        f"  episodes={stats['n']}  "
        f"min={stats['min']}  median={stats['median']:.0f}  "
        f"mean={stats['mean']:.2f}  p90={stats['p90']:.0f}  max={stats['max']}"
    )


def _stats_for_split(
    episodes: list[dict[str, Any]],
    per_env: bool,
) -> dict[str, Any]:
    counts, lengths = _count_actions(episodes)
    total_steps = int(sum(counts.values()))
    out: dict[str, Any] = {
        "n_episodes": len(episodes),
        "n_steps": total_steps,
        "length_stats": _length_stats(lengths),
        "action_counts": {int(k): int(v) for k, v in counts.items()},
        "action_pct": {
            int(k): (100.0 * v / total_steps) if total_steps else 0.0
            for k, v in counts.items()
        },
        "forward_9m_pct": (
            100.0 * counts.get(9, 0) / total_steps if total_steps else 0.0
        ),
        "stop_pct": 100.0 * counts.get(0, 0) / total_steps if total_steps else 0.0,
        "strafe_pct": (
            100.0 * (counts.get(6, 0) + counts.get(7, 0)) / total_steps
            if total_steps else 0.0
        ),
    }
    if per_env:
        per_env_data: dict[str, dict[str, Any]] = {}
        env_groups: dict[str, list[dict[str, Any]]] = {}
        for ep in episodes:
            env_groups.setdefault(episode_env_name(ep), []).append(ep)
        for env_name, eps in sorted(env_groups.items()):
            c, ll = _count_actions(eps)
            tot = int(sum(c.values()))
            per_env_data[env_name] = {
                "n_episodes": len(eps),
                "n_steps": tot,
                "length_stats": _length_stats(ll),
                "action_counts": {int(k): int(v) for k, v in c.items()},
                "forward_9m_pct": 100.0 * c.get(9, 0) / tot if tot else 0.0,
                "stop_pct": 100.0 * c.get(0, 0) / tot if tot else 0.0,
            }
        out["per_env"] = per_env_data
    return out


def _print_split(split: str, data: dict[str, Any], per_env: bool) -> None:
    print(f"\n=== split: {split} ===")
    print(
        f"  {data['n_episodes']} episodes  /  {data['n_steps']} steps  /  "
        f"forward_9m={data['forward_9m_pct']:.2f}%  stop={data['stop_pct']:.2f}%  "
        f"strafe={data['strafe_pct']:.2f}%"
    )
    print()
    print("Action distribution:")
    counts = Counter({int(k): int(v) for k, v in data["action_counts"].items()})
    print(_format_action_table(counts, data["n_steps"]))
    print()
    print("Episode lengths:")
    print(_format_length_table(data["length_stats"]))
    if per_env and "per_env" in data:
        print()
        print("Per-env breakdown:")
        for env_name, env_data in data["per_env"].items():
            print(
                f"  {env_name:<28}  "
                f"eps={env_data['n_episodes']:>4}  "
                f"steps={env_data['n_steps']:>6}  "
                f"forward_9m={env_data['forward_9m_pct']:>5.1f}%  "
                f"stop={env_data['stop_pct']:>5.1f}%  "
                f"len(mean/p90)={env_data['length_stats']['mean']:.1f}/"
                f"{env_data['length_stats']['p90']:.0f}"
            )


def _print_cross_split_comparison(per_split: dict[str, dict[str, Any]]) -> None:
    if len(per_split) < 2:
        return
    print("\n=== cross-split distribution comparison ===")
    print(
        f"  {'split':<10} {'episodes':>10} {'steps':>10} "
        f"{'fwd_9m%':>9} {'stop%':>8} {'len_mean':>10}"
    )
    print("  " + "-" * 60)
    for split, data in per_split.items():
        print(
            f"  {split:<10} {data['n_episodes']:>10} {data['n_steps']:>10} "
            f"{data['forward_9m_pct']:>8.2f}% {data['stop_pct']:>7.2f}% "
            f"{data['length_stats']['mean']:>10.2f}"
        )
    print(
        "\n  (large divergence in forward_9m% or stop% across splits would "
        "indicate a split-construction issue.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print action distribution and episode-length stats per OpenFly split."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "seen", "unseen"],
        help="Which splits to load. Default: train seen unseen.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Cap episodes per split (0 = no cap). Use for quick smoke runs.",
    )
    parser.add_argument(
        "--per_env",
        action="store_true",
        help="Also break each split down by env (env_*/...).",
    )
    parser.add_argument(
        "--out_dir",
        default="logs/openfly",
        help="Directory to drop the JSON snapshot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split: dict[str, dict[str, Any]] = {}
    t_total = time.time()
    for split in args.splits:
        t0 = time.time()
        episodes = load_episodes(split=split, max_episodes=args.max_episodes)
        data = _stats_for_split(episodes, per_env=args.per_env)
        data["load_seconds"] = round(time.time() - t0, 2)
        per_split[split] = data
        _print_split(split, data, per_env=args.per_env)

    _print_cross_split_comparison(per_split)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dataset_stats_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "splits_requested": args.splits,
        "max_episodes": args.max_episodes,
        "per_env": args.per_env,
        "total_seconds": round(time.time() - t_total, 2),
        "per_split": per_split,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved snapshot: {out_path}")


if __name__ == "__main__":
    main()
