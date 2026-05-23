#!/usr/bin/env python3
"""Bucket failed episodes from ``logs/benchmarks/openfly_*.json`` runs.

The :mod:`openfly.eval_benchmark` harness stores per-episode fields
(``success``, ``osr``, ``ne_m``, ``spl``, ``steps``, ``image_error``)
that are enough to assign every failure to a coarse failure mode:

* ``image_error``        — sim/bridge crash, not the policy's fault
* ``oracle_only``        — got within the success radius but never
                           emitted the stop action ("no_stop" / overshoot)
* ``stalled``            — used the full step budget without ever
                           reaching the success radius
* ``wandered``            — terminated early (presumably stopped or
                           collided) far from the goal
* ``near_miss``           — stopped close-ish to the goal but outside
                           the success radius

This is intentionally a coarse taxonomy: per-step trajectories are not
logged by the eval harness, so anything finer (e.g. "wrong turn at
landmark X") needs a separate trace replay. The buckets here are still
informative for comparing per-env failure distributions between
checkpoints.

Example
-------

::

    python -m openfly.scripts.analyse_failures \
        --inputs logs/benchmarks/openfly_unseen_paligemma_*.json \
        --output docs/failure_modes.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _classify(ep: dict[str, Any], *, max_steps: int, near_miss_m: float) -> str:
    """Map one failed episode to a failure-mode bucket."""
    if ep.get("image_error"):
        return "image_error"
    if ep.get("osr"):
        return "oracle_only"
    if ep.get("steps", 0) >= max_steps:
        return "stalled"
    if ep.get("ne_m", float("inf")) <= near_miss_m:
        return "near_miss"
    return "wandered"


def _iter_runs(patterns: Iterable[str]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for pat in patterns:
        for raw in sorted(glob.glob(pat)):
            if raw in seen:
                continue
            seen.add(raw)
            try:
                with open(raw, encoding="utf-8") as f:
                    yield json.load(f) | {"_path": raw}
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[failures] skipping {raw}: {exc}", file=sys.stderr)


def _summarise_run(run: dict[str, Any], *, near_miss_m: float) -> dict[str, Any]:
    max_steps = int(run.get("max_steps", 100))
    by_env_total: Counter[str] = Counter()
    by_env_failed: Counter[str] = Counter()
    by_env_mode: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ep in run.get("episodes", []):
        env = ep.get("env", "?")
        by_env_total[env] += 1
        if ep.get("success"):
            continue
        mode = _classify(ep, max_steps=max_steps, near_miss_m=near_miss_m)
        by_env_failed[env] += 1
        by_env_mode[env][mode] += 1
        if len(samples[env]) < 10:
            samples[env].append(
                {
                    "mode": mode,
                    "ne_m": round(ep.get("ne_m", float("nan")), 1),
                    "steps": ep.get("steps", -1),
                    "spl": round(ep.get("spl", 0.0), 3),
                    "image_path": ep.get("image_path", ""),
                    "instruction": ep.get("instruction", ""),
                }
            )

    return {
        "path": run.get("_path", ""),
        "policy": run.get("policy", "?"),
        "split": run.get("split", "?"),
        "checkpoint": run.get("checkpoint", ""),
        "max_steps": max_steps,
        "by_env_total": dict(by_env_total),
        "by_env_failed": dict(by_env_failed),
        "by_env_mode": {env: dict(c) for env, c in by_env_mode.items()},
        "samples": {env: rows for env, rows in samples.items()},
    }


_MODES = ["wandered", "stalled", "oracle_only", "near_miss", "image_error"]


def _render_markdown(reports: list[dict[str, Any]]) -> str:
    lines: list[str] = ["# Failure-mode analysis", ""]
    for report in reports:
        lines.append(f"## `{report['policy']}` on `{report['split']}`")
        if report["checkpoint"]:
            lines.append(f"- Checkpoint: `{report['checkpoint']}`")
        lines.append(f"- Source: `{report['path']}`")
        lines.append(f"- Max steps: {report['max_steps']}")
        lines.append("")
        header = "| Env | N | Failed | " + " | ".join(_MODES) + " |"
        sep = "|-----|--:|------:|" + "|".join(["----:"] * len(_MODES)) + "|"
        lines += [header, sep]
        for env in sorted(report["by_env_total"]):
            total = report["by_env_total"][env]
            failed = report["by_env_failed"].get(env, 0)
            modes = report["by_env_mode"].get(env, {})
            row = [
                f"`{env}`",
                str(total),
                f"{failed} ({(failed / max(total, 1)):.0%})",
            ]
            for m in _MODES:
                row.append(str(modes.get(m, 0)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for env in sorted(report["samples"]):
            rows = report["samples"][env]
            if not rows:
                continue
            lines.append(f"### Sampled failures — `{env}`")
            lines.append("| Mode | NE (m) | Steps | SPL | Instruction |")
            lines.append("|------|-------:|------:|----:|-------------|")
            for r in rows:
                instr = (r["instruction"] or "")[:80].replace("|", "/")
                lines.append(
                    f"| {r['mode']} | {r['ne_m']} | {r['steps']} | {r['spl']} | {instr} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more glob patterns of openfly_*.json eval files.",
    )
    p.add_argument(
        "--near_miss_m",
        type=float,
        default=30.0,
        help="Episodes failing within this distance are tagged 'near_miss' "
        "instead of 'wandered'.",
    )
    p.add_argument(
        "--output",
        default="",
        help="Write the Markdown summary here (default: stdout).",
    )
    p.add_argument(
        "--json_out",
        default="",
        help="Optional path to write the raw bucket counts as JSON.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = list(_iter_runs(args.inputs))
    if not runs:
        print("[failures] no matching eval files", file=sys.stderr)
        return 1
    reports = [_summarise_run(r, near_miss_m=args.near_miss_m) for r in runs]
    md = _render_markdown(reports)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[failures] wrote {len(reports)} reports → {args.output}")
    else:
        print(md)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        print(f"[failures] wrote JSON → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
