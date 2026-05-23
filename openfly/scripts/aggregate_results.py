#!/usr/bin/env python3
"""Aggregate ``logs/benchmarks/openfly_*.json`` runs into a results table.

Scans the benchmark output directory for JSON files written by
:mod:`openfly.eval_benchmark`, groups them by ``(policy, split)``, and
prints a Markdown table suitable for pasting into ``docs/RESEARCH.md`` or
the Jekyll site's ``results.md`` page.

When ``--per-env`` is passed each row is broken down by environment so
the cross-scene generalisation story (GTA vs UE smallcity vs GS sjtu02)
is visible at a glance.

Example
-------

    python -m openfly.scripts.aggregate_results \
        --logs_dir logs/benchmarks --per-env --output docs/results_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _iter_runs(logs_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(logs_dir.glob("openfly_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[aggregate] skipping {path.name}: {exc}", file=sys.stderr)
            continue
        data.setdefault("_path", str(path))
        yield data


def _fmt(v: float, ndigits: int = 3) -> str:
    return f"{v:.{ndigits}f}"


def _row_overall(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": run.get("policy", "?"),
        "split": run.get("split", "?"),
        "env_filter": run.get("env_filter", "") or "—",
        "n": run.get("n_episodes", 0),
        "sr": run.get("success_rate", 0.0),
        "osr": run.get("osr", 0.0),
        "ne": run.get("mean_ne_m", 0.0),
        "spl": run.get("mean_spl", 0.0),
        "ckpt": run.get("checkpoint", "") or "",
        "path": run.get("_path", ""),
    }


def _row_per_env(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_env = run.get("per_env") or {}
    if not per_env:
        # Older runs without the per-env breakdown still produce a single row
        # so downstream tooling does not silently lose them.
        rows.append({**_row_overall(run), "env": run.get("env_filter") or "all"})
        return rows
    for env_name, bucket in sorted(per_env.items()):
        rows.append(
            {
                "policy": run.get("policy", "?"),
                "split": run.get("split", "?"),
                "env": env_name,
                "n": bucket.get("n_episodes", 0),
                "sr": bucket.get("success_rate", 0.0),
                "osr": bucket.get("osr", 0.0),
                "ne": bucket.get("mean_ne_m", 0.0),
                "spl": bucket.get("mean_spl", 0.0),
                "ckpt": run.get("checkpoint", "") or "",
                "path": run.get("_path", ""),
            }
        )
    return rows


def _to_markdown(rows: list[dict[str, Any]], per_env: bool) -> str:
    if per_env:
        header = "| Policy | Split | Env | N | SR | OSR | NE (m) | SPL |"
        sep = "|--------|-------|-----|---|------|------|--------|------|"
    else:
        header = "| Policy | Split | Env filter | N | SR | OSR | NE (m) | SPL |"
        sep = "|--------|-------|------------|---|------|------|--------|------|"

    lines = [header, sep]
    for r in rows:
        if per_env:
            lines.append(
                f"| `{r['policy']}` | `{r['split']}` | `{r['env']}` | "
                f"{r['n']} | {_fmt(r['sr'])} | {_fmt(r['osr'])} | "
                f"{_fmt(r['ne'], 1)} | {_fmt(r['spl'])} |"
            )
        else:
            lines.append(
                f"| `{r['policy']}` | `{r['split']}` | `{r['env_filter']}` | "
                f"{r['n']} | {_fmt(r['sr'])} | {_fmt(r['osr'])} | "
                f"{_fmt(r['ne'], 1)} | {_fmt(r['spl'])} |"
            )
    return "\n".join(lines) + "\n"


def _to_csv(rows: list[dict[str, Any]], per_env: bool, fp) -> None:
    fields = (
        ["policy", "split", "env", "n", "sr", "osr", "ne", "spl", "ckpt", "path"]
        if per_env
        else ["policy", "split", "env_filter", "n", "sr", "osr", "ne", "spl", "ckpt", "path"]
    )
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs_dir",
        default="logs/benchmarks",
        help="Directory containing openfly_*.json eval outputs.",
    )
    parser.add_argument(
        "--per-env",
        action="store_true",
        help="Break each run out by env (use for unseen split with multiple scenes).",
    )
    parser.add_argument(
        "--policy",
        default="",
        help="Optional policy filter (substring match).",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Optional split filter (substring match).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write Markdown table here (default: stdout).",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Also write the rows as CSV at this path.",
    )
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"[aggregate] no such dir: {logs_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for run in _iter_runs(logs_dir):
        if args.policy and args.policy not in str(run.get("policy", "")):
            continue
        if args.split and args.split not in str(run.get("split", "")):
            continue
        if args.per_env:
            rows.extend(_row_per_env(run))
        else:
            rows.append(_row_overall(run))

    if not rows:
        print("[aggregate] no matching runs found", file=sys.stderr)
        return 1

    md = _to_markdown(rows, per_env=args.per_env)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[aggregate] wrote {len(rows)} rows → {args.output}")
    else:
        print(md)

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fp:
            _to_csv(rows, per_env=args.per_env, fp=fp)
        print(f"[aggregate] wrote CSV → {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
