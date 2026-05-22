#!/usr/bin/env python3
"""Unified benchmark runner for drone_project.

Examples:
  # OpenFly outdoor VLN (primary)
  python -m benchmarks.run openfly --split unseen --policy heuristic --max_episodes 5

  # CityNav oracle baseline (requires CITYNAV_ROOT)
  python -m benchmarks.run citynav --citynav_root /path/to/citynav --max_episodes 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="drone_project benchmark runner")
    p.add_argument("benchmark", choices=["openfly", "citynav"])
    args, passthrough = p.parse_known_args()

    if args.benchmark == "openfly":
        cmd = [sys.executable, "-m", "openfly.eval_benchmark", *passthrough]
    elif args.benchmark == "citynav":
        cmd = [sys.executable, "-m", "benchmarks.eval_citynav_oracle", *passthrough]
    else:
        raise SystemExit(f"Unknown benchmark: {args.benchmark}")

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
