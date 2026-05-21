#!/usr/bin/env python3
"""Unified benchmark runner for drone_project.

Examples:
  # HUGE-Bench (offline, HuggingFace — works without Isaac Sim)
  python -m benchmarks.run huge --backend waypoint_heuristic --split test_seen --max_batches 50
  python -m benchmarks.run huge --backend bc_checkpoint --checkpoint logs/huge_bench/.../model_20000.pt

  # CityNav oracle baseline (needs CITYNAV_ROOT + data)
  python -m benchmarks.run citynav --citynav_root /path/to/citynav --max_episodes 20

  # AirNav (needs AIRNAV_ROOT + multi-GB data + NavGym photos)
  python -m benchmarks.run airnav --airnav_root /path/to/AirNav --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="drone_project benchmark runner")
    p.add_argument("benchmark", choices=["huge", "citynav", "airnav", "openfly", "all"])
    args, passthrough = p.parse_known_args()

    if args.benchmark == "huge":
        cmd = [sys.executable, "-m", "benchmarks.eval_huge"] + passthrough
    elif args.benchmark == "citynav":
        cmd = [sys.executable, "-m", "benchmarks.eval_citynav_oracle"] + passthrough
    elif args.benchmark == "airnav":
        print(
            "AirNav requires the full AirNav repo, NavGym simulator, and ~tens of GB "
            "of rgbd/cityrefer/gsam data. See benchmarks/README.md and run:\n"
            "  cd $AIRNAV_ROOT && python light_model_eval.py\n"
            "To add a drone_project adapter, set AIRNAV_ROOT and follow setup_external.sh."
        )
        sys.exit(0)
    elif args.benchmark == "openfly":
        print(
            "OpenFly toolchain/dataset is not fully open-sourced yet.\n"
            "Track: https://shailab-ipec.github.io/openfly/"
        )
        sys.exit(0)
    elif args.benchmark == "all":
        cmd = [sys.executable, "-m", "benchmarks.eval_huge",
               "--backend", "waypoint_heuristic", "--split", "test_seen", "--max_batches", "20"]
        print("Running HUGE-Bench smoke only (citynav/airnav need external data)...")
    else:
        raise SystemExit(f"Unknown benchmark: {args.benchmark}")

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
