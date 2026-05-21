"""Record in-sim MP4 (onboard front+right cameras) — stage2 flying to a warehouse POI.

Prefer this launcher (sets xvfb + unbuffered stdout):
    bash ~/drone_project/vla_universal/run_play_sim_warehouse.sh

Or delegate to the smoke-test recorder:
    bash ~/drone_project/vla_universal/run_test_stage2.sh \\
        --scene warehouse_full --poi forklift_main --record_video --timeout_s 45
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    script = __file__.replace("play_sim_warehouse.py", "run_test_stage2.sh")
    cmd = ["bash", script, "--record_video", *sys.argv[1:]]
    if "--scene" not in sys.argv:
        cmd.extend(["--scene", "warehouse_full"])
    if "--poi" not in sys.argv:
        cmd.extend(["--poi", "forklift_main"])
    if "--timeout_s" not in sys.argv:
        cmd.extend(["--timeout_s", "45"])
    raise SystemExit(subprocess.call(cmd))
