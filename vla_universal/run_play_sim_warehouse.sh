#!/bin/bash
set -euo pipefail
source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
exec xvfb-run -a -s "-screen 0 1920x1080x24" bash /home/ubuntu/drone_project/vla_universal/run_test_stage2.sh \
  --record_video --scene warehouse_full --poi forklift_main --timeout_s 45 "$@"
