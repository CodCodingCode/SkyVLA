#!/bin/bash
set -euo pipefail
source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
exec stdbuf -oL ./isaaclab.sh -p /home/ubuntu/drone_project/vla_universal/test_stage2_warehouse.py "$@"
