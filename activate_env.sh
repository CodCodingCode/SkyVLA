#!/bin/bash
# Source this before running drone_project scripts:
#   source ~/drone_project/activate_env.sh

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate isaac

export OMNI_KIT_ACCEPT_EULA=yes
export ISAACSIM_PATH="${ISAACSIM_PATH:-$(python -c 'import isaacsim, os; print(os.path.dirname(isaacsim.__file__))' 2>/dev/null)}"
export TERM="${TERM:-xterm-256color}"

echo "Activated: conda env 'isaac'"
echo "ISAACSIM_PATH=$ISAACSIM_PATH"
echo "Isaac Lab: ~/IsaacLab"
echo "Pegasus:   ~/PegasusSimulator"
