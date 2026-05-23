#!/usr/bin/env bash
# Curriculum GRPO: easy -> medium -> hard reward presets on the
# PaliGemma BC/DAgger policy. See openfly/train_curriculum_grpo.py for
# the full set of knobs; this wrapper just activates the conda env and
# forwards all CLI args.
#
# Required:
#   --init_ckpt PATH   Bootstrap checkpoint (DAgger preferred, SFT OK).
#
# Example:
#   bash openfly/run_train_curriculum.sh \
#     --init_ckpt logs/openfly/dagger/<run>/last.pt \
#     --env_filter env_airsim_16 \
#     --steps_easy 80 --steps_medium 60 --steps_hard 60

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.train_curriculum_grpo "$@"
