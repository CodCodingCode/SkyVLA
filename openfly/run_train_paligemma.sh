#!/usr/bin/env bash
# Offline behaviour cloning of PaliGemmaVLNPolicy on OpenFly train.json.
set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.train_paligemma "$@"
