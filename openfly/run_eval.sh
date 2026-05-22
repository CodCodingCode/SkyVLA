#!/usr/bin/env bash
set -euo pipefail
DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"
python -m openfly.eval_benchmark "$@"
