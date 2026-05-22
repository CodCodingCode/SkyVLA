#!/usr/bin/env bash
# Clone external benchmark repos used by the optional CityNav oracle baseline.
# OpenFly is the primary benchmark and is set up via openfly/setup.sh.
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-$HOME/benchmarks_external}"
mkdir -p "$BENCH_DIR"
cd "$BENCH_DIR"

clone_if_missing() {
  local name="$1" url="$2"
  if [[ -d "$name" ]]; then
    echo "[skip] $name already exists"
  else
    git clone --depth 1 "$url" "$name"
  fi
}

clone_if_missing citynav https://github.com/water-cookie/citynav.git

cat <<EOF

Cloned under: $BENCH_DIR

Next steps — OpenFly (primary benchmark, set up separately):
  bash ~/drone_project/openfly/setup.sh
  bash ~/drone_project/openfly/download_airsim_scene.sh env_airsim_16

Next steps — CityNav oracle (optional, real urban point clouds):
  export CITYNAV_ROOT=$BENCH_DIR/citynav
  cd \$CITYNAV_ROOT && sh scripts/download_data.sh
  # plus SensatUrban PLY rasterization per citynav/README.md
  python -m benchmarks.run citynav --citynav_root \$CITYNAV_ROOT --max_episodes 20
EOF
