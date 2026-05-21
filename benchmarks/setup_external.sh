#!/usr/bin/env bash
# Clone external benchmark repos (does NOT download multi-GB datasets).
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
clone_if_missing AirNav   https://github.com/nopride03/AirNav.git

cat <<EOF

Cloned under: $BENCH_DIR

Next steps — HUGE-Bench (easiest, no sim):
  source ~/miniconda3/bin/activate isaac
  cd ~/drone_project
  pip install datasets pyarrow   # if missing
  python -m huge_bench.train_bc --max_steps 5000   # train adapter
  python -m benchmarks.run huge --backend bc_checkpoint --checkpoint logs/huge_bench/<run>/model_5000.pt

Next steps — CityNav (real urban point clouds + image cache):
  export CITYNAV_ROOT=$BENCH_DIR/citynav
  cd \$CITYNAV_ROOT && sh scripts/download_data.sh
  # + SensatUrban PLY rasterization per citynav/README.md
  python -m benchmarks.run citynav --citynav_root \$CITYNAV_ROOT --max_episodes 20

Next steps — AirNav:
  export AIRNAV_ROOT=$BENCH_DIR/AirNav
  # Download from https://huggingface.co/datasets/dpairnav/AirNav into \$AIRNAV_ROOT/data/
  cd \$AIRNAV_ROOT && pip install -r requirements.txt
  # See AirNav README for vLLM / Qwen baselines

OpenFly: not yet released — https://shailab-ipec.github.io/openfly/
EOF
