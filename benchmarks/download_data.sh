#!/usr/bin/env bash
# Pull benchmark datasets used by the Stage 6 / Stage 8 evaluation paths.
#
# - HUGE-Bench: huggingface dataset, fetched lazily by huge_bench/dataset.py
#   on first access. We just warm the cache here.
# - AirNav: requires manual setup of the upstream repo + dataset. We point
#   at the existing benchmarks/setup_external.sh helper for the repo and
#   document the dataset side here.
# - CityNav: same — repo via setup_external.sh, dataset via gdown link in
#   the upstream README. We surface the steps but don't automate the gdown
#   prompts.
set -euo pipefail

DRONE="${DRONE:-/home/ubuntu/drone_project}"
BENCH_EXT="${BENCH_EXT:-$HOME/benchmarks_external}"
AIRNAV_ROOT="${AIRNAV_ROOT:-$BENCH_EXT/AirNav}"
CITYNAV_ROOT="${CITYNAV_ROOT:-$BENCH_EXT/citynav}"

source /home/ubuntu/miniconda3/bin/activate isaac

echo "=== HUGE-Bench (HuggingFace dataset) ==="
python - <<'PY'
from huge_bench.dataset import HugeTask0
for split in ("train", "test_seen", "test_unseen"):
    print(f"[huge] warming {split} ...")
    ds = HugeTask0(split=split)
    print(f"  {split}: {len(ds)} frames across {ds.num_episodes()} episodes")
print("[huge] cache warmed.")
PY

echo ""
echo "=== AirNav ==="
if [[ -d "$AIRNAV_ROOT/.git" ]]; then
    echo "[airnav] repo present at $AIRNAV_ROOT"
else
    echo "[airnav] repo missing. Run: bash benchmarks/setup_external.sh"
fi
if [[ -f "$AIRNAV_ROOT/data/AirNav/val/airnav_val_seen.json" ]]; then
    echo "[airnav] dataset present."
else
    cat <<'EOF'
[airnav] dataset NOT present.
        The AirNav dataset is distributed as a HuggingFace dataset at
        https://huggingface.co/datasets/microsoft/AirVLN — accept the
        license then run, from $AIRNAV_ROOT:

            huggingface-cli download \
                microsoft/AirVLN \
                --repo-type dataset \
                --local-dir data/AirNav
EOF
fi

echo ""
echo "=== CityNav ==="
if [[ -d "$CITYNAV_ROOT/.git" ]]; then
    echo "[citynav] repo present at $CITYNAV_ROOT"
else
    echo "[citynav] repo missing. Run: bash benchmarks/setup_external.sh"
fi
if [[ -d "$CITYNAV_ROOT/data" ]]; then
    echo "[citynav] data present."
else
    cat <<'EOF'
[citynav] data NOT present.
        CityNav publishes its data via Google Drive (3D point cloud +
        instruction JSON). Follow the upstream README link at
        https://water-cookie.github.io/city-nav-proj/  and place the
        download into $CITYNAV_ROOT/data/.
EOF
fi

echo ""
echo "Done. Re-run benchmarks/run_all.sh once the missing pieces are present."
