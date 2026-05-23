#!/usr/bin/env bash
# Download the trajectory image trees for all 11 OpenFly train scenes from the
# IPEC-COMMUNITY/OpenFly Hugging Face dataset. The list is taken from
# Annotation/train.json so it stays in sync with the dataset itself (any
# UNSEEN scenes — env_game_gtav, env_ue_smallcity, env_gs_sjtu02 — are
# intentionally NOT downloaded).
#
# Layout produced (matches OPENFLY_IMAGE_ROOT default in activate.sh):
#   $OPENFLY_IMAGE_ROOT/<env_name>/<trajectory_id>/<frame>.png
#   e.g. /home/ubuntu/assets/OpenFly/images/Image/env_airsim_16/.../000.png
#
# Usage:
#   bash openfly/download_train_images.sh             # do the full download
#   bash openfly/download_train_images.sh --dry-run   # print envs/patterns, do not download
#
# Requires `huggingface-cli login` (or HF_TOKEN env var) — the dataset is
# gated. If auth is missing the script exits with a clear message instead
# of half-downloading.

set -uo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/SkyVLA}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *)
      echo "[download_train_images] unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# OPENFLY_IMAGE_ROOT defaults to .../images/Image but snapshot_download
# writes the "Image/" prefix itself, so target the parent directory.
LOCAL_DIR="$(dirname "${OPENFLY_IMAGE_ROOT}")"
mkdir -p "$LOCAL_DIR"

# Disk space pre-flight: full train set is ~100–150 GB; warn under 200 GB.
AVAIL_GB="$(df -BG --output=avail "$LOCAL_DIR" | tail -n1 | tr -dc '0-9')"
if [[ "$DRY_RUN" -eq 0 && -n "$AVAIL_GB" && "$AVAIL_GB" -lt 200 ]]; then
  echo "[download_train_images] ABORT: only ${AVAIL_GB}G free on $(df -h "$LOCAL_DIR" | tail -n1 | awk '{print $6}'); need >= 200G" >&2
  exit 3
fi

# Authentication check (informational; snapshot_download will still fail
# loudly if the user is unauthenticated AND the dataset stays gated).
if ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "[download_train_images] WARN: huggingface-cli whoami failed."
  echo "[download_train_images] WARN: run 'huggingface-cli login' (or export HF_TOKEN) before downloading."
  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "[download_train_images] ABORT: not authenticated; rerun after login."
    exit 4
  fi
fi

echo "[download_train_images] OPENFLY_ANNOTATION_DIR=$OPENFLY_ANNOTATION_DIR"
echo "[download_train_images] OPENFLY_IMAGE_ROOT=$OPENFLY_IMAGE_ROOT"
echo "[download_train_images] local_dir (snapshot target)=$LOCAL_DIR"
echo "[download_train_images] dry_run=$DRY_RUN"
echo "[download_train_images] free space at target=${AVAIL_GB:-unknown} GB"

export DRY_RUN LOCAL_DIR

python - <<'PY'
import json
import os
import sys
import time
from pathlib import Path

annotation_dir = Path(os.environ["OPENFLY_ANNOTATION_DIR"])
local_dir = Path(os.environ["LOCAL_DIR"])
dry_run = os.environ.get("DRY_RUN", "0") == "1"

train_json = annotation_dir / "train.json"
if not train_json.exists():
    sys.exit(f"[download_train_images] train.json not found at {train_json}")

with open(train_json) as fh:
    entries = json.load(fh)

envs = set()
for entry in entries:
    image_path = entry.get("image_path")
    if isinstance(image_path, list):
        image_path = image_path[0] if image_path else None
    if not isinstance(image_path, str):
        continue
    parts = image_path.split("/")
    # paths look like "Image/<env>/<traj>/<frame>.png"
    env = parts[1] if parts and parts[0] == "Image" else parts[0]
    if env:
        envs.add(env)

# Defence-in-depth: the OpenFly unseen scenes must never leak in here.
UNSEEN = {"env_game_gtav", "env_ue_smallcity", "env_gs_sjtu02"}
leaked = envs & UNSEEN
if leaked:
    sys.exit(f"[download_train_images] FATAL: unseen envs found in train.json: {sorted(leaked)}")

train_envs = sorted(envs)
allow_patterns = [f"Image/{env}/**" for env in train_envs]

print(f"[download_train_images] train envs ({len(train_envs)}):")
for env in train_envs:
    print(f"  - {env}")
print(f"[download_train_images] allow_patterns:")
for p in allow_patterns:
    print(f"  - {p}")

if dry_run:
    print("[download_train_images] dry-run: skipping snapshot_download")
    sys.exit(0)

from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(
    repo_id="IPEC-COMMUNITY/OpenFly",
    repo_type="dataset",
    allow_patterns=allow_patterns,
    local_dir=str(local_dir),
    max_workers=8,
)
dt = time.time() - t0
print(f"[download_train_images] snapshot_download finished in {dt/60:.1f} min")
print(f"[download_train_images] local path: {path}")
PY
rc=$?
if [[ "$rc" -ne 0 ]]; then
  echo "[download_train_images] FAILED (rc=$rc)" >&2
  exit "$rc"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[download_train_images] dry-run complete."
  exit 0
fi

# Final disk summary.
echo
echo "[download_train_images] disk usage by env:"
if [[ -d "$OPENFLY_IMAGE_ROOT" ]]; then
  du -sh "$OPENFLY_IMAGE_ROOT"/* 2>/dev/null | sort -h || true
  echo
  echo "[download_train_images] total:"
  du -sh "$OPENFLY_IMAGE_ROOT" 2>/dev/null || true
  echo
  echo "[download_train_images] png/jpg counts by env:"
  for d in "$OPENFLY_IMAGE_ROOT"/*; do
    if [[ -d "$d" ]]; then
      n=$(find "$d" \( -name "*.png" -o -name "*.jpg" \) | wc -l)
      printf "  %-30s %d frames\n" "$(basename "$d")" "$n"
    fi
  done
else
  echo "[download_train_images] WARN: $OPENFLY_IMAGE_ROOT not present after download"
fi

echo "[download_train_images] done."
