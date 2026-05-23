#!/usr/bin/env bash
# Phase-0 verification gates from docs/RESEARCH.md (and the older
# fix_skyvla_pipeline plan). Each gate prints a short PASS/FAIL line
# and exits non-zero on the first failure so this is safe to chain into
# a Makefile or pre-flight hook.
#
# Gates checked:
#   G0: NVIDIA driver + CUDA
#   G1: OpenFly annotation files reachable
#   G2: OPENFLY_IMAGE_ROOT exists and has at least one trajectory image
#   G3: openfly Python package importable + reward presets resolve
#   G4: SFT dataset can be built with --require_images
#   G5: AirSim env Gymnasium config accepts every reward preset
#   G6: Heuristic policy reset/act works
#   G7: AirSim bridge port (default 41451) reachable (informational)
#
# Bring-up of the UE binary itself is not automated here because it is
# host-specific; the script tells you what to do if G7 fails.

set -uo pipefail

PASS=0
FAIL=0

ok()   { echo "[gate] PASS  $1"; PASS=$((PASS+1)); }
fail() { echo "[gate] FAIL  $1"; FAIL=$((FAIL+1)); }

export DRONE_PROJECT="${DRONE_PROJECT:-$HOME/SkyVLA}"
export OPENFLY_ROOT="${OPENFLY_ROOT:-$HOME/OpenFly-Platform}"
export OPENFLY_ANNOTATION_DIR="${OPENFLY_ANNOTATION_DIR:-$HOME/assets/OpenFly/Annotation}"
export OPENFLY_IMAGE_ROOT="${OPENFLY_IMAGE_ROOT:-$HOME/assets/OpenFly/images/Image}"
export PYTHONPATH="${DRONE_PROJECT}:${PYTHONPATH:-}"

# G0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  ok "G0 nvidia-smi reports GPUs"
else
  fail "G0 nvidia-smi missing or broken"
fi

python - <<'PY' && ok "G0 torch.cuda.is_available()" || fail "G0 CUDA not available to torch"
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY

# G1
for f in train.json seen.json unseen.json; do
  if [[ -f "$OPENFLY_ANNOTATION_DIR/$f" ]]; then
    ok "G1 annotation $f"
  else
    fail "G1 missing $OPENFLY_ANNOTATION_DIR/$f"
  fi
done

# G2 — count PNGs/JPGs across ALL train env subdirectories under
# OPENFLY_IMAGE_ROOT so we can tell the difference between a single-env
# smoke download and a full 11-scene snapshot.
if [[ -d "$OPENFLY_IMAGE_ROOT" ]]; then
  n_imgs=$(find "$OPENFLY_IMAGE_ROOT" \( -name "*.png" -o -name "*.jpg" \) | wc -l)
  n_envs=$(find "$OPENFLY_IMAGE_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  echo "[gate] INFO  G2 frame count = $n_imgs across $n_envs env dir(s) under $OPENFLY_IMAGE_ROOT"
  if [[ "$n_imgs" -gt 10000 ]]; then
    if [[ "$n_imgs" -lt 100000 ]]; then
      echo "[gate] WARN  G2 only $n_imgs frames found (< 100k); full train download not complete yet (expected ~250k+)."
    fi
    ok "G2 OPENFLY_IMAGE_ROOT has $n_imgs trajectory frames across $n_envs env(s)"
  elif [[ "$n_imgs" -ge 1 ]]; then
    fail "G2 OPENFLY_IMAGE_ROOT has only $n_imgs frames (need > 10000 — run openfly/download_train_images.sh)"
  else
    fail "G2 OPENFLY_IMAGE_ROOT exists but contains no PNG/JPG frames"
  fi
else
  fail "G2 OPENFLY_IMAGE_ROOT does not exist: $OPENFLY_IMAGE_ROOT"
fi

# G3 + G5
python - <<'PY' && ok "G3 openfly package + reward presets" || fail "G3 openfly package import or preset lookup failed"
import sys
from openfly.rewards import REWARD_PRESETS, get_reward_preset
from openfly.envs.airsim_vln_env import AirSimVLNEnvConfig
for name in ("easy", "medium", "hard"):
    _ = get_reward_preset(name)
    cfg = AirSimVLNEnvConfig(reward_preset=name, env_filter="env_airsim_16", max_episodes=1)
    assert cfg.reward_config is not None
sys.exit(0)
PY

# G4 — full train split (no env_filter, no max_episodes cap) with
# require_images=True must index more than 50k steps. This is the gate
# that catches a partial image download.
python - <<'PY' && ok "G4 SFT dataset with --require_images (> 50k steps)" || fail "G4 SFT dataset indexed < 50000 steps with --require_images (run openfly/download_train_images.sh)"
import sys
from openfly.dataset import OpenFlyDataset
ds = OpenFlyDataset(
    split="train",
    require_images=True,
    history_frames=2,
)
n = len(ds)
print(f"[g4] indexed {n} steps from full train split with require_images=True")
sys.exit(0 if n > 50000 else 1)
PY

# G6
python - <<'PY' && ok "G6 heuristic policy reset/act" || fail "G6 heuristic policy failed"
import sys, numpy as np
from openfly.policies import build_policy
pol = build_policy("heuristic")
pol.reset("fly forward", [10.0, 20.0, 5.0])
_ = pol.act(np.zeros((224, 224, 3), dtype=np.uint8), pose=[0, 0, 0, 0], step=0, history=[])
sys.exit(0)
PY

# G7 (informational)
if command -v nc >/dev/null 2>&1; then
  if nc -z localhost 41451 2>/dev/null; then
    ok "G7 AirSim RPC port 41451 reachable on localhost"
  else
    echo "[gate] INFO  G7 AirSim RPC port not reachable; launch the UE binary:"
    echo "       bash \"\$OPENFLY_ROOT/envs/airsim/env_airsim_16/LinuxNoEditor/start.sh\""
  fi
else
  echo "[gate] INFO  G7 skipped (nc not installed)"
fi

echo
echo "[gate] summary: $PASS pass / $FAIL fail"
exit $(( FAIL > 0 ? 1 : 0 ))
