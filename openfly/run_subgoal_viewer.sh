#!/usr/bin/env bash
# Visualize SubgoalDiT predictions via nearest-neighbor RGB retrieval.
#
# Three phases:
#   1. build  — encode all val frames -> SigLIP feature bank   (~once per split)
#   2. eval   — sample subgoals from a DiT ckpt, retrieve NN   (~per ckpt)
#   3. view   — launch the Streamlit browser on the results    (interactive)
#
# Typical first-time usage:
#   bash openfly/run_subgoal_viewer.sh build
#   bash openfly/run_subgoal_viewer.sh eval <DIT_CKPT> --pretrained_path <PIXART_DIR>
#   bash openfly/run_subgoal_viewer.sh view
#
# All artifacts land under $DRONE_PROJECT/logs/openfly/subgoal_nn/.
# `view` defaults to the most recent eval directory; pass a path to pin one.

set -euo pipefail

DRONE_PROJECT="${DRONE_PROJECT:-$HOME/drone_project}"
# shellcheck disable=SC1091
source "$DRONE_PROJECT/openfly/activate.sh"

cd "$DRONE_PROJECT"

NN_ROOT="$DRONE_PROJECT/logs/openfly/subgoal_nn"
DEFAULT_SPLIT="unseen"
DEFAULT_BANK="$NN_ROOT/bank_${DEFAULT_SPLIT}"
DEFAULT_EVAL_DIR="$NN_ROOT/eval"

cmd="${1:-}"
shift || true

case "$cmd" in
  build)
    OUT_DIR="${OUT_DIR:-$DEFAULT_BANK}"
    echo "[run_subgoal_viewer] building bank for split=${DEFAULT_SPLIT} -> $OUT_DIR"
    python -m openfly.scripts.build_subgoal_nn_bank \
      --split "$DEFAULT_SPLIT" --out_dir "$OUT_DIR" "$@"
    ;;

  eval)
    CKPT="${1:-}"
    if [[ -z "$CKPT" ]]; then
      echo "usage: $0 eval <DIT_CKPT> [--pretrained_path PIXART_DIR] [...]" >&2
      exit 2
    fi
    shift
    ts="$(date +%Y%m%d_%H%M%S)"
    OUT_DIR="${OUT_DIR:-$NN_ROOT/eval_${ts}}"
    BANK="${BANK:-$DEFAULT_BANK}"
    echo "[run_subgoal_viewer] eval ckpt=$CKPT bank=$BANK -> $OUT_DIR"
    python -m openfly.scripts.eval_subgoal_nn \
      --ckpt "$CKPT" --bank "$BANK" --split "$DEFAULT_SPLIT" \
      --out_dir "$OUT_DIR" "$@"
    echo "[run_subgoal_viewer] when ready: bash $0 view $OUT_DIR/results.pt"
    ;;

  view)
    # Resolve results file: explicit path arg wins, otherwise pick the
    # newest results.pt under the eval root.
    RESULTS="${1:-}"
    if [[ -z "$RESULTS" ]]; then
      RESULTS="$(ls -1t "$NN_ROOT"/eval_*/results.pt 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "$RESULTS" || ! -f "$RESULTS" ]]; then
      echo "no results.pt found — run '$0 eval <CKPT>' first" >&2
      exit 2
    fi
    if ! python -c "import streamlit" 2>/dev/null; then
      echo "[run_subgoal_viewer] installing streamlit (one-time)…"
      pip install --quiet streamlit
    fi
    echo "[run_subgoal_viewer] viewing $RESULTS"
    exec streamlit run openfly/scripts/subgoal_viewer.py -- --results "$RESULTS"
    ;;

  ""|-h|--help|help)
    sed -n '2,16p' "$0"
    ;;

  *)
    echo "unknown command: $cmd" >&2
    echo "valid: build | eval | view" >&2
    exit 2
    ;;
esac
