"""Streamlit viewer for SubgoalDiT eval results.

Reads the ``results.pt`` produced by :mod:`openfly.scripts.eval_subgoal_nn`
and renders, per sample:

  * **current RGB**           — the frame the policy is observing
  * **real subgoal RGB**      — the ground-truth next-keyframe
  * **top-K NN of prediction**— the K real frames in the bank whose
                                pooled SigLIP features are closest to
                                the (mean) DiT prediction. Top-1 is the
                                "what the model thinks the subgoal looks
                                like" visualization.

Plus per-sample numerics: cos sim (mean and best-of-N), retrieval rank,
instruction / sub-instruction, pose delta, action context.

Usage:
  streamlit run openfly/scripts/subgoal_viewer.py -- \\
    --results ~/drone_project/logs/openfly/subgoal_nn/eval_<run>/results.pt

Notes:
* This file uses ``sys.argv`` parsing instead of ``argparse`` because
  Streamlit injects its own flags and ``argparse`` would choke on them.
* Images are read straight off disk on demand — no copying — so the
  viewer is cheap to launch even for large eval sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch


def _get_results_path() -> Path:
    # Streamlit forwards args after `--` into sys.argv. Tolerate either
    # `--results <path>` or a positional path.
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--results" and i + 1 < len(args):
            return Path(args[i + 1])
    for a in args:
        if a.endswith(".pt"):
            return Path(a)
    st.error("No --results <path-to-results.pt> argument provided.")
    st.stop()


@st.cache_data(show_spinner=True)
def _load_results(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["per_sample"], obj["summary"]


def _safe_image(col, path: str, caption: str) -> None:
    p = Path(path)
    if not p.is_file():
        col.warning(f"missing image:\n`{path}`")
        col.caption(caption)
        return
    col.image(str(p), caption=caption, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="SubgoalDiT viewer", layout="wide")
    st.title("SubgoalDiT — predictions vs ground truth")

    results_path = _get_results_path()
    per_sample, summary = _load_results(str(results_path))
    if not per_sample:
        st.error("No samples in results file.")
        st.stop()

    # ----- summary card --------------------------------------------------
    with st.expander("Run summary", expanded=True):
        cols = st.columns(4)
        cols[0].metric("N samples",            summary["n_samples"])
        cols[1].metric("Bank size",            summary["bank_size"])
        cols[2].metric("Samples-per-input N",  summary["num_samples_per_input"])
        cols[3].metric("DDIM steps",           summary["ddim_steps"])

        cols = st.columns(4)
        cols[0].metric("cos(mean pred, real)",
                       f"{summary['mean_cos_pred_vs_real']:.3f}")
        cols[1].metric("cos(best-of-N, real)",
                       f"{summary['mean_cos_best_of_n_vs_real']:.3f}")
        cols[2].metric("R@1", f"{summary['R@1']:.1%}")
        cols[3].metric("R@5", f"{summary['R@5']:.1%}")

        cols = st.columns(2)
        cols[0].metric("R@10",         f"{summary['R@10']:.1%}")
        cols[1].metric("Median rank",  summary.get("median_rank"))

        st.caption(
            f"Checkpoint: `{summary['ckpt']}`\n"
            f"Bank: `{summary['bank_dir']}`  |  split: `{summary['split']}`"
        )

    # ----- per-sample browser -------------------------------------------
    st.divider()

    sort_mode = st.sidebar.selectbox(
        "Sort samples by",
        ("dataset order",
         "cos(mean pred, real) — worst first",
         "cos(mean pred, real) — best first",
         "target rank — worst first",
         "target rank — best first"),
    )
    samples = list(enumerate(per_sample))
    if sort_mode == "cos(mean pred, real) — worst first":
        samples.sort(key=lambda kv: kv[1]["cos_mean_pred"])
    elif sort_mode == "cos(mean pred, real) — best first":
        samples.sort(key=lambda kv: -kv[1]["cos_mean_pred"])
    elif sort_mode == "target rank — worst first":
        samples.sort(key=lambda kv: -(kv[1]["target_rank"] if kv[1]["target_in_bank"] else -1))
    elif sort_mode == "target rank — best first":
        samples.sort(key=lambda kv: kv[1]["target_rank"] if kv[1]["target_in_bank"] else 10**9)

    only_in_bank = st.sidebar.checkbox("Only samples whose target is in bank", value=False)
    if only_in_bank:
        samples = [(i, s) for i, s in samples if s["target_in_bank"]]
        if not samples:
            st.warning("No samples with target in bank.")
            st.stop()

    pos = st.sidebar.slider("Position in sorted list", 0, len(samples) - 1, 0)
    raw_idx, sample = samples[pos]
    st.caption(f"Sorted position {pos + 1} / {len(samples)}  (original index {raw_idx})")

    # ----- header: instruction + numerics -------------------------------
    st.subheader(sample["instruction"])
    st.caption(f"Sub-instruction: *{sample['sub_instruction']}*")
    cols = st.columns(4)
    cols[0].metric("cos(mean pred, real)", f"{sample['cos_mean_pred']:.3f}")
    cols[1].metric("cos(best-of-N, real)", f"{sample['cos_best_of_n']:.3f}")
    cols[2].metric(
        "Target rank",
        sample["target_rank"] if sample["target_in_bank"] else "—",
    )
    cols[3].metric("Horizon (steps)", sample["horizon"])
    st.caption(
        f"Pose delta (body frame, dx dy dz dyaw): {sample['pose_delta']}  "
        f"|  Last action id: {sample['last_action']}"
    )

    # ----- image triptych -----------------------------------------------
    cols = st.columns(3)
    _safe_image(cols[0], sample["rgb_path"],         "Current frame")
    _safe_image(cols[1], sample["subgoal_rgb_path"], "Real subgoal")
    _safe_image(
        cols[2], sample["topk_paths"][0],
        f"NN of prediction (sim={sample['topk_sims'][0]:.3f})",
    )

    # ----- top-k strip ---------------------------------------------------
    k = len(sample["topk_paths"])
    st.subheader(f"Top-{k} nearest neighbors of predicted subgoal")
    cols = st.columns(k)
    for col, path, sim in zip(cols, sample["topk_paths"], sample["topk_sims"]):
        _safe_image(col, path, f"sim={sim:.3f}")


if __name__ == "__main__":
    main()
