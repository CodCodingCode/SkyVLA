#!/usr/bin/env python3
"""Sample subgoals from a trained DiT and retrieve nearest-neighbor RGB
frames from the precomputed bank — produces both interpretable metrics
(R@1 / R@5 / R@10 / best-of-N cos sim) and a per-sample results file
that the Streamlit viewer browses visually.

What this gives you that ``val_cos`` alone doesn't:

* **Retrieval rank.** "Out of N=10k val frames, what rank does the real
  subgoal hold when sorted by similarity to the prediction?" A median
  rank of 1–10 is great; a median of 5000 is essentially random.
* **R@k.** Recall@k headlines — how often the real subgoal is in the
  top-k. Much easier to reason about than a raw cosine.
* **Best-of-N cos sim.** Diffusion is stochastic; at intersections the
  model may sample a valid-but-different outcome. Reporting best-of-8
  separates "wrong" from "right alternative."
* **Visual ground-truth.** For each prediction, save the path of its
  top-k real-frame neighbors so the viewer can render them side-by-side
  with the actual subgoal.

The results file is the **only** artifact the Streamlit viewer needs;
it is fully self-contained (paths to RGBs the viewer reads directly).

Usage:
  python -m openfly.scripts.eval_subgoal_nn \\
    --ckpt   ~/drone_project/logs/openfly/subgoal_dit/<run>/best.pt \\
    --bank   ~/drone_project/logs/openfly/subgoal_nn/bank_unseen \\
    --split  unseen \\
    --num_samples 8 \\
    --out_dir ~/drone_project/logs/openfly/subgoal_nn/eval_<run>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_DRONE_ROOT = Path(__file__).resolve().parents[2]
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.dataset import OpenFlyDataset, collate
from openfly.train_subgoal_dit import (
    _build_processor,
    _tokenise_batch,
    _encode_frame_pair,
    _body_frame_pose_delta,
)
from openfly.train_paligemma_subgoal import _load_world_model
from vla.vla_policy import PaliGemmaFeatureExtractor


def _pool_norm(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the 256-token axis and L2-normalize — same recipe
    the bank used. Returned shape: ``(B, 2048)`` fp32."""
    pooled = tokens.mean(dim=1).float()
    return F.normalize(pooled, dim=-1)


def _topk_neighbors(
    queries: torch.Tensor,        # (B, D) unit-normed
    bank_features: torch.Tensor,  # (N, D) unit-normed
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top-k similarity scores, top-k indices into bank)."""
    sims = queries @ bank_features.T          # (B, N) cos sim
    top = sims.topk(k, dim=-1, largest=True)
    return top.values, top.indices


def _rank_of_target(
    queries: torch.Tensor,        # (B, D)
    bank_features: torch.Tensor,  # (N, D)
    target_idx: torch.Tensor,     # (B,) long — bank index of the real subgoal, -1 if not in bank
) -> torch.Tensor:
    """1-based rank of the target index per query (lower is better).
    Returns -1 for queries whose target is not in the bank."""
    out = torch.full_like(target_idx, fill_value=-1)
    sims = queries @ bank_features.T          # (B, N)
    for i in range(queries.shape[0]):
        ti = int(target_idx[i].item())
        if ti < 0:
            continue
        q_score = sims[i, ti]
        # Rank = 1 + (number of bank entries strictly more similar than the target).
        out[i] = 1 + int((sims[i] > q_score).sum().item())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True,
                        help="DiT checkpoint (best.pt is conventional — it carries "
                        "EMA weights when training was run with --ema_decay>0).")
    parser.add_argument("--pretrained_path", default=None,
                        help="PixArt-Σ snapshot dir; required if the ckpt is a "
                        "PixArtSubgoalDiT (the shape-match check would fail otherwise).")
    parser.add_argument("--bank", required=True,
                        help="Directory containing bank.pt + meta.json from "
                        "build_subgoal_nn_bank.py.")
    parser.add_argument("--split", default="unseen",
                        help="Split to evaluate on. Should match the split the "
                        "bank was built from so retrievals are over the same population.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap eval samples — keep this modest (default 200 via "
                        "limit below) since each sample costs N DDIM rollouts.")
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Diffusion samples per input (for best-of-N cos sim "
                        "and to surface diversity). 8 is a good default; "
                        "1 disables stochastic eval.")
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5,
                        help="How many nearest neighbors to retrieve per prediction.")
    parser.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    parser.add_argument("--paligemma_dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--out_dir",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project"))
                    / "logs" / "openfly" / "subgoal_nn" / "eval"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    # Default cap to 200 samples — N×20 DDIM rollouts add up fast.
    if args.max_samples == 0:
        args.max_samples = 200
        print(f"[eval_subgoal_nn] --max_samples defaulted to {args.max_samples}; "
              "pass an explicit value to override.")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_subgoal_nn] ckpt={args.ckpt} bank={args.bank} split={args.split}")
    print(f"[eval_subgoal_nn] out_dir={out_dir}")

    # ----- load bank ----------------------------------------------------
    bank_dir = Path(args.bank).resolve()
    bank = torch.load(bank_dir / "bank.pt", map_location="cpu", weights_only=False)
    with open(bank_dir / "meta.json") as f:
        bank_meta = json.load(f)
    bank_features: torch.Tensor = bank["features"].float().to(device)  # (N, 2048)
    bank_paths: list[str] = list(bank["paths"])
    path_to_bank_idx = {p: i for i, p in enumerate(bank_paths)}
    print(f"[eval_subgoal_nn] bank: {bank_features.shape[0]} frames, "
          f"D={bank_features.shape[1]}")

    # ----- models -------------------------------------------------------
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    paligemma = PaliGemmaFeatureExtractor(
        model_name=args.paligemma_model, lora_rank=8, lora_alpha=16.0,
        dtype=dtype_map[args.paligemma_dtype],
    ).to(device)
    paligemma.eval()
    for p in paligemma.parameters():
        p.requires_grad = False

    dit = _load_world_model(args.ckpt, pretrained_path=args.pretrained_path, device=device)
    processor = _build_processor(args.paligemma_model)

    # ----- data ---------------------------------------------------------
    ds = OpenFlyDataset(
        split=args.split,
        history_frames=0,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=True,
        oversample_stop=1.0,
        subgoal_pairing="semantic_only",  # deterministic — viewer reads the same target
    )
    if len(ds) == 0:
        raise RuntimeError("empty dataset")
    print(f"[eval_subgoal_nn] dataset: {len(ds)} samples")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    # ----- eval ---------------------------------------------------------
    per_sample: list[dict] = []
    cos_to_real_all: list[float] = []        # cos sim of *mean prediction* to real
    cos_to_real_best: list[float] = []       # best-of-N
    ranks: list[int] = []                    # -1 if target not in bank
    r1 = r5 = r10 = 0
    n_with_target_in_bank = 0
    t0 = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            subgoal_rgb = batch["subgoal_rgb"].to(device, non_blocking=True)
            pose = batch["pose"].to(device, non_blocking=True)
            subgoal_pose = batch["subgoal_pose"].to(device, non_blocking=True)
            last_action = batch["last_action"].to(device, non_blocking=True)
            horizon = batch["subgoal_horizon"].to(device, non_blocking=True)
            valid = batch["subgoal_valid"].to(device, non_blocking=True)

            input_ids, attention_mask = _tokenise_batch(
                processor, batch["instruction"], batch["sub_instruction"],
                rgb_dummy=rgb[0], device=device,
            )
            curr_tokens, tgt_tokens, text_embed = _encode_frame_pair(
                paligemma, rgb, subgoal_rgb, input_ids, attention_mask
            )
            pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)

            B = rgb.shape[0]
            # Sample N=num_samples subgoals per input. We do this as a
            # serial loop because batching N×B blows VRAM on big DiTs;
            # at N=8, B=8 the overhead is acceptable.
            pred_tokens_all = []  # list of (B, 256, 2048)
            for _ in range(args.num_samples):
                x0 = dit.ddim_sample(
                    curr_tokens=curr_tokens, text_embed=text_embed,
                    pose_delta=pose_delta, last_action=last_action, horizon=horizon,
                    num_steps=args.ddim_steps,
                )
                pred_tokens_all.append(x0)
            pred_stack = torch.stack(pred_tokens_all, dim=0)  # (N, B, 256, 2048)

            # Mean over N samples → one representative prediction per input.
            mean_pred_tokens = pred_stack.mean(dim=0)  # (B, 256, 2048)

            # Pool everything for retrieval & similarity scoring.
            mean_pred_pool = _pool_norm(mean_pred_tokens)               # (B, D)
            real_pool = _pool_norm(tgt_tokens)                          # (B, D)
            # Per-sample pool for best-of-N
            sample_pools = _pool_norm(
                pred_stack.reshape(-1, 256, 2048)
            ).reshape(args.num_samples, B, -1)                          # (N, B, D)

            # Cos sim of *mean prediction* to real subgoal tokens (just for reference).
            cos_mean = F.cosine_similarity(mean_pred_pool, real_pool, dim=-1)
            # Best-of-N: per sample, max cos sim across the N predictions.
            sample_cos = (sample_pools * real_pool.unsqueeze(0)).sum(dim=-1)  # (N, B)
            cos_best = sample_cos.max(dim=0).values                          # (B,)

            # Top-k retrieval over the bank, using the mean prediction.
            topk_sims, topk_idx = _topk_neighbors(mean_pred_pool, bank_features, args.topk)

            # Rank of the actual subgoal target inside the bank.
            sg_paths: list[str] = batch["subgoal_rgb_path"]
            target_idx = torch.tensor(
                [path_to_bank_idx.get(p, -1) for p in sg_paths],
                device=device, dtype=torch.long,
            )
            sample_ranks = _rank_of_target(mean_pred_pool, bank_features, target_idx)

            for i in range(B):
                if not bool(valid[i].item()):
                    continue
                rec = {
                    "instruction": batch["instruction"][i],
                    "sub_instruction": batch["sub_instruction"][i],
                    "rgb_path": batch["rgb_path"][i],
                    "subgoal_rgb_path": sg_paths[i],
                    "pose_delta": pose_delta[i].cpu().tolist(),
                    "horizon": int(horizon[i].item()),
                    "last_action": int(last_action[i].item()),
                    "cos_mean_pred": float(cos_mean[i].item()),
                    "cos_best_of_n": float(cos_best[i].item()),
                    "topk_paths": [bank_paths[int(j)] for j in topk_idx[i].cpu().tolist()],
                    "topk_sims": [float(s) for s in topk_sims[i].cpu().tolist()],
                    "target_in_bank": bool(target_idx[i].item() >= 0),
                    "target_rank": int(sample_ranks[i].item()),
                }
                per_sample.append(rec)

                cos_to_real_all.append(rec["cos_mean_pred"])
                cos_to_real_best.append(rec["cos_best_of_n"])
                if rec["target_in_bank"]:
                    n_with_target_in_bank += 1
                    r = rec["target_rank"]
                    ranks.append(r)
                    r1 += int(r <= 1)
                    r5 += int(r <= 5)
                    r10 += int(r <= 10)

            if step % 5 == 0:
                dt = time.time() - t0
                done = len(per_sample)
                print(
                    f"[eval_subgoal_nn] step={step:04d} done={done} "
                    f"elapsed={dt:.1f}s",
                    flush=True,
                )

    # ----- aggregate ----------------------------------------------------
    n_all = max(1, len(per_sample))
    n_bk = max(1, n_with_target_in_bank)
    summary = {
        "n_samples": len(per_sample),
        "n_with_target_in_bank": n_with_target_in_bank,
        "bank_size": bank_features.shape[0],
        "ddim_steps": args.ddim_steps,
        "num_samples_per_input": args.num_samples,
        "mean_cos_pred_vs_real": sum(cos_to_real_all) / n_all,
        "mean_cos_best_of_n_vs_real": sum(cos_to_real_best) / n_all,
        "median_rank": _median(ranks) if ranks else None,
        "R@1": r1 / n_bk,
        "R@5": r5 / n_bk,
        "R@10": r10 / n_bk,
        "ckpt": str(Path(args.ckpt).resolve()),
        "bank_dir": str(bank_dir),
        "split": args.split,
    }
    print("\n[eval_subgoal_nn] === summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")

    results_path = out_dir / "results.pt"
    summary_path = out_dir / "summary.json"
    torch.save({"per_sample": per_sample, "summary": summary}, results_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval_subgoal_nn] wrote {results_path}")
    print(f"[eval_subgoal_nn] wrote {summary_path}")
    print(f"[eval_subgoal_nn] view with: streamlit run openfly/scripts/subgoal_viewer.py -- "
          f"--results {results_path}")
    return 0


def _median(xs: list[int]) -> float:
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


if __name__ == "__main__":
    raise SystemExit(main())
