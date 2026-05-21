"""Offline target-prediction evaluation on HUGE-Bench for the hierarchical VLA.

Complements ``benchmarks/eval_huge.py`` (which scores ``HugeBCPolicy`` on
delta-action MSE) by scoring the Stage-6 hierarchical VLA on its actual
output: the predicted body-frame target.

Metric: target MSE (mean over xyz, in metres) and median displacement error
on test_seen / test_unseen, plus the cosine direction agreement with the
ground-truth body-frame future waypoint.

Backends:
  - vla_highlevel  — checkpoint from ``huge_bench/train_vla_highlevel.py``
                     or any ``vla/train.py`` checkpoint (shape-tolerant load)
  - zero           — predict ``target_body = 0`` (lower bound)

Launch:
    python -m benchmarks.eval_huge_vla \
        --backend vla_highlevel \
        --checkpoint logs/huge_bench_highlevel/<run>/model_5000.pt \
        --split test_seen
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from huge_bench.dataset_highlevel import (  # noqa: E402
    HugeTask0WithFutureWaypoint,
    collate_highlevel,
)
from huge_bench.train_vla_highlevel import (  # noqa: E402
    NUM_IMAGE_TOKENS,
    _build_obs,
    _load_processor,
    _load_resume,
)
from vla.vla_policy import HierarchicalVLAActor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HUGE-Bench target-prediction eval (hierarchical VLA)")
    p.add_argument("--backend", type=str, default="vla_highlevel",
                   choices=["vla_highlevel", "zero"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Required for vla_highlevel.")
    p.add_argument("--split", type=str, default="test_seen",
                   choices=["train", "test_seen", "test_unseen"])
    p.add_argument("--lookahead_frames", type=int, default=25,
                   help="Must match training (default 25 ~ 5s at 5Hz).")
    p.add_argument("--target_range", type=float, default=3.0,
                   help="Must match HierarchicalVLAActor.target_range (tanh cap).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--paligemma_model", type=str, default="google/paligemma-3b-pt-224")
    p.add_argument("--waypoint_checkpoint", type=str,
                   default=str(_DRONE_ROOT / "checkpoints" / "stage2_waypoint.pt"))
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def _predict_vla(actor, batch, processor, max_text_length, image_token_id,
                 target_range, device) -> torch.Tensor:
    obs = _build_obs(batch, processor, max_text_length, image_token_id, device)
    target_logits, _ = actor.forward_lora_grad(obs)
    return torch.tanh(target_logits) * target_range


def main() -> None:
    args = parse_args()
    if args.backend == "vla_highlevel" and not args.checkpoint:
        raise SystemExit("--checkpoint required for vla_highlevel backend")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[huge-vla] device={device} backend={args.backend} split={args.split}")

    ds = HugeTask0WithFutureWaypoint(
        split=args.split,
        lookahead_frames=args.lookahead_frames,
        max_target_norm=args.target_range,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_highlevel, drop_last=False,
    )

    actor = None
    processor = None
    image_token_id = None
    max_text_length = NUM_IMAGE_TOKENS + 24
    if args.backend == "vla_highlevel":
        print("[huge-vla] Building HierarchicalVLAActor...")
        actor = HierarchicalVLAActor(
            waypoint_checkpoint_path=args.waypoint_checkpoint,
            paligemma_model_name=args.paligemma_model,
            target_range=args.target_range,
        ).to(device)
        _load_resume(actor, args.checkpoint, device)
        actor.eval()
        processor = _load_processor(args.paligemma_model)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    sq_sum = np.zeros(3, dtype=np.float64)
    abs_dist: list[float] = []
    cos_sum = 0.0
    n = 0
    per_env: dict[str, list[float]] = defaultdict(list)

    for i, batch in enumerate(loader):
        if args.max_batches >= 0 and i >= args.max_batches:
            break
        target_body = batch["target_body"].to(device)
        if args.backend == "vla_highlevel":
            pred = _predict_vla(
                actor, batch, processor, max_text_length, image_token_id,
                args.target_range, device,
            )
        else:
            pred = torch.zeros_like(target_body)

        diff = (pred - target_body).cpu().numpy()
        sq_sum += (diff ** 2).sum(axis=0)
        per_sample_dist = np.linalg.norm(diff, axis=-1)
        abs_dist.extend(per_sample_dist.tolist())

        # Cosine direction agreement, masked to non-static labels.
        norm_label = target_body.norm(dim=-1)
        mask = (norm_label > 0.05).float()
        if mask.sum() > 0:
            cos = torch.nn.functional.cosine_similarity(pred, target_body, dim=-1, eps=1e-6)
            cos_sum += float((cos * mask).sum().item()) / max(1.0, float(mask.sum().item())) * diff.shape[0]

        for j, env_id in enumerate(batch["env_id"]):
            per_env[env_id].append(float(per_sample_dist[j]))
        n += diff.shape[0]
        if (i + 1) % 10 == 0:
            print(f"  batch {i + 1}: {n} samples")

    mse_per_dim = sq_sum / max(1, n)
    mse = float(mse_per_dim.mean())
    median_dist = float(np.median(abs_dist)) if abs_dist else float("nan")
    p90_dist = float(np.percentile(abs_dist, 90)) if abs_dist else float("nan")
    cos_avg = cos_sum / max(1, n)

    print(f"\n=== HUGE-Bench target-prediction {args.split} ({n} samples) ===")
    print(f"  target MSE (m^2):     {mse:.4f}  per-dim={mse_per_dim.round(4).tolist()}")
    print(f"  median |error| (m):   {median_dist:.3f}")
    print(f"  p90 |error| (m):      {p90_dist:.3f}")
    print(f"  cosine direction:     {cos_avg:.3f}")
    if per_env:
        print("  per-env (median |error|):")
        for env_id, vals in sorted(per_env.items()):
            print(f"    {env_id}: {np.median(vals):.3f}  (n={len(vals)})")

    if args.out_json:
        out = {
            "benchmark": "HUGE-Bench-task0-target",
            "backend": args.backend,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_samples": n,
            "mse_target": mse,
            "mse_per_dim": mse_per_dim.tolist(),
            "median_error_m": median_dist,
            "p90_error_m": p90_dist,
            "cosine_direction": cos_avg,
            "per_env_median_error_m": {k: float(np.median(v)) for k, v in per_env.items()},
            "lookahead_frames": args.lookahead_frames,
            "target_range": args.target_range,
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
