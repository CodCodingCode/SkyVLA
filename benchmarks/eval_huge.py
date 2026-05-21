"""Run HUGE-Bench task0 offline evaluation for drone_project policies.

Metrics: action MSE (normalized + raw), same as huge_bench/eval_bc.py.
Backends:
  - bc_checkpoint   — trained HugeBCPolicy (.pt from huge_bench/train_bc.py)
  - waypoint_heuristic — predict zero deltas (sanity baseline)
  - state_heuristic — point toward +Z goal encoded in instruction-free prior (weak)
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

from huge_bench.dataset import HugeTask0, collate_bc  # noqa: E402
from huge_bench.policy import HugeBCPolicy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="HUGE-Bench task0 offline eval")
    p.add_argument("--backend", type=str, default="bc_checkpoint",
                   choices=["bc_checkpoint", "waypoint_heuristic", "state_heuristic"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Required for bc_checkpoint")
    p.add_argument("--split", type=str, default="test_seen",
                   choices=["train", "test_seen", "test_unseen"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if omitted)")
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


class WaypointHeuristic:
    """Predict zero action deltas — lower bound for MSE."""

    def __call__(self, batch: dict) -> torch.Tensor:
        return torch.zeros_like(batch["action"])


class StateHeuristic:
    """Use normalized state as a crude proxy: small random walk toward origin in action space."""

    def __call__(self, batch: dict) -> torch.Tensor:
        # Predict opposite of current normalized state on xy (weak baseline)
        s = batch["state"]
        pred = torch.zeros_like(batch["action"])
        pred[:, :2] = -0.1 * s[:, :2]
        return pred


@torch.no_grad()
def predict_bc(batch: dict, model: HugeBCPolicy) -> torch.Tensor:
    return model(batch, with_grad_through_lora=False)


def main():
    args = parse_args()
    if args.backend == "bc_checkpoint" and not args.checkpoint:
        raise SystemExit("--checkpoint required for bc_checkpoint backend")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[huge] device={device} backend={args.backend} split={args.split}")

    ds = HugeTask0(split=args.split)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_bc, drop_last=False,
    )

    model = None
    heuristic = None
    if args.backend == "bc_checkpoint":
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt.get("args", {})
        model = HugeBCPolicy(
            max_text_length=cfg.get("max_text_length", 64),
            lora_rank=cfg.get("lora_rank", 8),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    elif args.backend == "waypoint_heuristic":
        heuristic = WaypointHeuristic()
    else:
        heuristic = StateHeuristic()

    per_dim_sq = np.zeros(4, dtype=np.float64)
    raw_per_dim_sq = np.zeros(4, dtype=np.float64)
    n = 0
    env_stats: dict[str, list[float]] = defaultdict(list)

    for i, batch in enumerate(loader):
        if args.max_batches >= 0 and i >= args.max_batches:
            break
        if model is not None:
            pred = predict_bc(batch, model)
        else:
            pred = heuristic(batch)

        target = batch["action"].to(device)
        raw_target = batch["raw_action"].to(device)
        raw_pred = ds.denormalize_action(pred.cpu())
        if isinstance(raw_pred, torch.Tensor):
            raw_pred = raw_pred.to(device)

        sq = ((pred.to(device) - target) ** 2).cpu().numpy()
        raw_sq = ((raw_pred - raw_target) ** 2).cpu().numpy()
        per_dim_sq += sq.sum(axis=0)
        raw_per_dim_sq += raw_sq.sum(axis=0)
        n += sq.shape[0]
        for j, env_id in enumerate(batch["env_id"]):
            env_stats[env_id].append(float(sq[j].mean()))
        if (i + 1) % 10 == 0:
            print(f"  batch {i + 1}: {n} samples")

    mse = float(np.mean(per_dim_sq / max(1, n)))
    raw_mse = float(np.mean(raw_per_dim_sq / max(1, n)))
    print(f"\n=== HUGE-Bench {args.split} ({n} samples) ===")
    print(f"  normalized MSE: {mse:.5f}")
    print(f"  raw MSE:        {raw_mse:.5f}")
    for name, v in zip(["dx", "dy", "dz", "dyaw"], per_dim_sq / max(1, n)):
        print(f"    {name}: {v:.5f}")

    if args.out_json:
        out = {
            "benchmark": "HUGE-Bench-task0",
            "backend": args.backend,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n_samples": n,
            "mse_normalized": mse,
            "mse_raw": raw_mse,
            "mse_per_dim_normalized": (per_dim_sq / max(1, n)).tolist(),
            "mse_per_dim_raw": (raw_per_dim_sq / max(1, n)).tolist(),
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
