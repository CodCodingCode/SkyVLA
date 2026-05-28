"""Standalone DiT evaluator: re-run val_cos on a saved checkpoint at
multiple DDIM step counts.

Why this script exists: every time the trainer's ``--val_ddim_steps``
default changes, historical ``val_cos`` numbers stop being directly
comparable to new ones. This script lets you re-measure a checkpoint
at any step count, in particular at the policy's deploy setting (4) —
which is what really matters for downstream SR. Output is a small
table: per-split × per-step-count cos sim and ε-prediction loss.

Reuses ``_eval_step`` from ``openfly.train_subgoal_dit`` directly so
there's no risk of drift between training-time val and this offline
measurement.

Typical use:
  # compare current ckpt against the historical 0.61 baseline at the
  # policy's deploy DDIM step count
  python -m openfly.scripts.eval_dit \\
      --checkpoint logs/openfly/subgoal_dit/20260525_184902/best.pt \\
      --val_split seen --val_ood_split unseen \\
      --val_max_episodes 100 \\
      --ddim_steps 4 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_DRONE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.dataset import OpenFlyDataset, collate
from openfly.models.subgoal_dit import SubgoalDiT
from openfly.train_subgoal_dit import (
    _build_processor,
    _eval_step,
)
from vla.vla_policy import PaliGemmaFeatureExtractor


def _build_dit_from_args(
    ckpt_args: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """Instantiate the DiT class that produced the saved checkpoint.

    The trainer stores its CLI args in the ckpt; ``pretrained_path``
    distinguishes the PixArt-Σ wrapper from the random-init SubgoalDiT.
    """
    if ckpt_args.get("pretrained_path"):
        from openfly.models.subgoal_dit_pixart import PixArtSubgoalDiT
        return PixArtSubgoalDiT(
            pretrained_path=ckpt_args["pretrained_path"],
            token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            pose_delta_dim=4,
            num_last_actions=9,
            num_timesteps=int(ckpt_args.get("num_timesteps", 1000)),
            freeze_backbone=bool(ckpt_args.get("freeze_backbone", False)),
            # dropout layers have no params; default 0.0 is fine even if
            # the saved checkpoint was trained with a different value.
            dropout=float(ckpt_args.get("dropout", 0.0)),
        ).to(device)
    return SubgoalDiT(
        token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
        hidden=int(ckpt_args.get("hidden", 1024)),
        depth=int(ckpt_args.get("depth", 12)),
        num_heads=int(ckpt_args.get("num_heads", 16)),
        text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
        pose_delta_dim=4,
        num_last_actions=9,
        num_timesteps=int(ckpt_args.get("num_timesteps", 1000)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
    ).to(device)


def _load_weights(
    dit: torch.nn.Module,
    ckpt: dict[str, Any],
    prefer_ema: bool,
) -> str:
    """Load DiT weights from ckpt. Returns a short description of which
    weights ended up in the model.

    ``best.pt`` files already store EMA weights directly in the ``dit``
    key (trainer line 1045-1055). ``last.pt`` stores live weights in
    ``dit`` and the EMA shadow under ``ema``. With ``prefer_ema``, this
    helper applies the shadow over the live weights when both are
    present — matching the trainer's val-time behaviour.
    """
    state = ckpt.get("dit") or ckpt.get("state_dict") or ckpt.get("model")
    if state is None:
        raise KeyError("Checkpoint has no 'dit' / 'state_dict' / 'model' key")
    missing, unexpected = dit.load_state_dict(state, strict=False)
    if missing:
        print(f"  [eval_dit] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [eval_dit] unexpected keys: {len(unexpected)}")
    source = "ckpt.dit"

    if prefer_ema and isinstance(ckpt.get("ema"), dict):
        shadow = ckpt["ema"].get("shadow")
        if isinstance(shadow, dict):
            applied = 0
            with torch.no_grad():
                for name, p in dit.named_parameters():
                    if name in shadow:
                        p.data.copy_(shadow[name].to(p.device, p.dtype))
                        applied += 1
            if applied:
                source = f"ckpt.ema.shadow ({applied} params)"
    return source


def _build_val_loader(
    split: str,
    *,
    ckpt_args: dict[str, Any],
    cli_args: argparse.Namespace,
    max_episodes_override: int,
) -> tuple[DataLoader | None, int]:
    if not split or split.lower() in {"none", ""}:
        return None, 0
    ds = OpenFlyDataset(
        split=split,
        history_frames=int(ckpt_args.get("history_frames", 0)),
        env_filter=ckpt_args.get("env_filter"),
        max_episodes=max_episodes_override,
        max_samples=cli_args.val_max_samples,
        require_images=True,
        oversample_stop=1.0,
        subgoal_pairing=ckpt_args.get("subgoal_pairing", "mixed"),
        subgoal_semantic_prob=float(ckpt_args.get("subgoal_semantic_prob", 0.25)),
        subgoal_uniform_max=int(ckpt_args.get("subgoal_uniform_max", 4)),
    )
    if len(ds) == 0:
        print(f"[eval_dit] WARNING: val split {split!r} loaded 0 samples")
        return None, 0
    loader = DataLoader(
        ds,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(ds)


def _run_eval(
    dit,
    paligemma,
    processor,
    loader: DataLoader,
    device: torch.device,
    ddim_steps: int,
) -> dict[str, float]:
    vl_sum, cos_sum, vn_sum = 0.0, 0.0, 0
    for batch in loader:
        m = _eval_step(
            dit, paligemma, processor, batch, device,
            num_ddim_steps=ddim_steps,
        )
        if m["n_valid"] == 0:
            continue
        vl_sum += m["val_loss"] * m["n_valid"]
        cos_sum += m["val_cos"] * m["n_valid"]
        vn_sum += m["n_valid"]
    return {
        "loss": vl_sum / max(vn_sum, 1),
        "cos": cos_sum / max(vn_sum, 1),
        "n_valid": vn_sum,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to a DiT best.pt / last.pt")
    p.add_argument(
        "--val_split", default="seen",
        help="Primary val split. Defaults to 'seen' (in-distribution).",
    )
    p.add_argument(
        "--val_ood_split", default="unseen",
        help="Second val split. Defaults to 'unseen' (OOD). Pass 'none' to disable.",
    )
    p.add_argument("--val_max_episodes", type=int, default=100)
    p.add_argument("--val_ood_max_episodes", type=int, default=100)
    p.add_argument("--val_max_samples", type=int, default=0)
    p.add_argument(
        "--ddim_steps", type=int, nargs="+", default=[4, 20],
        help="One or more DDIM step counts to evaluate at. Default: '4 20' "
        "to compare deploy-quality vs the historical baseline.",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--no_ema", action="store_true",
        help="Skip EMA shadow application; eval the live weights as stored "
        "in ckpt['dit']. best.pt already contains EMA, so this matters only "
        "for last.pt.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--paligemma_dtype", default=None,
        help="Override PaliGemma dtype (float16/bfloat16/float32). Default: "
        "use what the ckpt was trained with, or float16.",
    )
    args = p.parse_args()

    # Replicate the trainer's chdir so any relative paths resolve consistently.
    import os
    os.chdir(_DRONE_ROOT)

    device = torch.device(args.device)
    print(f"[eval_dit] loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args") or {}
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    print(
        f"[eval_dit] ckpt epoch={ckpt.get('epoch','?')} "
        f"val_cos@save={ckpt.get('val_cos', 'n/a')}"
    )

    # PaliGemma
    paligemma_model = ckpt_args.get("paligemma_model", "google/paligemma-3b-pt-224")
    dtype_name = args.paligemma_dtype or ckpt_args.get("paligemma_dtype", "float16")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    paligemma_dtype = dtype_map.get(dtype_name, torch.float16)

    paligemma = PaliGemmaFeatureExtractor(
        model_name=paligemma_model,
        lora_rank=8, lora_alpha=16.0,
        dtype=paligemma_dtype,
    ).to(device)
    paligemma.eval()
    for prm in paligemma.parameters():
        prm.requires_grad = False
    processor = _build_processor(paligemma_model)

    # DiT
    dit = _build_dit_from_args(ckpt_args, device)
    source = _load_weights(dit, ckpt, prefer_ema=not args.no_ema)
    dit.eval()
    print(f"[eval_dit] DiT weights from {source}")

    # Val loaders
    val_loader, n_val = _build_val_loader(
        args.val_split, ckpt_args=ckpt_args, cli_args=args,
        max_episodes_override=args.val_max_episodes,
    )
    val_ood_loader, n_ood = _build_val_loader(
        args.val_ood_split, ckpt_args=ckpt_args, cli_args=args,
        max_episodes_override=args.val_ood_max_episodes,
    )
    print(
        f"[eval_dit] val={n_val} ({args.val_split}) / "
        f"val_ood={n_ood} ({args.val_ood_split})  "
        f"steps={args.ddim_steps}"
    )

    # Evaluate
    results: list[dict[str, Any]] = []
    for steps in args.ddim_steps:
        for split_name, loader in (
            (args.val_split, val_loader),
            (args.val_ood_split, val_ood_loader),
        ):
            if loader is None:
                continue
            t0 = time.time()
            m = _run_eval(dit, paligemma, processor, loader, device, steps)
            m.update({"split": split_name, "ddim_steps": steps, "time_s": time.time() - t0})
            results.append(m)
            print(
                f"  split={split_name:<10} ddim={steps:<3}  "
                f"cos={m['cos']:.4f}  loss={m['loss']:.4f}  "
                f"n_valid={m['n_valid']}  ({m['time_s']:.1f}s)"
            )

    # Pretty summary table
    print("\n=== summary ===")
    splits = sorted({r["split"] for r in results})
    steps = sorted({r["ddim_steps"] for r in results})
    print(f"  {'split':<10}  " + "  ".join(f"ddim={s:<3}" for s in steps))
    for sp in splits:
        cells = []
        for st in steps:
            hit = next((r for r in results if r["split"] == sp and r["ddim_steps"] == st), None)
            cells.append(f"{hit['cos']:.4f}" if hit else "  -   ")
        print(f"  {sp:<10}  " + "  ".join(c.ljust(8) for c in cells))

    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(
            {"checkpoint": args.checkpoint, "ckpt_args": ckpt_args, "results": results},
            f, indent=2, default=str,
        )
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
