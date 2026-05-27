#!/usr/bin/env python3
"""Three-way evaluation of a P3 (BC + subgoals) checkpoint.

Why this script exists: the in-trainer ``val_acc`` reported by
``train_paligemma_subgoal.py`` uses ORACLE subgoals (the real
next-keyframe RGB encoded through PaliGemma). Because OpenFly's 8
discrete actions each produce a deterministic kinematic transform,
knowing ``(current_view, real_next_view)`` *uniquely identifies* the
action that was taken — the policy can read the answer off the
subgoal without doing any navigation. That metric is therefore
gameable and uninformative about real inference performance.

This script runs the same checkpoint three ways on the same val data:

* ``none``    — no subgoal pathway at all (policy is the BC baseline).
                Tells us: how much can the policy do without world-model help?
* ``oracle``  — feed the real ``subgoal_rgb`` encoded through PaliGemma.
                Tells us: upper-bound when the subgoal is perfect. This is
                the leakage-rich number; useful as a diagnostic ceiling.
* ``dit``     — sample subgoals from the P2 DiT via 4-step DDIM.
                Tells us: actual inference-time performance — the only
                number worth reporting in the paper.

The number we want for the headline claim "subgoals help" is the
**delta between ``none`` and ``dit``**. If they match, the world model
isn't contributing. If ``dit`` beats ``none`` by a meaningful margin
without quite matching ``oracle``, the world model is genuinely
shaping the policy's decisions (and the gap-to-oracle tells us how
much higher we could go with a better world model).

Per-class accuracy is also printed since OpenFly's action distribution
is heavily skewed (forward_9m ≈ 53%). A model can hit 0.6 overall by
just always predicting forward; the rare-class accuracies (stop,
turn-left, turn-right) are where subgoal conditioning should matter most.

Run:

    python -m openfly.eval_p3_subgoals \
      --p3_ckpt   logs/openfly/paligemma_subgoal/<run>/last.pt \
      --dit_path  logs/openfly/subgoal_dit/<run>/best.pt \
      --pretrained_path /path/to/pixart-snap \
      --modes     none,oracle,dit \
      --split     seen \
      --max_episodes 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openfly.actions import TRAINABLE_ACTION_NAMES
from openfly.dataset import NUM_OPENFLY_ACTIONS, OpenFlyDataset, collate
from openfly.models.paligemma_vln import PaliGemmaVLNPolicy
from openfly.train_paligemma_subgoal import (
    _build_processor,
    _tokenise_batch,
    _body_frame_pose_delta,
    _load_world_model,
    _oracle_subgoal_tokens,
    _dit_subgoal_tokens,
)


# ---------------------------------------------------------------------------
# Per-mode forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_batch(
    model: PaliGemmaVLNPolicy,
    dit,
    processor,
    batch: dict[str, Any],
    device: torch.device,
    mode: str,
    ddim_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one batch through the policy with the chosen subgoal mode.

    Returns
    -------
    preds: (B,) long — predicted action ids
    targets: (B,) long — ground-truth action ids
    """
    rgb = batch["rgb"].to(device, non_blocking=True)
    subgoal_rgb = batch["subgoal_rgb"].to(device, non_blocking=True)
    history = (
        batch["history"].to(device, non_blocking=True)
        if batch["history"].numel() > 0
        else batch["history"].to(device)
    )
    pose = batch["pose"].to(device, non_blocking=True)
    subgoal_pose = batch["subgoal_pose"].to(device, non_blocking=True)
    actions = batch["action_id"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    next_pose = batch["next_pose"].to(device, non_blocking=True)
    horizon = batch["subgoal_horizon"].to(device, non_blocking=True)
    progress = batch.get("progress")
    if progress is not None:
        progress = progress.to(device, non_blocking=True)

    input_ids, attention_mask = _tokenise_batch(
        processor, batch["instruction"], rgb_dummy=rgb[0], device=device,
        sub_instructions=batch.get("sub_instruction"),
    )

    # Build the subgoal tokens (or None) per mode.
    if mode == "none":
        subgoal_tokens = None
    elif mode == "oracle":
        subgoal_tokens = _oracle_subgoal_tokens(
            model.paligemma, subgoal_rgb, input_ids, attention_mask
        )
    elif mode == "dit":
        # Need raw curr_siglip + text_embed for the DiT.
        pv = model.paligemma.preprocess_images(rgb)
        gemma_feats, curr_siglip = model.paligemma.forward_tokens(
            pv, input_ids, attention_mask
        )
        model.paligemma.clear_cache()
        token_per_frame = 256
        text_feats = (
            gemma_feats[:, token_per_frame:]
            if gemma_feats.shape[1] > token_per_frame else gemma_feats
        )
        text_mask_part = (
            attention_mask[:, token_per_frame:]
            if attention_mask.shape[1] > token_per_frame else attention_mask
        )
        B = rgb.shape[0]
        seq_lengths = text_mask_part.sum(dim=1).clamp(min=1) - 1
        b_idx = torch.arange(B, device=device)
        text_embed = text_feats[b_idx, seq_lengths].float()
        pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)
        subgoal_tokens = _dit_subgoal_tokens(
            dit,
            curr_tokens=curr_siglip.float(),
            text_embed=text_embed,
            pose_delta=pose_delta,
            last_action=last_action,
            horizon=horizon,
            num_steps=ddim_steps,
        )
    else:
        raise ValueError(f"unknown mode: {mode!r}; choices: none|oracle|dit")

    out = model(
        instruction_input_ids=input_ids,
        instruction_attention_mask=attention_mask,
        rgb_current=rgb,
        rgb_history=history,
        pose=pose,
        last_action=last_action,
        next_pose=next_pose,
        progress=progress,
        with_grad=False,
        subgoal_tokens=subgoal_tokens,
    )
    preds = out["action_logits"].argmax(dim=-1)
    return preds.detach().cpu(), actions.detach().cpu()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _per_class_table(name: str, correct: np.ndarray, total: np.ndarray) -> str:
    parts = []
    for c in range(NUM_OPENFLY_ACTIONS):
        n = int(total[c])
        if n > 0:
            acc = float(correct[c]) / n
            parts.append(
                f"{TRAINABLE_ACTION_NAMES.get(c, str(c))}: {acc:.3f} ({correct[c]}/{n})"
            )
        else:
            parts.append(f"{TRAINABLE_ACTION_NAMES.get(c, str(c))}: NA")
    return f"  [{name}] " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--p3_ckpt", type=str, required=True,
                        help="P3 checkpoint (last.pt / best.pt) from train_paligemma_subgoal.")
    parser.add_argument("--dit_path", type=str, required=True,
                        help="P2 DiT checkpoint; same path used during P3 training.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="PixArt-Σ HF snapshot dir, if the DiT is a PixArtSubgoalDiT.")

    parser.add_argument("--modes", type=str, default="none,oracle,dit",
                        help="Comma-separated subset of {none, oracle, dit}.")
    parser.add_argument("--split", default="seen",
                        help="OpenFly split. 'seen' is standard for action-accuracy eval.")
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Cap loaded episodes (0 = all).")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap unrolled per-step samples (0 = all).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--history_frames", type=int, default=2)

    parser.add_argument("--ddim_steps", type=int, default=4,
                        help="DDIM sampling steps for the 'dit' mode.")
    parser.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument(
        "--out",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project"))
                    / "logs" / "benchmarks"),
        help="Directory to write the JSON result.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)
    device = torch.device(args.device)

    requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in requested:
        if m not in ("none", "oracle", "dit"):
            raise ValueError(f"unknown mode: {m!r}; choices: none|oracle|dit")
    print(f"[eval_p3_subgoals] modes: {requested}")

    # ----- data ---------------------------------------------------------
    ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=True,
        oversample_stop=1.0,   # NO oversampling at eval time
    )
    if len(ds) == 0:
        raise RuntimeError(f"empty dataset for split={args.split!r}")
    print(f"[eval_p3_subgoals] dataset: {len(ds)} samples")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    # ----- models -------------------------------------------------------
    needs_dit = "dit" in requested
    dit = None
    if needs_dit or "oracle" in requested:
        # We need DiT loaded for the 'dit' mode; not strictly required for
        # 'oracle' (we just use PaliGemma), but the trainer's loader prints
        # the val_cos metadata which is useful context, so we always load
        # if any mode might consume PaliGemma's feature extractor.
        if needs_dit:
            print("[eval_p3_subgoals] loading P2 world model …")
            dit = _load_world_model(
                args.dit_path, pretrained_path=args.pretrained_path, device=device,
            )

    print("[eval_p3_subgoals] loading P3 policy …")
    model = PaliGemmaVLNPolicy(
        history_frames=args.history_frames,
        paligemma_model_name=args.paligemma_model,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    ).to(device)
    p3_state = torch.load(args.p3_ckpt, map_location=device, weights_only=False)
    # Same shape-filter trick as the trainer: tolerate older checkpoints.
    own = model.state_dict()
    filtered = {
        k: v for k, v in p3_state["model"].items()
        if k in own and tuple(own[k].shape) == tuple(v.shape)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"[eval_p3_subgoals] P3 init — loaded={len(filtered)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.eval()
    processor = _build_processor(args.paligemma_model)

    # ----- eval loop ----------------------------------------------------
    # Per-mode running stats
    results: dict[str, dict[str, Any]] = {}
    for mode in requested:
        correct = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
        total = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
        n_total, n_correct = 0, 0
        t0 = time.time()
        print(f"\n[eval_p3_subgoals] === mode={mode!r} ===")
        for step, batch in enumerate(loader):
            preds, targets = _eval_batch(
                model, dit, processor, batch, device, mode, args.ddim_steps,
            )
            preds_np = preds.numpy()
            targets_np = targets.numpy()
            n_total += len(targets_np)
            n_correct += int((preds_np == targets_np).sum())
            for c in range(NUM_OPENFLY_ACTIONS):
                mask = targets_np == c
                if mask.any():
                    total[c] += int(mask.sum())
                    correct[c] += int(((preds_np == c) & mask).sum())
            if step % 50 == 0:
                acc_so_far = n_correct / max(n_total, 1)
                print(
                    f"  batch {step:04d} samples={n_total} "
                    f"running_acc={acc_so_far:.4f}",
                    flush=True,
                )

        acc = n_correct / max(n_total, 1)
        per_class = {
            TRAINABLE_ACTION_NAMES.get(c, str(c)): (int(correct[c]), int(total[c]))
            for c in range(NUM_OPENFLY_ACTIONS)
        }
        results[mode] = {
            "overall_acc": acc,
            "n_samples": n_total,
            "n_correct": n_correct,
            "per_class": per_class,
            "time_s": time.time() - t0,
        }
        print(f"  → overall acc {acc:.4f} on {n_total} samples ({time.time() - t0:.1f}s)")
        print(_per_class_table(mode, correct, total))

    # ----- delta + report ----------------------------------------------
    if "none" in results and "dit" in results:
        delta = results["dit"]["overall_acc"] - results["none"]["overall_acc"]
        results["delta_dit_minus_none"] = delta
        print(
            f"\n[eval_p3_subgoals] HEADLINE: dit_acc − none_acc = "
            f"{delta:+.4f} ({results['dit']['overall_acc']:.4f} − "
            f"{results['none']['overall_acc']:.4f})"
        )
    if "oracle" in results and "dit" in results:
        gap = results["oracle"]["overall_acc"] - results["dit"]["overall_acc"]
        results["gap_oracle_minus_dit"] = gap
        print(
            f"[eval_p3_subgoals] gap-to-oracle: oracle_acc − dit_acc = "
            f"{gap:+.4f} (how much better a perfect world model could do)"
        )

    # ----- save ---------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_p3_subgoals_{stamp}.json"
    payload = {
        "p3_ckpt": str(args.p3_ckpt),
        "dit_path": str(args.dit_path),
        "pretrained_path": args.pretrained_path,
        "split": args.split,
        "env_filter": args.env_filter,
        "max_episodes": args.max_episodes,
        "max_samples": args.max_samples,
        "ddim_steps": args.ddim_steps,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[eval_p3_subgoals] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
