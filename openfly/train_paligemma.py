#!/usr/bin/env python3
"""Offline behaviour cloning of `PaliGemmaVLNPolicy` on OpenFly trajectories.

Replaces the deleted Isaac PPO loop. The policy is supervised on the
discrete expert action labels in ``train.json`` (cross-entropy). When
``--aux_goal_weight`` is non-zero an auxiliary L1 loss against the
**next** trajectory pose pulls the action-head trunk toward a body-frame
next-step delta representation (cheap supervision for the short-horizon
planner — see Priority 7 in the accuracy-fix plan).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from openfly.actions import ACTION_NAMES
from openfly.dataset import NUM_OPENFLY_ACTIONS, OpenFlyDataset, collate
from openfly.models.paligemma_vln import (
    PaliGemmaVLNPolicy,
    lora_and_head_param_groups,
)


def _build_processor(model_name: str):
    """Return the PaliGemma processor for instruction tokenisation."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name)


def _tokenise_batch(
    processor,
    instructions: list[str],
    rgb_dummy: torch.Tensor,
    device: torch.device,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the PaliGemma processor on a batch of instructions.

    A dummy image is required because the processor expects vision input;
    the resulting image tokens are stripped downstream by
    ``PaliGemmaVLNPolicy._tokenize``.
    """
    batch = processor(
        text=[f"<image>\n{ins}" for ins in instructions],
        images=[rgb_dummy.cpu().numpy()] * len(instructions),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
    )


def _body_frame_delta(pose: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Rotate (goal - pose_xyz) into the drone's yaw-aligned body frame."""
    dx = goal[:, 0] - pose[:, 0]
    dy = goal[:, 1] - pose[:, 1]
    dz = goal[:, 2] - pose[:, 2]
    yaw = pose[:, 3]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    body_x = cos_y * dx + sin_y * dy
    body_y = -sin_y * dx + cos_y * dy
    return torch.stack([body_x, body_y, dz], dim=-1)


def _cache_key(split: str, max_episodes: int, env_filter: str | None) -> str:
    """Stable hash for the per-dataset action-count cache."""
    payload = json.dumps(
        {"split": split, "max_episodes": int(max_episodes), "env_filter": env_filter or ""},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _compute_action_counts(dataset: OpenFlyDataset) -> np.ndarray:
    """Count action ids over ``dataset._index`` (no image loads)."""
    counts = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
    episodes = dataset._episodes  # noqa: SLF001 — intentional read
    for ep_i, step in dataset._index:  # noqa: SLF001
        a = int(episodes[ep_i]["action"][step])
        if 0 <= a < NUM_OPENFLY_ACTIONS:
            counts[a] += 1
    return counts


def _class_weights_from_counts(
    counts: np.ndarray,
    *,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Inverse-frequency weights; clamped at ``max_weight`` for stability."""
    total = float(counts.sum())
    num_classes = counts.shape[0]
    safe = np.maximum(counts.astype(np.float64), 1.0)
    weights = total / (num_classes * safe)
    weights = np.minimum(weights, float(max_weight))
    return weights.astype(np.float32)


def _load_or_build_class_weights(
    dataset: OpenFlyDataset,
    *,
    cache_dir: Path,
    split: str,
    max_episodes: int,
    env_filter: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, weights), caching counts to JSON for reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(split, max_episodes, env_filter)
    cache_path = cache_dir / f"action_counts_{split}_{key}.json"
    counts: np.ndarray | None = None
    if cache_path.is_file():
        try:
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            cached = payload.get("counts")
            if (
                isinstance(cached, list)
                and len(cached) == NUM_OPENFLY_ACTIONS
                and int(payload.get("num_samples", -1)) == len(dataset)
            ):
                counts = np.asarray(cached, dtype=np.int64)
                print(
                    f"[train_paligemma] action counts loaded from {cache_path}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[train_paligemma] WARN failed to read {cache_path}: {exc}")
    if counts is None:
        counts = _compute_action_counts(dataset)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "split": split,
                        "max_episodes": int(max_episodes),
                        "env_filter": env_filter or "",
                        "num_samples": len(dataset),
                        "counts": counts.tolist(),
                    },
                    f,
                    indent=2,
                )
            print(f"[train_paligemma] action counts cached to {cache_path}")
        except OSError as exc:
            print(f"[train_paligemma] WARN failed to write {cache_path}: {exc}")
    weights = _class_weights_from_counts(counts)
    return counts, weights


def _format_per_class(
    name: str,
    correct: np.ndarray,
    total: np.ndarray,
) -> str:
    """One-line per-class accuracy table."""
    parts = []
    for c in range(NUM_OPENFLY_ACTIONS):
        n = int(total[c])
        if n > 0:
            acc = float(correct[c]) / n
            parts.append(f"{c}={acc:.2f}({n})")
        else:
            parts.append(f"{c}=NA")
    return f"[per-class {name}] " + " ".join(parts)


def _step(
    model: PaliGemmaVLNPolicy,
    batch: dict[str, Any],
    processor,
    device: torch.device,
    aux_goal_weight: float,
    *,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    history = (
        batch["history"].to(device, non_blocking=True)
        if batch["history"].numel() > 0
        else batch["history"].to(device)
    )
    pose = batch["pose"].to(device, non_blocking=True)
    actions = batch["action_id"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    next_pose = batch["next_pose"].to(device, non_blocking=True)

    input_ids, attention_mask = _tokenise_batch(
        processor, batch["instruction"], rgb_dummy=rgb[0], device=device
    )

    out = model(
        instruction_input_ids=input_ids,
        instruction_attention_mask=attention_mask,
        rgb_current=rgb,
        rgb_history=history,
        pose=pose,
        last_action=last_action,
        next_pose=next_pose,
        with_grad=True,
    )
    logits = out["action_logits"]
    ce = F.cross_entropy(logits, actions, weight=class_weights)

    preds = logits.argmax(dim=-1)
    metrics: dict[str, Any] = {
        "ce": ce.item(),
        "acc": (preds == actions).float().mean().item(),
        "preds": preds.detach(),
        "actions": actions.detach(),
    }
    loss = ce

    if aux_goal_weight > 0 and "goal_pred" in out:
        # Body-frame delta to the NEXT pose (short-horizon, ~one step ahead).
        target = _body_frame_delta(pose, next_pose[:, :3])
        target = target.clamp(min=-20.0, max=20.0)
        l1 = F.smooth_l1_loss(out["goal_pred"], target)
        loss = loss + aux_goal_weight * l1
        metrics["goal_l1"] = l1.item()

    return loss, metrics


def _accumulate_per_class(
    correct: np.ndarray,
    total: np.ndarray,
    preds: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    preds_np = preds.cpu().numpy()
    actions_np = actions.cpu().numpy()
    for c in range(NUM_OPENFLY_ACTIONS):
        mask = actions_np == c
        if mask.any():
            total[c] += int(mask.sum())
            correct[c] += int(((preds_np == c) & mask).sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--aux_goal_weight", type=float, default=0.1)
    parser.add_argument("--history_frames", type=int, default=2)
    parser.add_argument(
        "--oversample_stop",
        type=float,
        default=2.0,
        help="Per-step duplication factor for stop (action=0) samples in the "
        "dataset index. Set to 1.0 to disable.",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--paligemma_model",
        default="google/paligemma-3b-pt-224",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project")) / "logs" / "openfly" / "paligemma"),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--require_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require image frames on disk (default ON; pass --no-require-images for legacy behavior).",
    )
    parser.add_argument(
        "--compute_class_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute inverse-frequency class weights from the train split "
        "and pass them into cross_entropy (default ON).",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    base_out = Path(args.out_dir)
    out_dir = base_out / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_paligemma] writing to {out_dir}")

    full_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=args.require_images,
        oversample_stop=args.oversample_stop,
    )
    if len(full_ds) == 0:
        raise RuntimeError("Empty dataset — check OPENFLY_ANNOTATION_DIR / split.")

    # ----- class weights -------------------------------------------------
    class_weights_t: torch.Tensor | None = None
    counts = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
    if args.compute_class_weights:
        counts, weights = _load_or_build_class_weights(
            full_ds,
            cache_dir=base_out,
            split=args.split,
            max_episodes=args.max_episodes,
            env_filter=args.env_filter,
        )
        weight_str = ", ".join(
            f"{c}({ACTION_NAMES.get(c, str(c))})={weights[c]:.3f}"
            for c in range(NUM_OPENFLY_ACTIONS)
        )
        count_str = ", ".join(
            f"{c}={int(counts[c])}" for c in range(NUM_OPENFLY_ACTIONS)
        )
        print(f"[train_paligemma] action counts: {count_str}")
        print(f"[train_paligemma] class weights: {weight_str}")
        class_weights_t = torch.from_numpy(weights).to(device)
    else:
        print("[train_paligemma] class weights disabled (--no-compute-class-weights)")

    val_size = max(1, int(len(full_ds) * args.val_frac)) if args.val_frac > 0 else 0
    train_size = len(full_ds) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(0),
        )
    else:
        train_ds, val_ds = full_ds, None
    print(
        f"[train_paligemma] split: {train_size} train / {val_size} val "
        f"(samples per epoch: {math.ceil(train_size / args.batch_size)})"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    model = PaliGemmaVLNPolicy(
        history_frames=args.history_frames,
        paligemma_model_name=args.paligemma_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)
    processor = _build_processor(args.paligemma_model)

    optimizer = torch.optim.AdamW(
        lora_and_head_param_groups(model, lora_lr=args.lora_lr, head_lr=args.head_lr)
    )
    # Cache the target LRs so warmup can scale only the LoRA group.
    base_lrs = {
        pg.get("name", str(i)): float(pg["lr"])
        for i, pg in enumerate(optimizer.param_groups)
    }
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(
            f"[train_paligemma] resumed from {args.resume} @ epoch {start_epoch} "
            f"(global_step={global_step})"
        )

    def _apply_warmup(step_idx: int) -> None:
        if args.warmup_steps <= 0:
            return
        frac = min(1.0, (step_idx + 1) / float(args.warmup_steps))
        for pg in optimizer.param_groups:
            if pg.get("name") == "lora":
                pg["lr"] = base_lrs["lora"] * frac

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        n, ce_sum, acc_sum, goal_sum = 0, 0.0, 0.0, 0.0
        train_correct = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
        train_total = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
        for step, batch in enumerate(train_loader):
            _apply_warmup(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _step(
                model,
                batch,
                processor,
                device,
                args.aux_goal_weight,
                class_weights=class_weights_t,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            global_step += 1

            bs = batch["action_id"].shape[0]
            n += bs
            ce_sum += metrics["ce"] * bs
            acc_sum += metrics["acc"] * bs
            if "goal_l1" in metrics:
                goal_sum += metrics["goal_l1"] * bs
            _accumulate_per_class(
                train_correct, train_total, metrics["preds"], metrics["actions"]
            )

            if step % 20 == 0:
                cur_lora_lr = next(
                    (pg["lr"] for pg in optimizer.param_groups if pg.get("name") == "lora"),
                    float("nan"),
                )
                msg = (
                    f"epoch {epoch:02d} step {step:04d} gstep {global_step:06d} "
                    f"ce={metrics['ce']:.3f} acc={metrics['acc']:.3f} "
                    f"lora_lr={cur_lora_lr:.2e}"
                )
                if "goal_l1" in metrics:
                    msg += f" goal={metrics['goal_l1']:.3f}"
                print(msg, flush=True)

        train_log: dict[str, Any] = {
            "epoch": epoch,
            "train_ce": ce_sum / max(n, 1),
            "train_acc": acc_sum / max(n, 1),
            "train_goal_l1": goal_sum / max(n, 1),
            "time_s": time.time() - t0,
            "train_per_class_correct": train_correct.tolist(),
            "train_per_class_total": train_total.tolist(),
        }
        print(_format_per_class("train", train_correct, train_total))
        # Surface a few critical classes (stop + turns) explicitly.
        for c in (0, 2, 3):
            tot = int(train_total[c])
            cor = int(train_correct[c])
            acc = cor / tot if tot > 0 else float("nan")
            print(
                f"[train_paligemma] epoch {epoch} train class {c} "
                f"({ACTION_NAMES.get(c, str(c))}): {cor}/{tot} = {acc:.3f}"
            )

        if val_loader is not None:
            model.eval()
            val_correct = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
            val_total = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
            with torch.no_grad():
                vn, vce, vacc = 0, 0.0, 0.0
                for batch in val_loader:
                    _, m = _step(
                        model,
                        batch,
                        processor,
                        device,
                        0.0,
                        class_weights=None,
                    )
                    bs = batch["action_id"].shape[0]
                    vn += bs
                    vce += m["ce"] * bs
                    vacc += m["acc"] * bs
                    _accumulate_per_class(
                        val_correct, val_total, m["preds"], m["actions"]
                    )
            train_log["val_ce"] = vce / max(vn, 1)
            train_log["val_acc"] = vacc / max(vn, 1)
            train_log["val_per_class_correct"] = val_correct.tolist()
            train_log["val_per_class_total"] = val_total.tolist()
            print(_format_per_class("val", val_correct, val_total))
            for c in (0, 2, 3):
                tot = int(val_total[c])
                cor = int(val_correct[c])
                acc = cor / tot if tot > 0 else float("nan")
                print(
                    f"[train_paligemma] epoch {epoch} val class {c} "
                    f"({ACTION_NAMES.get(c, str(c))}): {cor}/{tot} = {acc:.3f}"
                )

        history.append(train_log)
        print(f"[train_paligemma] epoch {epoch} → {train_log}")

        ckpt_path = out_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "class_weights": (
                    class_weights_t.detach().cpu().tolist()
                    if class_weights_t is not None
                    else None
                ),
                "action_counts": counts.tolist(),
            },
            ckpt_path,
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"[train_paligemma] saved {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
