#!/usr/bin/env python3
"""Offline behaviour cloning of `PaliGemmaVLNPolicy` on OpenFly trajectories.

Replaces the deleted Isaac PPO loop. The policy is supervised on the
discrete expert action labels in ``train.json`` (cross-entropy). When
``--aux_goal_weight`` is non-zero an auxiliary L1 loss against the next
trajectory pose pulls the LSTM hidden state toward a body-frame goal
representation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from openfly.dataset import OpenFlyDataset, collate
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
    max_length: int = 256,
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


def _step(
    model: PaliGemmaVLNPolicy,
    batch: dict[str, Any],
    processor,
    device: torch.device,
    aux_goal_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    history = batch["history"].to(device, non_blocking=True) if batch["history"].numel() > 0 else batch["history"].to(device)
    pose = batch["pose"].to(device, non_blocking=True)
    goal = batch["goal"].to(device, non_blocking=True)
    actions = batch["action_id"].to(device, non_blocking=True)

    input_ids, attention_mask = _tokenise_batch(
        processor, batch["instruction"], rgb_dummy=rgb[0], device=device
    )

    out = model(
        instruction_input_ids=input_ids,
        instruction_attention_mask=attention_mask,
        rgb_current=rgb,
        rgb_history=history,
        pose=pose,
        with_grad=True,
    )
    logits = out["action_logits"]
    ce = F.cross_entropy(logits, actions)

    metrics: dict[str, float] = {
        "ce": ce.item(),
        "acc": (logits.argmax(dim=-1) == actions).float().mean().item(),
    }
    loss = ce

    if aux_goal_weight > 0 and "goal_pred" in out:
        target = _body_frame_delta(pose, goal)
        # Clip to avoid blowing up on huge inter-pose distances.
        target = target.clamp(min=-50.0, max=50.0)
        l1 = F.smooth_l1_loss(out["goal_pred"], target)
        loss = loss + aux_goal_weight * l1
        metrics["goal_l1"] = l1.item()

    return loss, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_lr", type=float, default=1e-6)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--aux_goal_weight", type=float, default=0.05)
    parser.add_argument("--history_frames", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
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
    parser.add_argument("--require_images", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_paligemma] writing to {out_dir}")

    full_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=args.require_images,
    )
    if len(full_ds) == 0:
        raise RuntimeError("Empty dataset — check OPENFLY_ANNOTATION_DIR / split.")

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
    ).to(device)
    processor = _build_processor(args.paligemma_model)

    optimizer = torch.optim.AdamW(
        lora_and_head_param_groups(model, lora_lr=args.lora_lr, head_lr=args.head_lr)
    )
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[train_paligemma] resumed from {args.resume} @ epoch {start_epoch}")

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        n, ce_sum, acc_sum, goal_sum = 0, 0.0, 0.0, 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _step(
                model, batch, processor, device, args.aux_goal_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            bs = batch["action_id"].shape[0]
            n += bs
            ce_sum += metrics["ce"] * bs
            acc_sum += metrics["acc"] * bs
            if "goal_l1" in metrics:
                goal_sum += metrics["goal_l1"] * bs

            if step % 20 == 0:
                msg = (
                    f"epoch {epoch:02d} step {step:04d} "
                    f"ce={metrics['ce']:.3f} acc={metrics['acc']:.3f}"
                )
                if "goal_l1" in metrics:
                    msg += f" goal={metrics['goal_l1']:.3f}"
                print(msg, flush=True)

        train_log = {
            "epoch": epoch,
            "train_ce": ce_sum / max(n, 1),
            "train_acc": acc_sum / max(n, 1),
            "train_goal_l1": goal_sum / max(n, 1),
            "time_s": time.time() - t0,
        }

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vn, vce, vacc = 0, 0.0, 0.0
                for batch in val_loader:
                    _, m = _step(model, batch, processor, device, 0.0)
                    bs = batch["action_id"].shape[0]
                    vn += bs
                    vce += m["ce"] * bs
                    vacc += m["acc"] * bs
            train_log["val_ce"] = vce / max(vn, 1)
            train_log["val_acc"] = vacc / max(vn, 1)

        history.append(train_log)
        print(f"[train_paligemma] epoch {epoch} → {train_log}")

        ckpt_path = out_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            },
            ckpt_path,
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"[train_paligemma] saved {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
