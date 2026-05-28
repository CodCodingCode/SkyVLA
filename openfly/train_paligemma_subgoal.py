#!/usr/bin/env python3
"""P3 trainer — BC + subgoal tokens, with a frozen world model.

This is the "sequential" half of the world-model story. The world
model from P2 is loaded and **frozen**; the PaliGemma policy from P1
(BC baseline) is loaded and **finetuned further** under standard
action cross-entropy, with subgoal tokens fed into its cross-attention
input stack.

Per-batch each sample independently flips a coin (``--dit_mix_prob``):

* heads → **oracle subgoal**: encode the dataset's ``subgoal_rgb``
  (the next-keyframe frame chosen by the π0.7-style 25 / 75 pairing)
  through PaliGemma, use those tokens as the subgoal supervision the
  policy attends over.
* tails → **DiT-generated subgoal**: sample from the frozen P2 DiT
  via short DDIM (default 4 steps), use the resulting SigLIP tokens.

Mixing both teaches the policy to use whatever subgoal it gets at
inference time — closing the train/test gap where the policy was
trained on perfect oracle subgoals but sees imperfect DiT samples in
deployment. π0.7 does the same.

The world model can be either:

* The vanilla from-scratch ``SubgoalDiT`` (depth 12, hidden 1024,
  ~150M params).
* The PixArt-Σ-pretrained ``PixArtSubgoalDiT`` (~620M params total).
  Auto-detected from the checkpoint's args / model_type field.

See ``docs/JOINT_TRAINING.md`` for the full design rationale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openfly.dataset import NUM_OPENFLY_ACTIONS, OpenFlyDataset, collate
from openfly.models.paligemma_vln import (
    PaliGemmaVLNPolicy,
    lora_and_head_param_groups,
)
from openfly.models.subgoal_dit import SubgoalDiT
from vla.vla_policy import PaliGemmaFeatureExtractor


# ---------------------------------------------------------------------------
# Processor + tokenisation (shared with train_paligemma.py)
# ---------------------------------------------------------------------------

def _build_processor(model_name: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name)


def _format_prompt(instruction: str, sub_instruction: str | None) -> str:
    base = f"<image>\nTask: {instruction}"
    if sub_instruction:
        return f"{base}\nNow: {sub_instruction}"
    return base


def _tokenise_batch(
    processor,
    instructions: list[str],
    rgb_dummy: torch.Tensor,
    device: torch.device,
    sub_instructions: list[str] | None = None,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sub_instructions is None:
        sub_instructions = [""] * len(instructions)
    texts = [
        _format_prompt(ins, sub) for ins, sub in zip(instructions, sub_instructions)
    ]
    batch = processor(
        text=texts,
        images=[rgb_dummy.cpu().numpy()] * len(instructions),
        return_tensors="pt", padding="longest",
        truncation=True, max_length=max_length,
    )
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
    )


# ---------------------------------------------------------------------------
# Body-frame helpers
# ---------------------------------------------------------------------------

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


def _body_frame_pose_delta(pose: torch.Tensor, subgoal_pose: torch.Tensor) -> torch.Tensor:
    dx = subgoal_pose[:, 0] - pose[:, 0]
    dy = subgoal_pose[:, 1] - pose[:, 1]
    dz = subgoal_pose[:, 2] - pose[:, 2]
    yaw = pose[:, 3]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    body_x = cos_y * dx + sin_y * dy
    body_y = -sin_y * dx + cos_y * dy
    dyaw = (subgoal_pose[:, 3] - yaw + math.pi) % (2 * math.pi) - math.pi
    return torch.stack([body_x, body_y, dz, dyaw], dim=-1).float()


# ---------------------------------------------------------------------------
# World-model load + subgoal generation
# ---------------------------------------------------------------------------

def _load_world_model(
    dit_path: str,
    *,
    pretrained_path: str | None,
    device: torch.device,
):
    """Load either a vanilla SubgoalDiT or a PixArtSubgoalDiT checkpoint.

    If ``pretrained_path`` is given, build a ``PixArtSubgoalDiT`` (since
    the DiT checkpoint's state_dict shape would only match the wrapped
    PixArt model). Otherwise build a vanilla ``SubgoalDiT`` with
    hyperparams recovered from the checkpoint's ``args`` blob.
    """
    ckpt = torch.load(dit_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})

    if pretrained_path is not None:
        from openfly.models.subgoal_dit_pixart import PixArtSubgoalDiT
        dit = PixArtSubgoalDiT(
            pretrained_path=pretrained_path,
            token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            num_timesteps=int(ckpt_args.get("num_timesteps", 1000)),
            freeze_backbone=False,  # checkpoint dictates which params exist
        )
    else:
        dit = SubgoalDiT(
            token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            hidden=int(ckpt_args.get("hidden", 1024)),
            depth=int(ckpt_args.get("depth", 12)),
            num_heads=int(ckpt_args.get("num_heads", 16)),
            text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            pose_delta_dim=4,
            num_last_actions=9,
            num_timesteps=int(ckpt_args.get("num_timesteps", 1000)),
        )

    missing, unexpected = dit.load_state_dict(ckpt["dit"], strict=False)
    if missing:
        print(f"[train_paligemma_subgoal] DiT load — missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[train_paligemma_subgoal] DiT load — unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    dit = dit.to(device).eval()
    for p in dit.parameters():
        p.requires_grad = False
    print(
        f"[train_paligemma_subgoal] DiT loaded ({type(dit).__name__}, val_cos@save="
        f"{ckpt.get('val_cos', float('nan')):.4f})"
    )
    return dit


@torch.no_grad()
def _oracle_subgoal_tokens(
    paligemma: PaliGemmaFeatureExtractor,
    subgoal_rgb: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Encode the real next-keyframe RGB through PaliGemma → (B, 256, 2048)."""
    pixel_values = paligemma.preprocess_images(subgoal_rgb)
    _, siglip = paligemma.forward_tokens(pixel_values, input_ids, attention_mask)
    paligemma.clear_cache()
    return siglip.float()


@torch.no_grad()
def _dit_subgoal_tokens(
    dit,
    curr_tokens: torch.Tensor,
    text_embed: torch.Tensor,
    pose_delta: torch.Tensor,
    last_action: torch.Tensor,
    horizon: torch.Tensor,
    num_steps: int = 4,
) -> torch.Tensor:
    """Sample subgoal SigLIP tokens from a frozen DiT via short DDIM."""
    return dit.ddim_sample(
        curr_tokens=curr_tokens,
        text_embed=text_embed,
        pose_delta=pose_delta,
        last_action=last_action.long(),
        horizon=horizon,
        num_steps=num_steps,
    )


# ---------------------------------------------------------------------------
# Per-step BC + subgoals
# ---------------------------------------------------------------------------

def _train_step(
    model: PaliGemmaVLNPolicy,
    dit,
    processor,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
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

    # Build subgoal tokens — per-sample mix of oracle vs DiT.
    B = rgb.shape[0]
    use_dit_mask = (torch.rand(B, device=device) < args.dit_mix_prob)

    # Oracle path: encode subgoal_rgb through PaliGemma (always do this;
    # cheap because PaliGemma is already loaded and forward is no_grad).
    oracle_tokens = _oracle_subgoal_tokens(
        model.paligemma, subgoal_rgb, input_ids, attention_mask
    )

    if use_dit_mask.any():
        # DiT path: need curr_siglip + text_embed for conditioning. Re-encode
        # current frame to get the raw SigLIP features. (The model's own
        # forward will encode it again, but with LoRA-gradient context;
        # for sampling we just need no-grad raw features.)
        with torch.no_grad():
            pv = model.paligemma.preprocess_images(rgb)
            gemma_feats, curr_siglip = model.paligemma.forward_tokens(pv, input_ids, attention_mask)
            model.paligemma.clear_cache()
            # Text summary from last non-pad text position (same as in P2 trainer).
            token_per_frame = 256
            text_feats = gemma_feats[:, token_per_frame:] if gemma_feats.shape[1] > token_per_frame else gemma_feats
            text_mask_part = attention_mask[:, token_per_frame:] if attention_mask.shape[1] > token_per_frame else attention_mask
            seq_lengths = text_mask_part.sum(dim=1).clamp(min=1) - 1
            b_idx = torch.arange(B, device=device)
            text_embed = text_feats[b_idx, seq_lengths].float()

            pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)
            dit_tokens = _dit_subgoal_tokens(
                dit,
                curr_tokens=curr_siglip.float(),
                text_embed=text_embed,
                pose_delta=pose_delta,
                last_action=last_action,
                horizon=horizon,
                num_steps=args.ddim_steps,
            )
    else:
        dit_tokens = oracle_tokens  # unused but satisfy shape

    # Combine: where use_dit_mask is True, use DiT tokens; else oracle.
    use_dit_mask_b = use_dit_mask.view(B, 1, 1).to(oracle_tokens.dtype)
    subgoal_tokens = use_dit_mask_b * dit_tokens + (1 - use_dit_mask_b) * oracle_tokens

    # Policy forward with the subgoal tokens injected.
    out = model(
        instruction_input_ids=input_ids,
        instruction_attention_mask=attention_mask,
        rgb_current=rgb,
        rgb_history=history,
        pose=pose,
        last_action=last_action,
        next_pose=next_pose,
        progress=progress,
        with_grad=True,
        subgoal_tokens=subgoal_tokens,
    )
    logits = out["action_logits"]
    ce = F.cross_entropy(logits, actions)

    loss = ce
    metrics: dict[str, Any] = {
        "ce": ce.item(),
        "acc": (logits.argmax(dim=-1) == actions).float().mean().item(),
        "n_dit": int(use_dit_mask.sum().item()),
        "n_oracle": int((~use_dit_mask).sum().item()),
    }

    if args.aux_goal_weight > 0 and "goal_pred" in out:
        target = _body_frame_delta(pose, next_pose[:, :3]).clamp(min=-20.0, max=20.0)
        l1 = F.smooth_l1_loss(out["goal_pred"], target)
        loss = loss + args.aux_goal_weight * l1
        metrics["goal_l1"] = l1.item()

    return loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data / training
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--aux_goal_weight", type=float, default=0.1)
    parser.add_argument("--history_frames", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument(
        "--val_split",
        type=str,
        default="seen",
        help="Held-out split for validation. 'seen' uses seen.json "
        "(trajectory holdout in training scenes); 'unseen' uses unseen.json "
        "(scene holdout). Replaces the previous sample-level random_split, "
        "which was leaking adjacent frames from the same trajectory into val.",
    )
    parser.add_argument(
        "--val_max_episodes",
        type=int,
        default=200,
        help="Cap on val episodes. 0 = use the whole held-out split.",
    )
    parser.add_argument("--num_workers", type=int, default=2)

    # Checkpoints
    parser.add_argument(
        "--bc_init_ckpt",
        type=str,
        default="",
        help="Optional path to a P1 BC checkpoint to initialize the policy. "
        "Empty (default) = train from scratch (recommended after the "
        "architecture change that removed last_action_embed + progress_proj; "
        "older checkpoints' action_head weights are shape-incompatible).",
    )
    parser.add_argument("--dit_path", type=str, required=True,
                        help="Path to P2 SubgoalDiT checkpoint (best.pt). Frozen.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="If the DiT checkpoint is a PixArtSubgoalDiT, "
                        "pass the original HF snapshot dir here so the "
                        "wrapper can reconstruct the backbone before loading.")

    # Subgoal mixing
    parser.add_argument("--dit_mix_prob", type=float, default=0.5,
                        help="Probability per-sample of using DiT-generated "
                        "subgoals (vs oracle-encoded real next-keyframe "
                        "RGB). 0.5 mirrors π0.7's mix.")
    parser.add_argument("--ddim_steps", type=int, default=4,
                        help="DDIM sampling steps when generating DiT "
                        "subgoals. 4 is the consistency-model target.")

    # PaliGemma
    parser.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")

    # Outputs
    parser.add_argument(
        "--out_dir",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project"))
                    / "logs" / "openfly" / "paligemma_subgoal"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args(argv)
    device = torch.device(args.device)
    base_out = Path(args.out_dir)
    out_dir = (base_out / time.strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_paligemma_subgoal] writing to {out_dir}")

    # ----- data ---------------------------------------------------------
    # Training data comes from --split (e.g. 'train'). Validation comes from
    # a SEPARATE held-out split (--val_split, default 'seen') so adjacent
    # frames of the same trajectory can never appear on both sides. The
    # previous random_split over a single dataset was producing val_acc=1.0
    # in epoch 1 because frame k landed in train and frame k+1 in val.
    train_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=True,
        oversample_stop=2.0,
    )
    if args.val_split and args.val_split.lower() != "none":
        val_ds = OpenFlyDataset(
            split=args.val_split,
            history_frames=args.history_frames,
            env_filter=args.env_filter,
            max_episodes=args.val_max_episodes,
            require_images=True,
            oversample_stop=1.0,  # honest distribution at eval time
        )
    else:
        val_ds = None
    print(
        f"[train_paligemma_subgoal] data: train_split={args.split} "
        f"n={len(train_ds)}  /  val_split={args.val_split} "
        f"n={len(val_ds) if val_ds is not None else 0}"
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, collate_fn=collate,
                   pin_memory=device.type == "cuda")
        if val_ds is not None else None
    )

    # ----- models -------------------------------------------------------
    print("[train_paligemma_subgoal] loading world model …")
    dit = _load_world_model(args.dit_path, pretrained_path=args.pretrained_path, device=device)

    print("[train_paligemma_subgoal] loading policy …")
    model = PaliGemmaVLNPolicy(
        history_frames=args.history_frames,
        paligemma_model_name=args.paligemma_model,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    ).to(device)
    if args.bc_init_ckpt:
        bc_state = torch.load(args.bc_init_ckpt, map_location=device, weights_only=False)
        # ``strict=False`` ignores missing/unexpected keys but still raises on
        # SHAPE mismatches. Older BC checkpoints predate the subgoal slot in
        # ``frame_embed`` and used a wider action head (336d → 288d after the
        # last_action_embed + progress_proj removal), so we filter
        # shape-mismatched keys explicitly and let them stay at fresh init.
        own = model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        shape_mismatches: list[str] = []
        for k, v in bc_state["model"].items():
            if k not in own:
                continue
            if tuple(own[k].shape) != tuple(v.shape):
                shape_mismatches.append(
                    f"{k} ({tuple(v.shape)} → {tuple(own[k].shape)})"
                )
                continue
            filtered[k] = v
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"[train_paligemma_subgoal] BC init from {args.bc_init_ckpt} — "
            f"loaded={len(filtered)} missing={len(missing)} "
            f"unexpected={len(unexpected)} shape_mismatch={len(shape_mismatches)}"
        )
        if shape_mismatches:
            print(
                "[train_paligemma_subgoal]   shape mismatches (using fresh init):"
            )
            for m in shape_mismatches:
                print(f"     {m}")
    else:
        print(
            "[train_paligemma_subgoal] no --bc_init_ckpt provided — training "
            "policy from scratch (PaliGemma backbone weights are frozen and "
            "loaded from HF regardless)."
        )
    processor = _build_processor(args.paligemma_model)

    optimizer = torch.optim.AdamW(
        lora_and_head_param_groups(model, lora_lr=args.lora_lr, head_lr=args.head_lr)
    )
    base_lrs = {pg.get("name", str(i)): float(pg["lr"]) for i, pg in enumerate(optimizer.param_groups)}

    def _apply_warmup(step_idx: int) -> None:
        if args.warmup_steps <= 0:
            return
        frac = min(1.0, (step_idx + 1) / float(args.warmup_steps))
        for pg in optimizer.param_groups:
            if pg.get("name") == "lora":
                pg["lr"] = base_lrs["lora"] * frac

    # ----- train --------------------------------------------------------
    history: list[dict[str, Any]] = []
    global_step = 0
    best_val_acc = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        n, ce_sum, acc_sum, dit_count, oracle_count = 0, 0.0, 0.0, 0, 0
        for step, batch in enumerate(train_loader):
            _apply_warmup(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _train_step(model, dit, processor, batch, device, args)
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
            dit_count += metrics["n_dit"]
            oracle_count += metrics["n_oracle"]
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch:02d} step {step:04d} gstep {global_step:06d} "
                    f"ce={metrics['ce']:.3f} acc={metrics['acc']:.3f} "
                    f"dit/oracle={metrics['n_dit']}/{metrics['n_oracle']}",
                    flush=True,
                )

        train_log: dict[str, Any] = {
            "epoch": epoch,
            "train_ce": ce_sum / max(n, 1),
            "train_acc": acc_sum / max(n, 1),
            "n_dit": dit_count, "n_oracle": oracle_count,
            "time_s": time.time() - t0,
        }
        # ----- val ------------------------------------------------------
        if val_loader is not None:
            model.eval()
            vn, vce, vacc = 0, 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    # Use ORACLE subgoals for val so the metric is clean
                    # (no DiT sampling noise polluting the comparison).
                    saved_prob = args.dit_mix_prob
                    args.dit_mix_prob = 0.0
                    _, m = _train_step(model, dit, processor, batch, device, args)
                    args.dit_mix_prob = saved_prob
                    bs = batch["action_id"].shape[0]
                    vn += bs
                    vce += m["ce"] * bs
                    vacc += m["acc"] * bs
            train_log["val_ce"] = vce / max(vn, 1)
            train_log["val_acc"] = vacc / max(vn, 1)
        print(f"[train_paligemma_subgoal] epoch {epoch} → {train_log}")
        history.append(train_log)

        # Save last + best
        ckpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "val_acc": train_log.get("val_acc"),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "last.pt.tmp"
        torch.save(ckpt_state, tmp)
        os.replace(tmp, out_dir / "last.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        val_acc = train_log.get("val_acc")
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            tmp = out_dir / "best.pt.tmp"
            torch.save(ckpt_state, tmp)
            os.replace(tmp, out_dir / "best.pt")
            print(f"[train_paligemma_subgoal] new best val_acc={best_val_acc:.4f} → saved best.pt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
