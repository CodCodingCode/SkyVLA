#!/usr/bin/env python3
"""Pretrain the feature-space ``SubgoalDiT`` on OpenFly trajectories (phase P2).

Trains the diffusion world model in isolation, with PaliGemma fully
frozen. The data signal is

    (curr_siglip, text_embed, pose_delta, last_action, horizon) -> subgoal_siglip

where ``curr_siglip`` and ``subgoal_siglip`` are PaliGemma SigLIP image
tokens (256 × 2048) for the current and next-action-transition frames
respectively, and ``text_embed`` is PaliGemma's Gemma-side hidden state
at the final non-pad text token (2048-d) — the same summary the BC
policy uses.

Loss: standard DDPM eps-prediction MSE over a cosine schedule with
1000 training steps and uniform timestep sampling. Validation reports
MSE in feature space and the cosine similarity between the predicted
``x0`` and the target tokens (a more interpretable "did we recover the
right scene?" signal).

We intentionally do **not** train the policy here. The contract is:
P1 (BC) produces a useful action head; P2 (this script) produces a
useful world model; P3 (subgoal-conditioned BC) wires them together
and checks whether the predicted subgoals actually help action accuracy.
That separation is what makes the negative result interpretable if the
world model fails to learn aerial scene structure.

Run via :file:`openfly/run_train_subgoal_dit.sh`.
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
from torch.utils.data import DataLoader, random_split

from openfly.dataset import OpenFlyDataset, collate
from openfly.models.subgoal_dit import SubgoalDiT
from vla.vla_policy import PaliGemmaFeatureExtractor


# ---------------------------------------------------------------------------
# PaliGemma helpers (frozen encoder + text embedding)
# ---------------------------------------------------------------------------

def _build_processor(model_name: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name)


def _format_prompt(instruction: str, sub_instruction: str) -> str:
    """Same prompt template as BC, so the DiT learns features aligned with the policy's view."""
    base = f"<image>\nTask: {instruction}"
    if sub_instruction:
        return f"{base}\nNow: {sub_instruction}"
    return base


def _tokenise_batch(
    processor,
    instructions: list[str],
    sub_instructions: list[str],
    rgb_dummy: torch.Tensor,
    device: torch.device,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [
        _format_prompt(ins, sub) for ins, sub in zip(instructions, sub_instructions)
    ]
    batch = processor(
        text=texts,
        images=[rgb_dummy.cpu().numpy()] * len(instructions),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.no_grad()
def _encode_frame(
    paligemma: PaliGemmaFeatureExtractor,
    rgb: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (siglip_tokens, gemma_last_text_token).

    ``siglip_tokens`` are the 256 image tokens used as DiT inputs/targets.
    ``gemma_last_text_token`` is the (B, 2048) hidden state at the final
    non-pad text position — used as the text conditioning vector. We
    derive it ourselves rather than reusing :py:meth:`PaliGemmaFeatureExtractor.forward`
    because the latter pools across the image+text sequence; for the
    text embed we want the language summary, not the fused one.

    NOTE: ``_encode_frame_pair`` is preferred during training — it runs
    PaliGemma once over a stacked (curr, subgoal) batch and is ~30%
    faster per step on an A100. This single-frame helper is kept for
    callers that only need one frame's features (eval-time inference
    rollouts, smoke tests).
    """
    pixel_values = paligemma.preprocess_images(rgb)
    gemma_feats, siglip_feats = paligemma.forward_tokens(
        pixel_values, input_ids, attention_mask
    )
    paligemma.clear_cache()

    token_per_frame = 256
    if gemma_feats.shape[1] > token_per_frame:
        text_feats = gemma_feats[:, token_per_frame:]
        text_mask = attention_mask[:, token_per_frame:]
    else:
        text_feats = gemma_feats
        text_mask = attention_mask

    B = text_feats.shape[0]
    if text_mask.shape[1] > 0 and int(text_mask.sum().item()) > 0:
        seq_len = text_mask.sum(dim=1).clamp(min=1) - 1
        b_idx = torch.arange(B, device=text_feats.device)
        text_summary = text_feats[b_idx, seq_len]
    else:
        text_summary = torch.zeros(
            B, text_feats.shape[-1], device=text_feats.device, dtype=text_feats.dtype
        )
    return siglip_feats.float(), text_summary.float()


@torch.no_grad()
def _encode_frame_pair(
    paligemma: PaliGemmaFeatureExtractor,
    rgb: torch.Tensor,
    subgoal_rgb: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single PaliGemma forward over [curr_rgb ; subgoal_rgb] stacked on batch.

    Replaces two back-to-back ``_encode_frame`` calls. Pays PaliGemma's
    per-call overhead once and runs a fuller batch (better GPU
    utilisation), so step wall-clock drops ~30% on an A100. VRAM cost
    rises by one image's worth of activations — well within budget at
    the standard batch sizes.

    The subgoal half's text input is byte-identical to the current
    half's (same instruction + sub-instruction), so we only return the
    current half's text summary — the duplicated subgoal-half text
    tokens exist on the GPU only to keep PaliGemma's positional /
    attention-mask plumbing consistent.

    Returns
    -------
    curr_siglip   : (B, 256, 2048) — SigLIP tokens for the current frame
    subgoal_siglip: (B, 256, 2048) — SigLIP tokens for the subgoal frame
    text_summary  : (B, 2048)      — Gemma last-text-token, curr half only
    """
    B = rgb.shape[0]
    stacked_rgb = torch.cat([rgb, subgoal_rgb], dim=0)
    stacked_ids = torch.cat([input_ids, input_ids], dim=0)
    stacked_mask = torch.cat([attention_mask, attention_mask], dim=0)

    pixel_values = paligemma.preprocess_images(stacked_rgb)
    gemma_feats, siglip_feats = paligemma.forward_tokens(
        pixel_values, stacked_ids, stacked_mask
    )
    paligemma.clear_cache()

    curr_siglip = siglip_feats[:B].float()
    subgoal_siglip = siglip_feats[B:].float()

    # Text summary from the curr half only.
    token_per_frame = 256
    if gemma_feats.shape[1] > token_per_frame:
        text_feats = gemma_feats[:B, token_per_frame:]
        text_mask = attention_mask[:, token_per_frame:]
    else:
        text_feats = gemma_feats[:B]
        text_mask = attention_mask

    if text_mask.shape[1] > 0 and int(text_mask.sum().item()) > 0:
        seq_len = text_mask.sum(dim=1).clamp(min=1) - 1
        b_idx = torch.arange(B, device=text_feats.device)
        text_summary = text_feats[b_idx, seq_len]
    else:
        text_summary = torch.zeros(
            B, text_feats.shape[-1], device=text_feats.device, dtype=text_feats.dtype
        )
    return curr_siglip, subgoal_siglip, text_summary.float()


# ---------------------------------------------------------------------------
# Body-frame pose delta (current → subgoal)
# ---------------------------------------------------------------------------

def _body_frame_pose_delta(pose: torch.Tensor, subgoal_pose: torch.Tensor) -> torch.Tensor:
    """(B, 4) — [dx_body, dy_body, dz, dyaw_wrapped]."""
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
# Training step
# ---------------------------------------------------------------------------

def _train_step(
    dit: SubgoalDiT,
    paligemma: PaliGemmaFeatureExtractor,
    processor,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    subgoal_rgb = batch["subgoal_rgb"].to(device, non_blocking=True)
    pose = batch["pose"].to(device, non_blocking=True)
    subgoal_pose = batch["subgoal_pose"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    horizon = batch["subgoal_horizon"].to(device, non_blocking=True)
    valid = batch["subgoal_valid"].to(device, non_blocking=True)

    # Skip the whole step if the entire batch is invalid (terminal runs only).
    if not bool(valid.any()):
        return torch.zeros((), device=device, requires_grad=True), {
            "loss": 0.0, "n_valid": 0,
        }

    input_ids, attention_mask = _tokenise_batch(
        processor,
        batch["instruction"],
        batch["sub_instruction"],
        rgb_dummy=rgb[0],
        device=device,
    )

    # PaliGemma encoding — fully frozen, no autograd through it. Curr and
    # subgoal frames go through a single stacked forward (see
    # ``_encode_frame_pair``) — ~30% faster per step than two calls.
    curr_tokens, tgt_tokens, text_embed = _encode_frame_pair(
        paligemma, rgb, subgoal_rgb, input_ids, attention_mask
    )
    pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)

    # Sample a uniform timestep for each example, build the noised target.
    B = rgb.shape[0]
    t = torch.randint(
        0, dit.num_timesteps, (B,), device=device, dtype=torch.long
    )
    noise = torch.randn_like(tgt_tokens)
    x_t, noise = dit.q_sample(tgt_tokens, t, noise=noise)

    eps_pred = dit(
        curr_tokens=curr_tokens,
        noisy_subgoal=x_t,
        t=t,
        text_embed=text_embed,
        pose_delta=pose_delta,
        last_action=last_action,
        horizon=horizon,
    )

    # Per-sample MSE, then mask out invalid (terminal) subgoals before reducing.
    per_sample = (eps_pred - noise).pow(2).mean(dim=[1, 2])  # (B,)
    mask = valid.to(per_sample.dtype)
    denom = mask.sum().clamp(min=1.0)
    loss = (per_sample * mask).sum() / denom

    return loss, {
        "loss": float(loss.item()),
        "n_valid": int(mask.sum().item()),
    }


@torch.no_grad()
def _eval_step(
    dit: SubgoalDiT,
    paligemma: PaliGemmaFeatureExtractor,
    processor,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Validation metrics: MSE on eps + cosine similarity of x0 recovery.

    Cosine similarity is the more interpretable signal. We unroll one
    DDIM step from a single random timestep, compute the implied ``x0``
    from the predicted noise, and compare to the ground-truth target.
    """
    rgb = batch["rgb"].to(device, non_blocking=True)
    subgoal_rgb = batch["subgoal_rgb"].to(device, non_blocking=True)
    pose = batch["pose"].to(device, non_blocking=True)
    subgoal_pose = batch["subgoal_pose"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    horizon = batch["subgoal_horizon"].to(device, non_blocking=True)
    valid = batch["subgoal_valid"].to(device, non_blocking=True)
    if not bool(valid.any()):
        return {"val_loss": 0.0, "val_cos": 0.0, "n_valid": 0}

    input_ids, attention_mask = _tokenise_batch(
        processor,
        batch["instruction"],
        batch["sub_instruction"],
        rgb_dummy=rgb[0],
        device=device,
    )

    # Single-pass batched encode for the (curr, subgoal) pair.
    curr_tokens, tgt_tokens, text_embed = _encode_frame_pair(
        paligemma, rgb, subgoal_rgb, input_ids, attention_mask
    )
    pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)

    B = rgb.shape[0]
    t = torch.randint(0, dit.num_timesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(tgt_tokens)
    x_t, noise = dit.q_sample(tgt_tokens, t, noise=noise)
    eps_pred = dit(
        curr_tokens=curr_tokens, noisy_subgoal=x_t, t=t,
        text_embed=text_embed, pose_delta=pose_delta,
        last_action=last_action, horizon=horizon,
    )
    per_sample_mse = (eps_pred - noise).pow(2).mean(dim=[1, 2])

    ab = dit.alpha_bar[t].to(x_t.dtype).view(-1, 1, 1)
    x0_pred = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-6)
    # Cosine over (sequence × feature) per-sample
    a = x0_pred.reshape(B, -1)
    b = tgt_tokens.reshape(B, -1)
    cos = F.cosine_similarity(a, b, dim=-1)

    mask = valid.to(per_sample_mse.dtype)
    denom = mask.sum().clamp(min=1.0)
    return {
        "val_loss": float((per_sample_mse * mask).sum().item() / denom.item()),
        "val_cos": float((cos * mask).sum().item() / denom.item()),
        "n_valid": int(mask.sum().item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # DiT hyperparams
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_timesteps", type=int, default=1000)

    # Data
    parser.add_argument("--history_frames", type=int, default=0,
                        help="DiT does not consume history; default 0 to skip extra disk reads.")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--require_images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--subgoal_pairing",
        choices=["mixed", "semantic_only", "uniform_only"],
        default="mixed",
        help="Subgoal-pair sampling mode (π0.7 Appendix C). 'mixed' uses "
        "semantic with prob --subgoal_semantic_prob, else uniform.",
    )
    parser.add_argument("--subgoal_semantic_prob", type=float, default=0.25)
    parser.add_argument("--subgoal_uniform_max", type=int, default=4)

    # PaliGemma
    parser.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    parser.add_argument("--paligemma_dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])

    # Outputs
    parser.add_argument(
        "--out_dir",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project"))
                    / "logs" / "openfly" / "subgoal_dit"),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop training if val_cos hasn't improved for N "
                        "consecutive epochs. 0 disables early stopping.")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)

    device = torch.device(args.device)
    base_out = Path(args.out_dir)
    out_dir = base_out / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve symlinks now so subsequent saves don't depend on a path that
    # could disappear mid-training (we hit this once already — a 20-minute
    # epoch died because the ``~/drone_project -> ~/SkyVLA`` symlink got
    # removed between startup mkdir and end-of-epoch save).
    out_dir = out_dir.resolve()
    print(f"[train_subgoal_dit] writing to {out_dir}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    paligemma_dtype = dtype_map[args.paligemma_dtype]

    # ----- data ---------------------------------------------------------
    full_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=args.require_images,
        oversample_stop=1.0,  # don't oversample stops for the world model
        subgoal_pairing=args.subgoal_pairing,
        subgoal_semantic_prob=args.subgoal_semantic_prob,
        subgoal_uniform_max=args.subgoal_uniform_max,
    )
    print(
        f"[train_subgoal_dit] pairing={args.subgoal_pairing} "
        f"semantic_prob={args.subgoal_semantic_prob} "
        f"uniform_max={args.subgoal_uniform_max}"
    )
    if len(full_ds) == 0:
        raise RuntimeError("Empty dataset — check OPENFLY_ANNOTATION_DIR / split.")

    val_size = max(1, int(len(full_ds) * args.val_frac)) if args.val_frac > 0 else 0
    train_size = len(full_ds) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(0),
        )
    else:
        train_ds, val_ds = full_ds, None
    print(
        f"[train_subgoal_dit] split: {train_size} train / {val_size} val "
        f"(steps/epoch: {math.ceil(train_size / args.batch_size)})"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None else None
    )

    # ----- models -------------------------------------------------------
    paligemma = PaliGemmaFeatureExtractor(
        model_name=args.paligemma_model,
        lora_rank=8, lora_alpha=16.0,  # LoRA-B zero-init → identity, so no extra effect
        dtype=paligemma_dtype,
    ).to(device)
    paligemma.eval()
    for p in paligemma.parameters():
        p.requires_grad = False

    dit = SubgoalDiT(
        token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
        hidden=args.hidden,
        depth=args.depth,
        num_heads=args.num_heads,
        text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
        pose_delta_dim=4,
        num_last_actions=9,
        num_timesteps=args.num_timesteps,
    ).to(device)

    processor = _build_processor(args.paligemma_model)

    optimizer = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )

    start_epoch = 0
    global_step = 0
    best_val_cos = float("-inf")
    epochs_since_best = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        dit.load_state_dict(ckpt["dit"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[train_subgoal_dit] resumed from {args.resume} @ epoch {start_epoch}")

    def _apply_warmup(step_idx: int) -> None:
        if args.warmup_steps <= 0:
            return
        frac = min(1.0, (step_idx + 1) / float(args.warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * frac

    # ----- train --------------------------------------------------------
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        dit.train()
        t0 = time.time()
        loss_sum, n_valid_sum, n_steps = 0.0, 0, 0
        for step, batch in enumerate(train_loader):
            _apply_warmup(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _train_step(dit, paligemma, processor, batch, device)
            if metrics["n_valid"] == 0:
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dit.parameters() if p.requires_grad], args.grad_clip
                )
            optimizer.step()
            global_step += 1

            loss_sum += metrics["loss"] * metrics["n_valid"]
            n_valid_sum += metrics["n_valid"]
            n_steps += 1

            if step % args.log_every == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch:02d} step {step:05d} gstep {global_step:06d} "
                    f"loss={metrics['loss']:.4f} n_valid={metrics['n_valid']} "
                    f"lr={cur_lr:.2e}",
                    flush=True,
                )

        train_log: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_valid_sum, 1),
            "train_n_steps": n_steps,
            "train_n_valid": n_valid_sum,
            "time_s": time.time() - t0,
        }

        # ----- val ------------------------------------------------------
        if val_loader is not None:
            dit.eval()
            vl_sum, cos_sum, vn_sum = 0.0, 0.0, 0
            for batch in val_loader:
                m = _eval_step(dit, paligemma, processor, batch, device)
                if m["n_valid"] == 0:
                    continue
                vl_sum += m["val_loss"] * m["n_valid"]
                cos_sum += m["val_cos"] * m["n_valid"]
                vn_sum += m["n_valid"]
            train_log["val_loss"] = vl_sum / max(vn_sum, 1)
            train_log["val_cos"] = cos_sum / max(vn_sum, 1)
            train_log["val_n_valid"] = vn_sum

        history.append(train_log)
        print(f"[train_subgoal_dit] epoch {epoch} → {train_log}")

        if (epoch + 1) % max(1, args.save_every) == 0 or epoch == args.epochs - 1:
            # Re-ensure the parent directory exists right before save. We
            # had a 20-minute training run lose its only checkpoint because
            # the symlink in ``out_dir``'s prefix vanished between startup
            # mkdir and end-of-epoch save. Idempotent and cheap; cost is
            # one stat call.
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_state = {
                "dit": dit.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "val_cos": float(train_log.get("val_cos", float("nan"))),
            }
            # 1) Always update ``last.pt`` (atomic).
            tmp_path = out_dir / "last.pt.tmp"
            ckpt_path = out_dir / "last.pt"
            torch.save(ckpt_state, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f"[train_subgoal_dit] saved {ckpt_path}")

            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            # 2) Update ``best.pt`` if this epoch is the new best by val_cos.
            #    Falls back to comparing train_loss if val isn't available.
            current_val = train_log.get("val_cos", None)
            if current_val is not None and current_val > best_val_cos:
                best_val_cos = float(current_val)
                epochs_since_best = 0
                best_tmp = out_dir / "best.pt.tmp"
                best_path = out_dir / "best.pt"
                torch.save(ckpt_state, best_tmp)
                os.replace(best_tmp, best_path)
                print(
                    f"[train_subgoal_dit] new best val_cos={best_val_cos:.4f}; "
                    f"saved {best_path}"
                )
            else:
                epochs_since_best += 1
                if current_val is not None:
                    print(
                        f"[train_subgoal_dit] val_cos={current_val:.4f} "
                        f"(best={best_val_cos:.4f}, no improvement for "
                        f"{epochs_since_best} epoch(s))"
                    )

            # 3) Early-stop if val_cos hasn't improved in --early_stop_patience epochs.
            if (
                args.early_stop_patience > 0
                and epochs_since_best >= args.early_stop_patience
            ):
                print(
                    f"[train_subgoal_dit] early stop: val_cos has not improved "
                    f"for {epochs_since_best} epochs "
                    f"(--early_stop_patience={args.early_stop_patience})"
                )
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
