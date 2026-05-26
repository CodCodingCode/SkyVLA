#!/usr/bin/env python3
"""P3.5 trainer — joint refine of (world model + policy).

After P3 has produced a policy that *can* use subgoals, this phase
**unfreezes the SubgoalDiT** for a short joint-training run. Both
the DiT and the policy are updated under a weighted sum:

    loss = λ_mse · diffusion_eps_mse(dit_predictions)
         + λ_ce  · action_cross_entropy(policy_predictions_given_dit_subgoals)

The key gradient flow this adds (vs the sequential P3) is

    action_CE → policy → cross_attn → subgoal_tokens
                        → DiT.token_out → DiT blocks → DiT.token_in / adapters

so the DiT now gets a signal that says "produce subgoals that help
the policy succeed," not just "produce subgoals that match real
frames." Visual fidelity and action utility are correlated but not
identical; this phase shrinks the gap.

PaliGemma's base weights are **still frozen** — only its LoRA + heads
are touched. The DiT runs in *training* mode and the inference DDIM
sampler is dropped to 4 steps so the backward pass through diffusion
is tractable on one A100.

Defaults are conservative — 1 epoch, lambda_mse=1.0, lambda_ce=0.3,
small LRs. The goal is alignment, not retraining. See
``docs/JOINT_TRAINING.md`` for the full design rationale.
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
from openfly.models.subgoal_dit import SubgoalDiT
from openfly.train_paligemma_subgoal import (
    _build_processor,
    _tokenise_batch,
    _body_frame_delta,
    _body_frame_pose_delta,
    _load_world_model,
    _oracle_subgoal_tokens,
)
from vla.vla_policy import PaliGemmaFeatureExtractor


# ---------------------------------------------------------------------------
# Per-step joint loss
# ---------------------------------------------------------------------------

def _joint_train_step(
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
    valid = batch["subgoal_valid"].to(device, non_blocking=True)
    progress = batch.get("progress")
    if progress is not None:
        progress = progress.to(device, non_blocking=True)

    input_ids, attention_mask = _tokenise_batch(
        processor, batch["instruction"], rgb_dummy=rgb[0], device=device,
        sub_instructions=batch.get("sub_instruction"),
    )

    # Encode current + subgoal frames through PaliGemma (frozen).
    # We need:
    #   - curr_siglip: input to DiT (both legs)
    #   - text_embed:  text conditioning for DiT
    #   - tgt_siglip:  target for the diffusion MSE leg
    with torch.no_grad():
        pv_curr = model.paligemma.preprocess_images(rgb)
        gemma_feats, curr_siglip = model.paligemma.forward_tokens(
            pv_curr, input_ids, attention_mask
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
        seq_lengths = text_mask_part.sum(dim=1).clamp(min=1) - 1
        B = rgb.shape[0]
        b_idx = torch.arange(B, device=device)
        text_embed = text_feats[b_idx, seq_lengths].float()

        tgt_tokens = _oracle_subgoal_tokens(
            model.paligemma, subgoal_rgb, input_ids, attention_mask
        )
        pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)

    # ---- DIFFUSION LEG: standard ε-prediction MSE on the unfrozen DiT ----
    t = torch.randint(0, dit.num_timesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(tgt_tokens)
    x_t, noise = dit.q_sample(tgt_tokens, t, noise=noise)
    eps_pred = dit(
        curr_tokens=curr_siglip.float(),
        noisy_subgoal=x_t,
        t=t,
        text_embed=text_embed,
        pose_delta=pose_delta,
        last_action=last_action.long(),
        horizon=horizon,
    )
    per_sample_mse = (eps_pred - noise).pow(2).mean(dim=[1, 2])  # (B,)
    valid_f = valid.to(per_sample_mse.dtype)
    denom = valid_f.sum().clamp(min=1.0)
    mse_loss = (per_sample_mse * valid_f).sum() / denom

    # ---- POLICY LEG: sample subgoal from the trainable DiT, run policy ----
    # Critical: the sampler must keep gradients flowing back into DiT params.
    # We use a short DDIM (default 4 steps) so backprop is tractable.
    subgoal_sample = _grad_aware_ddim_sample(
        dit,
        curr_tokens=curr_siglip.float(),
        text_embed=text_embed,
        pose_delta=pose_delta,
        last_action=last_action,
        horizon=horizon,
        num_steps=args.ddim_steps,
    )

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
        subgoal_tokens=subgoal_sample,
    )
    logits = out["action_logits"]
    ce_loss = F.cross_entropy(logits, actions)

    # ---- Joint backward ----
    loss = args.lambda_mse * mse_loss + args.lambda_ce * ce_loss

    metrics: dict[str, Any] = {
        "loss": float(loss.item()),
        "mse": float(mse_loss.item()),
        "ce": float(ce_loss.item()),
        "acc": float((logits.argmax(dim=-1) == actions).float().mean().item()),
        "n_valid": int(valid_f.sum().item()),
    }
    return loss, metrics


def _grad_aware_ddim_sample(
    dit,
    curr_tokens: torch.Tensor,
    text_embed: torch.Tensor,
    pose_delta: torch.Tensor,
    last_action: torch.Tensor,
    horizon: torch.Tensor,
    num_steps: int = 4,
) -> torch.Tensor:
    """DDIM sampler that KEEPS gradients flowing back through the DiT.

    The standard ``dit.ddim_sample`` wraps the loop in ``@torch.no_grad``
    because it's meant for inference. P3.5 needs gradients to propagate
    from the policy's action loss back into DiT params via the sampled
    subgoal — so we re-implement the loop here without the no-grad
    guard. Identical math, just leaves the autograd tape attached.

    Compatible with both ``SubgoalDiT`` and ``PixArtSubgoalDiT``; both
    expose ``alpha_bar``, ``num_timesteps``, and a ``forward`` with the
    same signature.
    """
    device = curr_tokens.device
    B, S, D = curr_tokens.shape
    ts = (
        torch.linspace(dit.num_timesteps - 1, 0, num_steps + 1, device=device)
        .round().long()
    )
    x = torch.randn((B, S, D), device=device, dtype=curr_tokens.dtype)
    for i in range(num_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        ab_cur = dit.alpha_bar[t_cur].to(x.dtype)
        ab_next = dit.alpha_bar[t_next].to(x.dtype)
        eps = dit(
            curr_tokens=curr_tokens,
            noisy_subgoal=x,
            t=t_cur.expand(B),
            text_embed=text_embed,
            pose_delta=pose_delta,
            last_action=last_action.long(),
            horizon=horizon,
        )
        x0_pred = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt().clamp(min=1e-6)
        x = ab_next.sqrt() * x0_pred + (1 - ab_next).clamp(min=0).sqrt() * eps
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data / training
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Smaller than P3 — backprop through DDIM is "
                        "expensive even at 4 steps. Increase only if VRAM allows.")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--history_frames", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=2)

    # Checkpoints
    parser.add_argument("--p3_ckpt", type=str, required=True,
                        help="P3 checkpoint (last.pt or best.pt) — policy "
                        "weights that already know how to use subgoals.")
    parser.add_argument("--dit_path", type=str, required=True,
                        help="P2 DiT checkpoint to initialize the world "
                        "model. Will be UNFROZEN and refined in this phase.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="If P2 DiT is a PixArtSubgoalDiT, pass the HF "
                        "snapshot dir here so the wrapper can rebuild it.")

    # Joint-loss tuning
    parser.add_argument("--lambda_mse", type=float, default=1.0,
                        help="Weight on the diffusion ε-MSE leg. Keep at "
                        "1.0 unless the DiT starts drifting from its P2 "
                        "trajectory (sign: val_cos in P3.5 dropping).")
    parser.add_argument("--lambda_ce", type=float, default=0.3,
                        help="Weight on the action-CE leg. 0.3 is the "
                        "default gentle pull. Try 0.1 if DiT collapses, "
                        "0.7 if joint phase has no effect.")
    parser.add_argument("--ddim_steps", type=int, default=4,
                        help="DDIM steps used for the policy-leg sampling. "
                        "More steps = better subgoal but heavier backward.")

    # LRs (param groups for DiT vs LoRA vs heads)
    parser.add_argument("--dit_backbone_lr", type=float, default=1e-6,
                        help="LR for DiT backbone params. Tiny — we're "
                        "refining, not retraining.")
    parser.add_argument("--dit_adapter_lr", type=float, default=1e-5,
                        help="LR for DiT adapter / I/O params. Only used "
                        "for PixArt-init DiT; vanilla DiT uses dit_backbone_lr.")
    parser.add_argument("--lora_lr", type=float, default=5e-6,
                        help="LR for PaliGemma LoRA. Smaller than P3's "
                        "1e-5 — the policy was already trained in P3.")
    parser.add_argument("--head_lr", type=float, default=1e-4,
                        help="LR for action head + cross-attn + heads.")

    # PaliGemma
    parser.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)

    # Outputs
    parser.add_argument(
        "--out_dir",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project"))
                    / "logs" / "openfly" / "joint_refine"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args(argv)
    device = torch.device(args.device)
    base_out = Path(args.out_dir)
    out_dir = (base_out / time.strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_joint_refine] writing to {out_dir}")
    print(
        f"[train_joint_refine] lambdas: mse={args.lambda_mse} ce={args.lambda_ce} "
        f"ddim_steps={args.ddim_steps}"
    )

    # ----- data ---------------------------------------------------------
    full_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_samples=args.max_samples,
        require_images=True,
        oversample_stop=1.0,
    )
    val_size = max(1, int(len(full_ds) * args.val_frac)) if args.val_frac > 0 else 0
    train_size = len(full_ds) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(0),
        )
    else:
        train_ds, val_ds = full_ds, None
    print(f"[train_joint_refine] split: {train_size} train / {val_size} val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    # ----- models -------------------------------------------------------
    print("[train_joint_refine] loading world model (will be UNFROZEN)…")
    dit = _load_world_model(
        args.dit_path, pretrained_path=args.pretrained_path, device=device,
    )
    # Re-enable gradients (load_world_model freezes by default for P3)
    for p in dit.parameters():
        p.requires_grad = True
    dit.train()

    print("[train_joint_refine] loading P3 policy…")
    model = PaliGemmaVLNPolicy(
        history_frames=args.history_frames,
        paligemma_model_name=args.paligemma_model,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    ).to(device)
    p3_state = torch.load(args.p3_ckpt, map_location=device, weights_only=False)
    # Filter shape-mismatched keys before loading (in case the P3 checkpoint
    # predates a later policy-architecture change).
    own = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    for k, v in p3_state["model"].items():
        if k not in own:
            continue
        if tuple(own[k].shape) != tuple(v.shape):
            shape_mismatches.append(f"{k} ({tuple(v.shape)} → {tuple(own[k].shape)})")
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"[train_joint_refine] P3 init — loaded={len(filtered)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_mismatch={len(shape_mismatches)}"
    )
    if shape_mismatches:
        print(f"[train_joint_refine]   shape mismatches (using fresh init):")
        for m in shape_mismatches:
            print(f"     {m}")
    processor = _build_processor(args.paligemma_model)

    # ----- optimizer with three groups: dit-backbone, dit-adapter, policy-lora, policy-heads
    policy_groups = lora_and_head_param_groups(model, lora_lr=args.lora_lr, head_lr=args.head_lr)
    has_param_groups_method = hasattr(dit, "param_groups")
    if has_param_groups_method:
        dit_groups = dit.param_groups(
            backbone_lr=args.dit_backbone_lr, adapter_lr=args.dit_adapter_lr,
        )
        # Tag with prefix so warmup can scale each individually.
        for pg in dit_groups:
            pg["name"] = f"dit_{pg['name']}"
    else:
        dit_groups = [{
            "params": [p for p in dit.parameters() if p.requires_grad],
            "lr": args.dit_backbone_lr, "name": "dit_all",
        }]
    optimizer = torch.optim.AdamW(
        dit_groups + policy_groups, weight_decay=0.01, betas=(0.9, 0.99),
    )
    base_lrs = {pg.get("name", str(i)): float(pg["lr"]) for i, pg in enumerate(optimizer.param_groups)}

    n_dit = sum(p.numel() for group in dit_groups for p in group["params"])
    n_policy = sum(p.numel() for group in policy_groups for p in group["params"])
    print(
        f"[train_joint_refine] trainable DiT={n_dit/1e6:.1f}M policy={n_policy/1e6:.1f}M"
    )

    def _apply_warmup(step_idx: int) -> None:
        if args.warmup_steps <= 0:
            return
        frac = min(1.0, (step_idx + 1) / float(args.warmup_steps))
        for i, pg in enumerate(optimizer.param_groups):
            name = pg.get("name", str(i))
            pg["lr"] = base_lrs[name] * frac

    # ----- train --------------------------------------------------------
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        dit.train()
        t0 = time.time()
        n, loss_sum, mse_sum, ce_sum, acc_sum = 0, 0.0, 0.0, 0.0, 0.0
        nan_skips = 0
        for step, batch in enumerate(train_loader):
            _apply_warmup(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _joint_train_step(model, dit, processor, batch, device, args)
            if not torch.isfinite(loss):
                nan_skips += 1
                if nan_skips <= 5 or nan_skips % 50 == 0:
                    print(f"[train_joint_refine] WARN non-finite loss at gstep {global_step}, skipping")
                continue
            loss.backward()
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"]],
                    args.grad_clip,
                )
                if not torch.isfinite(gn):
                    nan_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()
            global_step += 1

            bs = batch["action_id"].shape[0]
            n += bs
            loss_sum += metrics["loss"] * bs
            mse_sum += metrics["mse"] * bs
            ce_sum += metrics["ce"] * bs
            acc_sum += metrics["acc"] * bs
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch:02d} step {step:04d} gstep {global_step:06d} "
                    f"loss={metrics['loss']:.3f} mse={metrics['mse']:.3f} "
                    f"ce={metrics['ce']:.3f} acc={metrics['acc']:.3f}",
                    flush=True,
                )

        train_log: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n, 1),
            "train_mse": mse_sum / max(n, 1),
            "train_ce": ce_sum / max(n, 1),
            "train_acc": acc_sum / max(n, 1),
            "nan_skips": nan_skips,
            "time_s": time.time() - t0,
        }
        print(f"[train_joint_refine] epoch {epoch} → {train_log}")
        history.append(train_log)

        # Save BOTH the DiT and the policy — we'll need both for P5 inference.
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_state = {
            "dit": dit.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        }
        tmp = out_dir / "last.pt.tmp"
        torch.save(ckpt_state, tmp)
        os.replace(tmp, out_dir / "last.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"[train_joint_refine] saved {out_dir / 'last.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
