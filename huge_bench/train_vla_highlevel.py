"""Stage 6: hierarchical-VLA high-level offline SFT on HUGE-Bench.

Trains ONLY the high-level vision-language tower of
``vla.vla_policy.HierarchicalVLAActor``:

  - PaliGemma LoRA adapters
  - cross_attn / image_proj / text_proj
  - depth_encoder (sees zero depth here -- regularises only)
  - LSTM
  - target_mlp (final 3-D body-frame waypoint output)

The Stage-2 frozen waypoint MLP, the obs normalizer and the action std are
left untouched. Supervision is body-frame future-waypoint MSE (plus a
direction cosine term) computed by ``HugeTask0WithFutureWaypoint``.

Why this script and not ``huge_bench/train_bc.py``? ``HugeBCPolicy`` has its
own delta-action MLP head and bypasses the frozen waypoint policy entirely,
so its weights are NOT compatible with the Stage 4/5 RL stack. This trainer
produces a checkpoint whose state_dict drops directly into
``vla/train.py --resume_path`` (and the same shape-tolerant loader Stage 4
already uses).

Launch:
    source /home/ubuntu/miniconda3/bin/activate isaac
    cd /home/ubuntu/drone_project
    python -m huge_bench.train_vla_highlevel --max_steps 5000 \
        --resume_path logs/rsl_rl/vla_drone_direct/<run>/model_*.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from huge_bench.dataset_highlevel import HugeTask0WithFutureWaypoint, collate_highlevel  # noqa: E402
from vla.vla_policy import HierarchicalVLAActor  # noqa: E402

NUM_IMAGE_TOKENS = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 6 hierarchical-VLA SFT on HUGE-Bench")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-step batch (small: 4 cams * PaliGemma is memory-heavy).")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lookahead_frames", type=int, default=25,
                   help="Frames to look ahead for body-frame target label.")
    p.add_argument("--target_range", type=float, default=3.0,
                   help="Must match HierarchicalVLAActor.target_range (tanh cap).")
    p.add_argument("--head_lr", type=float, default=3e-4,
                   help="LR for cross-attn / target_mlp / depth_encoder / LSTM.")
    p.add_argument("--lora_lr", type=float, default=1e-6,
                   help="LR for PaliGemma LoRA adapters.")
    p.add_argument("--lora_warmup_steps", type=int, default=200,
                   help="Freeze LoRA for the first N steps so the head settles first.")
    p.add_argument("--direction_loss_weight", type=float, default=0.1,
                   help="Cosine direction loss weight (in addition to MSE).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_split", type=str, default="test_seen", choices=["test_seen", "test_unseen"])
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_dir", type=str, default=str(_DRONE_ROOT / "logs" / "huge_bench_highlevel"))
    p.add_argument("--resume_path", type=str, default=None,
                   help="Stage-5 VLA checkpoint (model_state_dict from vla/train.py).")
    p.add_argument("--waypoint_checkpoint", type=str,
                   default=str(_DRONE_ROOT / "checkpoints" / "stage2_waypoint.pt"),
                   help="Frozen Stage-2 waypoint policy.")
    p.add_argument("--paligemma_model", type=str, default="google/paligemma-3b-pt-224")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processor(model_name: str):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    proc.tokenizer.padding_side = "right"
    return proc


def _tokenize(processor, instructions: list[str], max_text_length: int, device: torch.device,
              image_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the env-side tokenization in vla/vla_drone_env.py."""
    prefixed = ["\n" + s for s in instructions]
    tok = processor.tokenizer(
        prefixed,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_text_length - NUM_IMAGE_TOKENS,
    )
    b = len(instructions)
    img_ids = torch.full((b, NUM_IMAGE_TOKENS), image_token_id, dtype=torch.long)
    img_mask = torch.ones(b, NUM_IMAGE_TOKENS, dtype=torch.long)
    input_ids = torch.cat([img_ids, tok["input_ids"]], dim=1).to(device)
    attention_mask = torch.cat([img_mask, tok["attention_mask"]], dim=1).to(device)
    return input_ids, attention_mask


def _build_obs(batch: dict, processor, max_text_length: int, image_token_id: int,
               device: torch.device) -> dict:
    """Build a HierarchicalVLAActor obs dict from a HUGE high-level batch.

    HUGE has ONE camera; we replicate it across the 4-camera obs slot. The
    cross-attn camera embedding still differentiates the slots, but the same
    visual content reaches each, so the model effectively learns "find the
    target in the image" without geometric direction priors. At Stage 4
    inference time the four real cameras provide the geometric signal.
    """
    image = batch["image"].to(device, non_blocking=True)              # (B, 224, 224, 3) in [-1, 1]
    image_01 = (image + 1.0) * 0.5                                    # -> [0, 1] for actor.preprocess_images
    B = image_01.shape[0]
    rgb = image_01.unsqueeze(1).expand(B, 4, 224, 224, 3).contiguous()

    # Depth and flight state are unavailable offline -> zeros (depth_dropout
    # behaviour during training will further mask them).
    depth = torch.zeros(B, 4, 224, 224, device=device, dtype=torch.float32)
    flight_state = torch.zeros(B, 9, device=device, dtype=torch.float32)
    pos_error_w = torch.zeros(B, 3, device=device, dtype=torch.float32)

    instructions = batch["instruction"]
    text_tokens, text_mask = _tokenize(
        processor, instructions, max_text_length, device, image_token_id
    )

    return {
        "rgb": rgb,
        "depth": depth,
        "policy": flight_state,
        "pos_error_w": pos_error_w,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _compute_loss(target_logits: torch.Tensor, target_body: torch.Tensor,
                  target_range: float, direction_w: float) -> tuple[torch.Tensor, dict]:
    pred_body = torch.tanh(target_logits) * target_range
    mse = F.mse_loss(pred_body, target_body)
    diag = {"mse": float(mse.detach().item())}
    if direction_w > 0.0:
        # Cosine direction loss, masked where the label is too short to have a
        # meaningful direction (ignore static frames near the goal).
        norm_label = target_body.norm(dim=-1)
        mask = (norm_label > 0.05).float()
        if mask.sum() > 0:
            cos = F.cosine_similarity(pred_body, target_body, dim=-1, eps=1e-6)
            dir_loss = ((1.0 - cos) * mask).sum() / mask.sum().clamp(min=1.0)
            diag["dir"] = float(dir_loss.detach().item())
            return mse + direction_w * dir_loss, diag
    return mse, diag


# ---------------------------------------------------------------------------
# Param groups
# ---------------------------------------------------------------------------

def _split_params(actor: HierarchicalVLAActor) -> tuple[list, list]:
    """Return (head_params, lora_params) — head = everything trainable except LoRA."""
    head_names = (
        "image_proj", "text_proj", "cross_attn", "camera_embed",
        "depth_encoder", "obj_classifier", "target_mlp", "lstm",
    )
    head_params = []
    lora_params = []
    for name, p in actor.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif any(k in name for k in head_names):
            head_params.append(p)
        # The Gaussian _std_param and frozen waypoint buffers are excluded.
    return head_params, lora_params


def _load_resume(actor: HierarchicalVLAActor, path: str, device: torch.device) -> None:
    """Shape-tolerant load from a vla/train.py checkpoint (policy.actor.* keys)."""
    print(f"[Stage6] Resuming from {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    src = ckpt.get("model_state_dict", ckpt)
    # vla/train.py stores keys under "actor." (VLAPolicy wraps actor + critic)
    stripped: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        new_k = k
        if new_k.startswith("actor."):
            new_k = new_k[len("actor."):]
        stripped[new_k] = v
    tgt = actor.state_dict()
    skipped = []
    filtered = {}
    for k, v in stripped.items():
        if k not in tgt:
            continue
        if tgt[k].shape != v.shape:
            skipped.append((k, tuple(v.shape), tuple(tgt[k].shape)))
            continue
        filtered[k] = v
    missing, unexpected = actor.load_state_dict(filtered, strict=False)
    print(
        f"  loaded {len(filtered)} keys; "
        f"missing={len(missing)} unexpected={len(unexpected)} shape_skipped={len(skipped)}"
    )
    for k, src_s, tgt_s in skipped[:5]:
        print(f"    skip {k}: ckpt {src_s} vs model {tgt_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Stage6] device={device}")

    # --- data ----------------------------------------------------------
    print("[Stage6] Building train dataset...")
    train_ds = HugeTask0WithFutureWaypoint(
        split="train",
        lookahead_frames=args.lookahead_frames,
        max_target_norm=args.target_range,
    )
    print(f"[Stage6] train: {len(train_ds)} frames across {train_ds.num_episodes()} episodes")
    val_ds = HugeTask0WithFutureWaypoint(
        split=args.val_split,
        lookahead_frames=args.lookahead_frames,
        max_target_norm=args.target_range,
    )
    print(f"[Stage6] val ({args.val_split}): {len(val_ds)} frames across {val_ds.num_episodes()} episodes")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_highlevel,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_highlevel,
        pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    # --- actor ---------------------------------------------------------
    print(f"[Stage6] Building HierarchicalVLAActor (waypoint={args.waypoint_checkpoint})")
    actor = HierarchicalVLAActor(
        waypoint_checkpoint_path=args.waypoint_checkpoint,
        paligemma_model_name=args.paligemma_model,
        target_range=args.target_range,
    ).to(device)

    if args.resume_path:
        _load_resume(actor, args.resume_path, device)

    head_params, lora_params = _split_params(actor)
    n_head = sum(p.numel() for p in head_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"[Stage6] head trainable: {n_head:,} | lora trainable: {n_lora:,}")

    head_opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=1e-4)
    lora_opt = torch.optim.AdamW(lora_params, lr=args.lora_lr, weight_decay=0.0)

    # --- tokenizer / processor ----------------------------------------
    print(f"[Stage6] Loading processor for {args.paligemma_model}...")
    processor = _load_processor(args.paligemma_model)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    max_text_length = NUM_IMAGE_TOKENS + 24  # 24 text tokens + 256 image placeholders

    # --- logging -------------------------------------------------------
    run_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[Stage6] Logging to {run_dir}")

    # --- train loop ---------------------------------------------------
    actor.train()
    step = 0
    accum = 0
    head_opt.zero_grad(set_to_none=True)
    lora_opt.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    t0 = time.time()
    running = {"loss": 0.0, "mse": 0.0, "dir": 0.0, "n": 0}

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        target_body = batch["target_body"].to(device, non_blocking=True)
        obs = _build_obs(batch, processor, max_text_length, image_token_id, device)

        # Disable LoRA gradient during warmup so the head settles before pulling
        # the backbone around.
        lora_active = step >= args.lora_warmup_steps
        if not lora_active:
            for p in lora_params:
                p.requires_grad = False
        else:
            for p in lora_params:
                p.requires_grad = True

        target_logits, _ = actor.forward_lora_grad(obs)
        loss, diag = _compute_loss(
            target_logits, target_body, args.target_range, args.direction_loss_weight
        )
        (loss / args.grad_accum).backward()
        accum += 1

        running["loss"] += float(loss.detach().item())
        running["mse"] += diag["mse"]
        running["dir"] += diag.get("dir", 0.0)
        running["n"] += 1

        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            head_opt.step()
            if lora_active:
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=0.5)
                lora_opt.step()
            head_opt.zero_grad(set_to_none=True)
            lora_opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if step % 10 == 0:
                n = max(1, running["n"])
                rate = step / max(1e-6, time.time() - t0)
                writer.add_scalar("train/loss", running["loss"] / n, step)
                writer.add_scalar("train/target_mse", running["mse"] / n, step)
                writer.add_scalar("train/dir_loss", running["dir"] / n, step)
                writer.add_scalar("train/steps_per_sec", rate, step)
                writer.add_scalar("train/lora_active", float(lora_active), step)
                print(
                    f"[Stage6] step {step:5d} | loss={running['loss']/n:.4f} "
                    f"mse={running['mse']/n:.4f} dir={running['dir']/n:.4f} "
                    f"lora={'on' if lora_active else 'off'} | {rate:.2f} it/s"
                )
                running = {"loss": 0.0, "mse": 0.0, "dir": 0.0, "n": 0}

            if step % args.val_every == 0:
                actor.eval()
                vmetrics = _validate(
                    actor, val_loader, processor, max_text_length, image_token_id,
                    device, args.val_batches, args.target_range, args.direction_loss_weight,
                )
                for k, v in vmetrics.items():
                    writer.add_scalar(f"val/{args.val_split}_{k}", v, step)
                print(f"[Stage6] step {step:5d} | val({args.val_split}) {vmetrics}")
                actor.train()

            if step % args.save_every == 0 or step == args.max_steps:
                ckpt_path = run_dir / f"model_{step}.pt"
                # Match vla/train.py format so the checkpoint is loadable by
                # subsequent --resume_path runs.
                torch.save(
                    {
                        "model_state_dict": {f"actor.{k}": v for k, v in actor.state_dict().items()},
                        "head_opt_state_dict": head_opt.state_dict(),
                        "lora_opt_state_dict": lora_opt.state_dict(),
                        "step": step,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"[Stage6] saved {ckpt_path}")

    writer.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(actor, loader, processor, max_text_length, image_token_id, device,
              max_batches, target_range, direction_w) -> dict:
    total = {"mse": 0.0, "dir": 0.0, "n": 0}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        target_body = batch["target_body"].to(device, non_blocking=True)
        obs = _build_obs(batch, processor, max_text_length, image_token_id, device)
        target_logits, _ = actor.forward_lora_grad(obs)
        _, diag = _compute_loss(target_logits, target_body, target_range, direction_w)
        total["mse"] += diag["mse"]
        total["dir"] += diag.get("dir", 0.0)
        total["n"] += 1
    n = max(1, total["n"])
    return {
        "target_mse": total["mse"] / n,
        "dir_loss": total["dir"] / n,
    }


if __name__ == "__main__":
    main()
