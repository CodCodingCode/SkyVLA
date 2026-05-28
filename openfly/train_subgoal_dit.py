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
from torch.utils.data import DataLoader

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
# EMA — exponential moving average of trainable parameters
# ---------------------------------------------------------------------------

class EMA:
    """Slow-moving copy of the trainable parameters.

    Standard diffusion-training hygiene. Random-t sampling makes the
    per-step gradient noisy; the live weights wobble around the optimum
    even after they've effectively converged. Evaluating and saving a
    decay=0.9999 EMA copy averages out that wobble — effective window of
    ~10k steps — and almost always lifts ``val_cos`` by a few points at
    near-zero compute cost.

    Memory cost: one extra copy of the trainable params on the model's
    device. For PixArt-Σ-XL (~600M trainable in fp32 ≈ 2.4 GB) this is
    noticeable but fits comfortably on an A100.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def swap_into(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Copy EMA weights into the live model. Returns a backup so the
        live weights can be restored with :meth:`swap_back`."""
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def swap_back(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.decay = float(sd["decay"])
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _train_step(
    dit: SubgoalDiT,
    paligemma: PaliGemmaFeatureExtractor,
    processor,
    batch: dict[str, Any],
    device: torch.device,
    min_snr_gamma: float = 0.0,
    repa_weight: float = 0.0,
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

    use_repa = repa_weight > 0.0 and getattr(dit, "repa_proj", None) is not None
    if use_repa:
        out = dit(
            curr_tokens=curr_tokens,
            noisy_subgoal=x_t,
            t=t,
            text_embed=text_embed,
            pose_delta=pose_delta,
            last_action=last_action,
            horizon=horizon,
            return_repa=True,
        )
        eps_pred = out["eps"]
        repa_hidden = out["repa_hidden"]
    else:
        eps_pred = dit(
            curr_tokens=curr_tokens,
            noisy_subgoal=x_t,
            t=t,
            text_embed=text_embed,
            pose_delta=pose_delta,
            last_action=last_action,
            horizon=horizon,
        )
        repa_hidden = None

    # Per-sample MSE, then mask out invalid (terminal) subgoals before reducing.
    per_sample = (eps_pred - noise).pow(2).mean(dim=[1, 2])  # (B,)

    # Min-SNR-γ loss weighting (Hang et al. 2023). Vanilla ε-MSE puts large
    # per-sample loss values on low-SNR (high-t) examples — they dominate the
    # gradient, low-t precision suffers, cos-sim plateaus. weight = min(snr,γ)/snr
    # caps the contribution from any one timestep so all t train more evenly.
    if min_snr_gamma > 0.0:
        ab_t = dit.alpha_bar[t.long()].to(per_sample.dtype)
        snr = ab_t / (1.0 - ab_t).clamp(min=1e-8)
        snr_weight = snr.clamp(max=min_snr_gamma) / snr.clamp(min=1e-8)
        per_sample = per_sample * snr_weight

    mask = valid.to(per_sample.dtype)
    denom = mask.sum().clamp(min=1.0)
    loss = (per_sample * mask).sum() / denom

    metrics: dict[str, float] = {
        "loss": float(loss.item()),
        "n_valid": int(mask.sum().item()),
    }

    # REPA cosine-alignment auxiliary loss.
    # Encourages an intermediate PixArt block (subgoal half) to align
    # with the CLEAN target SigLIP tokens, weighted per-token. Cosine
    # similarity in token space (dim=-1) is the natural metric — same
    # space as our val_cos. Loss = mean(1 - cos), masked by ``valid``.
    if use_repa and repa_hidden is not None:
        # repa_hidden: (B, S, token_dim); tgt_tokens: (B, S, token_dim)
        repa_cos = F.cosine_similarity(
            repa_hidden.to(tgt_tokens.dtype), tgt_tokens, dim=-1
        )  # (B, S)
        repa_loss_per_sample = (1.0 - repa_cos).mean(dim=-1)  # (B,)
        repa_loss = (repa_loss_per_sample * mask).sum() / denom
        loss = loss + repa_weight * repa_loss
        metrics["repa_loss"] = float(repa_loss.item())
        metrics["repa_cos"] = float(
            ((repa_cos.mean(dim=-1)) * mask).sum().item() / float(denom.item())
        )

    return loss, metrics


@torch.no_grad()
def _eval_step(
    dit: SubgoalDiT,
    paligemma: PaliGemmaFeatureExtractor,
    processor,
    batch: dict[str, Any],
    device: torch.device,
    *,
    num_ddim_steps: int = 20,
) -> dict[str, float]:
    """Validation metrics.

    Two numbers:

    * ``val_loss`` — single-step ε-MSE at a random timestep. Cheap,
      directly comparable to the training loss curve; mainly a sanity
      check that training is generalising.
    * ``val_cos`` — cosine similarity between **full DDIM-sampled**
      ``x0`` and the ground-truth subgoal tokens. This is the metric
      that actually mirrors how the policy will consume the DiT at
      inference time (the previous single-step recovery massively
      under- and over-stated quality depending on the random ``t``).
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

    # ---- val_loss: single-step ε-MSE (cheap, comparable to train loss) ----
    t = torch.randint(0, dit.num_timesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(tgt_tokens)
    x_t, noise = dit.q_sample(tgt_tokens, t, noise=noise)
    eps_pred = dit(
        curr_tokens=curr_tokens, noisy_subgoal=x_t, t=t,
        text_embed=text_embed, pose_delta=pose_delta,
        last_action=last_action, horizon=horizon,
    )
    per_sample_mse = (eps_pred - noise).pow(2).mean(dim=[1, 2])

    # ---- val_cos: full DDIM-sampled x0 vs ground-truth subgoal -----------
    # This is ~20× the compute of the old single-step recovery, but is the
    # only number that reflects what the policy will actually consume.
    x0_sampled = dit.ddim_sample(
        curr_tokens=curr_tokens,
        text_embed=text_embed,
        pose_delta=pose_delta,
        last_action=last_action,
        horizon=horizon,
        num_steps=num_ddim_steps,
    )
    a = x0_sampled.reshape(B, -1).float()
    b = tgt_tokens.reshape(B, -1).float()
    cos = F.cosine_similarity(a, b, dim=-1)

    mask = valid.to(per_sample_mse.dtype)
    denom = mask.sum().clamp(min=1.0)
    return {
        "val_loss": float((per_sample_mse * mask).sum().item() / denom.item()),
        "val_cos": float((cos * mask.to(cos.dtype)).sum().item() / denom.item()),
        "n_valid": int(mask.sum().item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _install_diagnostics(stack_dump_interval_s: float = 120.0) -> None:
    """Make silent-death debugging tractable.

    Without these, a long run that's hanging (DataLoader stuck on a
    corrupt frame, NCCL waiting, etc.) or being killed (timeout, OOM
    reaper, manual SIGTERM) gives us nothing in the log because Python
    stdout is block-buffered for non-TTY and frame state is lost on exit.

    What we install:
      * ``faulthandler.dump_traceback_later(N, repeat=True)`` — every N
        seconds, dumps a stack trace of every thread to stderr. If the
        process is stuck, you see WHERE it's stuck without needing
        py-spy / strace. Repeat=True means it fires once per interval
        until exit.
      * ``faulthandler.register(SIGUSR1)`` — send ``kill -USR1 <pid>``
        to dump a stack on demand without restarting.
      * ``atexit`` callback that prints the exit reason (clean / signal
        / unhandled exception) so the last line of the log is never
        ambiguous.
      * SIGTERM / SIGINT handlers that flush stdout/stderr before
        exiting so partial logs survive.
    """
    import atexit
    import faulthandler
    import signal
    import sys
    import threading
    import traceback

    faulthandler.enable()
    faulthandler.dump_traceback_later(
        float(stack_dump_interval_s), repeat=True, exit=False
    )
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)

    exit_reason: list[str] = ["clean"]

    def _atexit() -> None:
        print(
            f"[train_subgoal_dit] EXIT reason={exit_reason[0]}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()

    atexit.register(_atexit)

    def _term_handler(signum, frame) -> None:  # noqa: ARG001
        exit_reason[0] = f"signal({signum})"
        # Dump stacks so we know exactly what was running when killed.
        print(
            f"[train_subgoal_dit] received signal {signum}; "
            f"dumping live thread stacks before exit:",
            file=sys.stderr,
            flush=True,
        )
        for tid, frm in sys._current_frames().items():
            print(f"--- thread {tid} ---", file=sys.stderr, flush=True)
            traceback.print_stack(frm, file=sys.stderr)
        sys.stderr.flush()
        # Re-raise as SystemExit so atexit + finalizers still run.
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _term_handler)
    signal.signal(signal.SIGINT, _term_handler)

    # Mark exceptions in the reason so the EXIT line says "exception".
    _orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, tb):
        exit_reason[0] = f"exception({exc_type.__name__}: {exc_value})"
        _orig_excepthook(exc_type, exc_value, tb)

    sys.excepthook = _excepthook


def main(argv: list[str] | None = None) -> int:
    # Install diagnostics FIRST so even early failures (arg parsing,
    # config load, dataset construction) leave breadcrumbs.
    _install_diagnostics()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Exponential-moving-average decay over trainable params. "
        "0.9999 ≈ effective averaging window of ~10k steps; standard "
        "diffusion-training hygiene. Validation runs and best.pt are "
        "saved with EMA weights. 0 disables EMA entirely.",
    )
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR-γ loss weighting (Hang et al. 2023). 5.0 is the paper's "
        "default; 0 disables. Caps per-timestep loss so high-t (low-SNR) "
        "samples stop dominating the gradient — typically lifts val_cos by "
        "5–15 points on ε-prediction DiTs.",
    )

    # DiT hyperparams
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="REQUIRED. Path to a PixArt-Σ HF snapshot directory. The "
        "pretrained backbone (28 layers, hidden=1152, cross-attention to "
        "text) is the single biggest training-quality lever for this DiT — "
        "random init plateaus around val_cos≈0.6 even after long training, "
        "PixArt init reaches the same point in a fraction of the steps and "
        "keeps climbing. ``--depth``/``--hidden``/``--num_heads`` are "
        "ignored (PixArt config wins). Typical path on this machine: "
        "~/assets/pretrained/hf_cache/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/"
        "snapshots/<hash>/transformer",
    )
    parser.add_argument(
        "--freeze_backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the pretrained PixArt backbone and train only the SigLIP "
        "I/O adapters + extra-conditioning modules. Smaller VRAM footprint "
        "but loses the chance to adapt the backbone to aerial features. "
        "Only relevant with --pretrained_path.",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-6,
        help="LR for the pretrained PixArt backbone params. Only used with "
        "--pretrained_path (and ignored when --freeze_backbone is on). "
        "1e-6 is standard finetuning-of-pretrained-image-DiT territory.",
    )
    parser.add_argument(
        "--adapter_lr",
        type=float,
        default=1e-4,
        help="LR for the randomly-initialised SigLIP I/O adapters + extra "
        "conditioning modules. Only used with --pretrained_path. ~100× "
        "higher than --backbone_lr because these need to learn from scratch.",
    )

    # Data
    parser.add_argument("--history_frames", type=int, default=0,
                        help="DiT does not consume history; default 0 to skip extra disk reads.")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--per_env_max_episodes",
        type=int,
        default=0,
        help="Cap episodes-per-env when loading the training split. "
        "Balances contribution across all envs in train.json — without "
        "this, env_ue_bigcity dominates because it has the most "
        "downloaded frames locally. 0 = no cap (legacy behaviour). "
        "Typical value: 2000–3000 to match the smaller envs' available "
        "data while still capping the large ones.",
    )
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--val_split",
        type=str,
        default="unseen",
        help="OpenFly split to use for validation — defaults to 'unseen' "
        "(held-out environments). The previous random_split-on-train setup "
        "leaked at the per-step level (val step at episode E:47 shared "
        "trajectory with train E:46/E:48) and inflated val_cos. Pass "
        "'none' to disable validation entirely. Must differ from --split.",
    )
    parser.add_argument(
        "--val_max_episodes", type=int, default=0,
        help="Cap loaded episodes in the val split (debug). 0 = use all.",
    )
    parser.add_argument(
        "--val_max_samples", type=int, default=0,
        help="Cap unrolled steps in the val split (debug). 0 = use all.",
    )
    parser.add_argument(
        "--val_ood_split",
        type=str,
        default="",
        help="Optional SECOND validation split (e.g. 'seen' if --val_split=unseen "
        "or vice versa). When set, an extra val pass runs each epoch and "
        "writes val_ood_loss / val_ood_cos alongside the primary val. "
        "Distinguishes in-distribution generalisation from OOD generalisation; "
        "this is the diagnostic that catches DiT overfitting on train.json "
        "scene textures while looking fine on the in-dist val.",
    )
    parser.add_argument("--val_ood_max_episodes", type=int, default=0)
    parser.add_argument("--val_ood_max_samples", type=int, default=0)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Adapter / I/O dropout probability fed to PixArtSubgoalDiT "
        "(input + text + extra-cond paths) and to SubgoalDiT's transformer "
        "blocks (residual attn + MLP). 0.0 preserves prior behaviour. "
        "0.1–0.2 helps fight train-set overfit on smaller scene libraries.",
    )
    parser.add_argument(
        "--augment_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply input-side image augmentation (brightness/contrast/colour "
        "jitter/noise) to current + history frames during training. The "
        "subgoal target frame stays clean. Training only — val datasets "
        "never augment. Use to decorrelate scene texture from semantic "
        "content when train.json scene library is narrow.",
    )
    # ---- CFG-style conditioning dropout (training only) -----------------
    # GAIA-2 (arXiv 2503.20523) recipe: per-conditioner drop 50–80%, joint
    # drop 10%. Direct fix for "Unconditional Priors Matter"
    # (arXiv 2503.20240) — without this, the DiT memorises per-trajectory
    # conditioning patterns and val_cos on held-out splits stays near 0
    # (see docs/TRAIN.md gotcha #5+).
    parser.add_argument(
        "--cfg_drop_text", type=float, default=0.0,
        help="Per-sample probability of dropping the Gemma text embedding "
        "(replaced with the model's learned null embedding). GAIA-2 used 0.8.",
    )
    parser.add_argument("--cfg_drop_pose", type=float, default=0.0)
    parser.add_argument("--cfg_drop_action", type=float, default=0.0)
    parser.add_argument("--cfg_drop_horizon", type=float, default=0.0)
    parser.add_argument(
        "--cfg_drop_joint", type=float, default=0.0,
        help="Per-sample probability of dropping ALL conditioners together. "
        "Trains the pure unconditional branch. GAIA-2 used 0.1.",
    )
    # ---- REPA-style representation alignment ---------------------------
    parser.add_argument(
        "--repa_layer_idx", type=int, default=0,
        help="1-indexed PixArt transformer block whose hidden state to "
        "align (subgoal half) with the clean target SigLIP tokens via "
        "cosine similarity. 0 disables REPA. PixArt-Σ has 28 blocks; "
        "REPA paper (arXiv 2410.06940) suggests ~1/3 in, so 8–10. Only "
        "applied when ``--pretrained_path`` is set (PixArtSubgoalDiT).",
    )
    parser.add_argument(
        "--repa_weight", type=float, default=0.0,
        help="Weight on the REPA cosine-alignment auxiliary loss. 0.1 is "
        "a reasonable starting point per the REPA paper. Ignored when "
        "``--repa_layer_idx 0``.",
    )
    parser.add_argument(
        "--val_ddim_steps", type=int, default=4,
        help="DDIM sampling steps used for val_cos. Default 4 matches the "
        "POLICY's inference setting (paligemma_vln.PaliGemmaVLNPolicy "
        "subgoal_sample_steps=4). Measuring val at higher step counts "
        "inflates val_cos beyond what the policy actually gets at deploy "
        "time — earlier baselines (e.g. val_cos=0.61) used 20 and were "
        "over-optimistic. Bump to 20 only for an offline-only diagnostic "
        "of the DiT's denoising ceiling.",
    )
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

    # Persist args.json up front so we can diff run configs and so that a
    # mid-training crash still leaves provenance behind. (The previous
    # 20260528_011713 run lost its config entirely.)
    try:
        with open(out_dir / "args.json", "w", encoding="utf-8") as _f:
            json.dump(vars(args), _f, indent=2, default=str)
    except OSError as _exc:
        print(f"[train_subgoal_dit] WARN failed to persist args.json: {_exc}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    paligemma_dtype = dtype_map[args.paligemma_dtype]

    # ----- data ---------------------------------------------------------
    # Train comes from the train split. Validation comes from a SEPARATE
    # OpenFly split — by default ``unseen``, i.e. held-out environments
    # the train split never touched. We deliberately do NOT carve val out
    # of train with random_split: the previous setup did that at the
    # per-step level, so a val step at episode E step 47 had train steps
    # at E:46 and E:48 — visually near-identical frames, severe leak that
    # inflated val_cos. Use ``--val_split none`` to disable val entirely.
    train_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        per_env_max_episodes=args.per_env_max_episodes,
        require_images=args.require_images,
        oversample_stop=1.0,  # don't oversample stops for the world model
        subgoal_pairing=args.subgoal_pairing,
        subgoal_semantic_prob=args.subgoal_semantic_prob,
        subgoal_uniform_max=args.subgoal_uniform_max,
        augment_input=args.augment_input,
    )
    print(
        f"[train_subgoal_dit] pairing={args.subgoal_pairing} "
        f"semantic_prob={args.subgoal_semantic_prob} "
        f"uniform_max={args.subgoal_uniform_max}"
    )
    if len(train_ds) == 0:
        raise RuntimeError("Empty dataset — check OPENFLY_ANNOTATION_DIR / split.")

    val_ds: OpenFlyDataset | None = None
    if args.val_split and args.val_split.lower() not in {"none", ""}:
        if args.val_split == args.split:
            raise ValueError(
                f"--val_split={args.val_split!r} matches --split={args.split!r}; "
                "this would leak. Pick a different OpenFly split (unseen / seen / eval_test)."
            )
        val_ds = OpenFlyDataset(
            split=args.val_split,
            history_frames=args.history_frames,
            env_filter=args.env_filter,
            max_episodes=args.val_max_episodes,
            max_samples=args.val_max_samples,
            require_images=args.require_images,
            oversample_stop=1.0,  # never oversample stops in val — biases the metric
            subgoal_pairing=args.subgoal_pairing,
            subgoal_semantic_prob=args.subgoal_semantic_prob,
            subgoal_uniform_max=args.subgoal_uniform_max,
        )
        if len(val_ds) == 0:
            print(
                f"[train_subgoal_dit] WARNING: val split {args.val_split!r} loaded 0 "
                "samples — running without validation."
            )
            val_ds = None

    # Optional second val split. Lets us compare in-dist (e.g. 'seen') vs
    # OOD (e.g. 'unseen') cos sim per epoch — the only honest way to see
    # whether the DiT is generalising or just memorising training scenes.
    val_ood_ds: OpenFlyDataset | None = None
    if args.val_ood_split and args.val_ood_split.lower() not in {"none", ""}:
        if args.val_ood_split in (args.split, args.val_split):
            raise ValueError(
                f"--val_ood_split={args.val_ood_split!r} duplicates "
                f"--split or --val_split; pick a third distinct OpenFly split."
            )
        val_ood_ds = OpenFlyDataset(
            split=args.val_ood_split,
            history_frames=args.history_frames,
            env_filter=args.env_filter,
            max_episodes=args.val_ood_max_episodes,
            max_samples=args.val_ood_max_samples,
            require_images=args.require_images,
            oversample_stop=1.0,
            subgoal_pairing=args.subgoal_pairing,
            subgoal_semantic_prob=args.subgoal_semantic_prob,
            subgoal_uniform_max=args.subgoal_uniform_max,
        )
        if len(val_ood_ds) == 0:
            print(
                f"[train_subgoal_dit] WARNING: val_ood split "
                f"{args.val_ood_split!r} loaded 0 samples — disabling OOD val."
            )
            val_ood_ds = None

    print(
        f"[train_subgoal_dit] split: {len(train_ds)} train ({args.split}) / "
        f"{len(val_ds) if val_ds is not None else 0} val "
        f"({args.val_split if val_ds is not None else 'disabled'}) / "
        f"{len(val_ood_ds) if val_ood_ds is not None else 0} val_ood "
        f"({args.val_ood_split if val_ood_ds is not None else 'disabled'})  "
        f"steps/epoch: {math.ceil(len(train_ds) / args.batch_size)}  "
        f"augment_input={args.augment_input}  dropout={args.dropout}"
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
    val_ood_loader = (
        DataLoader(
            val_ood_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        if val_ood_ds is not None else None
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

    if args.pretrained_path:
        from openfly.models.subgoal_dit_pixart import PixArtSubgoalDiT
        dit = PixArtSubgoalDiT(
            pretrained_path=args.pretrained_path,
            token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            pose_delta_dim=4,
            num_last_actions=9,
            num_timesteps=args.num_timesteps,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout,
            cfg_drop_text=args.cfg_drop_text,
            cfg_drop_pose=args.cfg_drop_pose,
            cfg_drop_action=args.cfg_drop_action,
            cfg_drop_horizon=args.cfg_drop_horizon,
            cfg_drop_joint=args.cfg_drop_joint,
            repa_layer_idx=args.repa_layer_idx,
        ).to(device)
        print(
            f"[train_subgoal_dit] using PixArt-Σ pretrained backbone "
            f"from {args.pretrained_path} (freeze={args.freeze_backbone}, "
            f"dropout={args.dropout}, repa_layer={args.repa_layer_idx}, "
            f"repa_weight={args.repa_weight})"
        )
    else:
        dit = SubgoalDiT(
            token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            hidden=args.hidden,
            depth=args.depth,
            num_heads=args.num_heads,
            text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
            pose_delta_dim=4,
            num_last_actions=9,
            num_timesteps=args.num_timesteps,
            dropout=args.dropout,
        ).to(device)

    processor = _build_processor(args.paligemma_model)

    # When using the PixArt-pretrained backbone, use separate LRs for the
    # backbone (gentle, 1e-6) and the random-init adapters (aggressive,
    # 1e-4). Single global LR was the underlying cause of v1–v4 plateauing
    # at loss ~1.0 with PixArt init.
    if args.pretrained_path and not args.freeze_backbone:
        param_groups = dit.param_groups(
            backbone_lr=args.backbone_lr,
            adapter_lr=args.adapter_lr,
        )
        n_backbone = sum(p.numel() for p in param_groups[0]["params"])
        n_adapter = sum(p.numel() for p in param_groups[1]["params"])
        print(
            f"[train_subgoal_dit] param groups: "
            f"backbone={n_backbone/1e6:.1f}M @ lr={args.backbone_lr:.1e}, "
            f"adapter={n_adapter/1e6:.1f}M @ lr={args.adapter_lr:.1e}"
        )
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.99),
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in dit.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
        )
    # Snapshot the per-group base LRs so warmup scales each one to its own
    # target instead of clobbering them all to a single ``args.lr``.
    _base_lrs = {
        pg.get("name", str(i)): float(pg["lr"])
        for i, pg in enumerate(optimizer.param_groups)
    }

    ema: EMA | None = None
    if args.ema_decay > 0.0:
        ema = EMA(dit, decay=args.ema_decay)
        n_ema_params = sum(t.numel() for t in ema.shadow.values())
        print(
            f"[train_subgoal_dit] EMA enabled: decay={args.ema_decay} "
            f"shadow_params={n_ema_params/1e6:.1f}M"
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
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            print(f"[train_subgoal_dit] loaded EMA shadow from {args.resume}")
        elif ema is not None:
            print(
                f"[train_subgoal_dit] WARNING: --ema_decay set but checkpoint "
                f"{args.resume} has no 'ema' key; reinitialising shadow from "
                f"current weights (EMA history is lost)."
            )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[train_subgoal_dit] resumed from {args.resume} @ epoch {start_epoch}")

    def _apply_warmup(step_idx: int) -> None:
        if args.warmup_steps <= 0:
            return
        frac = min(1.0, (step_idx + 1) / float(args.warmup_steps))
        # Scale each group toward its OWN base LR — preserves the backbone
        # vs adapter ratio through warmup instead of dragging both to args.lr.
        for i, pg in enumerate(optimizer.param_groups):
            name = pg.get("name", str(i))
            pg["lr"] = _base_lrs[name] * frac

    # ----- train --------------------------------------------------------
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        dit.train()
        t0 = time.time()
        loss_sum, n_valid_sum, n_steps = 0.0, 0, 0
        nan_skip_count = 0
        for step, batch in enumerate(train_loader):
            _apply_warmup(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _train_step(
                dit, paligemma, processor, batch, device,
                min_snr_gamma=args.min_snr_gamma,
                repa_weight=args.repa_weight,
            )
            if metrics["n_valid"] == 0:
                continue
            # NaN guard: a single bad batch (numerically unlucky timestep
            # sample, fp16 overflow in the backbone, etc.) shouldn't be
            # allowed to corrupt the optimizer state. Zero the grads and
            # skip the step. The PixArt run @ lr=5e-5 spiked to NaN around
            # gstep 700 — this is the defense-in-depth fix even after we
            # also lowered the LR.
            if not torch.isfinite(loss):
                nan_skip_count += 1
                if nan_skip_count <= 5 or nan_skip_count % 50 == 0:
                    print(
                        f"[train_subgoal_dit] WARN non-finite loss at gstep "
                        f"{global_step} (total skipped: {nan_skip_count}); "
                        f"zero-grad and continue"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in dit.parameters() if p.requires_grad], args.grad_clip
                )
                # Same defense for blown-up gradients: clip_grad_norm_ returns
                # the pre-clipping norm; if it's non-finite, ``optimizer.step``
                # would still propagate NaNs into weights even though the
                # clip "succeeded" on individual elements. Skip.
                if not torch.isfinite(grad_norm):
                    nan_skip_count += 1
                    if nan_skip_count <= 5 or nan_skip_count % 50 == 0:
                        print(
                            f"[train_subgoal_dit] WARN non-finite grad_norm at "
                            f"gstep {global_step} (total skipped: "
                            f"{nan_skip_count}); zero-grad and continue"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()
            if ema is not None:
                ema.update(dit)
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
        def _run_val_pass(loader, prefix: str) -> None:
            if loader is None:
                return
            dit.eval()
            ema_backup = ema.swap_into(dit) if ema is not None else None
            try:
                vl_sum, cos_sum, vn_sum = 0.0, 0.0, 0
                for batch in loader:
                    m = _eval_step(
                        dit, paligemma, processor, batch, device,
                        num_ddim_steps=args.val_ddim_steps,
                    )
                    if m["n_valid"] == 0:
                        continue
                    vl_sum += m["val_loss"] * m["n_valid"]
                    cos_sum += m["val_cos"] * m["n_valid"]
                    vn_sum += m["n_valid"]
            finally:
                if ema is not None and ema_backup is not None:
                    ema.swap_back(dit, ema_backup)
            train_log[f"{prefix}_loss"] = vl_sum / max(vn_sum, 1)
            train_log[f"{prefix}_cos"] = cos_sum / max(vn_sum, 1)
            train_log[f"{prefix}_n_valid"] = vn_sum

        # Evaluate with EMA weights (this is the whole point of EMA);
        # ``_run_val_pass`` handles the swap and restores live weights so
        # training resumes against the optimizer state that goes with them.
        _run_val_pass(val_loader, "val")
        _run_val_pass(val_ood_loader, "val_ood")

        history.append(train_log)
        print(f"[train_subgoal_dit] epoch {epoch} → {train_log}")

        if (epoch + 1) % max(1, args.save_every) == 0 or epoch == args.epochs - 1:
            # Re-ensure the parent directory exists right before save. We
            # had a 20-minute training run lose its only checkpoint because
            # the symlink in ``out_dir``'s prefix vanished between startup
            # mkdir and end-of-epoch save. Idempotent and cheap; cost is
            # one stat call.
            out_dir.mkdir(parents=True, exist_ok=True)
            # ``last.pt`` is for resumption — carries raw (live) weights
            # plus the EMA shadow as a separate key. ``best.pt`` is for
            # inference — its ``dit`` key holds the EMA weights directly
            # so downstream loaders (eval_p3_subgoals, etc.) work unchanged.
            ckpt_state = {
                "dit": dit.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "val_cos": float(train_log.get("val_cos", float("nan"))),
            }
            if ema is not None:
                ckpt_state["ema"] = ema.state_dict()
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
                # Save EMA weights as the ``dit`` field of best.pt so
                # inference code loads the EMA-averaged checkpoint without
                # any change to its loader. The optimizer state is dropped
                # (best.pt is inference-only).
                if ema is not None:
                    best_state = {
                        **{k: v for k, v in ckpt_state.items()
                           if k not in {"dit", "optimizer", "ema"}},
                        "dit": ema.shadow,
                        "ema_decay": ema.decay,
                    }
                else:
                    best_state = ckpt_state
                torch.save(best_state, best_tmp)
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
