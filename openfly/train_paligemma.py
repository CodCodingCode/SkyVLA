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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from openfly.actions import (
    TRAINABLE_ACTION_IDS,
    TRAINABLE_ACTION_NAMES,
    action_id_to_logit_index,
)
from openfly.dataset import NUM_OPENFLY_ACTIONS, OpenFlyDataset, collate
from openfly.models.paligemma_vln import (
    PaliGemmaVLNPolicy,
    lora_and_head_param_groups,
)


def _build_processor(model_name: str):
    """Return the PaliGemma processor for instruction tokenisation."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name)


def _format_prompt(instruction: str, sub_instruction: str | None) -> str:
    """Compose the PaliGemma text prompt.

    The "Now:" line is only emitted when ``sub_instruction`` is non-empty
    so that inference paths without a high-level policy (which emit "")
    don't introduce a dangling label and surprise the model.
    """
    base = f"<image>\nTask: {instruction}"
    if sub_instruction:
        return f"{base}\nNow: {sub_instruction}"
    return base


def _tokenise_batch(
    processor,
    instructions: list[str],
    rgb_dummy: torch.Tensor,
    device: torch.device,
    max_length: int = 512,
    sub_instructions: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the PaliGemma processor on a batch of instructions.

    A dummy image is required because the processor expects vision input;
    the resulting image tokens are stripped downstream by
    ``PaliGemmaVLNPolicy._tokenize``.
    """
    if sub_instructions is None:
        sub_instructions = [""] * len(instructions)
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
    """Count actions over ``dataset._index`` as logit-index histograms.

    The dataset's index has already filtered to ``TRAINABLE_ACTION_IDS``,
    so every raw id we read here is guaranteed to remap cleanly.
    """
    counts = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
    episodes = dataset._episodes  # noqa: SLF001 — intentional read
    for ep_i, step in dataset._index:  # noqa: SLF001
        a = int(episodes[ep_i]["action"][step])
        if a in TRAINABLE_ACTION_IDS:
            counts[action_id_to_logit_index(a)] += 1
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


def _subset_sample_labels(
    train_subset: "Subset | OpenFlyDataset",
) -> np.ndarray:
    """Per-sample logit-index labels for a train Subset / Dataset.

    Reads ``parent._index[i]`` for each item in the subset and remaps the
    raw OpenFly action id to a logit index via ``action_id_to_logit_index``.
    Used to build the stratified ``WeightedRandomSampler``.
    """
    if isinstance(train_subset, Subset):
        parent = train_subset.dataset
        indices = train_subset.indices
    else:
        parent = train_subset
        indices = range(len(parent))
    labels = np.zeros(len(indices), dtype=np.int64)
    episodes = parent._episodes  # noqa: SLF001
    index = parent._index  # noqa: SLF001
    for k, i in enumerate(indices):
        ep_i, step = index[int(i)]
        labels[k] = action_id_to_logit_index(int(episodes[ep_i]["action"][step]))
    return labels


def _make_balanced_sampler(
    train_subset: "Subset | OpenFlyDataset",
    *,
    num_classes: int,
) -> tuple[WeightedRandomSampler, np.ndarray]:
    """Action-stratified ``WeightedRandomSampler`` over the train split.

    Per-sample sampling weight is ``1 / class_count(label)``, so each
    class is drawn with equal expected frequency. ``num_samples`` is set
    to ``len(train_subset)`` so each epoch sees the same volume as the
    plain ``shuffle=True`` loader (with replacement — minority classes
    get oversampled, ``fwd_9m`` undersampled). Returns the sampler and
    the per-class count tensor (for logging).
    """
    labels = _subset_sample_labels(train_subset)
    counts = np.bincount(labels, minlength=num_classes).astype(np.int64)
    safe = np.maximum(counts, 1).astype(np.float64)
    class_w = 1.0 / safe
    sample_w = class_w[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler, counts


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
    aux_progress_weight: float = 0.0,
    class_weights: torch.Tensor | None = None,
    use_progress: bool = True,
    use_sub_instruction: bool = True,
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
    progress = batch.get("progress")
    if progress is not None:
        progress = progress.to(device, non_blocking=True)
    # ``progress_target`` keeps the ground-truth for the stratified metrics
    # even when ``use_progress`` is off and we mask the model input.
    progress_target = progress
    if not use_progress:
        progress = None

    input_ids, attention_mask = _tokenise_batch(
        processor,
        batch["instruction"],
        rgb_dummy=rgb[0],
        device=device,
        sub_instructions=batch.get("sub_instruction") if use_sub_instruction else None,
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
    )
    logits = out["action_logits"]
    ce = F.cross_entropy(logits, actions, weight=class_weights)

    preds = logits.argmax(dim=-1)
    metrics: dict[str, Any] = {
        "ce": ce.item(),
        "acc": (preds == actions).float().mean().item(),
        "preds": preds.detach(),
        "actions": actions.detach(),
        # Always pass through the dataset's progress target so the trainer
        # can compute progress-bin-stratified stop-class metrics even when
        # ``use_progress`` is off (the baseline ablation arm).
        "progress_target": (
            progress_target.detach() if progress_target is not None else None
        ),
    }
    loss = ce

    if aux_goal_weight > 0 and "goal_pred" in out:
        # Body-frame delta to the NEXT pose (short-horizon, ~one step ahead).
        target = _body_frame_delta(pose, next_pose[:, :3])
        target = target.clamp(min=-20.0, max=20.0)
        l1 = F.smooth_l1_loss(out["goal_pred"], target)
        loss = loss + aux_goal_weight * l1
        metrics["goal_l1"] = l1.item()

    if (
        aux_progress_weight > 0
        and use_progress
        and "progress_pred" in out
        and progress_target is not None
    ):
        # Regress predicted progress against the dataset's ground-truth
        # path-fraction. Aux head reads from the no-progress feature
        # slice, so the supervision can't leak through its own input —
        # the gradient pushes ``scene + pose + last_action`` features
        # toward encoding trajectory phase.
        p_l1 = F.smooth_l1_loss(out["progress_pred"], progress_target)
        loss = loss + aux_progress_weight * p_l1
        metrics["progress_l1"] = p_l1.item()

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


# Progress-bin boundaries for the stop-class stratified metric. Three
# equal-width buckets over [0, 1]: early / mid / late phase of the
# trajectory. This is the key diagnostic for the progress-feature
# experiment — if the conditioning helps, the late bin's stop recall
# should rise substantially over the baseline arm.
_PROGRESS_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.34, "early"),
    (0.34, 0.67, "mid"),
    (0.67, 1.01, "late"),
)
N_PROGRESS_BINS: int = len(_PROGRESS_BINS)


def _progress_bin_index(progress: float) -> int:
    """Return 0 / 1 / 2 for early / mid / late."""
    for i, (lo, hi, _) in enumerate(_PROGRESS_BINS):
        if lo <= progress < hi:
            return i
    return N_PROGRESS_BINS - 1


def _accumulate_progress_stop(
    bin_samples: np.ndarray,           # (N_PROGRESS_BINS,)
    bin_stop_total: np.ndarray,        # (N_PROGRESS_BINS,) — gt action == 0
    bin_stop_correct: np.ndarray,      # (N_PROGRESS_BINS,) — pred == gt == 0
    bin_stop_predicted: np.ndarray,    # (N_PROGRESS_BINS,) — any pred == 0
    preds: torch.Tensor,
    actions: torch.Tensor,
    progress_target: torch.Tensor | None,
) -> None:
    """Bucket each step into early/mid/late and tally stop-class stats.

    Reports (per bin):
      * stop_recall   = correct / gt_stop_total  — does the model stop
                        when it should?  Late-bin recall is the headline.
      * stop_share    = gt_stop_total / samples — distribution of stops
                        across phases (concentrates in late by design).
      * stop_predicted = how often the model emits stop — useful for
                        spotting collapse to ``always stop``.

    When ``progress_target`` is None (e.g. legacy batches), the function
    is a no-op so callers can wire it unconditionally.
    """
    if progress_target is None:
        return
    preds_np = preds.cpu().numpy()
    actions_np = actions.cpu().numpy()
    prog_np = progress_target.cpu().numpy()
    for i in range(prog_np.shape[0]):
        b = _progress_bin_index(float(prog_np[i]))
        bin_samples[b] += 1
        if int(actions_np[i]) == 0:
            bin_stop_total[b] += 1
            if int(preds_np[i]) == 0:
                bin_stop_correct[b] += 1
        if int(preds_np[i]) == 0:
            bin_stop_predicted[b] += 1


def _format_progress_stop(
    name: str,
    bin_samples: np.ndarray,
    bin_stop_total: np.ndarray,
    bin_stop_correct: np.ndarray,
    bin_stop_predicted: np.ndarray,
) -> str:
    """One-line per-bin summary, mirrors ``_format_per_class``."""
    parts = []
    for b, (_, _, label) in enumerate(_PROGRESS_BINS):
        n = int(bin_samples[b])
        n_stop = int(bin_stop_total[b])
        n_pred = int(bin_stop_predicted[b])
        if n_stop > 0:
            recall = float(bin_stop_correct[b]) / n_stop
            recall_str = f"{recall:.2f}"
        else:
            recall_str = "NA"
        share = (n_stop / n) if n > 0 else 0.0
        parts.append(
            f"{label}=R{recall_str}(stops={n_stop}/{n} {100 * share:.1f}% "
            f"pred_stops={n_pred})"
        )
    return f"[progress-bin stop {name}] " + " ".join(parts)


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
    parser.add_argument(
        "--aux_progress_weight",
        type=float,
        default=0.1,
        help="Weight on the Smooth-L1 loss between the model's "
        "``progress_pred`` and the dataset's path-fraction target. "
        "Aux supervision; set 0 to disable.",
    )
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
    parser.add_argument(
        "--stratified_sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a ``WeightedRandomSampler`` over the train split so each "
        "minibatch sees a roughly action-balanced mix (combats OpenFly's "
        "~53%% fwd_9m dominance). When ON, --compute_class_weights is "
        "usually redundant — the sampler already corrects imbalance — so "
        "consider --no-compute_class_weights to avoid double-correction.",
    )
    parser.add_argument(
        "--use_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed the dataset's progress scalar into the action head "
        "(and into the aux progress regression). With --no-use_progress, "
        "the scalar is masked to None at the trainer boundary so "
        "progress_proj sees zeros and the aux progress loss is skipped. "
        "Use to compare baseline vs progress-conditioned model.",
    )
    parser.add_argument(
        "--use_sub_instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the templated sub-instruction to the PaliGemma prompt "
        "as a `Now: <sub>` line. With --no-use_sub_instruction the prompt "
        "collapses to the plain `Task: <ins>` form. Use to compare "
        "baseline vs sub-instruction-conditioned model.",
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
            f"{c}({TRAINABLE_ACTION_NAMES.get(c, str(c))})={weights[c]:.3f}"
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

    train_sampler: WeightedRandomSampler | None = None
    if args.stratified_sampler:
        train_sampler, strat_counts = _make_balanced_sampler(
            train_ds, num_classes=NUM_OPENFLY_ACTIONS
        )
        strat_counts_str = ", ".join(
            f"{c}({TRAINABLE_ACTION_NAMES.get(c, str(c))})={int(strat_counts[c])}"
            for c in range(NUM_OPENFLY_ACTIONS)
        )
        print(
            f"[train_paligemma] stratified sampler ON — pre-sampling "
            f"per-class counts: {strat_counts_str}"
        )
        if args.compute_class_weights:
            print(
                "[train_paligemma] WARNING: both --stratified_sampler and "
                "--compute_class_weights are ON. The sampler already "
                "balances classes; the CE weights will re-amplify rare "
                "classes (double-correction). Consider "
                "--no-compute_class_weights."
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    print(
        f"[train_paligemma] ablation flags: "
        f"use_progress={args.use_progress} "
        f"use_sub_instruction={args.use_sub_instruction} "
        f"aux_progress_weight={args.aux_progress_weight}"
    )

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        n, ce_sum, acc_sum, goal_sum, progress_sum = 0, 0.0, 0.0, 0.0, 0.0
        train_correct = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
        train_total = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
        # Progress-bin counters: early / mid / late.
        train_bin_samples = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
        train_bin_stop_total = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
        train_bin_stop_correct = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
        train_bin_stop_predicted = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
        for step, batch in enumerate(train_loader):
            _apply_warmup(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _step(
                model,
                batch,
                processor,
                device,
                args.aux_goal_weight,
                aux_progress_weight=args.aux_progress_weight,
                class_weights=class_weights_t,
                use_progress=args.use_progress,
                use_sub_instruction=args.use_sub_instruction,
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
            if "progress_l1" in metrics:
                progress_sum += metrics["progress_l1"] * bs
            _accumulate_per_class(
                train_correct, train_total, metrics["preds"], metrics["actions"]
            )
            _accumulate_progress_stop(
                train_bin_samples,
                train_bin_stop_total,
                train_bin_stop_correct,
                train_bin_stop_predicted,
                metrics["preds"],
                metrics["actions"],
                metrics.get("progress_target"),
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
                if "progress_l1" in metrics:
                    msg += f" progress={metrics['progress_l1']:.3f}"
                print(msg, flush=True)

        train_log: dict[str, Any] = {
            "epoch": epoch,
            "train_ce": ce_sum / max(n, 1),
            "train_acc": acc_sum / max(n, 1),
            "train_goal_l1": goal_sum / max(n, 1),
            "train_progress_l1": progress_sum / max(n, 1),
            "time_s": time.time() - t0,
            "train_per_class_correct": train_correct.tolist(),
            "train_per_class_total": train_total.tolist(),
            "train_progress_bin_samples": train_bin_samples.tolist(),
            "train_progress_bin_stop_total": train_bin_stop_total.tolist(),
            "train_progress_bin_stop_correct": train_bin_stop_correct.tolist(),
            "train_progress_bin_stop_predicted": train_bin_stop_predicted.tolist(),
        }
        print(_format_per_class("train", train_correct, train_total))
        print(
            _format_progress_stop(
                "train",
                train_bin_samples,
                train_bin_stop_total,
                train_bin_stop_correct,
                train_bin_stop_predicted,
            )
        )
        # Surface a few critical classes (stop + turns) explicitly.
        for c in (0, 2, 3):
            tot = int(train_total[c])
            cor = int(train_correct[c])
            acc = cor / tot if tot > 0 else float("nan")
            print(
                f"[train_paligemma] epoch {epoch} train class {c} "
                f"({TRAINABLE_ACTION_NAMES.get(c, str(c))}): {cor}/{tot} = {acc:.3f}"
            )

        if val_loader is not None:
            model.eval()
            val_correct = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
            val_total = np.zeros(NUM_OPENFLY_ACTIONS, dtype=np.int64)
            val_bin_samples = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
            val_bin_stop_total = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
            val_bin_stop_correct = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
            val_bin_stop_predicted = np.zeros(N_PROGRESS_BINS, dtype=np.int64)
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
                        use_progress=args.use_progress,
                        use_sub_instruction=args.use_sub_instruction,
                    )
                    bs = batch["action_id"].shape[0]
                    vn += bs
                    vce += m["ce"] * bs
                    vacc += m["acc"] * bs
                    _accumulate_per_class(
                        val_correct, val_total, m["preds"], m["actions"]
                    )
                    _accumulate_progress_stop(
                        val_bin_samples,
                        val_bin_stop_total,
                        val_bin_stop_correct,
                        val_bin_stop_predicted,
                        m["preds"],
                        m["actions"],
                        m.get("progress_target"),
                    )
            train_log["val_ce"] = vce / max(vn, 1)
            train_log["val_acc"] = vacc / max(vn, 1)
            train_log["val_per_class_correct"] = val_correct.tolist()
            train_log["val_per_class_total"] = val_total.tolist()
            train_log["val_progress_bin_samples"] = val_bin_samples.tolist()
            train_log["val_progress_bin_stop_total"] = val_bin_stop_total.tolist()
            train_log["val_progress_bin_stop_correct"] = val_bin_stop_correct.tolist()
            train_log["val_progress_bin_stop_predicted"] = val_bin_stop_predicted.tolist()
            print(_format_per_class("val", val_correct, val_total))
            print(
                _format_progress_stop(
                    "val",
                    val_bin_samples,
                    val_bin_stop_total,
                    val_bin_stop_correct,
                    val_bin_stop_predicted,
                )
            )
            for c in (0, 2, 3):
                tot = int(val_total[c])
                cor = int(val_correct[c])
                acc = cor / tot if tot > 0 else float("nan")
                print(
                    f"[train_paligemma] epoch {epoch} val class {c} "
                    f"({TRAINABLE_ACTION_NAMES.get(c, str(c))}): {cor}/{tot} = {acc:.3f}"
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
