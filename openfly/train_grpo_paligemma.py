#!/usr/bin/env python3
"""GRPO (Group-Relative Policy Optimisation) on PaliGemma BC.

Pattern follows the VLN-R1 / RFT recipe: per instruction sample
``K`` trajectories with the current policy, compute the group-relative
advantage on the episode reward from :mod:`openfly.rewards`, and apply
a clipped policy-gradient update with a KL anchor against a frozen
reference policy. LoRA + head parameters are updated; PaliGemma's
backbone stays frozen.

GRPO avoids fitting a value function (the per-group mean reward is the
baseline), which suits discrete-action VLN with sparse episode rewards
and slow AirSim rollouts.

Architecture sketch:

    for batch_step in range(steps):
      instructions = sample(B episodes)
      for each instruction:
        for k in range(K):
          rollout with sampling policy → trajectory
          score with rewards.compute_episode_reward
      group-relative advantage A_k = (R_k - mean(R)) / (std(R)+eps)
      loss = -sum_t advantage * log pi(a_t | s_t)
             + beta * KL(pi || pi_ref) on visited states
             + lambda * CE on a demo batch (optional anchor)
      step LoRA + head optimiser

Periodically writes a JSON metric log and the latest checkpoint;
running ``--eval_every`` triggers an in-loop eval on a small ``unseen``
subset using :func:`openfly.rollout.aggregate_metrics` for gating.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import (
    TRAINABLE_ACTION_IDS,
    action_id_to_logit_index,
    logit_index_to_action_id,
)
from openfly.dataset import NUM_OPENFLY_ACTIONS, OpenFlyDataset, collate
from openfly.envs import AirSimVLNEnv, AirSimVLNEnvConfig
from openfly.episodes import load_episodes
from openfly.models.paligemma_vln import (
    PaliGemmaVLNPolicy,
    lora_and_head_param_groups,
)
from openfly.rollout import (
    RolloutTrajectory,
    aggregate_metrics,
    collect_episode,
    save_trajectories_jsonl,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_prompt(instruction: str, sub_instruction: str | None) -> str:
    """Local copy of the trainer prompt template; keeps this module import-light."""
    base = f"<image>\nTask: {instruction}"
    if sub_instruction:
        return f"{base}\nNow: {sub_instruction}"
    return base


def _tokenise(
    processor,
    instruction: str,
    rgb: np.ndarray,
    device,
    max_length=256,
    sub_instruction: str | None = None,
):
    proc = processor(
        text=[_format_prompt(instruction, sub_instruction)],
        images=[rgb],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    return proc["input_ids"].to(device), proc["attention_mask"].to(device)


def _forward_logits(
    model: PaliGemmaVLNPolicy,
    processor,
    obs: dict,
    instruction: str,
    device: torch.device,
    *,
    with_grad: bool,
    progress: float | None = None,
    sub_instruction: str | None = None,
) -> torch.Tensor:
    """Single-step logits for one observation.

    ``last_action`` is sourced from the env's observation (``obs["last_action"]``
    — a raw OpenFly id; see :class:`openfly.envs.AirSimVLNEnv`). The env
    initialises this to ``-1`` at reset; ids outside the supervised set
    (including the reset value and any future strafes) fall back to the
    START sentinel (``NUM_OPENFLY_ACTIONS``) so the model's embedding
    table stays consistent between offline SFT and on-policy rollouts.
    ``next_pose`` is unknown at rollout time, so we pass the current pose
    (the model's ``goal_pred`` auxiliary output is not consumed during
    GRPO updates anyway).

    ``progress`` and ``sub_instruction`` are threaded through to match
    the SFT/eval conditioning. Callers that don't have them (e.g. the
    GRPO sampler when it doesn't know episode length) pass None — the
    model defaults progress to zeros and the prompt drops "Now:".
    """
    rgb_t = torch.from_numpy(obs["rgb"]).unsqueeze(0).to(device)
    hist_t = torch.from_numpy(obs["rgb_history"]).unsqueeze(0).to(device)
    pose_t = torch.from_numpy(obs["pose"]).unsqueeze(0).to(device)
    raw_la = int(obs.get("last_action", -1))
    if raw_la in TRAINABLE_ACTION_IDS:
        la_idx = action_id_to_logit_index(raw_la)
    else:
        la_idx = NUM_OPENFLY_ACTIONS  # START token
    last_action_t = torch.tensor([la_idx], dtype=torch.long, device=device)
    next_pose_t = pose_t.clone()
    progress_t: torch.Tensor | None = None
    if progress is not None:
        progress_t = torch.tensor([float(progress)], device=device, dtype=torch.float32)
    input_ids, attention_mask = _tokenise(
        processor, instruction, obs["rgb"], device, sub_instruction=sub_instruction
    )
    if with_grad:
        out = model(
            instruction_input_ids=input_ids,
            instruction_attention_mask=attention_mask,
            rgb_current=rgb_t,
            rgb_history=hist_t,
            pose=pose_t,
            last_action=last_action_t,
            next_pose=next_pose_t,
            progress=progress_t,
            with_grad=True,
        )
    else:
        with torch.no_grad():
            out = model(
                instruction_input_ids=input_ids,
                instruction_attention_mask=attention_mask,
                rgb_current=rgb_t,
                rgb_history=hist_t,
                pose=pose_t,
                last_action=last_action_t,
                next_pose=next_pose_t,
                progress=progress_t,
                with_grad=False,
            )
    return out["action_logits"]


def _step_progress(info: dict[str, Any], max_steps: int) -> float:
    """Budget-relative progress for on-policy rollouts."""
    step_idx = int(info.get("step_idx", 0))
    return min(1.0, float(step_idx) / float(max(max_steps - 1, 1)))


def _sampling_policy_fn(
    model: PaliGemmaVLNPolicy,
    processor,
    device: torch.device,
    *,
    temperature: float = 1.0,
    max_episode_steps: int = 60,
):
    """Build a ``policy_fn`` that samples from ``Categorical(logits)``."""

    def policy_fn(obs, info):
        with torch.no_grad():
            logits = _forward_logits(
                model,
                processor,
                obs,
                info.get("instruction", ""),
                device,
                with_grad=False,
                progress=_step_progress(info, max_episode_steps),
            )
            if temperature != 1.0:
                logits = logits / temperature
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logprob = dist.log_prob(a)
        # Env steps on raw OpenFly ids; convert at the boundary. The logit
        # index is recoverable from the raw id via ``action_id_to_logit_index``
        # downstream in the GRPO update, so we don't need to also record it.
        return logit_index_to_action_id(int(a.item())), {
            "logprob": float(logprob.item()),
            "logits": logits.detach().cpu(),
            # Record the per-step progress so the GRPO update can replay
            # the same conditioning without re-deriving it from a
            # potentially-stale ``info["step_idx"]``.
            "progress": _step_progress(info, max_episode_steps),
        }

    return policy_fn


def _eval_loop(
    env: AirSimVLNEnv,
    model: PaliGemmaVLNPolicy,
    processor,
    device: torch.device,
    *,
    n_episodes: int,
    max_episode_steps: int = 60,
) -> dict[str, float]:
    """Greedy rollout for gating; uses argmax not sampling."""

    def greedy_fn(obs, info):
        logits = _forward_logits(
            model,
            processor,
            obs,
            info.get("instruction", ""),
            device,
            with_grad=False,
            progress=_step_progress(info, max_episode_steps),
        )
        logit_idx = int(logits.argmax(dim=-1).item())
        return logit_index_to_action_id(logit_idx), {"source": "argmax"}

    model.eval()
    trajs = [collect_episode(env, greedy_fn, capture_obs=False) for _ in range(n_episodes)]
    model.train()
    return aggregate_metrics(trajs)


def _kl_term(
    logits_pi: torch.Tensor, logits_ref: torch.Tensor
) -> torch.Tensor:
    """KL(π || π_ref) for a single step's logits, in nats."""
    log_pi = F.log_softmax(logits_pi, dim=-1)
    p_pi = log_pi.exp()
    log_ref = F.log_softmax(logits_ref, dim=-1)
    return (p_pi * (log_pi - log_ref)).sum(dim=-1).mean()


def _compute_group_advantages(rewards: list[float]) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    mu = r.mean()
    sigma = r.std(unbiased=False).clamp(min=1e-3)
    return (r - mu) / sigma


def _demo_anchor_loss(
    model: PaliGemmaVLNPolicy,
    processor,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Standard CE on a batch from :class:`OpenFlyDataset`.

    Forwards the same ``last_action`` / ``next_pose`` / ``progress`` /
    ``sub_instruction`` fields the SFT trainer feeds so the action head
    sees identical conditioning across stages.
    """
    rgb = batch["rgb"].to(device, non_blocking=True)
    history = batch["history"].to(device, non_blocking=True)
    pose = batch["pose"].to(device, non_blocking=True)
    actions = batch["action_id"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    next_pose = batch["next_pose"].to(device, non_blocking=True)
    progress = batch.get("progress")
    if progress is not None:
        progress = progress.to(device, non_blocking=True)
    sub_instructions = batch.get("sub_instruction")
    if sub_instructions is None:
        prompts = [f"<image>\nTask: {ins}" for ins in batch["instruction"]]
    else:
        prompts = [
            _format_prompt(ins, sub)
            for ins, sub in zip(batch["instruction"], sub_instructions)
        ]

    proc = processor(
        text=prompts,
        images=[rgb[0].cpu().numpy()] * rgb.shape[0],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512,
    )
    input_ids = proc["input_ids"].to(device)
    attention_mask = proc["attention_mask"].to(device)
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
    return F.cross_entropy(out["action_logits"], actions)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init_ckpt", type=str, required=True, help="SFT PaliGemma checkpoint")
    p.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")

    p.add_argument("--steps", type=int, default=200, help="GRPO update steps")
    p.add_argument("--instructions_per_step", type=int, default=2)
    p.add_argument("--group_size", type=int, default=4, help="Trajectories per instruction (K)")
    p.add_argument("--max_episode_steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.02)
    p.add_argument("--demo_coef", type=float, default=0.05)
    p.add_argument("--demo_batch_size", type=int, default=4)
    p.add_argument("--clip_grad", type=float, default=1.0)

    p.add_argument("--lora_lr", type=float, default=1e-6)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--history_frames", type=int, default=2)
    p.add_argument("--env_filter", default="env_airsim_16")
    p.add_argument("--rollout_split", default="train")
    p.add_argument("--demo_split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out_dir",
        default=str(
            _DRONE_ROOT
            / "logs"
            / "openfly"
            / "grpo"
            / time.strftime("%Y%m%d_%H%M%S")
        ),
    )

    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_episodes", type=int, default=8)
    p.add_argument("--eval_split", default="unseen")
    p.add_argument(
        "--reward_preset",
        default=None,
        help="One of easy|medium|hard (see openfly.rewards.REWARD_PRESETS). "
        "When set, overrides the env's default RewardConfig and toggles "
        "dense progress shaping accordingly.",
    )
    p.add_argument(
        "--traj_category",
        default="",
        help="Optional trajectory-difficulty filter; episodes are kept only "
        "when their image_path contains this substring (e.g. 'short', "
        "'medium_average', 'long').",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[grpo] writing to {out_dir}")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.paligemma_model)

    model = PaliGemmaVLNPolicy(
        history_frames=args.history_frames,
        paligemma_model_name=args.paligemma_model,
    ).to(device)
    ckpt = torch.load(args.init_ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"[grpo] loaded {len(compatible)}/{len(state)} tensors from init ckpt")
    print(f"[grpo] loaded init ckpt {args.init_ckpt}")

    # Frozen reference for the KL anchor. We deep-copy on CPU first to dodge
    # GPU OOM on smaller cards; for GH200 unified-memory it's a no-op.
    ref_model = copy.deepcopy(model).to(device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    optimizer = torch.optim.AdamW(
        lora_and_head_param_groups(model, lora_lr=args.lora_lr, head_lr=args.head_lr)
    )

    train_env_cfg = AirSimVLNEnvConfig(
        split=args.rollout_split,
        env_filter=args.env_filter,
        max_steps=args.max_episode_steps,
        history_frames=args.history_frames,
        seed=args.seed,
        reward_preset=args.reward_preset,
    )
    train_env = AirSimVLNEnv(train_env_cfg)

    eval_env: AirSimVLNEnv | None = None
    if args.eval_every > 0:
        eval_env_cfg = AirSimVLNEnvConfig(
            split=args.eval_split,
            env_filter=args.env_filter,
            max_steps=args.max_episode_steps,
            history_frames=args.history_frames,
            seed=args.seed + 1,
            reward_preset=args.reward_preset,
        )
        eval_env = AirSimVLNEnv(eval_env_cfg)

    # Demo anchor: load offline OpenFlyDataset and stream batches.
    demo_loader = None
    if args.demo_coef > 0:
        demo_ds = OpenFlyDataset(
            split=args.demo_split,
            history_frames=args.history_frames,
            env_filter=args.env_filter,
        )
        if len(demo_ds) > 0:
            demo_loader = iter(
                torch.utils.data.DataLoader(
                    demo_ds,
                    batch_size=args.demo_batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    num_workers=0,
                    drop_last=True,
                )
            )

    train_episodes_meta = load_episodes(
        args.rollout_split,
        env_filter=args.env_filter,
    )
    if args.traj_category:
        n_before = len(train_episodes_meta)
        train_episodes_meta = [
            ep
            for ep in train_episodes_meta
            if args.traj_category in ep.get("image_path", "")
        ]
        print(
            f"[grpo] traj_category={args.traj_category!r} kept "
            f"{len(train_episodes_meta)}/{n_before} rollout episodes"
        )
    if not train_episodes_meta:
        raise RuntimeError("No training episodes for GRPO rollouts.")

    rng = np.random.default_rng(args.seed)
    sampling_fn = _sampling_policy_fn(
        model,
        processor,
        device,
        temperature=args.temperature,
        max_episode_steps=args.max_episode_steps,
    )

    log: list[dict[str, Any]] = []
    best_metric = -math.inf
    rollout_log_path = out_dir / "rollouts.jsonl"

    # Write a small manifest so curriculum-stage runs are easy to inspect.
    manifest = {
        "args": vars(args),
        "reward_preset": args.reward_preset or "default",
        "reward_config": train_env_cfg.reward_config.__dict__,
        "n_rollout_episodes": len(train_episodes_meta),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    for step in range(args.steps):
        t0 = time.time()
        group_rollouts: list[list[RolloutTrajectory]] = []
        chosen_episodes: list[dict[str, Any]] = []

        # 1. Roll K trajectories for each of `instructions_per_step` episodes.
        for _ in range(args.instructions_per_step):
            ep = train_episodes_meta[int(rng.integers(0, len(train_episodes_meta)))]
            chosen_episodes.append(ep)
            group = []
            for _k in range(args.group_size):
                traj = collect_episode(
                    train_env,
                    sampling_fn,
                    options={"episode": ep},
                    capture_obs=True,
                )
                group.append(traj)
            group_rollouts.append(group)
            save_trajectories_jsonl(group, rollout_log_path)

        # 2. Compute group-relative advantages.
        advantages_per_group: list[torch.Tensor] = []
        for group in group_rollouts:
            advantages_per_group.append(
                _compute_group_advantages([t.total_reward for t in group])
            )

        # 3. GRPO update with **incremental backward** to keep the
        #    activation graph bounded: a 3B PaliGemma + 20+ rollout
        #    timesteps would exceed 40 GB if we summed losses and called
        #    backward once. We pre-count contributions, scale each step's
        #    loss by 1/n_terms, and backward in-place so gradients
        #    accumulate without holding every step's graph in memory.
        n_terms = sum(
            min(len(traj.actions), len(traj.obs_rgb))
            for group in group_rollouts
            for traj in group
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0
        kl_running = 0.0
        rew_mean = 0.0
        rew_count = 0
        success_running = 0
        scale = 1.0 / max(n_terms, 1)

        for g_idx, group in enumerate(group_rollouts):
            adv_tensor = advantages_per_group[g_idx].to(device)
            for k_idx, traj in enumerate(group):
                rew_mean += traj.total_reward
                rew_count += 1
                success_running += int(traj.success)
                if not traj.obs_rgb:
                    continue
                advantage = adv_tensor[k_idx]
                for s_idx, action_id in enumerate(traj.actions):
                    if s_idx >= len(traj.obs_rgb):
                        break
                    obs = {
                        "rgb": traj.obs_rgb[s_idx],
                        "rgb_history": traj.obs_history[s_idx]
                        if s_idx < len(traj.obs_history)
                        else np.zeros_like(traj.obs_rgb[s_idx])[None],
                        "pose": traj.obs_poses[s_idx]
                        if s_idx < len(traj.obs_poses)
                        else np.zeros(4, dtype=np.float32),
                    }
                    # Recover the progress scalar captured at rollout time;
                    # falls back to the budget-relative ratio if the
                    # sampler didn't record one (e.g. older logs).
                    step_progress = float(
                        traj.extras[s_idx].get(
                            "progress",
                            min(1.0, float(s_idx) / float(max(args.max_episode_steps - 1, 1))),
                        )
                    )
                    logits_pi = _forward_logits(
                        model,
                        processor,
                        obs,
                        traj.instruction,
                        device,
                        with_grad=True,
                        progress=step_progress,
                    )
                    log_probs = F.log_softmax(logits_pi, dim=-1)
                    # ``action_id`` in traj is a raw OpenFly id (sim space);
                    # remap to the model's logit-index space before scoring.
                    a_idx = action_id_to_logit_index(int(action_id))
                    log_pi_a = log_probs[0, a_idx]
                    old_lp = float(traj.extras[s_idx].get("logprob", log_pi_a.detach().item()))
                    ratio = torch.exp(log_pi_a - old_lp)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                    ) * advantage
                    step_loss = -torch.min(unclipped, clipped)

                    if args.kl_coef > 0:
                        with torch.no_grad():
                            logits_ref = _forward_logits(
                                ref_model,
                                processor,
                                obs,
                                traj.instruction,
                                device,
                                with_grad=False,
                                progress=step_progress,
                            )
                        kl = _kl_term(logits_pi, logits_ref)
                        step_loss = step_loss + args.kl_coef * kl
                        kl_running += float(kl.detach().item())

                    (step_loss * scale).backward()
                    total_loss_val += float(step_loss.detach().item()) * scale

        demo_ce_val = 0.0
        if demo_loader is not None and args.demo_coef > 0:
            try:
                batch = next(demo_loader)
            except StopIteration:
                demo_ds = OpenFlyDataset(
                    split=args.demo_split,
                    history_frames=args.history_frames,
                    env_filter=args.env_filter,
                )
                demo_loader = iter(
                    torch.utils.data.DataLoader(
                        demo_ds,
                        batch_size=args.demo_batch_size,
                        shuffle=True,
                        collate_fn=collate,
                        num_workers=0,
                        drop_last=True,
                    )
                )
                batch = next(demo_loader)
            demo_ce = _demo_anchor_loss(model, processor, batch, device)
            (args.demo_coef * demo_ce).backward()
            demo_ce_val = float(demo_ce.detach().item())
            total_loss_val += args.demo_coef * demo_ce_val

        if n_terms > 0 or demo_loader is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip_grad
            )
            optimizer.step()

        step_log = {
            "step": step,
            "n_terms": n_terms,
            "loss": total_loss_val if n_terms > 0 else 0.0,
            "mean_reward": rew_mean / max(rew_count, 1),
            "success_rate": success_running / max(rew_count, 1),
            "kl": kl_running / max(n_terms, 1) if args.kl_coef > 0 else 0.0,
            "demo_ce": demo_ce_val,
            "elapsed_s": time.time() - t0,
        }
        log.append(step_log)
        print(
            f"[grpo] step {step:04d} | loss={step_log['loss']:+.3f} "
            f"R̄={step_log['mean_reward']:+.2f} SR={step_log['success_rate']:.2f} "
            f"KL={step_log['kl']:.3f} demo_ce={step_log['demo_ce']:.3f} "
            f"t={step_log['elapsed_s']:.1f}s"
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(log, f, indent=2)

        # 4. Periodic eval + checkpoint gating.
        if eval_env is not None and args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            eval_metrics = _eval_loop(
                eval_env,
                model,
                processor,
                device,
                n_episodes=args.eval_episodes,
                max_episode_steps=args.max_episode_steps,
            )
            print(f"[grpo] eval@{step}: {eval_metrics}")
            with open(out_dir / "eval.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **eval_metrics}) + "\n")
            score = eval_metrics["success_rate"] - 0.01 * eval_metrics["mean_ne_m"]
            if score > best_metric:
                best_metric = score
                torch.save(
                    {"model": model.state_dict(), "step": step, "args": vars(args)},
                    out_dir / "best.pt",
                )
                print(f"[grpo] new best (score={score:.3f}) → best.pt")

        torch.save(
            {"model": model.state_dict(), "step": step, "args": vars(args)},
            out_dir / "last.pt",
        )

    train_env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"\n[grpo] done. ckpts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
