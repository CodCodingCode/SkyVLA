#!/usr/bin/env python3
"""PPO + LoRA on the OpenFly-Agent 7B (OpenVLA-style VLA).

This is the heavier of the two RL pipelines. The 7B backbone stays
frozen; LoRA adapters on ``q_proj``/``v_proj`` plus a small value head
are the only trainable parameters (see
:class:`openfly.models.openfly_agent_rl.OpenFlyAgentRL`).

Why PPO and not GRPO? OpenVLA generates an 8-token action sequence
autoregressively, so the policy is naturally token-level. A clipped
PPO update with GAE on dense per-step rewards (progress shaping +
episode terms) gives a tighter signal than group-relative scoring of
discrete macros, and it amortises the cost of each AirSim rollout
across many gradient steps.

Layout:

1. Collect ``rollout_episodes`` trajectories with sampled actions; save
   per-step ``(obs, action_tokens, logprob, value, reward, done)``.
2. Compute GAE advantages and returns (gamma=0.99, lambda=0.95 by default).
3. Run ``ppo_epochs`` epochs of minibatch updates:
     * re-evaluate logprob+value+entropy under current policy,
     * clipped policy loss + MSE value loss + KL anchor to a frozen ref,
     * optional BC anchor: a small auxiliary 10-class head on the
       prefix's last hidden state, trained with CE against the
       goal-heuristic oracle's action id at each rollout state.
       Activated by ``--bc_coef > 0``; guards against forgetting the
       expert action distribution during PPO.
4. Save LoRA + value head deltas; periodic eval gates new checkpoints.

This file is a working scaffold — it depends on the OpenVLA HF wrapper
exposing ``predict_action`` + ``tokens_to_action`` (true for
``IPEC-COMMUNITY/openfly-agent-7b``). Different forks may need small
plumbing tweaks in ``models/openfly_agent_rl.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    goal_heuristic_action,
)
from openfly.envs import AirSimVLNEnv, AirSimVLNEnvConfig
from openfly.episodes import load_episodes
from openfly.rewards import DEFAULT_REWARD, RewardConfig, compute_step_progress
from openfly.rollout import RolloutTrajectory, aggregate_metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Standard GAE-lambda over one trajectory.

    Returns ``(advantages, returns)`` aligned with ``rewards``.
    """
    advantages: list[float] = [0.0] * len(rewards)
    returns: list[float] = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


def _collect_rollouts(
    env: AirSimVLNEnv,
    model,
    *,
    n_episodes: int,
    temperature: float,
    reward_config: RewardConfig,
    dense_progress: bool,
) -> list[dict[str, Any]]:
    """Roll out ``n_episodes`` with the current policy and capture per-step data.

    Returns a list of episode dicts with keys
    ``obs_rgb``, ``instructions``, ``histories``, ``action_tokens``,
    ``action_ids``, ``logprobs``, ``values``, ``rewards``, ``dones``,
    ``episode_return``, ``success``.
    """
    episodes: list[dict[str, Any]] = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        rgb_list: list[np.ndarray] = []
        instructions: list[str] = []
        history_list: list[list[int]] = []
        action_tokens: list[torch.Tensor] = []
        action_ids: list[int] = []
        expert_action_ids: list[int] = []
        logprobs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        history: list[int] = []
        prev_pose = list(obs["pose"].tolist())

        while True:
            rgb = obs["rgb"]
            action_id, lp, tokens, value = model.act_with_logprob(
                rgb,
                info.get("instruction", ""),
                history,
                temperature=temperature,
            )
            rgb_list.append(np.asarray(rgb, dtype=np.uint8))
            instructions.append(info.get("instruction", ""))
            history_list.append(list(history))
            action_tokens.append(tokens.cpu())
            action_ids.append(int(action_id))
            # BC anchor target: what the goal-heuristic oracle would have
            # picked at this state. Cheap to compute (no network).
            expert_action_ids.append(
                int(goal_heuristic_action(obs["pose"].tolist(), obs["goal"].tolist()))
            )
            logprobs.append(float(lp))
            values.append(float(value.item() if hasattr(value, "item") else value))

            next_obs, reward, terminated, truncated, info = env.step(int(action_id))
            # Add per-step progress shaping when env doesn't already.
            if dense_progress and not (terminated or truncated):
                reward += compute_step_progress(
                    prev_pose, next_obs["pose"].tolist(), next_obs["goal"].tolist(),
                    config=reward_config,
                )
            prev_pose = list(next_obs["pose"].tolist())
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
            history.append(int(action_id))

            if terminated or truncated:
                break
            obs = next_obs

        ep = {
            "obs_rgb": rgb_list,
            "instructions": instructions,
            "histories": history_list,
            "action_tokens": action_tokens,
            "action_ids": action_ids,
            "expert_action_ids": expert_action_ids,
            "logprobs": logprobs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "episode_return": float(sum(rewards)),
            "success": bool(info.get("success", False)),
            "ne_m": float(info.get("d_final", info.get("distance_to_goal", 0.0))),
            "steps": len(rewards),
        }
        episodes.append(ep)
    return episodes


def _ppo_update(
    model,
    ref_model,
    episodes: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
    bc_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    clip_grad: float,
    device: torch.device,
    gamma: float,
    lam: float,
) -> dict[str, float]:
    flat: list[dict[str, Any]] = []
    for ep in episodes:
        adv, ret = _compute_gae(
            ep["rewards"], ep["values"], ep["dones"], gamma=gamma, lam=lam
        )
        for i in range(len(ep["rewards"])):
            flat.append(
                {
                    "rgb": ep["obs_rgb"][i],
                    "instruction": ep["instructions"][i],
                    "history": ep["histories"][i],
                    "tokens": ep["action_tokens"][i],
                    "expert_action_id": ep["expert_action_ids"][i],
                    "old_logprob": ep["logprobs"][i],
                    "old_value": ep["values"][i],
                    "advantage": adv[i],
                    "return": ret[i],
                }
            )
    if not flat:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl_ref": 0.0, "bc_loss": 0.0}

    # Advantage normalisation across all transitions.
    advs = torch.tensor([f["advantage"] for f in flat], dtype=torch.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, f in enumerate(flat):
        f["advantage"] = float(advs[i].item())

    sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl_ref": 0.0,
        "approx_kl": 0.0,
        "bc_loss": 0.0,
    }
    n_updates = 0
    rng = np.random.default_rng()

    for _epoch in range(ppo_epochs):
        order = rng.permutation(len(flat))
        for s in range(0, len(flat), minibatch_size):
            mb = [flat[int(idx)] for idx in order[s : s + minibatch_size]]
            optimizer.zero_grad(set_to_none=True)
            mb_policy = torch.zeros((), device=device)
            mb_value = torch.zeros((), device=device)
            mb_kl = torch.zeros((), device=device)
            mb_ent = torch.zeros((), device=device)
            mb_bc = torch.zeros((), device=device)
            for t in mb:
                lp, ent, val = model.evaluate_actions(
                    t["rgb"], t["instruction"], t["history"], t["tokens"]
                )
                ratio = torch.exp(lp - t["old_logprob"])
                adv = torch.tensor(t["advantage"], device=device, dtype=lp.dtype)
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * adv
                policy_loss = -torch.min(unclipped, clipped)
                value_loss = (val - t["return"]) ** 2

                kl_ref = torch.tensor(0.0, device=device)
                if kl_coef > 0:
                    with torch.no_grad():
                        ref_lp, _, _ = ref_model.evaluate_actions(
                            t["rgb"], t["instruction"], t["history"], t["tokens"]
                        )
                    kl_ref = lp - ref_lp  # log-ratio surrogate KL

                bc_loss = torch.zeros((), device=device)
                if bc_coef > 0:
                    bc_logits = model.forward_bc_logits(
                        t["rgb"], t["instruction"], t["history"]
                    )
                    # ``expert_action_id`` is a raw OpenFly id from
                    # ``goal_heuristic_action`` (always in {0,1,2,3}), but
                    # the BC head outputs ``NUM_OPENFLY_ACTIONS`` (=8)
                    # logit-index classes — remap before CE. Any expert
                    # outside the supervised set (defensive) skips the BC
                    # term for that step.
                    expert_raw = int(t["expert_action_id"])
                    if expert_raw in TRAINABLE_ACTION_IDS:
                        target = torch.tensor(
                            action_id_to_logit_index(expert_raw),
                            device=device,
                            dtype=torch.long,
                        )
                        bc_loss = F.cross_entropy(
                            bc_logits.unsqueeze(0), target.unsqueeze(0)
                        )

                mb_policy = mb_policy + policy_loss
                mb_value = mb_value + value_loss
                mb_kl = mb_kl + kl_ref
                mb_ent = mb_ent + ent
                mb_bc = mb_bc + bc_loss
            mb_policy = mb_policy / max(len(mb), 1)
            mb_value = mb_value / max(len(mb), 1)
            mb_kl = mb_kl / max(len(mb), 1)
            mb_ent = mb_ent / max(len(mb), 1)
            mb_bc = mb_bc / max(len(mb), 1)

            loss = (
                mb_policy
                + value_coef * mb_value
                - entropy_coef * mb_ent
                + kl_coef * mb_kl
                + bc_coef * mb_bc
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip_grad
            )
            optimizer.step()

            sums["loss"] += float(loss.detach().item())
            sums["policy_loss"] += float(mb_policy.detach().item())
            sums["value_loss"] += float(mb_value.detach().item())
            sums["entropy"] += float(mb_ent.detach().item())
            sums["kl_ref"] += float(mb_kl.detach().item())
            sums["bc_loss"] += float(mb_bc.detach().item())
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in sums.items()}


def _greedy_eval(env: AirSimVLNEnv, model, *, n_episodes: int) -> dict[str, float]:
    from openfly.rollout import collect_episode

    def policy_fn(obs, info):
        action = model.act(
            obs["rgb"], info.get("instruction", ""), history=[], do_sample=False
        )
        return int(action), {"source": "argmax"}

    trajs: list[RolloutTrajectory] = [
        collect_episode(env, policy_fn, capture_obs=False) for _ in range(n_episodes)
    ]
    return aggregate_metrics(trajs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", default="IPEC-COMMUNITY/openfly-agent-7b")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)

    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--episodes_per_iter", type=int, default=4)
    p.add_argument("--max_episode_steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--minibatch_size", type=int, default=4)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--kl_coef", type=float, default=0.02)
    p.add_argument(
        "--bc_coef",
        type=float,
        default=0.0,
        help="Weight on the auxiliary BC (CE-to-expert-action) head loss. "
        "Set >0 to anchor the LoRA-updated representation against the "
        "goal-heuristic oracle's action ids — guards against the policy "
        "forgetting expert behaviour during PPO. Typical: 0.01–0.1.",
    )
    p.add_argument(
        "--bc_lr",
        type=float,
        default=None,
        help="LR for the BC head only. Defaults to --value_lr.",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--dense_progress", action="store_true", help="Enable step shaping reward")
    p.add_argument(
        "--reward_preset",
        default=None,
        help="Optional curriculum preset (easy|medium|hard). Overrides the env "
        "RewardConfig and forces --dense_progress on for 'easy'.",
    )

    p.add_argument("--lora_lr", type=float, default=1e-5)
    p.add_argument("--value_lr", type=float, default=3e-4)
    p.add_argument("--env_filter", default="env_airsim_16")
    p.add_argument("--rollout_split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_split", default="unseen")

    p.add_argument(
        "--out_dir",
        default=str(
            _DRONE_ROOT
            / "logs"
            / "openfly"
            / "ppo_agent"
            / time.strftime("%Y%m%d_%H%M%S")
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ppo-agent] writing to {out_dir}")

    from openfly.models.openfly_agent_rl import OpenFlyAgentRL

    model = OpenFlyAgentRL(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=str(device),
    )
    print("[ppo-agent] cloning frozen reference for KL anchor")
    ref_model = OpenFlyAgentRL(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=str(device),
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    optimizer = torch.optim.AdamW(
        model.trainable_param_groups(
            lora_lr=args.lora_lr,
            value_lr=args.value_lr,
            bc_lr=args.bc_lr,
        )
    )

    train_env_cfg = AirSimVLNEnvConfig(
        split=args.rollout_split,
        env_filter=args.env_filter,
        max_steps=args.max_episode_steps,
        seed=args.seed,
        reward_preset=args.reward_preset,
    )
    train_env = AirSimVLNEnv(train_env_cfg)
    # Honour the preset's dense-progress flag if it was set.
    args.dense_progress = args.dense_progress or train_env_cfg.dense_progress
    eval_env: AirSimVLNEnv | None = None
    if args.eval_every > 0:
        eval_env = AirSimVLNEnv(
            AirSimVLNEnvConfig(
                split=args.eval_split,
                env_filter=args.env_filter,
                max_steps=args.max_episode_steps,
                seed=args.seed + 1,
                reward_preset=args.reward_preset,
            )
        )

    history_log: list[dict[str, Any]] = []
    best_metric = -math.inf

    for it in range(args.iterations):
        t0 = time.time()
        episodes = _collect_rollouts(
            train_env,
            model,
            n_episodes=args.episodes_per_iter,
            temperature=args.temperature,
            reward_config=train_env_cfg.reward_config,
            dense_progress=args.dense_progress,
        )
        update_stats = _ppo_update(
            model,
            ref_model,
            episodes,
            optimizer,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            bc_coef=args.bc_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_grad=args.clip_grad,
            device=device,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )
        mean_ret = float(np.mean([ep["episode_return"] for ep in episodes]))
        success = float(np.mean([int(ep["success"]) for ep in episodes]))
        ne = float(np.mean([ep["ne_m"] for ep in episodes]))

        log_row = {
            "iteration": it,
            "mean_return": mean_ret,
            "success_rate": success,
            "mean_ne_m": ne,
            "elapsed_s": time.time() - t0,
            **update_stats,
        }
        history_log.append(log_row)
        print(
            f"[ppo-agent] it={it:03d} R̄={mean_ret:+.2f} SR={success:.2f} NE={ne:.1f}m "
            f"loss={update_stats['loss']:+.3f} v_loss={update_stats['value_loss']:.3f} "
            f"ent={update_stats['entropy']:.3f} kl={update_stats['kl_ref']:+.3f} "
            f"bc={update_stats['bc_loss']:.3f} "
            f"t={log_row['elapsed_s']:.1f}s"
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(history_log, f, indent=2)

        if eval_env is not None and (it + 1) % args.eval_every == 0:
            metrics = _greedy_eval(eval_env, model, n_episodes=args.eval_episodes)
            print(f"[ppo-agent] eval@{it}: {metrics}")
            with open(out_dir / "eval.jsonl", "a") as f:
                f.write(json.dumps({"iteration": it, **metrics}) + "\n")
            score = metrics["success_rate"] - 0.01 * metrics["mean_ne_m"]
            if score > best_metric:
                best_metric = score
                trainable_state = {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                    if any(
                        sub in k
                        for sub in ("lora_", "value_head", "bc_head")
                    )
                }
                torch.save(
                    {
                        "lora_state": trainable_state,
                        "iteration": it,
                        "args": vars(args),
                    },
                    out_dir / "best.pt",
                )
                print(f"[ppo-agent] new best ({score:.3f}) → best.pt")

        # Always save last (LoRA + value head only — backbone is frozen).
        last_state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if any(sub in k for sub in ("lora_", "value_head", "bc_head"))
        }
        torch.save(
            {"lora_state": last_state, "iteration": it, "args": vars(args)},
            out_dir / "last.pt",
        )

    train_env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"\n[ppo-agent] done. checkpoints in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
