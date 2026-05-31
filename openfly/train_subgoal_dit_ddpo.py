#!/usr/bin/env python3
"""DDPO / GRPO reinforcement-learning fine-tune of the PixArt subgoal world model.

This is the *training harness* around the tested RL core in
:mod:`openfly.dit_ddpo`. The math (stochastic-DDIM-as-MDP, per-step
Gaussian log-probs, GRPO advantages, clipped PPO surrogate, closed-form
KL-to-reference, reward models) lives there and is imported, never
re-implemented here. This file only does the plumbing: build the models,
encode batches into conditioning, draw GRPO groups, score them, run the
PPO update, and handle checkpoint/resume/W&B/diagnostics — mirroring
``openfly.train_subgoal_dit`` (the SFT trainer) conventions throughout.

The RL loop (one optimizer step per dataset batch of ``B`` contexts)
---------------------------------------------------------------------
  1. Encode the batch (frozen PaliGemma) -> curr_tokens (B,256,2048),
     text_embed (B,2048), pose_delta (B,4), last_action (B,), horizon (B,).
     Build a :class:`dit_ddpo.DiTConds`. Invalid (terminal) contexts are
     dropped via ``subgoal_valid`` before anything else.
  2. ``conds_g = conds.repeat(group_size)`` draws the GRPO group
     (``repeat_interleave`` layout). The reward's batch fields are
     repeated with the SAME interleave so trajectory ``k`` aligns with
     its conditioning + ground-truth action.
  3. ``sample_with_logprob(dit, conds_g, cfg)`` rolls out the K-step
     stochastic-DDIM chain under ``no_grad`` (dit in eval -> NO CFG
     dropout), storing per-step states and the detached "old" log-probs.
  4. The reward model scores each generated ``final_subgoal`` (policy
     log-likelihood of the GT action, simulator-free by default), and
     ``group_advantages`` standardises rewards within each group.
  5. For ``ppo_epochs`` inner passes over that SAME rollout buffer:
     recompute log-probs WITH grad (``trajectory_logprob``), form the
     clipped surrogate (``ppo_step_loss``), add ``kl_coef * KL`` to the
     frozen reference (``kl_to_reference``), backprop, NaN/Inf-guard,
     clip grad norm, step, EMA-update.

Train / inference parity
------------------------
* The DiT stays in ``.eval()`` for BOTH sampling and the PPO recompute.
  Sampling must avoid the model's CFG conditioning-dropout (it would add
  un-modelled randomness to the rollout); the recompute must use the same
  deterministic forward so old/new log-probs are comparable. ``eval`` for
  both is intentional and correct here — the DiT has no BatchNorm.
* ``num_sample_steps`` defaults to 4, matching the policy's deployment
  ``subgoal_sample_steps`` and the SFT ``--val_ddim_steps`` — we optimise
  the chain the policy actually consumes at inference time.

Reward (``--reward policy_likelihood``, default; simulator-free)
---------------------------------------------------------------
We wrap :class:`dit_ddpo.PolicyLikelihoodReward` with a CUSTOM ``score_fn``
(built by ``_make_policy_score_fn``) because the real
``PaliGemmaVLNPolicy.forward`` needs the full observation (RGB frames,
pose, instruction ids, ...) alongside ``subgoal_tokens`` — the library's
default extractor only passes ``subgoal_tokens`` and would not satisfy
that signature. The custom fn feeds the generated subgoal in via the
policy's documented ``subgoal_tokens=`` path, reads ``action_logits``
(N, 8), and returns the per-sample log-prob of the GT action
(``batch['action_id']``, already a logit index) == ``-cross_entropy``.
Higher => the subgoal makes the frozen policy more confident in the
expert action => a more useful world-model output.

  NOTE on the GT action key: the OpenFly collate emits the supervised
  label under ``action_id`` (a logit index in [0, 8)); the raw ``action``
  list from the annotation is NOT in the batch. We therefore read
  ``action_id`` and pass ``action_key='action_id'`` to the reward.

``--reward rollout`` is a hook for a true AirSim episode reward; it
requires a working ``reward_fn`` and is NOT wired to a simulator here
(AirSim availability is unconfirmed in this environment), so it raises
unless one is supplied. ``policy_likelihood`` is the safe default.

Run via :file:`openfly/run_train_subgoal_dit_ddpo.sh`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from openfly.dataset import OpenFlyDataset, collate
from openfly.dit_ddpo import (
    DDPOConfig,
    DiTConds,
    PolicyLikelihoodReward,
    RewardModel,
    RolloutReward,
    group_advantages,
    kl_to_reference,
    ppo_step_loss,
    sample_with_logprob,
    trajectory_logprob,
)
from openfly.models.paligemma_vln import PaliGemmaVLNPolicy
from openfly.models.subgoal_dit_pixart import PixArtSubgoalDiT

# Reuse the SFT trainer's battle-tested helpers verbatim (they are
# module-level and side-effect-free at import — verified). This keeps the
# PaliGemma encoding, tokenisation, pose-delta, EMA, checkpoint and
# diagnostics behaviour byte-identical between the SFT and RL trainers.
from openfly.train_subgoal_dit import (
    EMA,
    _atomic_save,
    _body_frame_pose_delta,
    _build_processor,
    _encode_frame_pair,
    _install_diagnostics,
    _tokenise_batch,
)
from vla.vla_policy import PaliGemmaFeatureExtractor

_TAG = "[train_subgoal_dit_ddpo]"

DEFAULT_INIT_DIT_CKPT = (
    "/home/ubuntu/SkyVLA/logs/openfly/subgoal_dit/B_pooled_cosloss_1p0/best.pt"
)
# PixArtSubgoalDiT calls ``from_pretrained(pretrained_path, subfolder="transformer")``,
# so this points at the snapshot DIR (not the nested transformer/). Same value
# the SFT run used (see its args.json). Override with --pretrained_path if the
# local HF snapshot hash differs.
DEFAULT_PRETRAINED_PATH = (
    str(Path.home() / "assets" / "pretrained" / "hf_cache"
        / "models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS"
        / "snapshots" / "76fb7eb5a9314bc1e4e479d2f13447517fca9be4")
)
# The collate's ground-truth supervised label (logit index in [0, 8)).
GT_ACTION_KEY = "action_id"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _shape_filtered_load(
    module: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """``load_state_dict(strict=False)`` after dropping shape-mismatched keys.

    An SFT ``best.pt`` may carry buffers / heads whose shapes differ from a
    freshly-built model, or extra/missing keys. We keep only keys present in
    the target with a matching shape, then load non-strict so missing /
    unexpected keys are tolerated. Returns ``(n_loaded, n_skipped)``.
    """
    own = module.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1
    module.load_state_dict(filtered, strict=False)
    return len(filtered), skipped


def _build_dit(args: argparse.Namespace, device: torch.device) -> PixArtSubgoalDiT:
    """Construct the PixArtSubgoalDiT exactly as the SFT trainer does.

    CFG conditioning-dropout is left OFF (defaults 0.0): RL rollouts must
    not inject extra randomness, and we want a deterministic forward for
    the PPO log-prob recompute. ``subgoal_pool='mean'`` -> ``subgoal_len==1``.
    REPA is off (no aux loss in RL).
    """
    return PixArtSubgoalDiT(
        pretrained_path=args.pretrained_path,
        token_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
        text_dim=PaliGemmaFeatureExtractor.FEATURE_DIM,
        pose_delta_dim=4,
        num_last_actions=9,
        num_timesteps=args.num_timesteps,
        freeze_backbone=args.freeze_backbone,
        dropout=0.0,
        cfg_drop_text=0.0,
        cfg_drop_pose=0.0,
        cfg_drop_action=0.0,
        cfg_drop_horizon=0.0,
        cfg_drop_joint=0.0,
        repa_layer_idx=0,
        subgoal_pool=args.subgoal_pool,
    ).to(device)


def _build_ckpt_state(
    dit: PixArtSubgoalDiT,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    global_step: int,
    args: argparse.Namespace,
    *,
    best_reward: float | None = None,
) -> dict[str, Any]:
    """Bundle full RL training state for resumption (mirrors the SFT shape)."""
    state: dict[str, Any] = {
        "dit": dit.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "args": vars(args),
    }
    if best_reward is not None:
        state["best_reward"] = float(best_reward)
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


# ---------------------------------------------------------------------------
# Conditioning construction (frozen PaliGemma) — mirrors SFT _train_step up
# to the DiT inputs, but produces a dit_ddpo.DiTConds over the valid rows.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_conds(
    dit: PixArtSubgoalDiT,
    paligemma: PaliGemmaFeatureExtractor,
    processor: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[DiTConds | None, torch.Tensor]:
    """Encode one dataset batch into DiT conditioning + a validity mask.

    Returns ``(conds, valid_mask)`` where ``conds`` is built over the VALID
    sub-batch only (invalid/terminal contexts carry no usable subgoal and
    are dropped) and ``valid_mask`` (B,) bool indexes the original rows that
    survived. ``conds`` is ``None`` if no row is valid (caller skips the
    step). Runs fully under ``no_grad`` — the encoder is frozen and the
    conditioning carries no policy gradient.
    """
    rgb = batch["rgb"].to(device, non_blocking=True)
    subgoal_rgb = batch["subgoal_rgb"].to(device, non_blocking=True)
    pose = batch["pose"].to(device, non_blocking=True)
    subgoal_pose = batch["subgoal_pose"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    horizon = batch["subgoal_horizon"].to(device, non_blocking=True)
    valid = batch["subgoal_valid"].to(device, non_blocking=True).bool()

    if not bool(valid.any()):
        return None, valid

    input_ids, attention_mask = _tokenise_batch(
        processor,
        batch["instruction"],
        batch["sub_instruction"],
        rgb_dummy=rgb[0],
        device=device,
    )

    # Single stacked PaliGemma forward over (curr, subgoal). We only need
    # curr_tokens + text_embed here; the subgoal frame's tokens are the SFT
    # reconstruction target, unused by RL (the reward is policy-likelihood,
    # not feature reconstruction).
    curr_tokens, _tgt_tokens, text_embed = _encode_frame_pair(
        paligemma, rgb, subgoal_rgb, input_ids, attention_mask
    )
    pose_delta = _body_frame_pose_delta(pose, subgoal_pose).to(device)

    # Restrict everything to valid rows so the GRPO groups and reward only
    # ever see usable contexts.
    sel = valid.nonzero(as_tuple=False).squeeze(-1)
    conds = DiTConds(
        curr_tokens=curr_tokens.index_select(0, sel),
        text_embed=text_embed.index_select(0, sel),
        pose_delta=pose_delta.index_select(0, sel),
        last_action=last_action.index_select(0, sel).long(),
        horizon=horizon.index_select(0, sel).long(),
        subgoal_len=int(dit.subgoal_len),
    )
    return conds, valid


def _repeat_reward_batch(
    batch: dict[str, Any],
    valid: torch.Tensor,
    group_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Select valid rows of the reward-relevant batch fields, then
    ``repeat_interleave(group_size)`` so they align with ``conds.repeat(g)``.

    Only fields the reward's ``score_fn`` consumes are carried (current RGB,
    history frames under key ``history``, pose, next_pose, last_action, GT
    ``action_id``, instruction text). Tensors move to ``device``; string
    fields (instructions) expand as Python lists with the same interleave.
    """
    sel = valid.nonzero(as_tuple=False).squeeze(-1)
    sel_cpu = sel.cpu()

    def _rep_tensor(x: torch.Tensor) -> torch.Tensor:
        x = x.index_select(0, sel.to(x.device))
        return x.repeat_interleave(group_size, dim=0).to(device, non_blocking=True)

    def _rep_list(xs: list[Any]) -> list[Any]:
        picked = [xs[i] for i in sel_cpu.tolist()]
        out: list[Any] = []
        for v in picked:
            out.extend([v] * group_size)
        return out

    rep: dict[str, Any] = {}
    for key in ("rgb", "history", "pose", "next_pose", "last_action", GT_ACTION_KEY):
        if key in batch and isinstance(batch[key], torch.Tensor):
            rep[key] = _rep_tensor(batch[key])
    for key in ("instruction", "sub_instruction"):
        if key in batch:
            rep[key] = _rep_list(batch[key])
    return rep


# ---------------------------------------------------------------------------
# Reward: policy log-likelihood of the ground-truth action given the subgoal
# ---------------------------------------------------------------------------

def _make_policy_score_fn(paligemma_processor: Any, device: torch.device):
    """Build the custom ``score_fn`` for :class:`PolicyLikelihoodReward`.

    Closure over the processor/device so the inner fn matches the
    ``score_fn(policy, final_subgoal, batch) -> (N,)`` contract. It runs the
    FROZEN policy through its real forward, supplying the generated subgoal
    via ``subgoal_tokens=`` and the full observation from the (group-
    repeated) batch, and returns the GT-action log-prob (``-CE``).
    """

    @torch.no_grad()
    def _score(policy: PaliGemmaVLNPolicy, final_subgoal: torch.Tensor,
               batch: dict[str, Any]) -> torch.Tensor:
        n = int(final_subgoal.shape[0])
        gt = batch[GT_ACTION_KEY].to(device).long().reshape(-1)  # (N,) logit idx

        rgb_current = batch["rgb"].to(device)
        rgb_history = batch.get("history")
        if rgb_history is None or rgb_history.numel() == 0:
            rgb_history = torch.empty(0, device=device)
        else:
            rgb_history = rgb_history.to(device)
        pose = batch["pose"].to(device).float()
        next_pose = batch.get("next_pose")
        next_pose = (next_pose.to(device).float() if next_pose is not None
                     else pose)
        last_action = batch["last_action"].to(device).long()

        # The policy owns its own PaliGemma; tokenise the (group-repeated)
        # instructions with the shared processor so ids/mask align row-wise.
        input_ids, attention_mask = _tokenise_batch(
            paligemma_processor,
            batch["instruction"],
            batch["sub_instruction"],
            rgb_dummy=rgb_current[0],
            device=device,
        )

        out = policy(
            instruction_input_ids=input_ids,
            instruction_attention_mask=attention_mask,
            rgb_current=rgb_current,
            rgb_history=rgb_history,
            pose=pose,
            last_action=last_action,
            next_pose=next_pose,
            with_grad=False,
            subgoal_tokens=final_subgoal.to(device),
        )
        logits = out["action_logits"].float()                 # (N, 8)
        log_probs = torch.log_softmax(logits, dim=-1)
        # log-likelihood of the GT action == -cross_entropy
        reward = log_probs.gather(-1, gt.unsqueeze(-1)).squeeze(-1)  # (N,)
        return reward.float().reshape(n)

    return _score


def _build_policy(args: argparse.Namespace, device: torch.device) -> PaliGemmaVLNPolicy:
    """Construct the frozen BC policy used by the likelihood reward.

    Built WITHOUT an internal ``subgoal_dit`` (subgoals are injected via the
    ``subgoal_tokens=`` path) and with history dropout off. Hyper-params
    mirror the P3 ``paligemma_subgoal`` checkpoint (history_frames=2,
    lora_rank=16, lora_alpha=32). The ckpt's policy weights live under the
    ``"model"`` key (per eval_p3_subgoals.py); loaded shape-filtered /
    non-strict, then everything is frozen + eval.
    """
    policy = PaliGemmaVLNPolicy(
        history_frames=args.policy_history_frames,
        embed_dim=args.policy_embed_dim,
        paligemma_model_name=args.paligemma_model,
        lora_rank=args.policy_lora_rank,
        lora_alpha=args.policy_lora_alpha,
        aux_goal_head=args.policy_aux_goal_head,
        aux_progress_head=args.policy_aux_progress_head,
        subgoal_dit=None,
        history_dropout_p=0.0,
    ).to(device)

    if args.policy_ckpt:
        ckpt = torch.load(args.policy_ckpt, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model", ckpt.get("policy", ckpt))
        else:
            sd = ckpt
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        n_load, n_skip = _shape_filtered_load(policy, sd)
        print(f"{_TAG} loaded policy ckpt {args.policy_ckpt} "
              f"({n_load} tensors, {n_skip} skipped)")
    else:
        print(f"{_TAG} WARNING: no --policy_ckpt given; reward uses an "
              "UNTRAINED BC policy (rewards will be near-uniform).")

    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _build_reward_model(
    args: argparse.Namespace, device: torch.device, processor: Any
) -> RewardModel:
    if args.reward == "policy_likelihood":
        policy = _build_policy(args, device)
        score_fn = _make_policy_score_fn(processor, device)
        return PolicyLikelihoodReward(
            policy, score_fn=score_fn, action_key=GT_ACTION_KEY
        )
    if args.reward == "rollout":
        # No confirmed simulator in this environment — RolloutReward with no
        # reward_fn raises on .score(), surfacing the misconfiguration loudly
        # rather than silently training on a fake signal.
        return RolloutReward(reward_fn=None)
    raise ValueError(f"unknown --reward {args.reward!r}")


# ---------------------------------------------------------------------------
# Argparse (mirror SFT naming + DDPO-specific args)
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    # ---- init / pretrained ----
    p.add_argument("--init_dit_ckpt", type=str, default=DEFAULT_INIT_DIT_CKPT,
                   help="SFT best.pt to initialise the trainable (and "
                   "reference) DiT from. Loaded shape-filtered/non-strict.")
    p.add_argument("--pretrained_path", type=str, default=DEFAULT_PRETRAINED_PATH,
                   help="PixArt-Σ HF snapshot dir (same as the SFT used).")

    # ---- DDPO / GRPO core ----
    p.add_argument("--eta", type=float, default=1.0,
                   help="DDIM stochasticity. 1.0 = full exploration noise.")
    p.add_argument("--num_sample_steps", type=int, default=4,
                   help="K = MDP horizon = DDIM reverse steps (deploy=4).")
    p.add_argument("--sigma_min", type=float, default=1e-3,
                   help="Per-step std floor so log-probs stay finite.")
    p.add_argument("--group_size", type=int, default=8,
                   help="G trajectories per context for the GRPO baseline.")
    p.add_argument("--ppo_epochs", type=int, default=4,
                   help="Inner PPO passes over each rollout buffer.")
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.1,
                   help="Weight on the KL-to-reference regulariser.")
    p.add_argument("--entropy_coef", type=float, default=0.0,
                   help="Optional entropy bonus (subtracted from the loss). "
                   "The schedule-only Gaussian entropy is constant w.r.t. "
                   "theta, so this is a no-op unless the core is extended; "
                   "kept for API parity. 0 disables.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--reward", choices=["policy_likelihood", "rollout"],
                   default="policy_likelihood")

    # ---- optimisation ----
    p.add_argument("--lr", type=float, default=1e-6,
                   help="AdamW LR on the DiT params (tiny — RL fine-tune).")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Linear LR warmup over N optimizer steps. 0 = off.")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Contexts per step (SMALL — multiplied by group_size).")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay over trainable params (0 disables).")

    # ---- run length ----
    p.add_argument("--max_steps", type=int, default=0,
                   help="Stop after N optimizer steps. 0 = use --epochs.")
    p.add_argument("--epochs", type=int, default=1,
                   help="Passes over the train split (ignored if --max_steps>0).")

    # ---- reward policy construction (P3 paligemma_subgoal defaults) ----
    p.add_argument("--policy_ckpt", type=str, default="",
                   help="BC policy checkpoint for the likelihood reward "
                   "(P3 paligemma_subgoal last.pt/best.pt). Empty => untrained.")
    p.add_argument("--policy_history_frames", type=int, default=2)
    p.add_argument("--policy_embed_dim", type=int, default=256)
    p.add_argument("--policy_lora_rank", type=int, default=16)
    p.add_argument("--policy_lora_alpha", type=float, default=32.0)
    p.add_argument("--policy_aux_goal_head",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--policy_aux_progress_head",
                   action=argparse.BooleanOptionalAction, default=True)

    # ---- model build ----
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--freeze_backbone",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--subgoal_pool", type=str, default="mean",
                   choices=["none", "mean"],
                   help="Must match the init ckpt's pooling (SFT used mean).")

    # ---- data (reuse SFT knobs) ----
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--history_frames", type=int, default=2,
                   help="Frames the dataset loads; the reward policy expects "
                   "--policy_history_frames of them.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--per_env_max_episodes", type=int, default=0)
    p.add_argument("--per_env_max_index_samples", type=int, default=0,
                   help="Balance usable step-pairs per env after image "
                   "filtering (same image-balancing knob as the SFT).")
    p.add_argument("--env_filter", type=str, default=None)
    p.add_argument("--require_images",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--subgoal_pairing",
                   choices=["mixed", "semantic_only", "uniform_only"],
                   default="mixed")
    p.add_argument("--subgoal_semantic_prob", type=float, default=0.25)
    p.add_argument("--subgoal_uniform_max", type=int, default=4)

    # ---- PaliGemma encoder ----
    p.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    p.add_argument("--paligemma_dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])

    # ---- outputs / resilience ----
    p.add_argument("--run_dir", type=str, required=True,
                   help="Exact run directory (pinned; required for resume).")
    p.add_argument("--auto_resume", action="store_true",
                   help="Resume from <run_dir>/last.pt if it exists.")
    p.add_argument("--ckpt_every_steps", type=int, default=200,
                   help="Save last.pt every N optimizer steps.")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- W&B (default-on per CLAUDE.md) ----
    p.add_argument("--wandb_project", type=str, default="skyvla-subgoal-dit")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    print(f"{_TAG} writing to {out_dir}")

    # ---- diagnostics (faulthandler heartbeat -> diagnostics.log, EXIT
    # reason line, SIGTERM/SIGINT stack dumps) — reuse the SFT installer. ----
    _diag_file = open(out_dir / "diagnostics.log", "a", buffering=1)
    print(f"{_TAG} diagnostics → {out_dir / 'diagnostics.log'}")
    _install_diagnostics(diag_file=_diag_file)

    # ---- auto-resume wiring ----
    resume_path: str | None = None
    if args.auto_resume:
        cand = out_dir / "last.pt"
        if cand.exists():
            resume_path = str(cand)
            print(f"{_TAG} auto-resuming from {cand}")
        else:
            print(f"{_TAG} --auto_resume set but {cand} missing; fresh start")

    # ---- persist args up front (provenance survives a mid-run crash) ----
    try:
        with open(out_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
    except OSError as exc:
        print(f"{_TAG} WARN failed to persist args.json: {exc}")

    # ---- W&B (best-effort; failures never kill training) ----
    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        key_file = Path("/home/ubuntu/SkyVLA/.wandb_key")
        if "WANDB_API_KEY" not in os.environ and key_file.exists():
            try:
                os.environ["WANDB_API_KEY"] = key_file.read_text().strip()
            except OSError as exc:
                print(f"{_TAG} WARN read .wandb_key: {exc}")
        try:
            import wandb as _wandb
            run_name = args.wandb_run_name or out_dir.name
            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                id=run_name,
                resume="allow",
                mode=args.wandb_mode,
                config=vars(args),
                dir=str(out_dir),
            )
            print(f"{_TAG} wandb: project={args.wandb_project} run={run_name} "
                  f"mode={args.wandb_mode} url={getattr(wandb_run, 'url', '?')}")
        except Exception as exc:
            print(f"{_TAG} WARN wandb init failed: {exc!r}")
            wandb_run = None

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    paligemma_dtype = dtype_map[args.paligemma_dtype]

    # ---- data (train split; episodes.py rule: load train.json, never random_split) ----
    train_ds = OpenFlyDataset(
        split=args.split,
        history_frames=args.history_frames,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        per_env_max_episodes=args.per_env_max_episodes,
        per_env_max_index_samples=args.per_env_max_index_samples,
        require_images=args.require_images,
        oversample_stop=1.0,
        subgoal_pairing=args.subgoal_pairing,
        subgoal_semantic_prob=args.subgoal_semantic_prob,
        subgoal_uniform_max=args.subgoal_uniform_max,
    )
    if len(train_ds) == 0:
        raise RuntimeError("Empty dataset — check OPENFLY_ANNOTATION_DIR / split.")
    print(f"{_TAG} train split {args.split}: {len(train_ds)} step-pairs; "
          f"batch_size={args.batch_size} group_size={args.group_size} "
          f"(effective rollout N = {args.batch_size * args.group_size})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    # ---- frozen PaliGemma encoder (for DiT conditioning) ----
    paligemma = PaliGemmaFeatureExtractor(
        model_name=args.paligemma_model,
        lora_rank=8, lora_alpha=16.0,  # zero-init LoRA-B -> identity (no effect)
        dtype=paligemma_dtype,
    ).to(device)
    paligemma.eval()
    for p in paligemma.parameters():
        p.requires_grad = False

    processor = _build_processor(args.paligemma_model)

    # ---- trainable policy DiT ----
    dit = _build_dit(args, device)
    # Gradient checkpointing on the PixArt backbone: the PPO recompute does
    # K=4 graph-retaining forwards per trajectory through the 610M backbone;
    # without this the activations OOM a 40GB A100 (two frozen PaliGemma-3B +
    # two DiTs already resident). Trades ~30% compute for a large memory cut.
    if getattr(dit, "backbone", None) is not None:
        try:
            dit.backbone.enable_gradient_checkpointing()
            print(f"{_TAG} gradient checkpointing enabled on DiT backbone")
        except Exception as exc:  # noqa: BLE001
            print(f"{_TAG} WARN could not enable grad checkpointing: {exc!r}")
    if args.init_dit_ckpt:
        ckpt = torch.load(args.init_dit_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("dit", ckpt) if isinstance(ckpt, dict) else ckpt
        n_load, n_skip = _shape_filtered_load(dit, sd)
        print(f"{_TAG} init DiT from {args.init_dit_ckpt} "
              f"({n_load} tensors, {n_skip} skipped)")
    else:
        print(f"{_TAG} WARNING: no --init_dit_ckpt; RL from un-finetuned DiT.")
    # eval mode for BOTH sampling and the PPO recompute — keeps CFG dropout
    # off and the forward deterministic (see module docstring).
    dit.eval()

    # ---- frozen reference DiT (KL anchor = the SFT init). Deep-copied
    # BEFORE any resume so it always anchors to the SFT policy, not the
    # resumed (already-RL'd) weights. ----
    ref_dit = copy.deepcopy(dit)
    ref_dit.requires_grad_(False)
    ref_dit.eval()

    # ---- reward model ----
    reward_model = _build_reward_model(args, device, processor)

    # ---- optimizer (DiT params only) + EMA ----
    optimizer = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )
    base_lr = args.lr
    ema: EMA | None = None
    if args.ema_decay > 0.0:
        ema = EMA(dit, decay=args.ema_decay)
        print(f"{_TAG} EMA enabled: decay={args.ema_decay}")

    # ---- resume (DiT + optimizer + global_step + EMA) ----
    global_step = 0
    best_reward = float("-inf")
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        dit.load_state_dict(ck["dit"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if ema is not None and "ema" in ck:
            ema.load_state_dict(ck["ema"])
        global_step = int(ck.get("global_step", 0))
        if "best_reward" in ck:
            best_reward = float(ck["best_reward"])
        print(f"{_TAG} resumed at gstep={global_step} "
              f"best_reward={best_reward:.4f}")

    cfg = DDPOConfig(
        eta=args.eta,
        num_sample_steps=args.num_sample_steps,
        sigma_min=args.sigma_min,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        group_size=args.group_size,
        ppo_epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)

    # EMA smoothers for the noisy per-step RL metrics (W&B charts) — same
    # ~50-step half-life convention as the SFT trainer.
    _ema_alpha = 0.02
    _ema_state: dict[str, float] = {}

    def _ema_update(key: str, val: float) -> float:
        prev = _ema_state.get(key)
        new = val if prev is None else (1 - _ema_alpha) * prev + _ema_alpha * val
        _ema_state[key] = new
        return new

    def _apply_warmup(step_idx: int) -> None:
        if args.warmup_steps <= 0:
            return
        frac = min(1.0, (step_idx + 1) / float(args.warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * frac

    nan_skip_total = 0
    update_attempts = 0  # backward attempts (denominator for nan_skip_ratio)
    stop = False

    print(f"{_TAG} starting RL: reward={args.reward} K={cfg.num_sample_steps} "
          f"eta={cfg.eta} kl_coef={cfg.kl_coef} clip={cfg.clip_ratio} "
          f"ppo_epochs={cfg.ppo_epochs} lr={args.lr}")

    epoch = 0
    while not stop:
        for batch in train_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

            # ---- 1. conditioning over valid rows ----
            conds, valid = _build_conds(dit, paligemma, processor, batch, device)
            if conds is None:
                continue  # no usable context in this batch

            # ---- 2. GRPO group (interleaved; reward fields aligned) ----
            conds_g = conds.repeat(cfg.group_size)
            reward_batch = _repeat_reward_batch(
                batch, valid, cfg.group_size, device
            )

            # ---- 3. rollout (no_grad, eval -> no CFG dropout) ----
            traj = sample_with_logprob(dit, conds_g, cfg, generator=generator)

            # ---- 4. reward + GRPO advantages ----
            rewards = reward_model.score(traj.final_subgoal, reward_batch)  # (N,)
            if not torch.isfinite(rewards).all():
                print(f"{_TAG} WARN non-finite rewards at gstep {global_step}; "
                      "skipping batch")
                continue
            adv = group_advantages(rewards, cfg.group_size).to(device)

            reward_mean = float(rewards.mean().item())
            reward_std = float(rewards.std(unbiased=False).item())
            adv_std = float(adv.std(unbiased=False).item())

            # ---- 5. PPO inner epochs over the SAME rollout buffer ----
            last_stats: dict[str, float] = {}
            kl_val = 0.0
            grad_norm_val = 0.0
            for _ in range(cfg.ppo_epochs):
                _apply_warmup(global_step)
                update_attempts += 1

                logp_new = trajectory_logprob(dit, traj, conds_g, cfg)  # (N,K)
                pg_loss, stats = ppo_step_loss(
                    logp_new, traj.logprob_old, adv, cfg.clip_ratio
                )
                kl = kl_to_reference(dit, ref_dit, traj, conds_g, cfg)  # (N,)
                loss = pg_loss + cfg.kl_coef * kl.mean()
                # Entropy of the schedule-fixed Gaussian carries no policy
                # gradient (sigma is constant w.r.t. theta), so the bonus is
                # a constant; wired in for API completeness / future use.
                if cfg.entropy_coef != 0.0:
                    loss = loss - cfg.entropy_coef * logp_new.new_zeros(())

                # NaN guard on the loss (mirror SFT): skip + zero-grad.
                if not torch.isfinite(loss):
                    nan_skip_total += 1
                    if nan_skip_total <= 5 or nan_skip_total % 50 == 0:
                        print(f"{_TAG} WARN non-finite loss at gstep "
                              f"{global_step} (skipped {nan_skip_total}); "
                              "zero-grad and continue")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in dit.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                # clip_grad_norm_ returns the pre-clip norm; a non-finite
                # norm means optimizer.step() would push NaNs into weights
                # even though per-element clipping "succeeded" — skip.
                if not torch.isfinite(grad_norm):
                    nan_skip_total += 1
                    if nan_skip_total <= 5 or nan_skip_total % 50 == 0:
                        print(f"{_TAG} WARN non-finite grad_norm at gstep "
                              f"{global_step} (skipped {nan_skip_total}); "
                              "zero-grad and continue")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                if ema is not None:
                    ema.update(dit)

                last_stats = stats
                kl_val = float(kl.mean().item())
                grad_norm_val = float(grad_norm.item())

            global_step += 1

            # ---- logging ----
            if last_stats:
                nan_ratio = nan_skip_total / max(update_attempts, 1)
                if global_step % args.log_every == 0:
                    print(
                        f"{_TAG} gstep {global_step:06d} "
                        f"reward_mean={reward_mean:+.4f} "
                        f"reward_std={reward_std:.4f} "
                        f"pg_loss={last_stats['pg_loss']:+.4f} kl={kl_val:.4f} "
                        f"clip_frac={last_stats['clip_frac']:.3f} "
                        f"approx_kl={last_stats['approx_kl']:+.4f} "
                        f"ratio={last_stats['ratio_mean']:.3f} "
                        f"grad_norm={grad_norm_val:.3f}",
                        flush=True,
                    )
                if wandb_run is not None:
                    try:
                        payload = {
                            "rl/reward_mean": reward_mean,
                            "rl/reward_std": reward_std,
                            "rl/adv_std": adv_std,
                            "rl/pg_loss": _ema_update("pg_loss", last_stats["pg_loss"]),
                            "rl/kl": _ema_update("kl", kl_val),
                            "rl/clip_frac": _ema_update("clip_frac", last_stats["clip_frac"]),
                            "rl/approx_kl": _ema_update("approx_kl", last_stats["approx_kl"]),
                            "rl/ratio_mean": _ema_update("ratio_mean", last_stats["ratio_mean"]),
                            "rl/grad_norm": _ema_update("grad_norm", grad_norm_val),
                            "rl/nan_skip_ratio": nan_ratio,
                        }
                        if cfg.entropy_coef != 0.0:
                            payload["rl/entropy"] = 0.0
                        wandb_run.log(payload, step=global_step)
                    except Exception as exc:
                        print(f"{_TAG} WARN wandb.log: {exc!r}")

            # ---- periodic checkpoint ----
            if (args.ckpt_every_steps > 0 and global_step > 0
                    and global_step % args.ckpt_every_steps == 0):
                out_dir.mkdir(parents=True, exist_ok=True)
                _atomic_save(
                    out_dir / "last.pt",
                    _build_ckpt_state(dit, optimizer, ema, global_step, args,
                                      best_reward=best_reward),
                )
                print(f"{_TAG} periodic save at gstep {global_step} → last.pt",
                      flush=True)

            # ---- best.pt on best mean reward ----
            if last_stats and reward_mean > best_reward:
                best_reward = reward_mean
                full = _build_ckpt_state(dit, optimizer, ema, global_step, args,
                                         best_reward=best_reward)
                best_state = {k: v for k, v in full.items()
                              if k not in {"optimizer", "ema"}}
                if ema is not None:
                    # best.pt holds EMA weights as `dit` (inference-ready),
                    # matching the SFT best.pt convention.
                    best_state["dit"] = ema.shadow
                    best_state["ema_decay"] = ema.decay
                tmp = out_dir / "best.pt.tmp"
                torch.save(best_state, tmp)
                os.replace(tmp, out_dir / "best.pt")
                print(f"{_TAG} new best reward_mean={best_reward:+.4f} → best.pt")

        epoch += 1
        if args.max_steps <= 0 and epoch >= args.epochs:
            stop = True

    # ---- final save ----
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(
        out_dir / "last.pt",
        _build_ckpt_state(dit, optimizer, ema, global_step, args,
                          best_reward=best_reward),
    )
    print(f"{_TAG} done: gstep={global_step} best_reward={best_reward:+.4f}; "
          f"saved {out_dir / 'last.pt'}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"{_TAG} WARN wandb finish: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
