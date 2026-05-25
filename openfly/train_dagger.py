#!/usr/bin/env python3
"""DAgger between SFT and online RL.

Idea (Ross & Bagnell 2010): the SFT policy makes mistakes that the
offline dataset never covers. Roll the current policy in AirSim, ask an
oracle (here :func:`openfly.actions.goal_heuristic_action`) what the
correct action was at each visited state, and fine-tune on the mixed
offline + corrected dataset. Repeating this for a few iterations gives
us a substantially more robust starting point for GRPO/PPO than vanilla
behaviour cloning on ``train.json`` alone.

Track scope (per plan): full DAgger loop for **PaliGemma (Track B)**;
for Track A (OpenFly-Agent 7B) we only collect the corrected JSONL —
upstream FSDP training can consume it later. The
``--track openfly-agent`` flag enables that collection-only mode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import (
    ACTION_NAMES,
    TRAINABLE_ACTION_IDS,
    action_id_to_logit_index,
    goal_heuristic_action,
    logit_index_to_action_id,
)
from openfly.dataset import (
    NUM_OPENFLY_ACTIONS,
    OpenFlyDataset,
    OpenFlySample,
    _build_sub_instruction,
    collate,
)
from openfly.envs import AirSimVLNEnv, AirSimVLNEnvConfig
from openfly.models.paligemma_vln import (
    PaliGemmaVLNPolicy,
    lora_and_head_param_groups,
)
from openfly.rollout import RolloutTrajectory, collect_episode


class DAggerBuffer(Dataset):
    """In-memory dataset of (rgb, history, instruction, pose, goal, expert_action).

    Mirrors :class:`OpenFlyDataset`'s sample shape so the same collate
    function can stack offline + DAgger batches.
    """

    def __init__(self, history_frames: int, image_size: int = 224) -> None:
        self.history_frames = history_frames
        self.image_size = image_size
        self._samples: list[OpenFlySample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> OpenFlySample:
        return self._samples[idx]

    def add_trajectory(self, traj: RolloutTrajectory) -> int:
        """Append oracle-corrected steps from one rollout.

        Stops are *not* corrected — relabelling a non-stop action to
        "stop" hits the policy with high CE on a single label and tends
        to collapse training. We just keep the trajectory honest.

        ``last_action`` records the previously *executed* (agent) action
        — ``traj.actions[i-1]`` (a raw OpenFly id from the sim) — because
        that's what produced the current observation. We remap it into
        the model's logit-index space here; un-supervised ids fall back
        to the START token (``NUM_OPENFLY_ACTIONS``). ``next_pose`` is
        the agent's pose at i+1, falling back to the current pose at
        the terminal step.
        """
        added = 0
        # Sub-instruction is templated from the *expert* (oracle) action at
        # this step extended into a single-action sub-segment. The oracle
        # only emits one action at a time, so this collapses to a single
        # primitive (e.g. "turn left 30 degrees"). Trajectory-length
        # progress mirrors the offline dataset's ``step / (n-1)``.
        traj_len = max(1, len(traj.obs_rgb))
        for i, rgb in enumerate(traj.obs_rgb):
            if i >= len(traj.obs_poses):
                break
            pose_arr = traj.obs_poses[i]
            pose = pose_arr.tolist() if hasattr(pose_arr, "tolist") else list(pose_arr)
            goal = list(traj.goal)
            # goal_heuristic_action only ever returns {0, 1, 2, 3} so the
            # remap is always defined; assert keeps that contract honest.
            expert_raw = int(goal_heuristic_action(pose, goal))
            expert = action_id_to_logit_index(expert_raw)
            sub_instruction = _build_sub_instruction([expert_raw], 0)
            progress = float(i) / float(max(1, traj_len - 1))
            history_arr = traj.obs_history[i] if i < len(traj.obs_history) else np.zeros(
                (self.history_frames, self.image_size, self.image_size, 3), dtype=np.uint8
            )
            if i == 0 or i - 1 >= len(traj.actions):
                last_action = NUM_OPENFLY_ACTIONS  # START token
            else:
                prev_raw = int(traj.actions[i - 1])
                last_action = (
                    action_id_to_logit_index(prev_raw)
                    if prev_raw in TRAINABLE_ACTION_IDS
                    else NUM_OPENFLY_ACTIONS
                )
            if i + 1 < len(traj.obs_poses):
                np_arr = traj.obs_poses[i + 1]
                next_pose = np.asarray(
                    np_arr.tolist() if hasattr(np_arr, "tolist") else list(np_arr),
                    dtype=np.float32,
                )
            else:
                next_pose = np.asarray(pose, dtype=np.float32)
            sample = OpenFlySample(
                rgb=np.asarray(rgb, dtype=np.uint8),
                history=np.asarray(history_arr, dtype=np.uint8),
                instruction=traj.instruction,
                action_id=int(expert),
                pose=np.asarray(pose, dtype=np.float32),
                goal=np.asarray(goal, dtype=np.float32),
                last_action=int(last_action),
                next_pose=next_pose,
                progress=progress,
                sub_instruction=sub_instruction,
            )
            self._samples.append(sample)
            added += 1
        return added


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_paligemma(args, device: torch.device) -> tuple[PaliGemmaVLNPolicy, Any]:
    from transformers import AutoProcessor

    model = PaliGemmaVLNPolicy(
        history_frames=args.history_frames,
        paligemma_model_name=args.paligemma_model,
    ).to(device)
    if args.sft_ckpt:
        ckpt = torch.load(args.sft_ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        model_state = model.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = len(state) - len(compatible)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"[dagger] loaded SFT ckpt {args.sft_ckpt}: "
            f"{len(compatible)} tensors, {skipped} shape-skipped, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    processor = AutoProcessor.from_pretrained(args.paligemma_model)
    return model, processor


def _format_prompt(instruction: str, sub_instruction: str | None) -> str:
    base = f"<image>\nTask: {instruction}"
    if sub_instruction:
        return f"{base}\nNow: {sub_instruction}"
    return base


def _tokenise_batch(
    processor,
    instructions: list[str],
    rgb_dummy,
    device,
    max_length=256,
    sub_instructions: list[str] | None = None,
):
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
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


def _train_step(
    model: PaliGemmaVLNPolicy,
    batch: dict[str, Any],
    processor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    history = batch["history"].to(device, non_blocking=True)
    pose = batch["pose"].to(device, non_blocking=True)
    actions = batch["action_id"].to(device, non_blocking=True)
    last_action = batch["last_action"].to(device, non_blocking=True)
    next_pose = batch["next_pose"].to(device, non_blocking=True)
    progress = batch.get("progress")
    if progress is not None:
        progress = progress.to(device, non_blocking=True)
    input_ids, attention_mask = _tokenise_batch(
        processor,
        batch["instruction"],
        rgb[0],
        device,
        sub_instructions=batch.get("sub_instruction"),
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
    ce = F.cross_entropy(logits, actions)
    metrics = {
        "ce": ce.item(),
        "acc": (logits.argmax(dim=-1) == actions).float().mean().item(),
    }
    return ce, metrics


def _policy_factory_paligemma(
    model: PaliGemmaVLNPolicy,
    processor,
    device: torch.device,
    max_steps: int = 60,
):
    """Return a ``policy_fn`` compatible with :func:`collect_episode`.

    ``max_steps`` is used as the denominator for the inference-time
    ``progress`` proxy. This matches what the env actually budgets so the
    scalar stays in [0, 1] and roughly aligns with the training signal
    (step / traj_len).
    """
    from openfly.dataset import NUM_OPENFLY_ACTIONS

    # Mutable closure cell so the same policy_fn instance can track
    # ``last_action`` across steps within an episode. Reset each time we
    # see a step-0 observation (info["step"] == 0 when emitted by the env).
    state = {"last_action": NUM_OPENFLY_ACTIONS}

    def policy_fn(obs, info):
        # Reset last_action at the start of each episode. AirSimVLNEnv
        # emits "step_idx" on every step (including 0); reset.info has no
        # step_idx, so we treat its absence the same as step 0.
        if int(info.get("step_idx", 0)) == 0:
            state["last_action"] = NUM_OPENFLY_ACTIONS
        rgb = obs["rgb"]
        history = obs["rgb_history"]
        pose = obs["pose"]
        rgb_t = torch.from_numpy(rgb).unsqueeze(0).to(device)
        hist_t = torch.from_numpy(history).unsqueeze(0).to(device)
        pose_t = torch.from_numpy(pose).unsqueeze(0).to(device)
        last_action_t = torch.tensor([state["last_action"]], dtype=torch.long, device=device)
        # ``next_pose`` is only consumed by the trainer's aux goal target;
        # at inference we duplicate the current pose.
        next_pose_t = pose_t.clone()
        # Budget-relative progress. Without a high-level expected-length
        # estimate at inference, this is the best proxy that aligns with
        # the [0, 1] signal the model saw during training.
        step_idx = int(info.get("step_idx", 0))
        progress_val = min(1.0, step_idx / float(max(max_steps - 1, 1)))
        progress_t = torch.tensor([progress_val], dtype=torch.float32, device=device)
        # No high-level policy plumbed in yet → empty sub-instruction.
        # The prompt template drops the "Now:" line in this case so
        # we don't fabricate a sub-step the model didn't choose.
        proc = processor(
            text=[_format_prompt(info.get("instruction", ""), None)],
            images=[rgb],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=256,
        )
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
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
            logits = out["action_logits"]
            logit_idx = int(logits.argmax(dim=-1).item())
        # ``state["last_action"]`` is fed back into the model embedding, so
        # we keep it in logit-index space. The env, however, expects a raw
        # OpenFly id; remap at the boundary.
        state["last_action"] = logit_idx
        return logit_index_to_action_id(logit_idx), {"logits": logits.detach().cpu()}

    return policy_fn


def _build_env(args) -> AirSimVLNEnv:
    cfg = AirSimVLNEnvConfig(
        split=args.rollout_split,
        env_filter=args.env_filter,
        max_steps=args.max_steps,
        history_frames=args.history_frames,
        image_size=args.image_size,
        seed=args.seed,
    )
    return AirSimVLNEnv(cfg)


def _serialise_jsonl(traj: RolloutTrajectory, path: Path) -> None:
    """Track A collection: write `(image_path, instruction, action, pose)` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, action in enumerate(traj.actions):
            if i >= len(traj.poses):
                break
            row = {
                "instruction": traj.instruction,
                "image_path": traj.image_path,
                "step": i,
                "pose": traj.poses[i],
                "agent_action": int(action),
                "expert_action": int(
                    goal_heuristic_action(traj.poses[i], traj.goal)
                ),
            }
            f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=("paligemma", "openfly-agent"), default="paligemma")
    p.add_argument("--sft_ckpt", type=str, default="", help="SFT checkpoint to warm-start")
    p.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")

    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--episodes_per_iter", type=int, default=200)
    p.add_argument("--epochs_per_iter", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--history_frames", type=int, default=2)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument(
        "--rollout_split",
        default="train",
        help="Episode pool used by AirSim rollouts",
    )
    p.add_argument(
        "--offline_split",
        default="train",
        help="Offline OpenFlyDataset split mixed with DAgger buffer (50/50).",
    )
    p.add_argument("--env_filter", default="env_airsim_16")
    p.add_argument("--offline_mix", type=float, default=0.5, help="Fraction sampled from OpenFlyDataset")
    p.add_argument("--offline_max_episodes", type=int, default=0, help="0 = all (smoke runs use small N)")
    p.add_argument("--steps_per_update", type=int, default=0, help="0 = one full epoch over the mixed dataset; smaller values cap update length for smoke runs")
    p.add_argument("--lora_lr", type=float, default=1e-6)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(
            _DRONE_ROOT
            / "logs"
            / "openfly"
            / "dagger"
            / time.strftime("%Y%m%d_%H%M%S")
        ),
    )
    return p.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dagger] writing to {out_dir}")

    env = _build_env(args)

    track_a_jsonl: Path | None = None
    model: PaliGemmaVLNPolicy | None = None
    processor = None
    optimizer = None
    offline_ds: OpenFlyDataset | None = None
    buffer = DAggerBuffer(history_frames=args.history_frames, image_size=args.image_size)

    if args.track == "paligemma":
        model, processor = _load_paligemma(args, device)
        optimizer = torch.optim.AdamW(
            lora_and_head_param_groups(model, lora_lr=args.lora_lr, head_lr=args.head_lr)
        )
        offline_ds = OpenFlyDataset(
            split=args.offline_split,
            history_frames=args.history_frames,
            env_filter=args.env_filter,
            max_episodes=args.offline_max_episodes,
        )
        print(f"[dagger] offline OpenFlyDataset size={len(offline_ds)}")
        policy_fn = _policy_factory_paligemma(
            model, processor, device, max_steps=args.max_steps
        )
    else:
        # Track A: collection-only. We rely on the existing OpenFlyAgentPolicy
        # for rollouts so the corrected JSONL feeds an upstream re-SFT later.
        from openfly.policies import OpenFlyAgentPolicy

        agent_policy = OpenFlyAgentPolicy(model_id=args.sft_ckpt or "IPEC-COMMUNITY/openfly-agent-7b")
        track_a_jsonl = out_dir / "openfly_agent_dagger.jsonl"

        def policy_fn(obs, info):
            agent_policy.reset(info.get("instruction", ""), obs["goal"].tolist())
            action = agent_policy.act(
                obs["rgb"],
                obs["pose"].tolist(),
                int(obs["step_idx"]),
                history=[],
            )
            return int(action), {"source": "openfly-agent"}

    history_log: list[dict[str, Any]] = []
    for it in range(args.iterations):
        print(f"\n[dagger] === iteration {it} ===")
        rollouts: list[RolloutTrajectory] = []
        n_added = 0
        for ep in range(args.episodes_per_iter):
            traj = collect_episode(env, policy_fn, capture_obs=(args.track == "paligemma"))
            rollouts.append(traj)
            if args.track == "paligemma":
                n_added += buffer.add_trajectory(traj)
            elif track_a_jsonl is not None:
                _serialise_jsonl(traj, track_a_jsonl)
            if ep % 10 == 0:
                print(
                    f"  ep {ep:03d} steps={traj.steps} reward={traj.total_reward:+.2f} "
                    f"SR={int(traj.success)} NE={traj.ne_m:.1f}"
                )

        from openfly.rollout import aggregate_metrics

        agg = aggregate_metrics(rollouts)
        agg.update({"iteration": it, "buffer_size": len(buffer)})
        history_log.append(agg)
        print(f"[dagger] iter {it} rollout summary: {agg}")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history_log, f, indent=2)

        if args.track != "paligemma":
            continue

        # ----- supervised update on offline + DAgger buffer ----------------
        assert model is not None and processor is not None
        assert optimizer is not None and offline_ds is not None
        if len(buffer) == 0:
            print("[dagger] no on-policy samples gathered; skipping update")
            continue

        # Build a 50/50 mix via concat dataset and weighted sampler.
        mixed = ConcatDataset([offline_ds, buffer])
        weights = [args.offline_mix / len(offline_ds)] * len(offline_ds) + [
            (1.0 - args.offline_mix) / len(buffer)
        ] * len(buffer)
        # ``num_samples`` controls steps per iteration. By default we keep
        # one full epoch over ``mixed``; smoke runs override with a small
        # cap so the loop finishes in seconds.
        if args.steps_per_update > 0:
            num_samples = args.steps_per_update * args.batch_size
        else:
            num_samples = len(mixed)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=num_samples, replacement=True
        )
        loader = DataLoader(
            mixed,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )

        model.train()
        for epoch in range(args.epochs_per_iter):
            for step, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = _train_step(model, batch, processor, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                if step % 20 == 0:
                    print(
                        f"  iter {it} epoch {epoch} step {step:04d} "
                        f"ce={metrics['ce']:.3f} acc={metrics['acc']:.3f}"
                    )

        ckpt_path = out_dir / f"dagger_iter{it:02d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": it,
                "args": vars(args),
                "buffer_size": len(buffer),
            },
            ckpt_path,
        )
        # Refresh latest checkpoint pointer
        torch.save({"model": model.state_dict(), "iteration": it}, out_dir / "last.pt")
        print(f"[dagger] saved {ckpt_path}")

    env.close()
    print(f"\n[dagger] done. checkpoints in {out_dir}")
    if track_a_jsonl is not None:
        print(f"[dagger] Track A JSONL: {track_a_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
