#!/usr/bin/env python3
"""Run GRPO in a reward-sparsity curriculum: easy -> medium -> hard.

This is a thin orchestrator around :mod:`openfly.train_grpo_paligemma`.
Each stage launches the underlying GRPO trainer as a subprocess with the
corresponding ``--reward_preset`` and step budget, threading the previous
stage's ``last.pt`` into ``--init_ckpt`` for the next stage.

The motivation is documented in :doc:`docs/RESEARCH.md`: aerial VLN
under purely terminal sparse reward is a hard exploration problem once
the imitation prior is exhausted, so we ramp the supervision down rather
than jump to it.

Layout under ``--out_root``::

    <out_root>/
      stage_easy/    # reward_preset=easy, dense progress on
      stage_medium/  # reward_preset=medium, terminal NE + success
      stage_hard/    # reward_preset=hard, success + SPL only
      curriculum_manifest.json

Each stage directory follows the standard
``logs/openfly/grpo/<run>/`` layout (``last.pt``, ``history.json``,
``rollouts.jsonl``, ``manifest.json``) so existing eval and aggregation
tooling works without changes.

Example
-------

::

    bash openfly/run_train_curriculum.sh \\
        --init_ckpt logs/openfly/dagger/<run>/last.pt \\
        --env_filter env_airsim_16 \\
        --steps_easy 80 --steps_medium 60 --steps_hard 60

After the run, evaluate the final stage on each unseen environment via
``openfly/run_eval.sh --policy grpo --paligemma_ckpt
<out_root>/stage_hard/last.pt --env_filter env_game_gtav`` (and friends).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))


@dataclass(frozen=True)
class Stage:
    name: str
    reward_preset: str
    steps: int
    # Optional trajectory-category filter (e.g. "short" -> "long") used as a
    # secondary axis of curriculum. Pass empty string to skip.
    traj_category: str = ""


def _default_stages(args: argparse.Namespace) -> list[Stage]:
    """Translate CLI knobs into the (easy, medium, hard) stage list."""
    return [
        Stage(
            name="easy",
            reward_preset="easy",
            steps=args.steps_easy,
            traj_category=args.traj_easy,
        ),
        Stage(
            name="medium",
            reward_preset="medium",
            steps=args.steps_medium,
            traj_category=args.traj_medium,
        ),
        Stage(
            name="hard",
            reward_preset="hard",
            steps=args.steps_hard,
            traj_category=args.traj_hard,
        ),
    ]


def _passthrough_grpo_args(args: argparse.Namespace) -> list[str]:
    """Forward GRPO knobs that are not curriculum-specific."""
    fwd: list[str] = [
        "--paligemma_model", args.paligemma_model,
        "--instructions_per_step", str(args.instructions_per_step),
        "--group_size", str(args.group_size),
        "--max_episode_steps", str(args.max_episode_steps),
        "--temperature", str(args.temperature),
        "--clip_ratio", str(args.clip_ratio),
        "--kl_coef", str(args.kl_coef),
        "--demo_coef", str(args.demo_coef),
        "--demo_batch_size", str(args.demo_batch_size),
        "--clip_grad", str(args.clip_grad),
        "--lora_lr", str(args.lora_lr),
        "--head_lr", str(args.head_lr),
        "--history_frames", str(args.history_frames),
        "--env_filter", args.env_filter,
        "--rollout_split", args.rollout_split,
        "--demo_split", args.demo_split,
        "--seed", str(args.seed),
        "--device", args.device,
        "--eval_every", str(args.eval_every),
        "--eval_episodes", str(args.eval_episodes),
        "--eval_split", args.eval_split,
    ]
    return fwd


def _run_stage(
    stage: Stage,
    *,
    init_ckpt: Path,
    out_dir: Path,
    common_args: list[str],
    dry_run: bool,
) -> Path:
    """Launch one GRPO stage and return the resulting ``last.pt``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        sys.executable, "-m", "openfly.train_grpo_paligemma",
        "--init_ckpt", str(init_ckpt),
        "--out_dir", str(out_dir),
        "--steps", str(stage.steps),
        "--reward_preset", stage.reward_preset,
    ]
    if stage.traj_category:
        cmd += ["--traj_category", stage.traj_category]
    cmd += common_args

    print(f"\n[curriculum] === stage {stage.name} (preset={stage.reward_preset}, "
          f"steps={stage.steps}, traj={stage.traj_category or '*'}) ===")
    print(f"[curriculum] cmd: {' '.join(cmd)}")
    if dry_run:
        return out_dir / "last.pt"

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_DRONE_ROOT))
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"GRPO stage {stage.name!r} failed with exit code {proc.returncode}"
        )

    last_ckpt = out_dir / "last.pt"
    if not last_ckpt.is_file():
        raise FileNotFoundError(
            f"Stage {stage.name!r} did not produce {last_ckpt}; "
            f"check {out_dir / 'history.json'}."
        )
    return last_ckpt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--init_ckpt",
        required=True,
        help="Bootstrap checkpoint (DAgger preferred, SFT acceptable).",
    )
    p.add_argument(
        "--out_root",
        default=str(
            _DRONE_ROOT
            / "logs"
            / "openfly"
            / "curriculum"
            / time.strftime("%Y%m%d_%H%M%S")
        ),
        help="Root directory for all stage outputs.",
    )

    # Per-stage step budgets.
    p.add_argument("--steps_easy", type=int, default=80)
    p.add_argument("--steps_medium", type=int, default=60)
    p.add_argument("--steps_hard", type=int, default=60)

    # Optional trajectory-difficulty curriculum (substring on image_path).
    p.add_argument("--traj_easy", default="")
    p.add_argument("--traj_medium", default="")
    p.add_argument("--traj_hard", default="")

    p.add_argument(
        "--skip",
        default="",
        help="Comma-separated stage names to skip (e.g. 'easy' if you already "
        "have a checkpoint from a previous run and want to resume).",
    )
    p.add_argument(
        "--resume_from",
        default="",
        help="Path to a checkpoint from a previous stage; combined with --skip "
        "to resume a partial curriculum.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command for each stage without executing it.",
    )

    # GRPO passthrough flags (mirrors train_grpo_paligemma).
    p.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    p.add_argument("--instructions_per_step", type=int, default=2)
    p.add_argument("--group_size", type=int, default=4)
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
    p.add_argument(
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_episodes", type=int, default=8)
    p.add_argument("--eval_split", default="unseen")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stages = _default_stages(args)
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    common = _passthrough_grpo_args(args)

    manifest = {
        "init_ckpt": args.init_ckpt,
        "out_root": str(out_root),
        "stages": [s.__dict__ for s in stages],
        "skip": sorted(skip),
        "resume_from": args.resume_from,
        "common_grpo_args": common,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_root / "curriculum_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    current_ckpt = Path(args.resume_from) if args.resume_from else Path(args.init_ckpt)
    if not current_ckpt.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Initial checkpoint not found: {current_ckpt}")

    stage_outputs: dict[str, str] = {}
    for stage in stages:
        if stage.name in skip:
            print(f"[curriculum] skipping stage {stage.name}")
            continue
        out_dir = out_root / f"stage_{stage.name}"
        current_ckpt = _run_stage(
            stage,
            init_ckpt=current_ckpt,
            out_dir=out_dir,
            common_args=common,
            dry_run=args.dry_run,
        )
        stage_outputs[stage.name] = str(current_ckpt)

    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["stage_outputs"] = stage_outputs
    manifest["final_ckpt"] = str(current_ckpt)
    with open(out_root / "curriculum_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[curriculum] done. Final checkpoint: {current_ckpt}")
    print(f"[curriculum] manifest: {out_root / 'curriculum_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
