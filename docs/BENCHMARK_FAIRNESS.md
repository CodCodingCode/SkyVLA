# Benchmark Fairness

Short reference for "is what we're doing fair, and what can we honestly claim about the numbers?" — read this before posting Stage 6/7 results anywhere public.

## TL;DR

Training on a benchmark's **train** split and reporting the held-out **test** numbers is the standard protocol followed by every leaderboard paper we compare against (AirVLN-R1, Seq2Seq, CMA, MGP, π₀-style VLAs). It is **not** cheating. What is cheating, or at least misleading, is:

1. Training on the test split.
2. Doing oracle-goal eval (handing the policy ground-truth target coordinates instead of language) but reporting the result as a leaderboard number.
3. Calling fine-tuned numbers "zero-shot generalization".

## What the curriculum trains on what

| Stage | Data source | Splits used | Notes |
|---|---|---|---|
| 1–2 | Procedurally generated empty-arena rollouts | n/a | No benchmark contact. |
| 4–5 | Same empty-arena, with PaliGemma | n/a | RL-only; no benchmark data. |
| 6 | HUGE-Bench task0 LeRobot trajectories | **train** only | Labels are body-frame future waypoints, not raw delta actions. |
| 7 | HUGE-Bench digital-twin Isaac scenes | **train** scene_id list, when HUGE sim drops | RL on the same scenes the dataset was captured in. |
| 8 | Eval only | `test_seen` and `test_unseen` separately | Reported as fine-tuned numbers. |

The key constraint enforced in [`huge_bench/train_vla_highlevel.py`](../huge_bench/train_vla_highlevel.py) and [`huge_bench/dataset_highlevel.py`](../huge_bench/dataset_highlevel.py) is that the dataset is constructed with `split="train"`. Validation inside the training loop uses `--val_split test_seen` purely for tracking; the final eval numbers come from a separate run of [`benchmarks/eval_huge_vla.py`](../benchmarks/eval_huge_vla.py) on both held-out splits.

## What is fair to claim

- "Hierarchical VLA fine-tuned on HUGE-Bench train; reports `test_seen=X`, `test_unseen=Y`." — Yes.
- "High-level vision-language head was fine-tuned, low-level controller is held frozen at the Stage-2 RL checkpoint." — Yes; this is a valid hierarchical method and worth flagging because it isolates which part of the system the benchmark exercises.
- "Beats the `HugeBCPolicy` action-MSE baseline by Z." — Yes, as long as the baseline is the same one shipped here ([`huge_bench/policy.py`](../huge_bench/policy.py)) trained on the same train split.

## What is not fair

- "Zero-shot HUGE-Bench results." — Not after Stage 6 SFT. Without Stage 6, the empty-arena Stage-5 checkpoint is the actual zero-shot baseline; expect it to do poorly because the empty-arena task distribution doesn't overlap with HUGE outdoor scenes.
- "Beats AirVLN-R1." — Almost certainly false until we add AirNav-specific SFT (optional Stage 7b in the plan). Quoting AirNav SR/SPL after only HUGE training would be misleading because the discrete-action protocol differs and the input distribution is unrelated.
- "Stage 7 numbers from HUGE Isaac." — Not until the upstream sim is released. Until then, [`vla_huge/`](../vla_huge) is a scaffold; any number it produces is on the warehouse placeholder asset, not real digital-twin scenes, and shouldn't be quoted as a HUGE benchmark result.

## Practical: how to label results

Suggested wording for a results table or blog post:

> Hierarchical VLA, **fine-tuned on HUGE-Bench task0 train split**, evaluated on test_seen / test_unseen. Low-level controller frozen at Stage-2 PPO checkpoint. Metric: median body-frame target displacement error (m).

That's accurate, calibrates expectations, and makes the contribution legible without overclaiming generalization.

## Cross-references

- Plan: see the project's `.cursor/plans/benchmark_curriculum_extension_*.plan.md`.
- Implementation: [`huge_bench/train_vla_highlevel.py`](../huge_bench/train_vla_highlevel.py), [`benchmarks/eval_huge_vla.py`](../benchmarks/eval_huge_vla.py).
- Curriculum overview: [`CURRICULUM.md`](CURRICULUM.md).
