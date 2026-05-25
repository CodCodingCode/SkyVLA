---
layout: default
title: Implementation — How SkyVLA's OpenFly stack works
description: One-page tour of the SkyVLA / OpenFly implementation — env, data, policy, world model, training pipeline, eval.
permalink: /implementation/
---

# How the SkyVLA OpenFly implementation works

A one-page tour of the moving parts in this repository. The deeper
documents are linked at the bottom — this page is for getting oriented
in 5 minutes.

## 1. The benchmark

[OpenFly](https://arxiv.org/abs/2502.18041) is an outdoor aerial
vision-language navigation benchmark. The "drone" is a **kinematic
camera** that teleports between poses — no physics, no flight
controller, no collision response. Each step is one A\* macro action
from an 8-class space:

| ID | Action | Effect |
|----|--------|--------|
| 0 | `stop`         | terminate |
| 1 / 8 / 9 | `forward 3m / 6m / 9m` | translate along yaw |
| 2 / 3 | `turn left / right 30°` | rotate |
| 4 / 5 | `up / down 3m` | altitude change |

A trajectory in OpenFly's `train.json` is a recorded A\* path:

```json
{
  "image_path": "env_airsim_18/astar_data/low_short/2025-1-8_19-2-1_xxx",
  "gpt_instruction": "Proceed in a straight line past the building, then turn left.",
  "action":     [9, 9, 9, 3, 1, 0],
  "index_list": ["t0", "t1", "t2", "t3", "t4", "t5"],
  "pos":        [[x0,y0,z0], …],
  "yaw":        [yaw0, …]
}
```

Each step has one rendered RGB frame on disk and one pose. There is no
video framerate — frames are action-step keyframes. One step ≈ one
second of drone motion at typical speeds.

The split that drives the research is **per-environment unseen**:

- `env_game_gtav` — cross-renderer shift (no GTA in training)
- `env_ue_smallcity` — new Unreal Engine layout (same engine)
- `env_gs_sjtu02` — new 3D-Gaussian-Splatting reconstruction

## 2. The architecture in one picture

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  RGB ─► PaliGemma 3B ─► SigLIP tokens (256 × 2048)                   │
│                  │                                                   │
│                  ▼                                                   │
│            ┌───────────────────┐                                     │
│            │  SubgoalDiT (50M) │  feature-space diffusion            │
│            │  predicts next-   │  conditioned on text + pose delta   │
│            │  keyframe tokens  │                                     │
│            └─────────┬─────────┘                                     │
│                      │                                               │
│   curr + predicted-subgoal + history ─► cross-attn ─► action head    │
│                                                          │           │
│                                                          ▼           │
│                                              discrete action (0..7)  │
└──────────────────────────────────────────────────────────────────────┘
```

Two models, trained largely independently, composed at inference:

- **Action policy** — [`PaliGemmaVLNPolicy`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/models/paligemma_vln.py).
  PaliGemma 3B with LoRA on `q_proj/k_proj/v_proj/o_proj`, a small
  cross-attention pool, and an 8-class action head. ~3M trainable
  parameters; PaliGemma itself is frozen.
- **World model** — [`SubgoalDiT`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/models/subgoal_dit.py).
  ~50M-param Diffusion Transformer that operates **entirely in
  PaliGemma's 2048-d SigLIP token space** — no pixel decoder, no VAE.
  Predicts the SigLIP tokens of the next-keyframe view.

Why feature-space diffusion? PaliGemma's cross-attention already eats
SigLIP tokens; predicting pixels just to re-encode them is wasted
compute. Feature-space inference is ~5× faster than pixel-space (tens
of milliseconds vs. seconds).

## 3. The dataset, and how subgoal pairs are sampled

For world-model training the dataset emits per-step samples of:

```python
(current_rgb, subgoal_rgb, instruction, sub_instruction, pose, subgoal_pose, ...)
```

Subgoal pair selection follows the recipe in
[π0.7 Appendix C](https://arxiv.org/abs/2604.15483):

| Mode | Probability | Future frame |
|------|-------------|--------------|
| **Semantic** (end-of-segment) | **0.25** | The frame where the current same-action run ends — variable horizon, language-aligned. |
| **Uniform** | **0.75** | `t + k`, `k ~ Uniform(1, 4)` actions ahead — dense, short-horizon supervision. |

Both modes draw frames that already exist on disk in the recorded A\*
trajectory. The world model never generates training data for itself.
Pairing is implemented in [`openfly/dataset.py`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/dataset.py)
and exposed via `--subgoal_pairing {mixed,semantic_only,uniform_only}`.

## 4. Why a world model at all

A monolithic VLA (image + instruction → action) does the full
*"interpret language → plan → pick action"* chain inside one forward
pass. It works in seen scenes by memorising shortcuts; it doesn't
transfer.

A world model gives the policy a **concrete visual target** instead of
a sentence. Action selection collapses from "reason about the future"
to "pick the action that moves my current view toward this imagined
target." It is the same idea SuSIE and π0.7 use; the SkyVLA-specific
choice is doing it in SigLIP feature space, with an explicit pose-delta
conditioning input to handle OpenFly's kinematic teleport.

For the long-form motivation see the
[whitepaper](whitepaper) and the [research plan](research-plan).

## 5. Training tracks

The repo ships six tracks, three pre-existing and three new for the
world-model direction:

| Track | Trains | Notes |
|-------|--------|-------|
| **B1 BC SFT** | PaliGemma LoRA + heads | Offline imitation on `train.json`. [Code](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_paligemma.py) |
| **B2 DAgger** | LoRA + heads | On-policy correction over AirSim scenes. [Code](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_dagger.py) |
| **B3 PPO (OpenFly-Agent 7B)** | LoRA + value head | OpenVLA 7B with PPO + LoRA. [Code](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_ppo_openfly_agent.py) |
| **B4 GRPO (PaliGemma)** | LoRA + heads | On-policy RL with reward presets. [Code](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_grpo_paligemma.py) |
| **B5 Curriculum GRPO** | same as B4 | Easy → medium → hard reward sparsity. [Code](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_curriculum_grpo.py) |
| **P2 Subgoal-DiT pretrain** *(new)* | DiT only (PaliGemma frozen) | Offline feature-space diffusion. [Code](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_subgoal_dit.py) |

Phases that touch the world model:

```
P1 BC ─► P2 World-model pretrain ─► P3 BC with subgoals ─► P4 CM distill (opt.)
   │                                       │
   └──────────────────────────────────────► P5 DAgger + subgoals
                                                  │
                                                  ▼
                                    P6 PPO/GRPO + curriculum + subgoals
                                                  │
                                                  ▼
                                          Per-env eval
```

The world model is frozen everywhere outside P2 and P4. RL never trains
it.

## 6. Reward presets and the curriculum

The curriculum lives in [`openfly/rewards.py`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/rewards.py)
and is driven by [`train_curriculum_grpo.py`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/train_curriculum_grpo.py).
Three presets, each loaded as the next stage's `--init_ckpt`:

| Preset | `progress_scale` | `ne_scale` | `success_scale` | `dense_progress` | Interpretation |
|--------|---|---|---|---|----|
| `easy`   | 0.1 | 1/40 | 15.0 | True  | Thick step shaping + soft terminal terms |
| `medium` | 0.0 | 1/40 | 15.0 | False | Terminal NE penalty + soft success |
| `hard`   | 0.0 | 0.0  | 20.0 | False | Almost-binary success + SPL only |

Success radius (20 m) and SPL weighting stay constant across stages so
the reward stays comparable to OpenFly's eval metrics — see
[`BENCHMARK_FAIRNESS.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/BENCHMARK_FAIRNESS.md).

## 7. Evaluation

One harness, [`openfly/eval_benchmark.py`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/eval_benchmark.py),
handles every policy (`heuristic`, `paligemma`, `dagger`, `grpo`,
`openfly_agent`, `ppo`). Each run writes
`logs/benchmarks/openfly_<split>_<policy>_<env>.json` with:

- **SR** — success rate within 20 m, requires `stop` action
- **OSR** — oracle success: any point within 20 m, ignores `stop`
- **NE** — mean navigation error in metres
- **SPL** — success weighted by path length
- per-env breakdown, episode-level metadata, `image_error` flags

Aggregation: [`openfly/scripts/aggregate_results.py`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/scripts/aggregate_results.py)
rolls all `logs/benchmarks/*.json` into a Markdown / CSV table.
Failure-mode analysis (wandered / stalled / oracle-only / near-miss):
[`openfly/scripts/analyse_failures.py`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/scripts/analyse_failures.py).

## 8. Running the stack

```bash
# 1. Setup (one-time)
bash openfly/setup.sh
bash openfly/download_train_images.sh           # ~100 GB of train frames

# 2. P1 BC baseline
bash openfly/run_train_paligemma.sh --epochs 10 --batch_size 8

# 3. P2 SubgoalDiT pretrain (new)
bash openfly/run_train_subgoal_dit.sh \
  --epochs 5 --batch_size 8 \
  --depth 8 --hidden 768 --num_heads 12 \
  --subgoal_pairing mixed --subgoal_semantic_prob 0.25

# 4. Per-env unseen eval
for ENV in env_game_gtav env_ue_smallcity env_gs_sjtu02; do
  bash openfly/run_eval.sh --split unseen --policy paligemma \
    --paligemma_ckpt <ckpt> --env_filter "$ENV" --max_episodes 50
done
```

See [`docs/setup.md`](setup) and
[`docs/A100_SETUP.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/A100_SETUP.md)
for full host bring-up.

## 9. Where the deeper docs live

| Doc | Contents |
|-----|----------|
| [Whitepaper](whitepaper) | Vision + motivation in one page |
| [Research plan](research-plan) | Long-form experimental design |
| [Results](results) | The per-env unseen table (populated as runs complete) |
| [Setup](setup) | Quickstart |
| [Benchmark fairness](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/BENCHMARK_FAIRNESS.md) | What is and is not claimable |
| [Next steps](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/NEXT_STEPS.md) | Engineering checklist |
| [GitHub repository](https://github.com/CodCodingCode/SkyVLA) | Code |
