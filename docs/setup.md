---
layout: default
title: Setup
description: Where to find the host setup, training, and eval guides.
permalink: /setup/
---

# Setup

Full end-to-end host setup lives in
[`docs/A100_SETUP.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/A100_SETUP.md).
This page is a short index of the most useful entry points.

## Host bring-up

OpenFly's AirSim binaries are **x86_64 only** — they do not run on
aarch64 hosts (GH200 Grace, Apple Silicon). The recommended host is an
A100 40 GB SXM4 or larger x86 + NVIDIA box.

Quickstart:

```bash
git clone https://github.com/CodCodingCode/SkyVLA.git ~/SkyVLA
cd ~/SkyVLA

bash openfly/setup.sh                          # conda env + annotation JSON
bash openfly/download_scene.sh env_airsim_16   # training scene
# For the per-env unseen eval you also need:
bash openfly/download_scene.sh env_game_gtav
bash openfly/download_scene.sh env_ue_smallcity
bash openfly/download_scene.sh env_gs_sjtu02

source openfly/activate.sh
bash openfly/scripts/verify_phase0.sh          # automated infra gates
bash openfly/run_eval.sh --split seen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5
```

If `nvidia-smi` reports a driver mismatch or `vulkaninfo` only sees
`llvmpipe`, fix that first. The
[fix_skyvla_pipeline plan](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/A100_SETUP.md)
covers the driver / sim / image-download path in detail.

## Training entry points

| Stage | Script |
|-------|--------|
| SFT | [`openfly/run_train_paligemma.sh`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/run_train_paligemma.sh) |
| DAgger | [`openfly/run_train_dagger.sh`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/run_train_dagger.sh) |
| PPO (OpenFly-Agent 7B) | [`openfly/run_train_ppo_agent.sh`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/run_train_ppo_agent.sh) |
| GRPO (single stage) | [`openfly/run_train_grpo.sh`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/run_train_grpo.sh) |
| **GRPO curriculum** | [`openfly/run_train_curriculum.sh`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/run_train_curriculum.sh) |

## Eval entry points

| What | Command |
|------|---------|
| Single split eval | `bash openfly/run_eval.sh --split unseen --policy paligemma --paligemma_ckpt <ckpt>` |
| Per-env unseen eval | `--env_filter env_game_gtav` (or `env_ue_smallcity` / `env_gs_sjtu02`) |
| Aggregate runs into a Markdown table | `python -m openfly.scripts.aggregate_results --logs_dir logs/benchmarks --per-env` |

## Repo-only references

The Jekyll site is intentionally a short, public-facing summary. The
long-form documents in the repository are:

- [`README.md`](https://github.com/CodCodingCode/SkyVLA) — repo overview.
- [`openfly/README.md`](https://github.com/CodCodingCode/SkyVLA/tree/main/openfly) — full training/eval reference.
- [`docs/RESEARCH.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/RESEARCH.md) — canonical research plan with code links.
- [`docs/BENCHMARK_FAIRNESS.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/BENCHMARK_FAIRNESS.md) — what is and is not claimable.
- [`docs/A100_SETUP.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/A100_SETUP.md) — end-to-end host bring-up.
- [`docs/NEXT_STEPS.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/NEXT_STEPS.md) — engineering checklist.
