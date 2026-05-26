# SkyVLA

[![Site](https://img.shields.io/badge/site-skyvla-blue.svg)](https://codcodingcode.github.io/SkyVLA/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Aerial vision-language navigation for the [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) benchmark.

```
RGB ─► PaliGemma 3B (frozen + LoRA) ─► curr SigLIP ──┬──► SubgoalDiT (~150M) ─► predicted subgoal
                                                     │                                  │
                                                     └────► cross-attn + action head ◄──┘
                                                                       │
                                                                       ▼
                                                              discrete action (0..7)
```

Three training phases:

1. **P1 — behaviour cloning.** PaliGemma + LoRA + action head, offline on OpenFly's `train.json`.
2. **P2 — world model.** `SubgoalDiT` is a feature-space DDPM (~150M, from-scratch DiT) that predicts the next-keyframe SigLIP tokens from the current frame, instruction, and pose delta. PaliGemma is frozen for this stage so the diffusion loss is the only signal. A web-pretrained PixArt-Σ backbone with a thin SigLIP adapter was tried as a π0.7-style init — it didn't transfer (see whitepaper §10).
3. **P3 — subgoal-conditioned policy.** The frozen world model feeds the policy via cross-attention. Online RL — GRPO on PaliGemma, PPO on the OpenFly-Agent 7B baseline — updates only the action head, with an easy → medium → hard reward curriculum on the GRPO run.

Eval uses OpenFly's seen / unseen splits, with a per-env breakdown for the three unseen scenes (`env_game_gtav`, `env_ue_smallcity`, `env_gs_sjtu02`).

## Quick start

```bash
git clone https://github.com/CodCodingCode/SkyVLA.git ~/SkyVLA
cd ~/SkyVLA

bash openfly/setup.sh                              # conda env, OpenFly-Platform clone, annotations
bash openfly/download_scene.sh env_airsim_16       # ~2 GB
source openfly/activate.sh
bash openfly/run_eval.sh --split unseen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5
```

Eval JSON lands in `logs/benchmarks/`.

## Train

```bash
export OPENFLY_IMAGE_ROOT=~/assets/OpenFly/images/Image

# P1 — BC
bash openfly/run_train_paligemma.sh --epochs 10 --batch_size 8

# P2 — world model (PaliGemma frozen, SigLIP-token diffusion)
bash openfly/run_train_subgoal_dit.sh

# P3 — RL on the subgoal-conditioned policy
bash openfly/run_train_curriculum.sh \
  --init_ckpt logs/openfly/paligemma/<run>/last.pt \
  --steps_easy 80 --steps_medium 60 --steps_hard 60

# Eval any checkpoint
bash openfly/run_eval.sh --split unseen --policy paligemma \
  --paligemma_ckpt logs/openfly/<run>/last.pt
```

A second training track ships for the upstream OpenFly-Agent (OpenVLA 7B) via FSDP — see [`openfly/README.md`](openfly/README.md).

## Layout

```
openfly/
  eval_benchmark.py              eval harness
  train_paligemma.py             P1 — BC
  train_subgoal_dit.py           P2 — SigLIP-token diffusion world model
  train_grpo_paligemma.py        P3 — GRPO on PaliGemma
  train_curriculum_grpo.py       P3 — easy → medium → hard reward curriculum
  train_ppo_openfly_agent.py     P3 — PPO + LoRA + value head on OpenFly-Agent 7B
  models/
    paligemma_vln.py             BC backbone
    subgoal_dit.py               world model (vanilla DiT)
    subgoal_dit_pixart.py        failed ablation — PixArt-Σ frozen backbone + thin adapter
    openfly_agent_rl.py          7B + value head
  envs/airsim_vln_env.py         gymnasium wrapper around the AirSim / UE bridge
  rewards.py, rollout.py         episode rewards + trajectory collection
vla/                             portable PaliGemma feature extractor + design notes
docs/                            research plan, setup, fairness, Jekyll site
logs/                            training and benchmark outputs (gitignored)
```

## Docs

| File | What's in it |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | research question, splits, reward curriculum, experiment matrix |
| [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) | motivation, architecture, expected contribution |
| [`docs/implementation.md`](docs/implementation.md) | one-page tour: env, data, policy, world model, training, eval |
| [`docs/A100_SETUP.md`](docs/A100_SETUP.md) | end-to-end host bring-up on an x86_64 A100 |
| [`docs/BENCHMARK_FAIRNESS.md`](docs/BENCHMARK_FAIRNESS.md) | what each leaderboard number can and can't claim |
| [`vla/VLA_SYSTEM.md`](vla/VLA_SYSTEM.md) | PaliGemma + LoRA backbone notes |
| [Project site](https://codcodingcode.github.io/SkyVLA/) | the same content, browsable |

## Requirements

- x86_64 Linux with an NVIDIA GPU. The upstream UE scene binaries are x86 only — no GH200 / aarch64.
- 24 GB VRAM covers P1–P3 except the OpenFly-Agent 7B FSDP track, which needs more.
- Python 3.10 inside a conda env named `openfly` (created by `openfly/setup.sh`).

## License

MIT.
