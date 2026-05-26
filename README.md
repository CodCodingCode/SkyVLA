# SkyVLA

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![OpenFly](https://img.shields.io/badge/OpenFly-VLN-1f6feb.svg)](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
[![Site](https://img.shields.io/badge/site-codcodingcode.github.io%2FSkyVLA-blue.svg)](https://codcodingcode.github.io/SkyVLA/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Outdoor aerial vision-language navigation built on the [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) benchmark. The repository ships:

- a thin evaluation harness around OpenFly's seen / unseen splits and SR / OSR / NE / SPL metrics, with a **per-env breakdown** for the three unseen scenes (`env_game_gtav`, `env_ue_smallcity`, `env_gs_sjtu02`);
- a wrapper around the official OpenFly-Agent (OpenVLA 7B) FSDP fine-tune;
- a custom PaliGemma + LoRA + MLP behaviour-cloning policy with its own offline trainer; and
- an RL pipeline on top of the AirSim bridge — GRPO (PaliGemma) with an **easy &rarr; medium &rarr; hard reward curriculum**, and PPO (OpenFly-Agent 7B with LoRA + value head). RL bootstraps directly from the BC checkpoint (no DAgger stage — PPO's on-policy rollouts subsume it, and the geometric oracle DAgger needs was too weak to teach obstacle avoidance in OpenFly's kinematic env).

The research narrative — what we are actually trying to learn from these splits — lives in [`docs/RESEARCH.md`](docs/RESEARCH.md) and on the project site at <https://codcodingcode.github.io/SkyVLA/>. See [`vla/VLA_SYSTEM.md`](vla/VLA_SYSTEM.md) for design notes on the PaliGemma + LoRA backbone, and [`docs/A100_SETUP.md`](docs/A100_SETUP.md) for end-to-end setup on an x86_64 A100 host.

## Quick start

```bash
git clone https://github.com/CodCodingCode/SkyVLA.git ~/drone_project
cd ~/drone_project

# 1. One-time setup: conda env, OpenFly-Platform clone, annotation JSON.
bash openfly/setup.sh

# 2. Download at least one AirSim scene (several GB).
bash openfly/download_airsim_scene.sh env_airsim_16

# 3. Sanity-check the eval harness with the oracle policy.
source openfly/activate.sh
bash openfly/run_eval.sh \
  --split unseen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5
```

Eval results land in `logs/benchmarks/openfly_*.json`.

## Train

Two interchangeable training tracks ship with the repo. Both target the OpenFly eval harness above.

```bash
# Track A — official OpenFly-Agent (OpenVLA 7B) via upstream FSDP. See openfly/README.md
#           for the TFDS build steps that must run first.
export OPENFLY_TFDS_DIR=~/openfly_tfds
bash openfly/run_train_agent.sh --run_root_dir runs/openfly_agent_7b

# Track B — custom PaliGemma BC on train.json (single GPU, offline).
# Grab the trajectory frames from IPEC-COMMUNITY/OpenFly (all 11 train scenes; ~100 GB):
bash openfly/download_train_images.sh
export OPENFLY_IMAGE_ROOT=~/assets/OpenFly/images/Image
bash openfly/run_train_paligemma.sh --epochs 10 --batch_size 8

# Evaluate the custom checkpoint:
bash openfly/run_eval.sh --split unseen --policy paligemma \
  --paligemma_ckpt logs/openfly/paligemma/<run>/last.pt \
  --env_filter env_airsim_16
```

See [`openfly/README.md`](openfly/README.md) for the full eval / train reference.

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) | What SkyVLA is trying to achieve — motivation, architecture, expected contribution |
| [`docs/implementation.md`](docs/implementation.md) | One-page tour of the implementation (env, data, policy, world model, training tracks, eval) |
| [`openfly/README.md`](openfly/README.md) | OpenFly eval, both training tracks, environment variables |
| [`vla/VLA_SYSTEM.md`](vla/VLA_SYSTEM.md) | PaliGemma + LoRA feature extractor design notes |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Living research plan — research question, splits, reward curriculum, experiment matrix, results |
| [`docs/A100_SETUP.md`](docs/A100_SETUP.md) | End-to-end setup of the OpenFly RL stack on an x86_64 A100 host |
| [`docs/BENCHMARK_FAIRNESS.md`](docs/BENCHMARK_FAIRNESS.md) | What is and is not claimable from each leaderboard number |
| [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) | Engineering checklist feeding into `RESEARCH.md` |
| [Project site](https://codcodingcode.github.io/SkyVLA/) | Public Jekyll site published from `docs/` on `main` |

## Repository layout

```
drone_project/
├── README.md, LICENSE, requirements.txt
├── openfly/                 OpenFly eval, dataset, training, RL pipeline, policies
│   ├── eval_benchmark.py    Main eval harness (heuristic / OpenFly-Agent / PaliGemma / GRPO / PPO)
│   ├── train_paligemma.py   Offline BC trainer for the custom model
│   ├── train_grpo_paligemma.py   GRPO RL fine-tune for the PaliGemma policy
│   ├── train_curriculum_grpo.py  Reward-sparsity curriculum (easy -> medium -> hard) on top of GRPO
│   ├── train_ppo_openfly_agent.py   PPO + LoRA + value head for OpenFly-Agent 7B
│   ├── scripts/aggregate_results.py   Roll logs/benchmarks/*.json into Markdown/CSV
│   ├── scripts/analyse_failures.py    Per-env failure-mode breakdown for unseen runs
│   ├── run_train_agent.sh   Wrapper for upstream OpenVLA FSDP training
│   ├── envs/airsim_vln_env.py    gymnasium env wrapping the AirSim bridge
│   ├── rewards.py / rollout.py   Episode rewards + trajectory collection
│   ├── dataset.py           PyTorch dataset over OpenFly trajectories
│   ├── models/paligemma_vln.py / models/openfly_agent_rl.py
│   ├── actions.py / episodes.py / platform.py / policies.py
├── vla/                     Portable PaliGemma feature extractor + LoRA + design notes
├── docs/                    RESEARCH, A100_SETUP, BENCHMARK_FAIRNESS, NEXT_STEPS + Jekyll site
└── logs/                    Training and benchmark outputs (gitignored)
```

## Prerequisites

- NVIDIA GPU (24 GB+ VRAM recommended; OpenFly-Agent 7B FSDP needs more).
- Linux **x86_64** (the upstream AirSim Unreal Engine binaries are not built for aarch64 — see [`docs/A100_SETUP.md`](docs/A100_SETUP.md)).
- Python 3.10 inside a conda environment named `openfly` (`openfly/setup.sh` creates it).

## Citation

```bibtex
@software{codcodingcode_skyvla,
  author = {CodCodingCode},
  title  = {SkyVLA: outdoor aerial vision-language navigation with OpenFly},
  year   = {2026},
  url    = {https://github.com/CodCodingCode/SkyVLA}
}
```

## License

Released under the [MIT License](LICENSE).
