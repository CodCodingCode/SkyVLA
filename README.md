# drone_project

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![OpenFly](https://img.shields.io/badge/OpenFly-VLN-1f6feb.svg)](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Outdoor aerial vision-language navigation built on the [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) benchmark. The repository ships:

- a thin evaluation harness around OpenFly's seen / unseen splits and SR / OSR / NE / SPL metrics;
- a wrapper around the official OpenFly-Agent (OpenVLA 7B) FSDP fine-tune; and
- a custom PaliGemma + LoRA + LSTM behaviour-cloning policy with its own offline trainer.

All Isaac Sim / Isaac Lab code (the previous indoor curriculum) has been removed; see [`vla/VLA_SYSTEM.md`](vla/VLA_SYSTEM.md) for the architectural lineage of the policy that now lives under [`openfly/models/`](openfly/models/).

## Quick start

```bash
git clone https://github.com/CodCodingCode/drone_project.git ~/drone_project
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
export OPENFLY_IMAGE_ROOT=~/assets/OpenFly/images
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
| [`openfly/README.md`](openfly/README.md) | OpenFly eval, both training tracks, environment variables |
| [`vla/VLA_SYSTEM.md`](vla/VLA_SYSTEM.md) | PaliGemma + LoRA + cross-attention + LSTM design notes |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark coverage and the optional CityNav oracle baseline |
| [`docs/BENCHMARK_FAIRNESS.md`](docs/BENCHMARK_FAIRNESS.md) | What is and is not claimable from each leaderboard number |
| [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) | Roadmap for extending the trainer and eval coverage |

## Repository layout

```
drone_project/
├── README.md, LICENSE, requirements.txt
├── openfly/                 OpenFly eval, dataset, training, policies
│   ├── eval_benchmark.py    Main eval harness (heuristic / OpenFly-Agent / PaliGemma)
│   ├── train_paligemma.py   Offline BC trainer for the custom model
│   ├── run_train_agent.sh   Wrapper for upstream OpenVLA FSDP training
│   ├── dataset.py           PyTorch dataset over OpenFly trajectories
│   ├── models/paligemma_vln.py
│   ├── actions.py / episodes.py / platform.py / policies.py
├── vla/                     Portable PaliGemma feature extractor + LoRA + design docs
├── benchmarks/              OpenFly-first benchmark runner (CityNav oracle optional)
├── checkpoints/             Legacy Isaac-trained weights (reference only — not used by OpenFly)
├── docs/                    BENCHMARKS, BENCHMARK_FAIRNESS, NEXT_STEPS
└── logs/                    Training and benchmark outputs
```

## Prerequisites

- NVIDIA GPU (24 GB+ VRAM recommended; OpenFly-Agent 7B FSDP needs more).
- Linux x86_64 or aarch64.
- Python 3.10 inside a conda environment named `openfly` (`openfly/setup.sh` creates it).
- Optional: `git-lfs` if you want the legacy checkpoints in [`checkpoints/`](checkpoints).

## Citation

```bibtex
@software{codcodingcode_drone_project,
  author = {CodCodingCode},
  title  = {drone_project: outdoor aerial VLN with OpenFly},
  year   = {2026},
  url    = {https://github.com/CodCodingCode/drone_project}
}
```

## License

Released under the [MIT License](LICENSE).
