# drone_project

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76b900.svg)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-76b900.svg)](https://github.com/isaac-sim/IsaacLab)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Language-grounded drone navigation in NVIDIA Isaac Sim. A Crazyflie quadcopter is trained with PPO across a four-stage curriculum so it can fly to objects described in natural language ("fly to the red cube", "go to the forklift") using only an onboard camera and the command text.

## Architecture

The policy is built up across four stages, each one extending the previous one with strictly more task structure while preserving the low-level flight skills already learned. See [docs/CURRICULUM.md](docs/CURRICULUM.md) for the full story, including the weight-transfer mechanics that bridge each stage.

| Stage | Module | What it teaches |
|-------|--------|-----------------|
| 1 | [hover/](hover) | Altitude hold, stability, thrust control |
| 2 | [waypoint_nav/](waypoint_nav) | Point-to-point navigation with an in-env reward curriculum |
| 3 | [lang_nav/](lang_nav) | Pick the correct object from a CLIP-encoded language command |
| 4 | [vla/](vla) | Full VLA: PaliGemma decides where to go, the frozen Stage 2 policy flies |

Optional and experimental modules (SigLIP variant, Pi0, warehouse and Cesium fine-tunes, behaviour-cloning baseline) are documented in [docs/ADVANCED.md](docs/ADVANCED.md).

## Prerequisites

- NVIDIA GPU with at least 24 GB VRAM (A10, H100, GH200 all tested)
- Ubuntu 22.04, x86_64 or aarch64
- Isaac Sim 5.1 and Isaac Lab 2.3.2
- Pegasus Simulator for multirotor flight dynamics
- Python 3.11 inside a conda environment named `isaac`
- `git-lfs` for the pretrained checkpoints in [checkpoints/](checkpoints)

## Quick start

```bash
git lfs install
git clone https://github.com/CodCodingCode/drone_project.git ~/drone_project
cd ~/drone_project
bash setup.sh
source activate_env.sh
```

`setup.sh` installs Miniconda (if needed), clones Isaac Lab, creates the `isaac` conda env, and installs Isaac Sim plus all Python dependencies. `activate_env.sh` is meant to be sourced before running any training script and sets `OMNI_KIT_ACCEPT_EULA`, `ISAACSIM_PATH`, and `TERM`.

If you only need the Python deps in a pre-existing environment, `requirements.txt` lists everything that is not part of the Isaac Sim install.

## Training

```bash
# Stage 1: hover
./isaaclab.sh -p ~/drone_project/hover/train.py \
    --num_envs 1024 --max_iterations 1500 --headless

# Stage 2: waypoint navigation, resumed from hover
python ~/drone_project/scripts/transfer_hover_to_waypoint.py \
    --hover_checkpoint ~/drone_project/checkpoints/stage1_hover.pt \
    --output_path logs/rsl_rl/waypoint_nav/pretrained_init.pt
./isaaclab.sh -p ~/drone_project/waypoint_nav/train.py \
    --num_envs 1024 --max_iterations 2000 --headless \
    --resume_path logs/rsl_rl/waypoint_nav/pretrained_init.pt

# Stage 3: language navigation (CLIP)
bash ~/drone_project/scripts/train_lang_nav.sh

# Stage 4: full VLA (PaliGemma)
./isaaclab.sh -p ~/drone_project/vla/train.py \
    --num_envs 256 --max_iterations 5000 --headless --enable_cameras
```

Pretrained checkpoints for the first two stages live in [checkpoints/](checkpoints) (Git LFS) so you can skip straight to Stage 3 or Stage 4 if you only want to iterate on the language and VLA pieces.

## Repository layout

```
drone_project/
├── README.md, LICENSE, requirements.txt
├── activate_env.sh, setup.sh
├── checkpoints/        Pretrained Stage 1 and Stage 2 weights (LFS)
├── docs/               CURRICULUM.md, ADVANCED.md
├── scripts/            Transfer scripts, Stage 3 launcher, FPV camera test
├── hover/              Stage 1 environment + agent
├── waypoint_nav/       Stage 2 environment + agent
├── lang_nav/           Stage 3 environment + agent (CLIP)
├── vla/                Stage 4 environment + agent (PaliGemma)
└── lang_nav_siglip/, pi/, vla_warehouse/, vla_cesium/, vla_universal/, huge_bench/
                        Advanced and experimental modules (see docs/ADVANCED.md)
```

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{codcodingcode_drone_project,
  author = {CodCodingCode},
  title  = {drone_project: language-grounded drone navigation in Isaac Sim},
  year   = {2026},
  url    = {https://github.com/CodCodingCode/drone_project}
}
```

## License

Released under the [MIT License](LICENSE).
