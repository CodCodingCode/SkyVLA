# OpenFly — outdoor aerial vision-language navigation

`drone_project` runs entirely on [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform): AirSim / Unreal Engine / 3D Gaussian Splatting scenes, the official seen / unseen / eval_test splits, and the SR / OSR / NE / SPL metrics.

## Quick start

> ⚠️ The OpenFly AirSim binaries are **x86_64 only** — they cannot run on aarch64 hosts (GH200 Grace, Apple Silicon). For x86 + NVIDIA GPU setup (A100 / H100 / L40 / 4090 / etc.) see [`docs/A100_SETUP.md`](../docs/A100_SETUP.md).

```bash
# 1. One-time setup (conda env, clone platform, HF annotations)
bash ~/drone_project/openfly/setup.sh

# 2. Download at least one AirSim scene (several GB)
bash ~/drone_project/openfly/download_airsim_scene.sh env_airsim_16

# 3. Evaluate the oracle heuristic to verify the sim path
source ~/drone_project/openfly/activate.sh
bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5

# 4. Evaluate the official OpenFly-Agent (7B; flash-attn + GPU)
bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy openfly-agent \
  --env_filter env_airsim_16 --max_episodes 10
```

Results land in `logs/benchmarks/openfly_*.json`.

## Action space

10 discrete macros are defined in [`actions.py`](actions.py) and match upstream `train/eval.py`:

| id | macro | id | macro |
|----|-------|----|-------|
| 0 | stop | 5 | down 3 m |
| 1 | forward 3 m | 6 | strafe left 3 m |
| 2 | turn left 30° | 7 | strafe right 3 m |
| 3 | turn right 30° | 8 | forward 6 m |
| 4 | up 3 m | 9 | forward 9 m |

`action_id_to_vector(id)` returns the 8-dim training vector OpenFly's TFDS pipeline expects, and `target_body_to_openfly(pose, goal)` discretises a continuous body-frame target — useful when adapting policies that natively predict a vector goal.

## Policies

| `--policy` | Description |
|------------|-------------|
| `heuristic` | Oracle: face the goal, fly forward, stop within 20 m. Sanity check only. |
| `openfly-agent` | Hugging Face `IPEC-COMMUNITY/openfly-agent-7b`. |
| `paligemma` | Custom PaliGemma BC checkpoint produced by `train_paligemma.py`. |
| `dagger` | PaliGemma checkpoint after DAgger refinement (`train_dagger.py`). |
| `grpo` | PaliGemma checkpoint after GRPO online RL (`train_grpo_paligemma.py`). |
| `ppo` | OpenFly-Agent LoRA + value head from PPO (`train_ppo_openfly_agent.py`); pass `--ppo_ckpt`. |

## Training

Two interchangeable tracks ship with the repo. Both target the same eval harness above.

### Track A — official OpenFly-Agent (OpenVLA 7B)

Wraps upstream `OpenFly-Platform/train/train.py` without reimplementing FSDP locally.

```bash
# 1. Install upstream extras (rlds, tensorflow_datasets) once
cd $OPENFLY_ROOT/train && pip install -r requirements.txt

# 2. Build the TFDS dataset from train.json
cd $OPENFLY_ROOT/train/dataset_builder/vln
tfds build --data_dir $OPENFLY_TFDS_DIR

# 3. Launch training (defaults to 8 GPUs FSDP)
export OPENFLY_TFDS_DIR=~/openfly_tfds
bash ~/drone_project/openfly/run_train_agent.sh \
  --run_root_dir runs/openfly_agent_7b
```

The resulting checkpoint is consumed by `--policy openfly-agent --model_id <path>`.

### Track B — custom PaliGemma BC on `train.json`

Single-GPU offline behaviour cloning over the OpenFly trajectories. The model is `paligemma-3b-pt-224` (LoRA on `q_proj` / `v_proj`) plus an LSTM and a 10-class action head.

```bash
# 1. Make sure train.json + the trajectory frames are on disk
#    OPENFLY_IMAGE_ROOT must point at extracted RGB folders.
export OPENFLY_IMAGE_ROOT=~/assets/OpenFly/images

# 2. Train (logs/openfly/paligemma/<run>)
bash ~/drone_project/openfly/run_train_paligemma.sh \
  --epochs 10 --batch_size 8

# 3. Evaluate the trained policy
bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy paligemma \
  --paligemma_ckpt logs/openfly/paligemma/last/model.pt \
  --env_filter env_airsim_16
```

The custom training entrypoint lives in [`train_paligemma.py`](train_paligemma.py); the model in [`models/paligemma_vln.py`](models/paligemma_vln.py).

## RL track (after SFT)

Once you have an SFT checkpoint from Track A or B, the repo ships a three-stage RL pipeline that talks to AirSim through a Gymnasium wrapper.

```mermaid
flowchart LR
    SFT[SFT checkpoint] --> DAgger
    DAgger --> GRPO[GRPO online RL\n(PaliGemma)]
    DAgger --> PPO[PPO + LoRA\n(OpenFly-Agent 7B)]
    GRPO --> Eval
    PPO --> Eval
```

### Step 0 — Smoke-test the env

```bash
source ~/drone_project/openfly/activate.sh
python -m openfly.scripts.smoke_rl_env --episodes 3 --split seen
```

This drives [`AirSimVLNEnv`](envs/airsim_vln_env.py) with the oracle heuristic and writes a JSONL of trajectories under `logs/openfly/rollouts/`. Use it whenever AirSim changes or you tweak [`rewards.py`](rewards.py).

### Step 1 — DAgger (between SFT and RL)

```bash
bash ~/drone_project/openfly/run_train_dagger.sh \
  --sft_ckpt logs/openfly/paligemma/<run>/last.pt \
  --iterations 3 --episodes_per_iter 200
```

Each iteration: rollout the current policy in AirSim, relabel visited states with [`goal_heuristic_action`](actions.py), and fine-tune on a 50/50 mix of the OpenFly offline dataset and the corrected buffer. For Track A (OpenFly-Agent), pass `--track openfly-agent` to collect a corrected JSONL only.

### Step 2 — Track B online RL (GRPO)

```bash
bash ~/drone_project/openfly/run_train_grpo.sh \
  --init_ckpt logs/openfly/dagger/<run>/last.pt \
  --steps 200 --group_size 4 --instructions_per_step 2
```

GRPO samples `K` trajectories per instruction, scores each with [`compute_episode_reward`](rewards.py), and applies a clipped policy-gradient update with a KL anchor against the frozen init checkpoint. Best checkpoints are gated on a small `unseen` eval (`--eval_every 10 --eval_episodes 8`).

### Step 3 — Track A online RL (PPO + LoRA)

```bash
bash ~/drone_project/openfly/run_train_ppo_agent.sh \
  --iterations 30 --episodes_per_iter 4 --kl_coef 0.02
```

PPO on the OpenVLA 7B with LoRA on `q_proj`/`v_proj` and a small value head ([`models/openfly_agent_rl.py`](models/openfly_agent_rl.py)). Heavier than GRPO; keep batches small and rely on the GH200 unified memory for the prefix forward pass.

### Eval an RL checkpoint

```bash
# PaliGemma after GRPO
bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy grpo \
  --paligemma_ckpt logs/openfly/grpo/<run>/best.pt \
  --env_filter env_airsim_16

# OpenFly-Agent after PPO
bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy ppo \
  --ppo_ckpt logs/openfly/ppo_agent/<run>/best.pt \
  --env_filter env_airsim_16
```

The `dagger` / `grpo` / `paligemma` aliases all load the same `PaliGemmaVLNPolicy` state dict; the name only changes the label in `logs/benchmarks/`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENFLY_ROOT` | `~/OpenFly-Platform` | Upstream platform clone |
| `OPENFLY_ANNOTATION_DIR` | `~/assets/OpenFly/Annotation` | Annotation JSON files |
| `OPENFLY_IMAGE_ROOT` | `OPENFLY_ROOT/uav_vln_data` | Extracted trajectory frames |
| `OPENFLY_TFDS_DIR` | (unset) | Output dir for `tfds build` |

## Files

| File | Role |
|------|------|
| `setup.sh` | Clone platform, build conda `openfly`, fetch annotations |
| `activate.sh` | Source before any OpenFly work |
| `eval_benchmark.py` | Main eval harness |
| `actions.py` | Discrete action space + adapters |
| `episodes.py` | Annotation loaders |
| `dataset.py` | PyTorch dataset over trajectories |
| `platform.py` | Import upstream AirSim / UE / GS bridges |
| `policies.py` | Policy adapters (heuristic, OpenFly-Agent, PaliGemma, PPO) |
| `models/paligemma_vln.py` | Custom PaliGemma + LoRA + LSTM action head |
| `models/openfly_agent_rl.py` | OpenFly-Agent 7B + LoRA + value head for PPO |
| `train_paligemma.py` | Offline BC entrypoint for the custom model |
| `train_dagger.py` | DAgger between SFT and online RL |
| `train_grpo_paligemma.py` | GRPO online RL on the PaliGemma policy |
| `train_ppo_openfly_agent.py` | PPO + LoRA online RL on OpenFly-Agent 7B |
| `envs/airsim_vln_env.py` | Gymnasium wrapper around the AirSim bridge |
| `rewards.py` | Episode-level OpenFly-aligned reward |
| `rollout.py` | Shared trajectory collector used by every RL trainer |
| `scripts/smoke_rl_env.py` | Sanity-check the RL env with the heuristic |
| `run_eval.sh` / `run_train_agent.sh` / `run_train_paligemma.sh` | CLI wrappers (SFT) |
| `run_train_dagger.sh` / `run_train_grpo.sh` / `run_train_ppo_agent.sh` | CLI wrappers (RL) |

## Metrics (OpenFly standard)

- **SR** — success rate (within 20 m of goal)
- **OSR** — oracle success (ever within 20 m during the episode)
- **NE** — navigation error (final distance to goal, metres)
- **SPL** — success weighted by path length ratio

## References

- Paper: [OpenFly arXiv:2502.18041](https://arxiv.org/abs/2502.18041)
- Dataset: [HF IPEC-COMMUNITY/OpenFly](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly)
- Platform: [SHAILAB-IPEC/OpenFly-Platform](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
