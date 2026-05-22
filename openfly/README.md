# OpenFly — outdoor aerial vision-language navigation

`drone_project` runs entirely on [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform): AirSim / Unreal Engine / 3D Gaussian Splatting scenes, the official seen / unseen / eval_test splits, and the SR / OSR / NE / SPL metrics.

## Quick start

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
| `policies.py` | Policy adapters (heuristic, OpenFly-Agent, PaliGemma) |
| `models/paligemma_vln.py` | Custom PaliGemma + LoRA + LSTM action head |
| `train_paligemma.py` | Offline BC entrypoint for the custom model |
| `run_eval.sh` / `run_train_agent.sh` / `run_train_paligemma.sh` | CLI wrappers |

## Metrics (OpenFly standard)

- **SR** — success rate (within 20 m of goal)
- **OSR** — oracle success (ever within 20 m during the episode)
- **NE** — navigation error (final distance to goal, metres)
- **SPL** — success weighted by path length ratio

## References

- Paper: [OpenFly arXiv:2502.18041](https://arxiv.org/abs/2502.18041)
- Dataset: [HF IPEC-COMMUNITY/OpenFly](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly)
- Platform: [SHAILAB-IPEC/OpenFly-Platform](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
