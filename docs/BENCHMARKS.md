# Benchmarks

This project uses [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) as its primary benchmark and exposes a small number of supplementary baselines through `benchmarks/run.py`.

| Benchmark | Metrics | Status | Adapter |
|-----------|---------|--------|---------|
| [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) | SR, OSR, NE, SPL | Primary | [`openfly/eval_benchmark.py`](../openfly/eval_benchmark.py) |
| [CityNav](https://github.com/water-cookie/citynav) (oracle baseline) | NE, SR, OSR | Optional | [`benchmarks/eval_citynav_oracle.py`](../benchmarks/eval_citynav_oracle.py) |

## OpenFly

OpenFly evaluates outdoor aerial vision-language navigation across AirSim, Unreal Engine, and 3D Gaussian Splatting scenes. Episodes from the official `seen.json` and `unseen.json` annotation files are run with the upstream simulation bridges.

```bash
bash ~/drone_project/openfly/setup.sh
bash ~/drone_project/openfly/download_airsim_scene.sh env_airsim_16
source ~/drone_project/openfly/activate.sh

python -m benchmarks.run openfly \
  --split unseen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5
```

Three policies are available out of the box:

- `heuristic` — oracle that turns toward the episode goal and lands within 20 m. Used as a sanity check on the simulator and metrics.
- `openfly-agent` — the official `IPEC-COMMUNITY/openfly-agent-7b` from Hugging Face.
- `paligemma` — the custom PaliGemma + LoRA + LSTM checkpoint produced by [`openfly/train_paligemma.py`](../openfly/train_paligemma.py). Pass `--paligemma_ckpt <path>`.

See [`openfly/README.md`](../openfly/README.md) for the full eval and training reference.

## CityNav (oracle baseline)

The CityNav adapter is **not** a fair vision-language comparison: it hands the policy the ground-truth goal XYZ and only measures whether the discretiser in [`benchmarks/adapters/discrete.py`](../benchmarks/adapters/discrete.py) can reach it. Useful as a regression test for the body-frame action mapping.

```bash
bash benchmarks/setup_external.sh
export CITYNAV_ROOT=$HOME/benchmarks_external/citynav
python -m benchmarks.run citynav --citynav_root "$CITYNAV_ROOT" --max_episodes 50
```

## Running everything

```bash
RUN_OPENFLY_AGENT=1 \
PALIGEMMA_CKPT=logs/openfly/paligemma/<run>/last.pt \
bash benchmarks/run_all.sh
```

Results land under `logs/benchmarks/`.

## What we report

- Always specify the split (`seen`, `unseen`, or a single env via `--env_filter`).
- Always specify the policy and (for `paligemma`) the checkpoint path.
- The OpenFly summary JSON includes per-episode SR / OSR / NE / SPL plus the simulator's exit reason — keep the file alongside any leaderboard claim.

See [`BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md) for what is and is not claimable from each evaluation mode.
