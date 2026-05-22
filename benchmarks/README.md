# Benchmarks for drone_project

The repository targets **OpenFly** — outdoor aerial vision-language navigation in AirSim / Unreal Engine / 3D Gaussian Splatting scenes — as its primary benchmark. The runner in this folder also keeps an oracle baseline against [CityNav](https://github.com/water-cookie/citynav) for sanity-checking the body-frame discretisation logic; both are wired through `benchmarks/run.py`.

| Benchmark | Metric | Adapter | Data / sim |
|-----------|--------|---------|------------|
| **OpenFly** | SR, OSR, NE, SPL | [`openfly/eval_benchmark.py`](../openfly/eval_benchmark.py) | OpenFly-Platform clone + AirSim/UE/3DGS scene |
| CityNav (oracle) | NE, SR, OSR | [`eval_citynav_oracle.py`](eval_citynav_oracle.py) | CityNav clone + point clouds + image cache |

## Quick start (OpenFly)

```bash
bash ~/drone_project/openfly/setup.sh
bash ~/drone_project/openfly/download_airsim_scene.sh env_airsim_16
source ~/drone_project/openfly/activate.sh

python -m benchmarks.run openfly \
  --split unseen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5
```

To evaluate the OpenFly-Agent baseline or your custom PaliGemma checkpoint, swap `--policy openfly-agent` or `--policy paligemma --paligemma_ckpt <path>` (see [`openfly/README.md`](../openfly/README.md)).

## Quick start (CityNav oracle)

CityNav uses discrete actions in a real-city point-cloud simulator. The oracle baseline maps a known goal XYZ through `benchmarks/adapters/discrete.py:target_body_to_citynav` and measures whether the geometry alone gets the agent within 20 m. It is **not** a fair VLA comparison — use it only to validate the discretiser.

```bash
bash benchmarks/setup_external.sh
export CITYNAV_ROOT=$HOME/benchmarks_external/citynav
python -m benchmarks.run citynav --citynav_root "$CITYNAV_ROOT" --max_episodes 50
```

## Running everything

`benchmarks/run_all.sh` chains the OpenFly heuristic and (optionally) OpenFly-Agent / PaliGemma evals, and tags on the CityNav oracle when the dataset is available.

```bash
RUN_OPENFLY_AGENT=1 \
PALIGEMMA_CKPT=logs/openfly/paligemma/<run>/last.pt \
bash benchmarks/run_all.sh
```

Results land under `logs/benchmarks/`.
