# External benchmarks for drone_project

This folder wires our models to public aerial VLN benchmarks. Each benchmark has different requirements.

| Benchmark | Metric | Our adapter | Data / sim needed |
|-----------|--------|-------------|-------------------|
| **HUGE-Bench** | Action MSE (offline) | `huge_bench/` + `eval_huge.py` | HuggingFace only |
| **CityNav** | NE, SR, OSR (sim) | `eval_citynav_oracle.py` | Point clouds + image cache |
| **AirNav** | NE, SR, OSR, SPL (sim) | Manual (see AirNav repo) | Multi-GB rgbd + NavGym |
| **OpenFly** | SR (sim) | Not available | Toolchain not open yet |

## Quick start (HUGE-Bench)

Works without Isaac Sim. Uses the same PaliGemma path as Stage 4 / `huge_bench/`.

```bash
source ~/miniconda3/bin/activate isaac
cd ~/drone_project

# 1) Sanity baseline (no GPU, fast)
python -m benchmarks.run huge \
  --backend waypoint_heuristic \
  --split test_seen --max_batches 100 --device cpu

# 2) Train BC head on HUGE trajectories (~hours on GPU)
bash huge_bench/run_train_bc.sh

# 3) Eval trained checkpoint
python -m benchmarks.run huge \
  --backend bc_checkpoint \
  --checkpoint logs/huge_bench/<timestamp>/model_20000.pt \
  --split test_seen --out_json /tmp/huge_test_seen.json
```

**Note:** Our curriculum VLA checkpoint (`vla/train.py`) is **not** directly comparable to HUGE action deltas. The closest match is `HugeBCPolicy` (PaliGemma + MLP → `(dx,dy,dz,dyaw)`). To eval a trained VLA `.pt` you would need a separate export script; none is checked in yet.

## CityNav

CityNav uses discrete actions in a real-city point-cloud simulator. Full language-conditioned eval requires their CMA/Seq2Seq training stack.

We provide an **oracle waypoint baseline** — goal XYZ from the episode, mapped to discrete moves — to test whether our navigation geometry transfers:

```bash
bash benchmarks/setup_external.sh
export CITYNAV_ROOT=$HOME/benchmarks_external/citynav
# Follow citynav/README.md: download_data.sh, rasterize, build image cache
python -m benchmarks.run citynav --citynav_root $CITYNAV_ROOT --max_episodes 50
```

This is **not** a fair VLA comparison (oracle goals). For that, fine-tune `HierarchicalVLAActor` on CityNav RGB+depth+instructions or train their baselines.

## AirNav

AirNav eval runs inside `NavGym` with pre-rendered views. Setup:

```bash
export AIRNAV_ROOT=$HOME/benchmarks_external/AirNav
# Download https://huggingface.co/datasets/dpairnav/AirNav → $AIRNAV_ROOT/data/
cd $AIRNAV_ROOT && pip install -r requirements.txt
python light_model_eval.py   # CMA/Seq2Seq baselines
```

A drone_project discrete adapter (`benchmarks/adapters/discrete.py`) can map body-frame targets to AirNav's 4 actions; full integration is left to a follow-up once data is on disk.

## OpenFly

OpenFly ([site](https://shailab-ipec.github.io/openfly/)) lists 100k trajectories but the toolchain was not fully open at last check. Track the repo for release.

## Mapping our stack → benchmark action spaces

| Our output | HUGE-Bench | CityNav / AirNav |
|------------|------------|------------------|
| Stage-2 waypoint MLP (thrust) | N/A (continuous sim) | via `target_body_to_*` discretization |
| VLA target body (3,) | N/A | discrete macro actions |
| HugeBCPolicy (4 deltas) | **native** | would need delta integrator |

## GPU / environment issues

If you see `cudaGetDeviceCount` / NVML driver mismatch, use `--device cpu` for HUGE smoke tests or fix the host NVIDIA driver before sim benchmarks.

## Results directory

Write JSON metrics with `--out_json`:

```bash
python -m benchmarks.eval_huge ... --out_json logs/benchmarks/huge_test_seen.json
```
