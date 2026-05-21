# External benchmarks for drone_project

This folder wires our models to public aerial VLN benchmarks. Each benchmark has different requirements.

| Benchmark | Metric | Our adapter | Data / sim needed |
|-----------|--------|-------------|-------------------|
| **HUGE-Bench** (action MSE baseline) | normalized + raw delta-action MSE | `huge_bench/policy.py` + [`eval_huge.py`](eval_huge.py) | HuggingFace only |
| **HUGE-Bench** (hierarchical VLA target) | target MSE + median displacement + cosine direction | [`eval_huge_vla.py`](eval_huge_vla.py) (Stage 6 / Stage 7 ckpts) | HuggingFace only |
| **CityNav** | NE, SR, OSR (sim) | `eval_citynav_oracle.py` | Point clouds + image cache |
| **AirNav** | NE, SR, OSR, SPL (sim) | Manual (see AirNav repo) | Multi-GB rgbd + NavGym |
| **OpenFly** | SR (sim) | Not available | Toolchain not open yet |

`eval_huge.py` and `eval_huge_vla.py` are NOT redundant — they score **different models**. The first scores the legacy [`HugeBCPolicy`](../huge_bench/policy.py) on raw delta actions (the "we trained a separate BC head" baseline). The second scores the hierarchical [`HierarchicalVLAActor`](../vla/vla_policy.py) on its actual output: the body-frame target waypoint, with the Stage-2 controller still frozen. See [BENCHMARK_FAIRNESS.md](../docs/BENCHMARK_FAIRNESS.md) for what's claimable from each.

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

**Two evaluation paths.** The legacy `HugeBCPolicy` predicts raw `(dx, dy, dz, dyaw)` deltas, which match the dataset's action labels and produce an action-MSE leaderboard number. The Stage-6 hierarchical VLA predicts a body-frame **target** instead and re-uses the frozen Stage-2 waypoint controller for low-level flight; for it we score the predicted target against a body-frame future-waypoint label computed from the trajectory. Both numbers are useful: the BC MSE is directly comparable to other delta-action baselines on the HUGE leaderboard, while the target MSE shows whether the high-level head learned to localise the goal.

```bash
# Stage-6 hierarchical VLA target eval (after huge_bench/train_vla_highlevel.py)
python -m benchmarks.eval_huge_vla \
    --backend vla_highlevel \
    --checkpoint logs/huge_bench_highlevel/<run>/model_5000.pt \
    --split test_seen --out_json logs/benchmarks/huge_vla_target_test_seen.json
```

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
