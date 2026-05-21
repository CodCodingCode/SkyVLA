# External benchmarks

This project can be evaluated on public aerial vision-and-language navigation (VLN) benchmarks via the [`benchmarks/`](../benchmarks/) harness. Results are written to `logs/benchmarks/` (not checked into git).

## Supported benchmarks

| Benchmark | Metrics | Status | Our adapter |
|-----------|---------|--------|-------------|
| [HUGE-Bench](https://huggingface.co/datasets/yu781986168/HUGE_Dataset_task0) | Action MSE (offline) | Runnable | `huge_bench/` + `benchmarks/eval_huge.py` |
| [CityNav](https://arxiv.org/abs/2406.14240) | NE, SR, OSR | Partial | `benchmarks/eval_citynav_oracle.py` |
| [AirNav](https://huggingface.co/datasets/dpairnav/AirNav) | NE, SR, OSR, SPL | Data-heavy | Manual (AirNav repo) |
| [OpenFly](https://shailab-ipec.github.io/openfly/) | SR | Not open yet | — |

## Quick run

```bash
source ~/miniconda3/bin/activate isaac
cd ~/drone_project
pip install pyarrow   # once, for HUGE parquet

# All runnable benchmarks (skips missing data)
bash benchmarks/run_all.sh
```

Or per benchmark:

```bash
python -m benchmarks.run huge --backend waypoint_heuristic --split test_seen
python -m benchmarks.run citynav --citynav_root $HOME/benchmarks_external/citynav
```

Clone external repos (does not download multi-GB assets):

```bash
bash benchmarks/setup_external.sh
```

## Initial results (May 2026)

Run on **NVIDIA A100-SXM4-40GB**, driver **580.105.08**.

### HUGE-Bench task0 (zero-action baseline)

| Split | Samples | Norm. MSE | Raw MSE |
|-------|---------|-----------|---------|
| test_seen | 19,157 | 1.212 | 0.160 |
| test_unseen | 14,856 | 1.234 | 0.160 |

This is a **sanity lower bound** (predict zero deltas). The trained model path uses `HugeBCPolicy` (PaliGemma + MLP); see below.

### CityNav (oracle waypoint mapper, 500 eps/split)

| Split | NE (m) | SR | OSR |
|-------|--------|-----|-----|
| val_seen | 17.5 | 100% | 100% |
| val_unseen | 17.5 | 100% | 100% |

**Caveat:** Goals are given as oracle XYZ — this tests navigation geometry only, not language grounding. Paper baselines (CMA, Seq2Seq) use RGB + depth + instructions and typically achieve much lower SR on the full task.

### Blocked / pending

- **HUGE BC (PaliGemma):** requires `huggingface-cli login` and accepting the [PaliGemma license](https://huggingface.co/google/paligemma-3b-pt-224).
- **AirNav:** needs full `rgbd-new` / NavGym photo cache (multi-GB).
- **OpenFly:** toolchain not fully released.
- **Curriculum VLA:** no Stage 4 checkpoint in repo yet — Isaac Sim eval not benchmarked.

## Model ↔ benchmark mapping

| drone_project output | HUGE-Bench | CityNav / AirNav |
|----------------------|------------|------------------|
| Stage 2 waypoint MLP | N/A (continuous sim) | `target_body_to_*` discretization |
| Stage 4 VLA target (3D body) | N/A | discrete macro-actions (future) |
| `HugeBCPolicy` (dx,dy,dz,dyaw) | **native** | future delta integrator |

Stage 4 VLA (`vla/train.py`) is **not** directly comparable to HUGE action deltas without an export adapter.

## Trained HUGE-Bench eval

```bash
# 1) Accept license + login
huggingface-cli login

# 2) Train (no Isaac Sim)
bash huge_bench/run_train_bc.sh

# 3) Eval
python -m benchmarks.run huge \
  --backend bc_checkpoint \
  --checkpoint logs/huge_bench/<run>/model_20000.pt \
  --split test_seen --out_json logs/benchmarks/huge_bc_test_seen.json
```

## Driver note (Lambda / A100)

If `nvidia-smi` fails with **Driver/library version mismatch**, install utils matching the loaded kernel module:

```bash
sudo apt-get install -y nvidia-utils-580-server
sudo apt-get purge libnvidia-compute-580 libnvidia-compute-570  # stale rc configs only
```

## References

- CityNav: [water-cookie/citynav](https://github.com/water-cookie/citynav)
- AirNav: [nopride03/AirNav](https://github.com/nopride03/AirNav)
- HUGE-Bench dataset: [yu781986168/HUGE_Dataset_task0](https://huggingface.co/datasets/yu781986168/HUGE_Dataset_task0)

See also [benchmarks/README.md](../benchmarks/README.md) for file-level detail.
