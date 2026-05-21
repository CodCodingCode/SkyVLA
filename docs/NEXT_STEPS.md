# Next steps: curriculum learning and benchmarks

Roadmap for training the full language-grounded stack and evaluating it on public benchmarks.

## Where you are now

| Stage | Checkpoint in repo | Benchmark coverage |
|-------|-------------------|-------------------|
| 1 Hover | `checkpoints/stage1_hover.pt` | — |
| 2 Waypoint | `checkpoints/stage2_waypoint.pt` | CityNav oracle geometry only |
| 3 Lang (CLIP) | *train required* | — |
| 4 VLA (PaliGemma) | *train required* | HUGE-Bench via `huge_bench/` (after HF login) |

The four-stage curriculum is documented in [CURRICULUM.md](CURRICULUM.md). Weight transfer between stages preserves flight skills while growing observation size.

## Recommended training order

### 1. Finish Stage 2 on your GPU (if not already strong)

Stage 2 is the **frozen low-level controller** for Stages 3–4 and `vla_universal/`. Re-train or extend if waypoint play drifts on new hardware (e.g. A100 after GH200):

```bash
source activate_env.sh
python scripts/transfer_hover_to_waypoint.py \
  --hover_checkpoint checkpoints/stage1_hover.pt \
  --output_path logs/rsl_rl/waypoint_nav/pretrained_init.pt
./isaaclab.sh -p waypoint_nav/train.py \
  --num_envs 1024 --max_iterations 2000 --headless \
  --resume_path logs/rsl_rl/waypoint_nav/pretrained_init.pt
```

**Done when:** drone reaches goals reliably in `waypoint_nav/play.py` or `play_fast.py` (video in `videos/`).

### 2. Stage 3 — language + CLIP (object disambiguation)

```bash
bash scripts/train_lang_nav.sh
```

**Done when:** policy reaches the correct cube/sphere/cylinder from varied phrases in `lang_nav/play.py`.

**Tips:**
- Requires `--enable_cameras`.
- Uses frozen CLIP; GRU policy learns to *select* among three objects.
- Reuses Stage 2 flight columns via partial weight load (see CURRICULUM.md).

### 3. Stage 4 — full VLA (PaliGemma + frozen waypoint)

**Prerequisites:**
- HuggingFace login + [PaliGemma license](https://huggingface.co/google/paligemma-3b-pt-224).
- Stage 2 checkpoint path wired in `vla/train.py` (default: `checkpoints/stage2_waypoint.pt`).

```bash
huggingface-cli login
./isaaclab.sh -p vla/train.py \
  --num_envs 256 --max_iterations 5000 --headless --enable_cameras
```

**Done when:** `vla/play.py --checkpoint <run>/model_*.pt` flies to the commanded object and stops.

**Curriculum inside Stage 4:** precision phase after ~200 iterations (tighter stopping, less fly-through). LoRA on PaliGemma enables after iteration 50.

### 4. Domain fine-tunes (optional, richer scenes)

| Module | When to use |
|--------|-------------|
| `vla_warehouse/` | Warehouse / hospital / office USD scenes |
| `vla_cesium/` | Large-scale city tiles |
| `vla_universal/` | **No training** — scan → semantic map → navigate any USD scene |

Fine-tunes subclass `VLADroneEnv` and only change scene + POI sampling; same PPO + hierarchical actor.

### 5. Offline HUGE-Bench BC (parallel track)

Does not need Isaac Sim; validates PaliGemma + language + images on real trajectories:

```bash
huggingface-cli login
bash huge_bench/run_train_bc.sh
python -m benchmarks.run huge --backend bc_checkpoint \
  --checkpoint logs/huge_bench/<run>/model_20000.pt --split test_seen
```

Compare MSE against baselines in [BENCHMARKS.md](BENCHMARKS.md).

## Benchmark roadmap (after training)

| Priority | Task | Requires |
|----------|------|----------|
| 1 | HUGE-Bench BC eval (test_seen / test_unseen) | HF login + `run_train_bc.sh` |
| 2 | Export Stage 4 VLA → benchmark adapter | Trained `vla/*.pt` + new script |
| 3 | CityNav language-conditioned eval | CityNav image cache + rasterized maps + fine-tune or discrete adapter |
| 4 | AirNav NavGym rollouts | Full [AirNav dataset](https://huggingface.co/datasets/dpairnav/AirNav) |
| 5 | OpenFly | Upstream release |

## GH200 → A100 migration checklist

- [x] Align NVIDIA user-space with kernel (`nvidia-utils-580-server` on Lambda images).
- [ ] Re-verify Isaac Sim launches: `source activate_env.sh` then a short `hover/train.py` smoke run.
- [ ] Confirm `stage2_waypoint.pt` still flies stable; re-train Stage 2 if sim dynamics feel different.
- [ ] Set `huggingface-cli login` on the new machine before Stage 4 or `huge_bench/`.

## Suggested milestones

1. **M1 — Low-level flight:** Stage 2 checkpoint + waypoint video.
2. **M2 — Language in sim:** Stage 3 success on three-object arena.
3. **M3 — Full VLA in sim:** Stage 4 checkpoint + `vla/play.py` video.
4. **M4 — External validity:** HUGE-Bench BC MSE below heuristic baseline; one CityNav/AirNav number with fair protocol.
5. **M5 — Real scenes:** `vla_universal` scan+navigate on warehouse USD without per-scene training.

## Files to read

- [CURRICULUM.md](CURRICULUM.md) — stage design and weight transfer
- [ADVANCED.md](ADVANCED.md) — SigLIP, Pi0, warehouse, Cesium, huge_bench
- [BENCHMARKS.md](BENCHMARKS.md) — benchmark commands and initial numbers
- [benchmarks/README.md](../benchmarks/README.md) — harness implementation
