# Curriculum Learning

This project trains a Crazyflie quadcopter to follow natural-language commands inside Isaac Sim. We do not learn the full task end-to-end. Instead, we walk the policy up a four-stage curriculum, where each stage extends the previous one with strictly more task structure while preserving the low-level flight skills that have already been learned.

The same `256 x 256` MLP geometry is reused across every stage. Whenever the observation space grows, the input layer is expanded in place and only the columns that correspond to flight state are copied over from the prior checkpoint; new vision and language columns start out zero-initialized so the policy keeps flying competently from step 0.

## Pipeline at a glance

```mermaid
flowchart TB
    subgraph stage1 [Stage1_Hover]
        H1["15-dim obs, 256x256 MLP"]
        H2["Reward: altitude hold + stability"]
    end
    subgraph stage2 [Stage2_WaypointNav]
        W1["Same 15-dim obs and MLP"]
        W2["In-env curriculum: nav fade-in, goal distance ramp, dwell tightening"]
    end
    subgraph stage3 [Stage3_LangNav_CLIP]
        L1["1033-dim obs, GRU + frozen CLIP text and image"]
        L2["Pick the correct object from a language command"]
    end
    subgraph stage4 [Stage4_VLA_PaliGemma]
        V1["RGB + tokens, frozen Stage 2 low-level policy"]
        V2["LoRA fine-tune the VLM that decides where to go"]
    end
    stage1 -->|"transfer_hover_to_waypoint.py or --resume_path"| stage2
    stage2 -->|"partial load, flight-state weights only"| stage3
    stage2 -->|"frozen waypoint MLP inside HierarchicalVLAActor"| stage4
```

## Stage-by-stage breakdown

| Stage | Module | What the drone learns | Key files |
|-------|--------|-----------------------|-----------|
| 1 | [hover/](../hover) | Basic flight: altitude hold, stability, thrust control | [hover_env.py](../hover/hover_env.py), [train.py](../hover/train.py) |
| 2 | [waypoint_nav/](../waypoint_nav) | Point-to-point navigation with a reward curriculum (survival warmup, nav fade-in, goal-distance ramp, dwell tightening) | [waypoint_nav_env.py](../waypoint_nav/waypoint_nav_env.py), [train.py](../waypoint_nav/train.py) |
| 3 | [lang_nav/](../lang_nav) | Language-grounded object selection via frozen CLIP text and image embeddings | [lang_drone_env.py](../lang_nav/lang_drone_env.py), [clip_grounder.py](../lang_nav/clip_grounder.py) |
| 4 | [vla/](../vla) | Full VLA: PaliGemma predicts the target, the frozen Stage 2 policy executes flight, with a precision curriculum after roughly 200 iterations | [vla_drone_env.py](../vla/vla_drone_env.py), [vla_policy.py](../vla/vla_policy.py), [VLA_SYSTEM.md](../vla/VLA_SYSTEM.md) |

### Stage 1 — Hover

The drone sits in an empty arena and learns to hold altitude at a randomly sampled target while staying upright. The observation is a 15-dim vector of linear velocity, angular velocity, projected gravity, body-frame target position, and world-frame position error. The reward combines XY and Z distance shaping, an uprightness term, velocity penalties, an alive bonus, and a `+5` success bonus when the drone is within `0.8 m` of the target. Training runs across 1024 parallel environments for 1500 PPO iterations.

### Stage 2 — Waypoint navigation

The observation space is identical to Stage 1, which is what makes weight transfer trivial. What changes is the task: the drone now flies to a distant goal, dwells there, and then a fresh waypoint respawns mid-episode. The policy is initialized from the hover checkpoint either with [transfer_hover_to_waypoint.py](../scripts/transfer_hover_to_waypoint.py) (a direct copy that resets the optimizer) or by passing the hover `.pt` straight to `--resume_path`.

Most of the curriculum work happens inside the environment itself:

- **Survival warmup.** For the first 5,000 steps only the alive and uprightness terms are active.
- **Navigation fade-in.** Over the next 10,000 steps a `nav_multiplier` ramps from 0 to 1, gradually mixing in the navigation reward terms.
- **Goal distance ramp.** Goals start out 0.5 to 1.5 m away and grow to 1.0 to 3.0 m over 60,000 steps.
- **Dwell zone curriculum.** The acceptance radius tightens from 1.0 m down to 0.15 m over 40,000 steps so the drone is forced to stop precisely.

### Stage 3 — Language navigation

The policy now receives a 1033-dim observation: a 9-dim flight-state slice plus a 512-dim CLIP text embedding for the command and a 512-dim CLIP image embedding from the onboard camera. The architecture switches from a pure MLP to a GRU recurrent policy with `256` hidden units. Training runs at 1024 envs for 3000 iterations with `--enable_cameras`.

Stage 3 reuses the same survival-warmup and nav-fade-in pattern from Stage 2. Reaching the correct object within `0.35 m` yields `+10`, reaching the wrong object yields `-5`, plus dwell, proximity, and pinpoint bonuses. Loading is shape-tolerant: keys whose shapes do not match (for example because the GRU block didn't exist in the Stage 2 MLP) are skipped instead of erroring.

### Stage 4 — Full VLA

The VLA stage replaces frozen CLIP with PaliGemma 3B (LoRA-fine-tuned) and switches to a hierarchical actor. PaliGemma plus a cross-attention head plus an LSTM produce a target position, which is fed into the **frozen Stage 2 waypoint MLP** to compute the actual thrust-and-moment action. This way the high-level policy only has to learn "where to go" while the low-level controller is taken as solved. A short precision curriculum kicks in after about 200 iterations: loose distance and proximity rewards fade out while hover and success rewards amplify, teaching the drone to stop at the target rather than fly through it. Training runs at 256 envs for 5000 iterations with separate optimizers for PPO, the auxiliary cross-attention head (`3e-4`), and LoRA (`1e-6`, enabled after iteration 50).

For the full VLA architecture spec (token layout, auxiliary losses, LoRA targets) see [VLA_SYSTEM.md](../vla/VLA_SYSTEM.md).

## Weight transfer mechanics

All [transfer scripts](../scripts) follow the same recipe:

1. Load the prior-stage checkpoint.
2. Copy the **flight-state columns** (the first 9 obs dims: linear velocity, angular velocity, projected gravity) into the expanded input layer.
3. **Zero-init** every new vision or language input column so the first hidden layer initially ignores them and the policy keeps the flight skills it just learned.
4. Copy the hidden and output MLP weights directly. The `256 x 256` geometry is preserved across stages by design.
5. Reset the optimizer state and iteration counter, and reset the observation normalizer's count to a small value (around 100) so it re-converges quickly on the new input statistics.

The exact dimension expansion per script:

| Script | Stages | Input dim |
|--------|--------|-----------|
| [transfer_hover_to_waypoint.py](../scripts/transfer_hover_to_waypoint.py) | hover -> waypoint | 15 -> 15 (direct copy) |
| [transfer_waypoint_to_vla_siglip.py](../scripts/transfer_waypoint_to_vla_siglip.py) | waypoint -> lang_nav (SigLIP) | 15 -> 1546 |
| [transfer_waypoint_to_vla.py](../scripts/transfer_waypoint_to_vla.py) | waypoint -> VLA action head | 15 -> 2057 |
| [transfer_waypoint_to_pi0.py](../scripts/transfer_waypoint_to_pi0.py) | waypoint -> Pi0 action head | 15 -> 2057 |

The `2057`-dim layout is `[PaliGemma features (2048) | flight state (9)]`; the flight columns sit at the end, which is why the transfer code copies them into `cols[2048:2057]` rather than `cols[0:9]`.

The resume behaviour also differs by stage:

- **Stage 2** uses RSL-RL's `runner.load(...)` for a full state-dict load.
- **Stage 3** does a partial load that skips shape mismatches, so the same checkpoint can migrate from a Stage 2 MLP into the Stage 3 GRU.
- **Stage 4** runs a custom PPO loop and applies the same shape-tolerant filter on `model_state_dict`.

## End-to-end command sequence

```bash
source ~/drone_project/activate_env.sh
cd ~/IsaacLab

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

## Benchmark-aligned extension (Stages 5–8)

Stages 1–4 train the architecture; Stages 5–8 train the **vision-language head only** on benchmark data so we can report numbers on HUGE-Bench (and later AirNav) without disturbing the frozen Stage-2 controller. The high-level rules:

- The Stage-2 waypoint MLP is **never** updated past Stage 2. It is loaded as frozen buffers inside [`HierarchicalVLAActor`](../vla/vla_policy.py).
- Only the PaliGemma LoRA adapters, cross-attention head, depth encoder, LSTM and `target_mlp` are trainable. The action MLP already knows how to fly.
- Benchmark SFT/RL is done on the benchmark **train** split; eval is on `test_seen` / `test_unseen` only. See [BENCHMARK_FAIRNESS.md](BENCHMARK_FAIRNESS.md) for what's claimable.

```mermaid
flowchart LR
    S4[Stage4 vla RL empty arena] --> S5[Stage5 core VLA RL]
    S5 --> S6[Stage6 HUGE highlevel SFT offline]
    S6 --> S7[Stage7 HUGE Isaac RL when sim drops]
    S7 --> S8[Stage8 unified eval]
```

### Stage 5 — Core VLA RL

This is the same trainer as Stage 4 — listed separately because it is the **prerequisite checkpoint** that Stages 6 and 7 resume from. Run from the empty-arena env until reward plateaus, then save the checkpoint that subsequent stages will fine-tune.

```bash
./isaaclab.sh -p ~/drone_project/vla/train.py \
    --num_envs 256 --max_iterations 5000 --headless --enable_cameras
# -> logs/rsl_rl/vla_drone_direct/<run>/model_*.pt
```

### Stage 6 — HUGE high-level offline SFT

[`huge_bench/train_vla_highlevel.py`](../huge_bench/train_vla_highlevel.py) loads the Stage-5 checkpoint and supervises the high-level head only. Labels are **body-frame future waypoints** computed from each episode's trajectory by [`huge_bench/dataset_highlevel.py`](../huge_bench/dataset_highlevel.py) (look ahead K frames, project the displacement into the body frame at the current yaw). Loss is target MSE plus a small cosine direction term.

This is **not** the same as the legacy [`huge_bench/train_bc.py`](../huge_bench/train_bc.py): that script trains a separate delta-action head ([`HugeBCPolicy`](../huge_bench/policy.py)) and bypasses the frozen waypoint policy. Keep it as a baseline action-MSE number for the leaderboard, but the model that flies in Isaac is the hierarchical one trained here.

```bash
python -m huge_bench.train_vla_highlevel \
    --max_steps 5000 \
    --resume_path logs/rsl_rl/vla_drone_direct/<run>/model_<N>.pt
# -> logs/huge_bench_highlevel/<run>/model_*.pt
```

### Stage 7 — HUGE Isaac RL (scaffold)

The HUGE-Bench Isaac Sim toolchain (3DGS-Mesh USDs + scene loaders) is not yet public. [`vla_huge/`](../vla_huge) follows the same shim pattern as [`vla_warehouse/`](../vla_warehouse): a `sys.modules`-patched `train.py` that swaps the env without duplicating the training loop. Until the digital-twin USDs ship, every entry in [`vla_huge/scenes.py`](../vla_huge/scenes.py) resolves to `warehouse_full.usd`, so Stage 7 functions today as a smoke test of the Stage-6-checkpoint resume path.

```bash
./isaaclab.sh -p ~/drone_project/vla_huge/train.py \
    --num_envs 64 --max_iterations 3000 \
    --headless --enable_cameras --scene_id 1_office \
    --resume_path logs/huge_bench_highlevel/<run>/model_<N>.pt
```

When HUGE drops the sim, edit `HugeScene.usd_path` in `scenes.py` to point at the real digital-twin assets. See [`vla_huge/README.md`](../vla_huge/README.md) for details.

### Stage 8 — Unified benchmark eval

[`benchmarks/run_all.sh`](../benchmarks/run_all.sh) now scores three things on HUGE-Bench:

| Backend | Metric | Source |
|---|---|---|
| `eval_huge.py --backend bc_checkpoint` | normalized + raw delta-action MSE | `HugeBCPolicy` (legacy baseline) |
| `eval_huge_vla.py --backend vla_highlevel` | target MSE + median displacement + cosine direction on body-frame future waypoint | `HierarchicalVLAActor` from Stage 6 (or Stage 5 if no Stage 6 ckpt yet) |
| `eval_huge.py --backend waypoint_heuristic` | zero-action MSE | lower bound |

Both VLA backends evaluate on `test_seen` and `test_unseen` separately; results land under `logs/benchmarks/`.

### One-shot runner

[`scripts/train_curriculum_benchmark.sh`](../scripts/train_curriculum_benchmark.sh) chains Stages 5 → 6 → 7 → 8, gating each stage on the previous stage's checkpoint and writing per-stage logs under `logs/curriculum/stageN/`.

For domain fine-tunes (warehouse scenes, Cesium tiles), behaviour-cloning baselines, and the Pi0 alternative, see [ADVANCED.md](ADVANCED.md).
