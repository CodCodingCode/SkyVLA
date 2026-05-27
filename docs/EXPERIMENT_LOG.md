---
layout: default
title: Experiment log — what worked, what didn't, and why
description: Research diary for the SkyVLA world-model + policy pipeline.
permalink: /experiment-log/
---

# SkyVLA experiment log

A condensed record of the work, the experiments, and the lessons.
Written to be self-contained: a future session reading just this doc
should be able to pick up where we are without rereading scrollback.

Cross-references:
- [`docs/WHITEPAPER.md`](WHITEPAPER.md) — what we're trying to achieve
- [`docs/implementation.md`](implementation.md) — how the stack works
- [`docs/RESEARCH.md`](RESEARCH.md) — long-form research plan
- [`docs/JOINT_TRAINING.md`](JOINT_TRAINING.md) — sequential vs joint training design
- [`docs/BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md) — what each number can claim

## 1. Project in one paragraph

SkyVLA is outdoor aerial vision-language navigation on the
[OpenFly benchmark](https://arxiv.org/abs/2502.18041). The thesis is
that **handing the policy a visual subgoal** (an imagined next-keyframe
image / SigLIP feature tensor) **makes action selection easier on
out-of-distribution scenes** — the same idea π0.7 (Physical
Intelligence, 2026) and SuSIE (Black et al., ICLR 2024) used in
manipulation. The novel piece for SkyVLA is doing it in **PaliGemma's
SigLIP token space**, not in pixels, plus pose-delta conditioning
because OpenFly's drone teleports deterministically given the action
sequence.

## 2. The OpenFly benchmark — what it actually is

Before any model design, understand the environment.

* **The drone is a kinematic camera.** Each step is one of 8 discrete
  macros (stop / forward 3-6-9m / turn ±30° / up-down 3m). Pose
  updates via `simSetVehiclePose(target_pose, True)` — the `True`
  means *ignore collisions*. The drone phases through buildings.
* **Trajectories in `train.json` are A\* paths** generated against a
  voxelized point cloud per scene. So expert trajectories don't
  collide, but the simulator at eval time doesn't enforce that the
  policy avoids them either.
* **The paper (§6.2) DOES count collisions** as failure for the SR
  metric, checked externally against the scene's point cloud:
  > "Each environment provides corresponding point clouds that enable
  > collision checking. If a collision occurs, the task is counted
  > as a failure."
  The **reference code apparently doesn't implement this check** —
  neither our earlier code nor `OpenFly-Platform/train/eval.py`
  queries any point cloud at eval time. So vanilla SR numbers from
  either codebase are *inflated* relative to a paper-faithful eval.
* **We added the point-cloud check ourselves** in commit `62f72e9`
  via `openfly/envs/scene_occupancy.py`. Loads `.pcd`, voxelizes,
  exposes `is_occupied(world_xyz)`. Lazy-loaded on first reset into
  a scene; cached as int64-key arrays in `.voxel_cache/`. Triggered
  per-step in `airsim_vln_env.py:step()`. The existing
  `collision_penalty = 0.5` in `rewards.py` (easy/medium presets)
  was wired but never fired before this — now it does.
* **PCD coverage is partial**: 8 of 14 scenes have published `.pcd`
  files on HF (`IPEC-COMMUNITY/OpenFly_DataGen/pcd_map/*.pcd`) —
  all 6 airsim scenes plus `env_ue_bigcity` and `env_ue_smallcity`.
  `env_gs_*` and `env_game_gtav` are not published. The env returns
  `collided=False` for those scenes (with a one-time warning) and
  continues running.

### Eval-set facts (do not relitigate)

| Split | File | Episodes | Use |
|---|---|---|---|
| `train` | `train.json` | 100,226 | SFT, RL rollouts |
| `seen` | `seen.json` | 1,800 | dev benchmark, same 11 scenes as train |
| `unseen` | `unseen.json` | 1,200 | **headline test set** — 3 NEW envs |
| `eval_test` | `OpenFly-Platform/configs/eval_test.json` | 240 | demo set, mostly train-env samples — **not** a meaningful held-out test |

Unseen envs are heterogeneous (test different OOD shifts):

| Env | Episodes | Shift type |
|---|---|---|
| `env_game_gtav` | 406 | renderer + game world |
| `env_ue_smallcity` | 404 | UE layout (same engine as train) |
| `env_gs_sjtu02` | 390 | 3D-GS reconstruction |

## 3. Architecture

```
RGB ─► PaliGemma 3B ──► curr SigLIP tokens (256 × 2048)
        (LoRA only             │
        trainable,             ▼
        base frozen)    ┌─────────────────────┐
                        │  SubgoalDiT (~150M  │  feature-space diffusion
                        │  from-scratch DiT)  │  on SigLIP tokens
                        │  (frozen after P2)  │  no pixels, no VAE
                        └──────────┬──────────┘
                                   │  predicted subgoal SigLIP
   curr + history + predicted-subgoal ──► cross-attn ──► action head
                                                            │
                                                            ▼
                                                discrete action (0..7)
```

* **Policy** ([`paligemma_vln.py`](../openfly/models/paligemma_vln.py)):
  PaliGemma 3B + LoRA on `q/k/v/o`, history pooled to `[CLS]` per
  frame, cross-attention over the image-token stack, 8-class action
  head. `forward()` accepts an optional `subgoal_tokens` kwarg
  (`(B, 256, 2048)` SigLIP tokens) — appended to `image_tokens` with
  a dedicated frame-embedding slot before the cross-attn.
* **World model** ([`subgoal_dit.py`](../openfly/models/subgoal_dit.py)):
  vanilla from-scratch DiT (depth 12, hidden 1024). AdaLN-Zero,
  cosine schedule, DDIM sampler. Conditions on time + Gemma text
  summary + body-frame pose delta + last-action + horizon-to-subgoal
  embedding.
* **PixArt-Σ variant** ([`subgoal_dit_pixart.py`](../openfly/models/subgoal_dit_pixart.py)):
  wraps `diffusers.PixArtTransformer2DModel` (610M params, web
  pretrained). Replaces VAE patch-embed + proj_out with SigLIP-shaped
  I/O adapters. **Negative result; see §6.**

## 4. Training pipeline

The pipeline is **sequential**, not joint:

```
P1 BC ─► P2 World model ─► P3 BC+subgoals ─► (P4 CM distill) ─► P5 PPO/GRPO+curriculum
            │                      │
            └─ frozen everywhere ──┘
               outside of P2 / P4
```

Phase summary:

| Phase | Trains | Frozen | Loss | Wall time |
|---|---|---|---|---|
| P1 BC | PaliGemma LoRA + heads | PaliGemma base | action CE | ~6 hr |
| P2 World model | SubgoalDiT | PaliGemma | ε-MSE | ~3 hr |
| P3 BC+subgoals | PaliGemma LoRA + heads | DiT, PaliGemma base | action CE (with subgoal tokens) | ~3-6 hr |
| P3.5 Joint refine *(optional)* | DiT + LoRA + heads | PaliGemma base | λ_mse · MSE + λ_ce · CE | ~3 hr |
| P5 PPO/GRPO + curriculum | action head (+ optional LoRA) | DiT, PaliGemma base | policy gradient | ~24 hr per stage |

**No DAgger stage** — removed in commits `4d2dd70` (code) and
`912abe9` (docs). Rationale: PPO's on-policy rollouts subsume
DAgger's distribution-shift fix, and OpenFly's geometric oracle
(direction-to-goal, no obstacle map) was too weak to teach
obstacle avoidance in a kinematic env. The 3-line removal note
appears in every doc with "no DAgger" framing.

## 5. The π0.7 recipe we copied (and what we adapted)

Verified from the π0.7 PDF (arXiv 2604.15483), not from blog posts:

| Element | π0.7 | SkyVLA |
|---|---|---|
| World model arch | BAGEL-init, 14B params, pixel-space | SubgoalDiT, 150M from scratch, SigLIP-feature-space |
| Subgoal pairing | 25% end-of-segment + 75% uniform 0-4s ahead | **same 25/75 mix** ([dataset.py:344](../openfly/dataset.py#L344)) |
| Train policy on generated subgoals | yes (mix oracle + WM samples) | yes (50/50, `--dit_mix_prob 0.5`) |
| Text conditioning | T5-XXL embeddings via cross-attn | Gemma summary projected to backbone hidden, our own adapter (see PixArt §6) |
| Multi-camera | 3 cameras | 1 (OpenFly single-camera) |
| Multi-task action head | flow matching | discrete CE on 8 macros |

Specific quote that grounded the pairing implementation (PDF, App C):

> "with probability 0.25, we sample the end-of-segment images
> (consistent with the prediction target for the world model), and
> with probability 0.75 we sample future images uniformly from 0–4
> seconds ahead of the current timestep."

For OpenFly, one macro action ≈ one second of motion, so the
"0–4 seconds" maps to "k ∈ {1, 2, 3, 4}" actions ahead. Implemented
in `OpenFlyDataset` with CLI knobs `--subgoal_pairing
{mixed,semantic_only,uniform_only}`, `--subgoal_semantic_prob`,
`--subgoal_uniform_max`.

## 6. Experiment chronology

Chronological — what we tried, the result, the lesson.

### P2 — World model

| Run | Config | val_cos | Lesson |
|---|---|---|---|
| P2-run1 | depth=8, hidden=768, ~42k samples × 1 epoch | 0.575 | smaller config works |
| P2-run2 | depth=12, hidden=1024, ~42k samples × 3 epochs | 0.612 | +1.6× params for +0.035 |
| P2-run3 | depth=12, hidden=1024, ~70k samples × 2 epochs | 0.612 | more data → same ceiling. **From-scratch plateaus at ~0.612 on this data scale.** |

After run3 we ran the data-bottleneck test (the user named it).
Adding 70k samples vs 42k did not raise the ceiling — only improved
*sample efficiency* (2 epochs vs 3 to reach the same val_cos).
Conclusion: "data is the bottleneck" is half-true. More data helps
get to the ceiling faster, but doesn't raise the ceiling. Something
deeper limits us at ~0.61.

Most likely deeper cause: **PaliGemma's SigLIP features are
*discriminative* by training objective, not generative**. The
feature-space distribution may simply not have a clean denoising
trajectory at the ε-MSE objective we use. We can't fix this without
either (a) a better generative latent space or (b) more data than
our 100k trajectories.

### The PixArt-Σ experiments (v1 – v5, all P2)

Motivation: π0.7 initialized its world model from BAGEL (web-pretrained
image generation model) because robot data alone can't teach a model
"what real scenes look like." We tried the analog with PixArt-Σ.

| Variant | Setup | Result | Diagnosis |
|---|---|---|---|
| **v1** | lr=5e-5 (from-scratch's default), warmup 300, full FT | **NaN @ gstep 700** | LR too high for pretrained backbone; gradient explosion |
| **v2** | lr=1e-5 (5× lower), warmup 1000, full FT | descending @ gstep 700 then **killed by harness** | LR fix worked, but I switched to v3 before finishing |
| **v3** | `--freeze_backbone`, lr=5e-5 (adapter-only) | **plateau loss 0.97** for 2k+ steps | Pretrained features didn't transfer through a thin adapter at the SigLIP-vs-VAE-latent feature-space distance |
| **v4** | Full FT lr=1e-5 (same as v2) | **plateau loss 0.99** for 700+ steps before harness kill | Wrapper bugs — see below |
| **v5** | Three wrapper fixes + param groups | **descended to loss 0.65; val_cos 0.5964** | Wrapper was the problem; PixArt actually trains. **Matched** from-scratch in 1 epoch (vs 3). Didn't *beat* it. |

#### The three wrapper bugs we fixed in v5

Documented in commit `9f8166e`. The wrapper around
`diffusers.PixArtTransformer2DModel` had three issues that combined
to make the model stuck at the "predict zero" loss floor:

1. **Bypassed the pretrained `caption_projection`.** Old wrapper
   projected Gemma summary 2048 → 4096 then ran it through
   PixArt's `caption_projection` (trained for T5-XXL embeddings).
   Statistics didn't match. Cross-attn got garbage. Now we learn a
   fresh `Linear(2048 → 1152)` directly producing
   `encoder_hidden_states`. **This was the biggest one.**

2. **Separate zero-init `extra_modulation` branch** for our pose /
   last-action / horizon conditioning. Old wrapper folded these into
   PixArt's `embedded_timestep`, polluting the pretrained
   `scale_shift_table` modulation distribution. New: extras feed a
   separate zero-init `Linear(1152 → 2×1152)` whose output adds a
   shift/scale to the final hidden. DiT-Zero-style; starts as
   identity, grows in only if training data warrants it.

3. **Parameter groups with different LRs.** Pretrained backbone
   (610M) gets `lr=1e-6`; random-init adapters (10.4M) get `lr=1e-4`
   — 100× ratio. Single global LR was making the adapters never
   learn fast enough to escape near-zero predictions while
   simultaneously pushing the backbone in directions it didn't want
   to go. Standard finetuning recipe.

After v5: PixArt-Σ-init *works mechanically* (the wrapper no longer
sabotages it) but **doesn't beat from-scratch DiT** in val_cos. It
matched 0.612 in one epoch vs three. Sample-efficient, not
absolute-better.

**The honest research conclusion**: π0.7's BAGEL-init thesis
("web-pretrained world model gives a head start") translates to
SkyVLA only as a *sample-efficiency* win, not an *absolute-ceiling*
win. The SigLIP-token-vs-VAE-latent domain gap absorbed most of the
prior. For a paper, this is a partial positive result, not a clean
one.

### P3 — BC + subgoals

| Attempt | Config | Result | Lesson |
|---|---|---|---|
| P3-run1 | full data, BC ckpt from May 25 | **State-dict shape mismatch** at load | BC ckpt predates the subgoal-slot frame_embed change (`[3, 256]` → `[4, 256]`) and the progress_head reshape |
| P3-run2 | Same after `strict=False` + shape-filter fix (commit `fdbdfdd`) | **Process died silently overnight at gstep ~14,276 in epoch 0**; no save | Per-epoch step count was way bigger than estimated (~18k not ~6k). Save only happens end-of-epoch, so 4.5 hr of training was lost |
| P3-run3 | `--max_samples 15000 --epochs 2` (current) | TBD | Cap epoch length so first save happens fast (~75 min). Re-evaluate after first checkpoint. |

#### What we observed in P3 training

CE pinned at ~0.000, accuracy at 1.000 within ~500 steps from launch.
This happens **on both oracle subgoal batches AND DiT-generated
subgoal batches**, even pure-DiT batches (e.g., `dit/oracle=4/0`
showing CE=0.000).

Two interpretations:

* **Optimistic**: the DiT subgoals at val_cos 0.60 carry enough
  action signal that the policy can extract it reliably. The
  observation that pure-DiT batches don't degrade is evidence
  *against* simple oracle-leak memorization.
* **Pessimistic**: the policy has learned to predict actions from
  other available signals (`last_action`, `pose`, `progress`,
  current frame alone) and the subgoal pathway is essentially
  ignored. In which case adding subgoals isn't helping.

**The in-trainer `val_acc` cannot distinguish these** because the
val loop uses oracle subgoals (`dit_mix_prob=0` during val). With
oracle subgoals, even a model that ignores them entirely would
likely score high. **We must run the 3-way eval to know.**

## 7. The 3-way eval (the only number worth reporting)

Built in commit `581c96d`:
[`openfly/eval_p3_subgoals.py`](../openfly/eval_p3_subgoals.py).

Runs a saved P3 checkpoint three ways on the same val split:

| Mode | What it measures | What its high acc means |
|---|---|---|
| `none` | No subgoal pathway (policy is BC baseline) | The policy's intrinsic capability |
| `oracle` | Real `subgoal_rgb` through PaliGemma | Upper-bound, leakage-rich diagnostic |
| **`dit`** | 4-step DDIM samples from the P2 DiT | **Real inference-time performance** |

**Headline derived stat**: `dit_acc − none_acc`. The "does the world
model help?" delta.

**Gap-to-perfect**: `oracle_acc − dit_acc`. How much room is left
for a better world model.

Per-class accuracy is printed because OpenFly's action distribution
is heavily skewed (~53% `forward_9m`). An overall 0.6 acc could be
"always predict forward" — the rare-class accuracies (`stop`,
`turn_left`, `turn_right`) are where subgoals should help most.

Decision rule:

| Result | Action |
|---|---|
| `dit > none + 5%` | World model helps. Proceed to P3.5 or RL. |
| `dit ≈ none ± 2%` | Subgoals don't move the needle. Either world model too weak, or pathway not being used. |
| `dit < none − 2%` | Subgoals are hurting. Debug cross-attn pathway. |
| Per-class: `stop / turn_*` recall ↑ specifically | World model is closing the rare-action gap (best case) |

## 8. What the policy actually trains on

Useful sanity check that confused us earlier.

**Subgoal definition**: the frame at the end of the current
same-action run (semantic mode) or `t + k` for `k ∈ {1, 2, 3, 4}`
actions ahead (uniform mode). Mix is 25/75 per π0.7. Variable
horizon depending on run length.

**Subgoal source during training**:
- 50% of samples: encode the real `subgoal_rgb` through PaliGemma
  ("oracle path")
- 50% of samples: sample subgoal SigLIP tokens from the frozen P2
  DiT via 4-step DDIM ("DiT path")

Mixing both teaches the policy to use whatever subgoal it gets at
inference — closing the train/test gap. Same trick as π0.7.

**Action target**: ground-truth `action_id` from the dataset
(8-way CE). Action target is the *immediate next action* — the
"plan" is implicit.

**Pose target** (auxiliary, weighted 0.1): body-frame next-pose
delta. Smooth L1 loss. Goes through the `goal_pred` head.

## 9. Information leakage — important caveat we caught late

The "oracle" subgoal is the frame AFTER the action was executed.
For OpenFly's deterministic kinematic actions, knowing
`(current_view, next_view)` *uniquely identifies* the action.

This means:
1. **Training with oracle subgoals 50% of the time** gives the
   policy a strong signal — the policy learns to use the
   "look at future view, infer action" trick.
2. **Validation with oracle-only subgoals** (the trainer's default)
   produces a leakage-rich val_acc that doesn't tell us about real
   inference. **The 3-way eval is what we actually trust.**

This is the same setup π0.7 uses — they train with real future
frames and rely on the policy learning to also use the DiT-generated
ones via the train-time mix. Not a bug; a known property. We just
need to evaluate carefully.

## 10. Negative results worth documenting

These are useful to remember so we don't repeat the mistakes.

### DAgger removed entirely

* Geometric "turn-toward-goal" oracle has no obstacle map.
* In OpenFly's kinematic teleport sim, the drone can pass through
  buildings — so the oracle's "always forward" label is sometimes
  *worse* than the BC policy's prior.
* PPO's on-policy rollouts subsume DAgger's purpose (distribution
  shift fix) anyway. So nothing of value is lost.
* Removed in code commit `4d2dd70`, docs commit `912abe9`.

### PixArt-Σ frozen-backbone (v3)

* Loss plateaued at ~0.97 indefinitely.
* 10.4M trainable adapter params could not bridge the
  SigLIP-token-vs-VAE-latent feature distance through PixArt's
  frozen 610M body.
* Implication: web-pretrained DiTs are not "universal feature
  extractors" — their priors are tied to the latent space they
  were trained on.

### "Data is the bottleneck" hypothesis (P2 70k-sample run)

* Adding 70k → 42k samples kept val_cos at 0.612.
* Sample efficiency improved (2 epochs vs 3), ceiling didn't.
* Implication: the limiting factor in our P2 is *something else* —
  likely the discriminative-vs-generative nature of SigLIP space.
  Web init might help (π0.7's BAGEL-init story), but PixArt-Σ
  specifically didn't deliver (matched, didn't beat).

### Single global LR for finetuning a pretrained model

* v1-v4 PixArt experiments all used a single `--lr` for all 624M
  params.
* Pretrained backbone wants 1e-6; random adapters want 1e-4.
* Compromise LR (1e-5 in v2) was simultaneously too high for the
  backbone (NaN at v1, near-NaN at v4) and too low for the adapters
  (stuck at random init).
* Lesson: when finetuning a pretrained model with new heads, use
  param groups. Default in v5 trainer.

### Saving only end-of-epoch when epochs are huge

* P3-run2 hit gstep 14,276 in epoch 0 (out of ~18k for the full
  dataset), then died overnight.
* All 4.5 hr of training lost because the trainer only saves at
  end-of-epoch.
* Lesson: cap `max_samples` so epoch length matches save cadence,
  OR add periodic intra-epoch saves. We chose the former for P3.

## 11. Files / code map

### Code

| Concern | File |
|---|---|
| OpenFly env (gymnasium) | [`openfly/envs/airsim_vln_env.py`](../openfly/envs/airsim_vln_env.py) |
| Point-cloud collision check | [`openfly/envs/scene_occupancy.py`](../openfly/envs/scene_occupancy.py) |
| Action space (8-class) | [`openfly/actions.py`](../openfly/actions.py) |
| Dataset (with 25/75 subgoal pairing) | [`openfly/dataset.py`](../openfly/dataset.py) |
| Episode loader | [`openfly/episodes.py`](../openfly/episodes.py) |
| Reward presets (easy/medium/hard) | [`openfly/rewards.py`](../openfly/rewards.py) |
| Rollout collector | [`openfly/rollout.py`](../openfly/rollout.py) |
| PaliGemma feature extractor | [`vla/vla_policy.py`](../vla/vla_policy.py) |
| Policy (BC backbone with subgoal slot) | [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) |
| World model (from-scratch DiT) | [`openfly/models/subgoal_dit.py`](../openfly/models/subgoal_dit.py) |
| World model (PixArt-Σ-init) | [`openfly/models/subgoal_dit_pixart.py`](../openfly/models/subgoal_dit_pixart.py) |
| P1 BC trainer | [`openfly/train_paligemma.py`](../openfly/train_paligemma.py) |
| P2 world model trainer | [`openfly/train_subgoal_dit.py`](../openfly/train_subgoal_dit.py) |
| P3 BC+subgoals trainer | [`openfly/train_paligemma_subgoal.py`](../openfly/train_paligemma_subgoal.py) |
| P3.5 joint-refine trainer | [`openfly/train_joint_refine.py`](../openfly/train_joint_refine.py) |
| GRPO + reward curriculum | [`openfly/train_grpo_paligemma.py`](../openfly/train_grpo_paligemma.py), [`openfly/train_curriculum_grpo.py`](../openfly/train_curriculum_grpo.py) |
| PPO on OpenFly-Agent 7B | [`openfly/train_ppo_openfly_agent.py`](../openfly/train_ppo_openfly_agent.py) |
| Per-env benchmark eval | [`openfly/eval_benchmark.py`](../openfly/eval_benchmark.py) |
| **3-way subgoal eval** (none/oracle/dit) | [`openfly/eval_p3_subgoals.py`](../openfly/eval_p3_subgoals.py) |

### Docs

| File | Contents |
|---|---|
| [`docs/RESEARCH.md`](RESEARCH.md) | Canonical research plan, experimental matrix |
| [`docs/WHITEPAPER.md`](WHITEPAPER.md) | Vision / motivation |
| [`docs/implementation.md`](implementation.md) | One-pager for the site |
| [`docs/JOINT_TRAINING.md`](JOINT_TRAINING.md) | Sequential vs joint training design |
| [`docs/BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md) | What each leaderboard number can claim |
| [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) | Engineering checklist |
| [`docs/A100_SETUP.md`](A100_SETUP.md) | Host bring-up |
| [`docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | **You are here** |

### On-disk artifacts (this machine)

| What | Where |
|---|---|
| OpenFly annotations | `/home/ubuntu/assets/OpenFly/Annotation/{train,seen,unseen}.json` |
| Per-trajectory frames | `/home/ubuntu/assets/OpenFly/images/Image/<env>/<traj>/<frame_idx>.png` |
| Scene point clouds (.pcd) | `/home/ubuntu/assets/OpenFly/openfly_datagen/pcd_map/<env>.pcd` (8 envs, ~3.8 GB total) |
| PixArt-Σ pretrained snapshot | `/home/ubuntu/assets/pretrained/hf_cache/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/snapshots/<hash>/` |
| P1 BC checkpoint | `~/SkyVLA/logs/openfly/paligemma/20260525_165732/last.pt` |
| P2 world model (v5 PixArt) | `~/SkyVLA/logs/openfly/subgoal_dit/20260526_205830/{best,last}.pt` (val_cos 0.5964) |
| Repo | `/home/ubuntu/SkyVLA/` (symlinked at `~/drone_project`) |

## 12. Standard commands

### Train

```bash
# P1 BC (random head, full train.json)
bash openfly/run_train_paligemma.sh --epochs 10 --batch_size 8

# P2 World model (frozen PaliGemma, mixed 25/75 pairing)
bash openfly/run_train_subgoal_dit.sh \
  --split train --epochs 3 --batch_size 8 \
  --depth 12 --hidden 1024 --num_heads 16 \
  --subgoal_pairing mixed --subgoal_semantic_prob 0.25 --subgoal_uniform_max 4 \
  --save_every 1 --early_stop_patience 2

# P2 with PixArt-Σ init (the v5 config that worked)
bash openfly/run_train_subgoal_dit.sh \
  --epochs 1 --batch_size 4 \
  --warmup_steps 1000 --grad_clip 0.5 \
  --backbone_lr 1e-6 --adapter_lr 1e-4 \
  --pretrained_path /home/ubuntu/assets/pretrained/hf_cache/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/snapshots/<hash>

# P3 BC + subgoals
bash openfly/run_train_paligemma_subgoal.sh \
  --bc_init_ckpt logs/openfly/paligemma/20260525_165732/last.pt \
  --dit_path     logs/openfly/subgoal_dit/20260526_205830/best.pt \
  --pretrained_path /home/ubuntu/assets/pretrained/.../snapshots/<hash> \
  --max_samples 15000 --epochs 2 --batch_size 4 \
  --dit_mix_prob 0.5 --ddim_steps 4

# P3.5 Joint refine (optional, after P3 produces a usable ckpt)
bash openfly/run_train_joint_refine.sh \
  --p3_ckpt  logs/openfly/paligemma_subgoal/<run>/best.pt \
  --dit_path logs/openfly/subgoal_dit/20260526_205830/best.pt \
  --pretrained_path /home/ubuntu/assets/pretrained/.../snapshots/<hash> \
  --epochs 1 --batch_size 2 --ddim_steps 4 \
  --lambda_mse 1.0 --lambda_ce 0.3
```

### Eval

```bash
# 3-way action-accuracy eval (the real headline number)
python -m openfly.eval_p3_subgoals \
  --p3_ckpt   logs/openfly/paligemma_subgoal/<run>/last.pt \
  --dit_path  logs/openfly/subgoal_dit/20260526_205830/best.pt \
  --pretrained_path /home/ubuntu/assets/pretrained/.../snapshots/<hash> \
  --modes none,oracle,dit \
  --split seen \
  --max_episodes 200

# OpenFly per-env unseen benchmark (the paper-quality number)
for ENV in env_game_gtav env_ue_smallcity env_gs_sjtu02; do
  bash openfly/run_eval.sh \
    --split unseen --policy paligemma \
    --paligemma_ckpt <ckpt> --env_filter "$ENV" --max_episodes 50
done
```

### One-time data setup

```bash
bash openfly/setup.sh                           # conda env, OpenFly-Platform, annotations
bash openfly/download_train_images.sh           # ~100 GB of train frames
bash openfly/download_scene.sh env_airsim_16    # ~2 GB scene binary
bash openfly/download_scene.sh env_airsim_16 --with-pcd   # also fetch point cloud for collision check
```

## 13. Current state (as of this writeup)

* **P1 BC checkpoint exists.** `paligemma/20260525_165732/last.pt`.
* **P2 world model exists.** `subgoal_dit/20260526_205830/best.pt`,
  val_cos 0.5964 (the v5 PixArt run, after the three wrapper fixes).
* **P3 is running** (task `bjlwmxhaf`, launched 16:31 UTC) on a
  15,000-sample cap. First save lands ~75 min from launch.
* **3-way eval script exists and is smoke-tested.** Ready to run
  the moment P3 saves.

## 14. Open questions / next experiments

1. **The decisive one: does `dit_acc > none_acc` on the 3-way eval?**
   This is the binary signal that determines whether the world-model
   pathway is contributing. We'll have it within hours.

2. **If positive on `seen`**: does it hold per-env on `unseen`?
   Specifically the layout-shift (`env_ue_smallcity`) and
   recon-shift (`env_gs_sjtu02`) envs are where we expect the
   world model to help most. Renderer-shift (`env_game_gtav`) is
   the hardest — we expect it to fail there.

3. **If P3 passes**: does P3.5 joint refine improve over P3?
   Joint training has been shown to help in π0.7. Our P3.5
   trainer ships with `λ_ce = 0.3, λ_mse = 1.0, 4-step DDIM with
   gradient backprop`. Cost ~3 hr.

4. **P2 ceiling**: can we push val_cos above 0.65 via either
   (a) much more data, (b) a different pretrained init like Sana
   or HunyuanDiT, or (c) a fundamentally different objective
   (consistency models, flow matching)? Open.

5. **OpenFly point-cloud collision faithfulness**: we now check
   collisions on 8/14 envs. The remaining 6 (`env_gs_*`,
   `env_game_gtav`) silently pass through obstacles. For the
   unseen split this affects `env_gs_sjtu02` (passes through) and
   `env_game_gtav` (passes through). Only `env_ue_smallcity` is
   covered. **Numbers from the GS / GTAV scenes are not
   collision-faithful** — flag this in the paper.

## 15. The three most important lessons in one paragraph each

**Lesson 1**: When evaluating a model that consumes a "goal" input,
**always check whether the goal trivially contains the answer**. We
mixed oracle and DiT subgoals during training, fine. But we
evaluated with oracle-only subgoals, which makes the val_acc gameable.
The 3-way eval (`none / oracle / dit`) is the proper diagnostic.

**Lesson 2**: When finetuning a pretrained model with new heads,
**use separate LRs for the backbone and the random-init heads**
(`backbone_lr=1e-6`, `adapter_lr=1e-4` is a sane default). A single
global LR ends up either pushing the backbone too hard or starving
the adapters of gradient. Five PixArt training runs failed for
exactly this reason before we caught it.

**Lesson 3**: **Save checkpoints periodically, not just at
end-of-epoch**, when epochs might take hours. We lost 4.5 hr of
P3 training because the process died at 77% through epoch 0 and
the next save would have been 25 more minutes away. Cheap insurance:
cap `max_samples` so each epoch is small, or add intra-epoch saves.
