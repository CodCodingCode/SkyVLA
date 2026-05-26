---
layout: default
title: Whitepaper — Generative visual subgoals for aerial VLN
description: What SkyVLA is trying to achieve.
permalink: /whitepaper/
---

# SkyVLA: Generative Visual Subgoals for Aerial Vision-Language Navigation

> Status: active research. Numbers in [results](research.md) are
> placeholders until P2 / P3 finish. The long-form experimental plan
> is in [`RESEARCH.md`](RESEARCH.md), the engineering checklist in
> [`NEXT_STEPS.md`](NEXT_STEPS.md), and what each number can claim is in
> [`BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md).

## 1. The problem

Off-the-shelf VLA fine-tunes do well on OpenFly's `seen` split — the
OpenFly-Agent (OpenVLA 7B) gets clean numbers there. On `unseen` they
collapse, and the three unseen scenes don't fail for the same reason:

| Unseen env | Shift type | Why it matters |
|---|---|---|
| `env_game_gtav` | New renderer + game world | Hardest visual OOD |
| `env_ue_smallcity` | New UE layout, same engine | Layout / semantics OOD |
| `env_gs_sjtu02` | New 3D Gaussian-Splat reconstruction | Real-to-sim recon OOD |

SkyVLA exists to find out **which of these shifts a policy can fix at
training time, and with what mechanism**.

## 2. Why action policies alone aren't enough

A monolithic VLA folds three jobs into one forward pass:

1. Ground "gray building" in the current pixels.
2. Plan a multi-step trajectory implicitly.
3. Pick the action consistent with step 2.

In familiar scenes the model memorises shortcuts for all three at once.
Under a new renderer the shortcuts don't transfer, and there's no
intermediate representation to fall back on.

π0.7 and SuSIE both argue that handing the policy a *visual* target
collapses step 3 to "pick the action that moves my current view toward
this image." We're applying the same lever to aerial VLN.

## 3. What we're building

Two models, trained largely independently, composed at inference:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Instruction + RGB                                                  │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐   curr SigLIP   ┌────────────────────┐         │
│  │  PaliGemma 3B   │ ───────────────►│ SubgoalDiT (~150M) │         │
│  │  (frozen +      │   instruction   │ feature-space      │         │
│  │  LoRA, ~3M)     │ ───────────────►│ diffusion          │         │
│  └─────────────────┘   pose delta    │                    │         │
│           │                          └─────────┬──────────┘         │
│           │   curr SigLIP                      │  predicted         │
│           └─────────────┬──────────────────────┘  subgoal SigLIP    │
│                         ▼                                           │
│             ┌────────────────────────┐                              │
│             │  Cross-attn fusion +   │                              │
│             │  action head (8-way)   │                              │
│             └───────────┬────────────┘                              │
│                         ▼                                           │
│                  next discrete action                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

`SubgoalDiT` is a Diffusion Transformer that predicts the next-keyframe
view in PaliGemma's own 2048-d SigLIP token space — no pixels, no VAE.
Two consequences:

- Inference cost drops from seconds (pixel diffusion) to tens of
  milliseconds. That matters when one PPO iteration burns thousands of
  rollouts.
- The output feeds straight into PaliGemma's cross-attention input.
  Predicting pixels and re-encoding them with SigLIP throws away effort
  and adds drift.

The world model is ~150M params, trained from scratch (depth 12, hidden
1024) with a NaN-guarded skip-step for LR stability. We tried a
PixArt-Σ web-pretrained backbone as a π0.7-style "start from a model
that already knows what scenes look like" alternative; it didn't
transfer. The negative result and the mechanism are in §10.

The policy backbone is the same
[`PaliGemmaVLNPolicy`](../openfly/models/paligemma_vln.py) the BC and
GRPO tracks already use. The only change is that its cross-attention now
attends over a slot of predicted-subgoal tokens alongside the current
frame and history.

## 4. Where the ideas come from

| Component | Source |
|---|---|
| Image subgoals as a steering lever for VLAs | π0.7 (Physical Intelligence, 2026) |
| Diffusion subgoal generation | SuSIE (Black et al., ICLR 2024) |
| 25 % end-of-segment + 75 % uniform 0–4 s pairing | π0.7 Appendix C |
| Training the policy on world-model-generated subgoals | π0.7 |
| Curriculum sparse reward in online RL | The existing GRPO curriculum in this repo (B5 in the experimental matrix) |
| The OpenFly benchmark and the seen / unseen split | Gao et al., 2025 |

Two design choices that are SkyVLA's:

- **Feature-space diffusion over SigLIP tokens.** π0.7 and SuSIE both
  generate pixels (or pixel latents). Generating in token space matches
  what the downstream consumer eats.
- **Pose-delta conditioning.** OpenFly's drone teleports — its pose
  trajectory is deterministic given the action sequence. We feed the
  body-frame pose delta to the next subgoal explicitly, so the world
  model can't cheat by re-deriving forward kinematics. It has to predict
  visual content given the pose.

## 5. Training pipeline

| Phase | Trains | Purpose |
|---|---|---|
| **P1 — BC** | PaliGemma LoRA + action head + cross-attn | Baseline policy from OpenFly expert demos |
| **P2 — World model** | `SubgoalDiT` (PaliGemma frozen) | `(curr_siglip, instruction, pose_delta) → subgoal_siglip` with π0.7's 25 / 75 pairing |
| **P3 — BC + subgoals** | PaliGemma LoRA + heads (DiT frozen) | Teach the policy to use subgoal tokens. Mix oracle and DiT-generated subgoals 50 / 50 to close the train / test gap |
| **P4 — CM distill** (optional) | A consistency-model student from P2 | 20-step DDIM → 4-step CM, lower inference cost |
| **P5 — PPO / GRPO + curriculum** | Action head (and optionally LoRA) | Online RL with the easy → medium → hard sparse-reward curriculum |
| **Eval** | nothing | Per-env unseen breakdown |

The world model is never trained inside an RL loop. P2 and P4 are
offline; everything after sees it as a frozen feature provider, like
PaliGemma. There is no DAgger stage — PPO's on-policy rollouts subsume
the distribution-shift fix, and OpenFly's geometric oracle is too weak
to teach obstacle avoidance in a kinematic env.

## 6. What we're testing

The unseen split tests three hypotheses:

1. **Renderer OOD (`env_game_gtav`)**: does a world model with web-scale
   visual priors transfer to a new renderer where the policy alone
   doesn't?
2. **Layout OOD (`env_ue_smallcity`)**: does instruction-conditioned
   subgoal prediction help the policy generalise to a new layout under
   the same renderer?
3. **Recon OOD (`env_gs_sjtu02`)**: does the world model close the gap
   between AirSim training trajectories and a 3D-GS reconstruction at
   test time?

All three can fail. The most likely partial outcome is that subgoals
help on layout and recon shifts (where language-conditioned priors
transfer) but fail on renderer shift, where the SigLIP feature
distribution itself moves.

## 7. What we're not claiming

Per [`BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md):

- Not "image subgoals solve aerial VLN" — too broad for one repo.
- Not zero-shot generalisation — we fine-tune on `train`.
- Not superiority over OpenFly-Agent 7B without an identical eval
  harness (same `--max_steps`, same episode count).
- Not city-scale exploration — OpenFly episodes are ~70 m local hops.

The claim is narrower: the per-env unseen breakdown will tell us *which*
OOD shift visual subgoals close, and the curriculum-RL ablation will
tell us whether RL adds anything once subgoals are in place.

## 8. Why feature-space, not pixels

- **Speed.** A 12-layer DiT over 256 + 256 SigLIP tokens runs in ~30–60 ms
  after consistency distillation. Pixel diffusion at comparable quality
  costs 0.5–3 s — fine for π0.7's tabletop pace, fatal for thousands of
  PPO rollouts per iteration.
- **Consumer match.** PaliGemma's cross-attention already eats SigLIP
  tokens. Predicting pixels just to re-encode them with SigLIP wastes
  effort and adds drift.

The trade-off is interpretability. A SigLIP token grid isn't
human-readable. A small SigLIP → RGB decoder for visualisation is a
future engineering item, not on the critical path for the research
question.

## 9. Where the risk lives

Three failure modes, ordered by how likely they are to bite:

1. **SigLIP may not be diffusion-friendly.** It was trained as a
   discriminative encoder, not a generative latent space. If its token
   geometry is too sharp for denoising, the DiT will produce blurry /
   off-distribution subgoals. P2 validation cosine similarity is the
   cheap probe — if it stays near zero after a couple of epochs, the
   bet is wrong.
2. **OpenFly's templated sub-instructions are information-poor.** "Move
   forward 6 meters" doesn't carry rich linguistic content; conditioning
   on it may collapse the per-instruction posterior. Mitigation: feed
   the full GPT-4o instruction alongside the template, and accept that
   the conditioning ceiling is lower than π0.7's.
3. **The drone teleports through obstacles.** OpenFly's A* paths are
   collision-free, but the renderer doesn't enforce occlusion. A world
   model trained only on this data may happily synthesise subgoals
   inside a building. Mitigation: the point-cloud collision penalty in
   [`rewards.py`](../openfly/rewards.py) for the RL phase, fed by the
   `.pcd` voxel maps we ship for the AirSim and smallcity scenes.

## 10. Ablations: what didn't work

Worth reporting because each one rules out a recipe other people would
also try.

- **Frozen PixArt-Σ backbone + thin SigLIP I/O adapter** (`subgoal_dit_pixart.py`,
  ~620M total / ~14M trainable adapter heads). The bet: web-pretrained
  scene priors from a DiT-XL/2 trained on 33M image-text pairs should
  transfer to aerial subgoal prediction through a small adapter that
  translates between SigLIP token space and the backbone's VAE-latent
  hidden dim. The reality: PaliGemma's SigLIP feature distribution and
  PixArt's VAE-latent hidden distribution are far enough apart that a
  14M-param adapter can't bridge them. Validation cosine similarity
  stayed near zero. The pretrained backbone needs to actually adapt for
  the priors to be useful — a frozen backbone with a thin adapter is
  not the right partition.

  This is the cleanest comparison we could have run against the
  from-scratch DiT, and the negative result is the takeaway:
  *frozen pretrained-DiT init fails for SigLIP-feature-space subgoal
  diffusion*. The from-scratch ~150M DiT (full finetune, NaN-guarded
  step) remains the only world model in the live pipeline. A
  full-finetune PixArt-Σ run is open but not on the critical path.

## 11. The honest endpoint

A clean negative result is publishable —
*"web-pretrained visual subgoals don't close OpenFly's renderer OOD
gap, here is the feature-space evidence why"* — and a clean positive
result is a CoRL / ICRA-shaped paper:

> Feature-space generative subgoals from a world model improve aerial
> VLN generalisation on layout and reconstruction shifts but not on
> cross-renderer shifts, and curriculum sparse reward compounds the
> gain on the closable shifts.

Either way, the per-env breakdown is the headline figure. The
contribution is empirical and mechanistic, not architectural — but the
mechanism (instruction-conditioned visual subgoals as a regulariser for
OOD generalisation in aerial VLN) hasn't been characterised on outdoor
aerial data before.

## 12. Pointers into the code

| Concern | File |
|---|---|
| OpenFly env wiring + action space | [`openfly/envs/airsim_vln_env.py`](../openfly/envs/airsim_vln_env.py), [`openfly/actions.py`](../openfly/actions.py) |
| PaliGemma BC policy + subgoal slot | [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) |
| Feature-space DiT — live world model (~150M, from-scratch) | [`openfly/models/subgoal_dit.py`](../openfly/models/subgoal_dit.py) |
| Feature-space DiT — failed ablation (PixArt-Σ frozen + adapter; see §10) | [`openfly/models/subgoal_dit_pixart.py`](../openfly/models/subgoal_dit_pixart.py) |
| P2 world-model trainer | [`openfly/train_subgoal_dit.py`](../openfly/train_subgoal_dit.py) |
| Dataset + 25 / 75 subgoal pairing | [`openfly/dataset.py`](../openfly/dataset.py) |
| Reward presets + curriculum | [`openfly/rewards.py`](../openfly/rewards.py), [`openfly/train_curriculum_grpo.py`](../openfly/train_curriculum_grpo.py) |
| Per-env unseen eval harness | [`openfly/eval_benchmark.py`](../openfly/eval_benchmark.py) |
| Long-form research plan | [`docs/RESEARCH.md`](RESEARCH.md) |
| Engineering checklist | [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) |
