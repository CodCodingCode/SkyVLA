---
layout: default
title: Whitepaper — Generative visual subgoals for aerial VLN
description: What SkyVLA is trying to achieve.
permalink: /whitepaper/
---

# SkyVLA: Generative Visual Subgoals for Aerial Vision-Language Navigation

> **Status:** active research, code in this repository. Numbers in the
> [results table](research.md) are placeholders until P2 / P3 finish.
> This document is the elevator pitch; the long-form experimental plan is
> in [`RESEARCH.md`](RESEARCH.md), the engineering checklist in
> [`NEXT_STEPS.md`](NEXT_STEPS.md), and what is and isn't claimable is in
> [`BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md).

## 1. The problem

Outdoor aerial vision-language navigation (VLN) — fly a drone to a place
described in natural language — is the kind of task where current
vision-language-action (VLA) models are simultaneously impressive and
brittle. On OpenFly's `seen` split, off-the-shelf VLA fine-tunes such as
the OpenFly-Agent (OpenVLA 7B) clear the benchmark cleanly. On the
`unseen` split — three never-trained outdoor scenes
(`env_game_gtav`, `env_ue_smallcity`, `env_gs_sjtu02`) — the same models
collapse. They learned scene-specific cues rather than a generalizable
navigation behaviour, and the per-env breakdown shows the failure mode
differs by scene type:

| Unseen env | Shift type | Why it matters |
|------------|------------|----------------|
| `env_game_gtav` | New renderer + game world | Hardest visual OOD |
| `env_ue_smallcity` | New UE layout, same engine | Layout / semantics OOD |
| `env_gs_sjtu02` | New 3D-Gaussian-Splatting recon | Real-to-sim recon OOD |

The research question driving SkyVLA is **which of these shifts a
policy can actually fix at training time, and with what mechanism**.

## 2. Why action policies alone are not enough

A monolithic VLA — image + instruction → action — is doing a lot of
implicit work in one forward pass. To execute *"fly past the gray
building, then turn left at the intersection"* the model must:

1. Ground "gray building" in the current pixels.
2. Plan a multi-step trajectory in its head.
3. Pick the next action consistent with that implicit plan.

In familiar scenes the model memorises shortcuts. In unfamiliar ones —
especially under a different renderer — the shortcuts don't transfer,
and there is no explicit intermediate representation to fall back on.

The hypothesis we are testing is borrowed from π0.7 and SuSIE:
**handing the policy a visual target makes the action-selection problem
shorter and more transferrable**. If the model can imagine what the
camera should be seeing at the end of the current sub-instruction, the
policy's job collapses to "pick the action that moves my current view
toward that target."

## 3. What we are building

SkyVLA composes two models that are trained largely independently and
combined at inference time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Instruction + RGB                                                  │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐   curr SigLIP   ┌────────────────────┐         │
│  │  PaliGemma 3B   │ ───────────────►│  SubgoalDiT (50M)  │         │
│  │  (frozen +      │   instruction   │  feature-space     │         │
│  │  LoRA, ~3M)     │ ───────────────►│  diffusion         │         │
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

The novel piece is the **`SubgoalDiT`**: a Diffusion Transformer that
predicts the next-keyframe view *in PaliGemma's own 2048-d SigLIP token
space*, never in pixels. Operating in feature space (a) skips VAE
decoding entirely — predictions feed straight back into PaliGemma's
cross-attention input — and (b) drops inference cost from seconds (pixel
diffusion) to tens of milliseconds (after consistency-model
distillation). It is, to our knowledge, the first feature-space visual
subgoal generator applied to aerial VLN.

The policy backbone is unchanged: PaliGemma 3B with LoRA, the same
[`PaliGemmaVLNPolicy`](../openfly/models/paligemma_vln.py) that powers
the BC and GRPO tracks already in this repository. The only
architectural change is that the cross-attention block now also attends
over a slot of "predicted-subgoal" tokens alongside the current frame
and history.

## 4. Where the ideas come from

| Component | Source |
|-----------|--------|
| Image subgoals as a steering lever for VLAs | [π0.7](https://arxiv.org/abs/2604.15483) (Physical Intelligence, 2026) |
| Diffusion-based subgoal generation | [SuSIE](https://arxiv.org/abs/2310.10639) (Black et al., ICLR 2024) |
| 25 % end-of-segment + 75 % uniform 0–4 s pairing | π0.7 Appendix C |
| Train policy on world-model-generated subgoals | π0.7 ("mitigate train-test mismatch") |
| Curriculum sparse reward in online RL | This repo's pre-existing [GRPO curriculum](../openfly/train_curriculum_grpo.py) (B5 in the experimental matrix) |
| The OpenFly benchmark and the seen/unseen split | [Gao et al., 2025](https://arxiv.org/abs/2502.18041) |

Two design choices specific to SkyVLA are not in any of the above:

- **Feature-space diffusion over SigLIP tokens.** π0.7 and SuSIE both
  generate pixels (or pixel latents). Feature-space generation matches
  what the consumer (PaliGemma's cross-attention) actually eats and
  shortens the inference loop.
- **Pose-delta conditioning.** The OpenFly drone teleports — its pose
  trajectory is deterministic given the action sequence — so we feed
  the body-frame pose delta to the next subgoal explicitly. Without it
  the world model could simply re-derive forward-kinematics; with it,
  the model is forced to predict visual content given the pose, not the
  pose itself.

## 5. Training pipeline

| Phase | Trains | Purpose |
|-------|--------|---------|
| **P1 BC**         | PaliGemma LoRA + action head + cross-attn | Baseline policy from OpenFly expert demonstrations |
| **P2 World model**| `SubgoalDiT` (PaliGemma frozen) | Learn `(curr_siglip, instruction) → subgoal_siglip`. Uses the π0.7 25/75 pairing on real OpenFly frames. |
| **P3 BC + subgoals** | PaliGemma LoRA + heads (DiT frozen) | Teach the policy to use subgoal tokens. Mixes oracle (real-frame) and DiT-generated subgoals 50/50 to close the train/test gap. |
| **P4 CM distill** *(optional)* | A consistency-model student from the P2 teacher | Drop inference cost from 20-step DDIM to 4-step CM. |
| **P5 PPO/GRPO + curriculum + subgoals** | Action head (+ optional LoRA) | Online RL with the easy → medium → hard sparse-reward curriculum already in [`train_curriculum_grpo.py`](../openfly/train_curriculum_grpo.py). PPO's on-policy rollouts subsume DAgger's distribution-shift fix, so we initialize directly from P3 — no DAgger stage. |
| **Eval** | nothing | Per-env unseen breakdown over `env_game_gtav`, `env_ue_smallcity`, `env_gs_sjtu02`. |

The world model is *never trained inside an RL loop*. P2 and P4 are
offline; P3–P6 see it as a frozen feature provider, the same way they
see PaliGemma.

## 6. What we are testing

The unseen split exists to test exactly three hypotheses:

1. **Renderer OOD** (`env_game_gtav`) — does a world model with web-scale
   visual priors transfer to a new renderer, where the policy alone
   doesn't?
2. **Layout OOD** (`env_ue_smallcity`) — does instruction-conditioned
   subgoal prediction help the policy generalise to a new layout under
   the same renderer?
3. **Recon OOD** (`env_gs_sjtu02`) — does the world model close the gap
   between AirSim trajectories (its main training source) and a 3D-GS
   reconstruction at test time?

The honest framing is that all three of these can fail. The most likely
*partial* outcome — and the one most worth a paper if it lands — is
that the world model helps on **layout / recon shifts** (where its
language-conditioned priors transfer) but fails on **renderer shift**
(where the SigLIP feature distribution itself moves under the policy's
feet).

## 7. What we are not claiming

This is a systems + empirical study, not a new VLN architecture or
benchmark. Following [`BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md):

- **Not** claiming "image-subgoals solve aerial VLN" — too broad,
  insufficient prior.
- **Not** claiming zero-shot generalisation — we fine-tune on `train`.
- **Not** claiming superiority over OpenFly-Agent 7B without an
  identical eval harness (same `--max_steps`, same episode count).
- **Not** claiming city-scale exploration — OpenFly episodes are ~70 m
  local hops.

We are claiming that **the per-env unseen breakdown will identify which
type of OOD shift visual subgoals close**, with the curriculum-RL
ablation telling us whether RL adds anything once subgoals are in
place.

## 8. Why feature-space, and why not pixels

Two reasons feature-space wins for SkyVLA specifically:

- **Speed.** A 12-layer DiT over 256+256 SigLIP tokens runs in
  ~30–60 ms after consistency-model distillation. Pixel diffusion at
  the same quality costs 0.5–3 s — fine for π0.7's tabletop pace, fatal
  for thousands of rollouts per PPO iteration.
- **Match to downstream consumer.** PaliGemma's cross-attention block
  eats SigLIP tokens. Predicting pixels and then re-encoding them with
  SigLIP throws away effort and adds drift. Predicting in token space
  directly is the natural fit.

The trade-off is interpretability: a SigLIP token grid is not human-readable.
A pixel decoder (e.g. a small SigLIP → RGB decoder on top of the
predicted tokens) is a P-future engineering item but not on the
critical path for the research question.

## 9. Where the risk lives

Three failure modes, in order of how likely they are to bite:

1. **SigLIP features may not be diffusion-friendly.** SigLIP was
   trained to be a discriminative encoder, not a generative latent
   space. If the geometry of the token distribution is too sharp for
   denoising, the DiT will produce blurry / off-distribution subgoals.
   This is the failure mode validated cheaply in P2: if validation
   cosine similarity between predicted and true subgoal tokens stays
   near zero after a couple of epochs, the bet is wrong.
2. **OpenFly's templated sub-instructions are information-poor.**
   Sub-instructions like "move forward 6 meters" don't carry rich
   linguistic content; conditioning on them may collapse the
   per-instruction posterior. The mitigation is feeding the *full*
   GPT-4o instruction in addition to the template — and accepting that
   the conditioning ceiling is lower than π0.7's.
3. **The drone teleports through obstacles.** OpenFly's A* planner
   produces collision-free paths, but the rendering pipeline doesn't
   enforce occlusion. A world model trained only on this data may
   happily synthesise subgoals inside a building. The longer-term fix
   is to initialise the world model from a public DiT (Stable Diffusion 3
   / PixArt-α) with web-scale physical-scene priors; the v1 from-scratch
   model in this repo is a stepping-stone for that.

## 10. The honest endpoint

A clean negative result here is still publishable — *"web-pretrained
visual subgoals do not close OpenFly's renderer OOD gap, here is the
feature-space evidence why"* — and a clean positive result is a CoRL /
ICRA-shaped paper:

> **"Feature-space generative subgoals from a world model improve
> aerial VLN generalisation on layout and reconstruction shifts but
> not on cross-renderer shifts, and curriculum sparse reward
> compounds the gain on the closable shifts."**

Either way, the per-env breakdown is the headline figure. The
contribution is empirical and mechanistic, not architectural — but the
mechanism (instruction-conditioned visual subgoals as a regulariser for
OOD generalisation in aerial VLN) is one that hasn't been characterised
on outdoor aerial data before.

## 11. Pointers into the code

| Concern | File |
|---------|------|
| OpenFly env wiring & action space | [`openfly/envs/airsim_vln_env.py`](../openfly/envs/airsim_vln_env.py), [`openfly/actions.py`](../openfly/actions.py) |
| PaliGemma BC policy + subgoal slot | [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) |
| Feature-space DiT | [`openfly/models/subgoal_dit.py`](../openfly/models/subgoal_dit.py) |
| P2 world-model trainer | [`openfly/train_subgoal_dit.py`](../openfly/train_subgoal_dit.py) |
| Dataset + 25/75 pairing | [`openfly/dataset.py`](../openfly/dataset.py) |
| Reward presets + curriculum | [`openfly/rewards.py`](../openfly/rewards.py), [`openfly/train_curriculum_grpo.py`](../openfly/train_curriculum_grpo.py) |
| Per-env unseen eval harness | [`openfly/eval_benchmark.py`](../openfly/eval_benchmark.py) |
| Long-form research plan | [`docs/RESEARCH.md`](RESEARCH.md) |
| Engineering checklist | [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) |
