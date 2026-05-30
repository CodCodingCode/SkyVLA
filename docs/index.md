---
layout: default
title: SkyVLA
description: Aerial vision-language navigation on OpenFly — generative visual subgoals + curriculum sparse RL.
---

# SkyVLA

Aerial vision-language navigation on the [OpenFly](https://arxiv.org/abs/2502.18041) benchmark. Two models, two ideas:

1. A **feature-space diffusion world model** that imagines the next-keyframe view (π0.7 / SuSIE-style, but in SigLIP token space — no pixels).
2. A **PaliGemma-VLN policy** that picks one of 8 discrete macros per step, conditioned on the imagined subgoal.

## Research question

Do generative visual subgoals + a reward-sparsity curriculum during online RL improve navigation on **genuinely new scenes** — and which type of domain shift does each piece actually close?

## Live demo

P3 policy (PaliGemma BC + SubgoalDiT) navigating the unseen `env_ue_smallcity` Unreal scene. 1920×1080 FPV with HUD; a companion top-down view tracks position vs goal.

<video controls preload="metadata" width="100%" style="max-width: 880px; border-radius: 6px;">
  <source src="https://github.com/CodCodingCode/SkyVLA/raw/main/videos/p3_realsim.mp4" type="video/mp4">
  <a href="https://github.com/CodCodingCode/SkyVLA/raw/main/videos/p3_realsim.mp4">Watch the demo (mp4)</a>
</video>

Reproducer: [`docs/RECORDING_DEMOS.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/RECORDING_DEMOS.md). Top-down companion: [`videos/p3_realsim_topdown.mp4`](https://github.com/CodCodingCode/SkyVLA/blob/main/videos/p3_realsim_topdown.mp4).

## What's new — May 2026 WM breakthrough

The `SubgoalDiT` val cosine similarity sat at the noise floor for two weeks despite training loss dropping cleanly. Diagnosis: a degenerate ε-MSE minimum. Fix: mean-pool the subgoal target to 1 × 2048, add a direct cos-loss on the reconstructed `x0`, keep REPA on as a representation regulariser. First time clearing the noise floor — `val_cos: 0 → 0.107` across two epochs with the **unseen split matching the seen split** (the generalisation signal we were missing). Full writeup: [Implementation §4½](implementation/#4-breaking-the-modal-collapse--what-training-the-wm-actually-took) · Live training: [W&B dashboard](https://wandb.ai/nathanyan2008p-personal/skyvla-subgoal-dit).

## The stack

```
RGB ─► PaliGemma 3B ─► curr SigLIP ──┬──► SubgoalDiT  (~150M) ──► predicted subgoal SigLIP
                                     │                                       │
                                     └─────────────► cross-attn + action head ◄──┘
                                                            │
                                                            ▼
                                                discrete action (0..7)
```

PaliGemma is frozen except for LoRA. The DiT trains offline (P2) and is then frozen for the rest of the pipeline. RL only updates the action head.

## The OOD set

We never report unseen as one averaged number. The three unseen envs test three different shifts:

<div class="env-grid" markdown="1">
<div class="env-card" markdown="1">
### env_game_gtav
**Renderer shift.** Zero GTA episodes in training. Hardest visual OOD.
</div>
<div class="env-card" markdown="1">
### env_ue_smallcity
**Layout shift.** Same Unreal engine as `ue_bigcity` in train, different city geometry.
</div>
<div class="env-card" markdown="1">
### env_gs_sjtu02
**Recon shift.** Same 3D-Gaussian-Splatting pipeline as `gs_sjtu01` in train, different campus.
</div>
</div>

## Borrowed vs. new

| Borrowed | Source |
|----------|--------|
| Visual subgoals as a steering lever for VLAs | [π0.7](https://arxiv.org/abs/2604.15483) (Physical Intelligence, 2026) |
| Diffusion-based subgoal generation | [SuSIE](https://arxiv.org/abs/2310.10639) (Black et al., ICLR 2024) |
| 25 % end-of-segment + 75 % uniform 0–4 s pairing | π0.7 Appendix C |
| Curriculum sparse reward in online RL | This repo |

| New here | What |
|----------|------|
| Feature-space diffusion over SigLIP tokens (not pixels) | Skips VAE decode — predictions feed straight into PaliGemma's cross-attention. ~5× faster than pixel diffusion. |
| Pose-delta conditioning | OpenFly teleports kinematically. Feeding the body-frame delta explicitly forces the DiT to predict visual content given the pose, not the pose itself. |
| First feature-space visual-subgoal generator for aerial VLN | To our knowledge. |

## Where to look next

- [**Implementation tour**](implementation/) — five-minute walk through env, data, policy, world model, training tracks, eval.
- [**Whitepaper**](whitepaper/) — what we are trying to achieve and what we will not claim.
- [Research plan](research-plan/) — long-form experimental matrix.
- [Results](results/) — per-env unseen table, populated as runs complete.
- [Setup](setup/) — quickstart on an A100 host.
- [GitHub](https://github.com/CodCodingCode/SkyVLA) — code.

<details markdown="1">
<summary>BibTeX</summary>

```bibtex
@software{codcodingcode_skyvla,
  author = {CodCodingCode},
  title  = {SkyVLA: outdoor aerial vision-language navigation with OpenFly},
  year   = {2026},
  url    = {https://github.com/CodCodingCode/SkyVLA}
}
```
</details>
