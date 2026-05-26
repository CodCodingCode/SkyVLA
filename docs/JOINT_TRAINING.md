---
layout: default
title: Joint training — P3 sequential + P3.5 joint refine
description: How the world model and the policy are trained together (eventually).
permalink: /joint-training/
---

# Joint training: P3 sequential + P3.5 joint refine

## Why this doc exists

The SkyVLA training pipeline is **mostly sequential** — P1 BC, then P2
world model, then P3 BC + subgoals, then P5 RL — with each phase
training one model under one loss. That's the safe engineering default
and it's what most precedents in the field (SuSIE, GR-2,
RT-Trajectory) use.

But pure sequential leaves a real win on the table: the world model
in P2 optimises for *visual accuracy* (cosine similarity to the real
next-keyframe SigLIP tokens), not for *policy usefulness*. Those
metrics are correlated but not identical. A subgoal that captures
"vague aerial scene forward" might score val_cos 0.6 yet provide more
decision-relevant information than a 0.7 subgoal that captures every
building texture but blurs turn direction. π0.7's policy and world
model are trained with overlapping objectives precisely so the world
model learns to produce subgoals the policy can *use*, not just
subgoals that look right.

This document describes the two-stage strategy we ship:

* **P3 BC + subgoals (sequential)** — the foundation. PaliGemma LoRA
  + heads trains under standard action cross-entropy, with the frozen
  P2 DiT providing subgoal tokens via cross-attention. Mixes oracle
  (real next-keyframe encoded through PaliGemma) and DiT-generated
  subgoals so the policy learns to use both kinds. This is the
  "standard" path for an aerial-VLN paper.
* **P3.5 joint refine (optional, the π0.7 move)** — short follow-up
  phase where the DiT *and* the policy are unfrozen together and
  trained under a weighted sum of `λ_mse · ε-MSE + λ_ce · action_CE`.
  The DiT learns to produce subgoals that help the policy succeed,
  not just subgoals that match real frames. ~1 epoch is usually
  enough; the goal is alignment, not from-scratch training.

If P3 alone produces a meaningful action-accuracy improvement over P1
BC, the joint refine is gravy. If P3 doesn't improve over P1, the
joint refine might rescue it by giving the DiT a different signal
than pure MSE.

## Phase ownership recap

| Phase | Trains | Frozen | Loss | Wall time |
|---|---|---|---|---|
| **P1 BC** | PaliGemma LoRA + action head + cross-attn | PaliGemma base | action CE | ~6 hr |
| **P2 World model** | DiT (vanilla or PixArt) | PaliGemma entirely | ε-MSE | ~3 hr |
| **P3 BC + subgoals** | PaliGemma LoRA + heads | DiT, PaliGemma base | action CE (with subgoal tokens in scope) | ~6 hr |
| **P3.5 joint refine** | DiT + PaliGemma LoRA + heads | PaliGemma base | λ_mse · ε-MSE + λ_ce · action_CE | ~3 hr |
| **P4 CM distill** *(optional)* | student CM | teacher DiT | consistency-model loss | ~2 hr |
| **P5 PPO/GRPO + curriculum** | action head (+ optional LoRA) | DiT, PaliGemma base | policy gradient | ~24 hr per stage |

After P3.5, the joint-refined DiT is **frozen again** for P5. RL never
trains the world model. The world model is, at all times, a
feature provider — the question is just how thoroughly its features
were aligned to the policy's needs before RL starts.

## P3: BC + subgoals (the sequential trainer)

### Idea

Take a frozen P2 DiT, attach it to the existing
[`PaliGemmaVLNPolicy`](../openfly/models/paligemma_vln.py)'s subgoal
slot, and run standard BC training. The cross-attention block now
attends over `[history_CLS, current_SigLIP, predicted_subgoal_SigLIP]`
instead of `[history_CLS, current_SigLIP]`.

Two sources of subgoal tokens during training, mixed per-sample:

1. **Oracle subgoals** — encode `subgoal_rgb` (from the dataset's
   25/75 pairing) through PaliGemma to get real
   `(B, 256, 2048)` SigLIP tokens. Perfect-quality target.
2. **DiT-generated subgoals** — sample from the frozen P2 DiT via 4-step
   DDIM. Same shape; quality bounded by the DiT's val_cos.

Mixing ratio is a knob: `--dit_mix_prob` (default 0.5). Per-batch
each sample independently flips a coin. **Why mix instead of one or
the other:**

- Oracle-only: policy gets a "lucky" subgoal it'll never have at
  inference time → train/test gap, policy can't recover from imperfect
  DiT samples.
- DiT-only: policy can never tell whether a bad action was because the
  subgoal was bad or because the policy didn't use it well → harder
  optimisation landscape.
- Mix: policy sees both, learns to extract whatever signal exists.
  π0.7 does exactly this.

### Trainable / frozen

```
Trainable:
  PaliGemma LoRA (q/k/v/o, rank 16)        ~2M
  Cross-attention pool                      ~1M
  Action head (8-class) + aux heads         ~1M
  Frame CLS pool                            ~50k
                                          ─────
                                           ~4M

Frozen:
  PaliGemma base (3B)
  SubgoalDiT (150M from-scratch OR 620M PixArt)
```

### Per-step forward (simplified)

```python
# 1. Encode current frame + (optionally) subgoal frame through PaliGemma
curr_tokens, text_embed = encode_frame(rgb, instruction)
if random() < args.dit_mix_prob:
    # DiT-generated subgoal
    subgoal_tokens = dit.ddim_sample(curr_tokens, text_embed, ...)
else:
    # Oracle subgoal — encode the real next-keyframe RGB
    subgoal_tokens, _ = encode_frame(subgoal_rgb, instruction)

# 2. Run policy with both
action_logits = policy(
    curr_tokens=curr_tokens,
    subgoal_tokens=subgoal_tokens,    # NEW input slot
    text=text_embed,
    pose=pose,
    last_action=last_action,
    history=history,
)

# 3. Standard BC loss
loss = F.cross_entropy(action_logits, gt_action)
```

The policy's `forward()` already accepts a `subgoal_dit` field; we
extend it to also accept a pre-computed `subgoal_tokens` tensor so the
trainer can skip the DiT sampling step on oracle batches.

### Run

```bash
bash openfly/run_train_paligemma_subgoal.sh \
  --bc_init_ckpt logs/openfly/paligemma/<run>/last.pt \
  --dit_path     logs/openfly/subgoal_dit/<run>/best.pt \
  --dit_mix_prob 0.5 \
  --epochs       5 \
  --batch_size   4 \
  --lora_lr      1e-5 \
  --head_lr      3e-4
```

### Decision rule

- **P3 SR (seen) > P1 SR + 3%** → subgoals are useful. Proceed to P5
  RL (with subgoals).
- **P3 SR ≈ P1 SR** → subgoals at this val_cos don't help. Either
  (a) run P3.5 joint refine to give the DiT a different signal, or
  (b) push P2 val_cos higher first (more data, larger model).
- **P3 SR < P1 SR** → subgoals are *hurting*. Likely a bug in the
  subgoal pathway through cross-attention. Debug before continuing.

## P3.5: joint refine (the optional π0.7 move)

### Idea

After P3 has produced a policy that can use subgoals, **unfreeze the
DiT** for a short joint-training phase. Both the DiT and the policy
are updated, but under a *combined* loss:

```
loss = λ_mse · diffusion_eps_mse(dit_predictions)
     + λ_ce  · action_cross_entropy(policy_predictions_given_dit_subgoals)
```

The key gradient flow we add is: **action_CE → policy → cross_attn →
subgoal_tokens → DiT.token_out → DiT blocks**. The DiT now gets a
signal that says "produce subgoals that help the policy succeed,"
not just "produce subgoals that match real frames."

In π0.7's framing this is what aligns the world model to be a *useful*
prior generator for the action policy specifically — visual fidelity
and action utility are correlated but not identical, and the joint
phase shrinks the gap.

### Trainable / frozen

```
Trainable:
  SubgoalDiT (entire model — was frozen in P3)
  PaliGemma LoRA + heads (same as P3)
                                          ─────
                                          ~150M-620M depending on world model

Frozen:
  PaliGemma base (3B)
```

The DiT's pretrained-or-trained weights are *finetuned* here. The LR
should be small (DiT base LR ~1e-6, adapter LR ~1e-5 if PixArt-init)
because we're refining, not retraining.

### Per-step forward

```python
# 1. Encode current + subgoal frames
curr_tokens, text_embed = encode_frame(rgb, instruction)
tgt_tokens, _           = encode_frame(subgoal_rgb, instruction)

# 2. DIFFUSION LEG: train DiT to predict subgoal_siglip from noisy version
t           = randint(0, num_timesteps, (B,))
x_t, noise  = dit.q_sample(tgt_tokens, t)
eps_pred    = dit(curr_tokens, x_t, t, text_embed, ...)
mse_loss    = (eps_pred - noise).pow(2).mean()

# 3. POLICY LEG: sample a subgoal from the trainable DiT, feed policy
#    Use a SHORTER DDIM (4-step CM if available; 8-step otherwise) so
#    the gradient path through diffusion is manageable.
#    Set DiT.eval() momentarily? NO — we want gradients to flow.
subgoal_sample = dit.ddim_sample(
    curr_tokens=curr_tokens.detach(),  # detach curr inputs so we don't double-count
    text_embed=text_embed,
    ...,
    num_steps=4,  # << was 20 in P2, now 4 for tractable backprop
)
action_logits = policy(
    curr_tokens=curr_tokens,
    subgoal_tokens=subgoal_sample,
    ...,
)
ce_loss = F.cross_entropy(action_logits, gt_action)

# 4. Joint backward
loss = args.lambda_mse * mse_loss + args.lambda_ce * ce_loss
loss.backward()
```

**Two important practical choices** in this loop:

1. **Backprop through diffusion sampling.** This is the expensive part.
   We use a 4-step DDIM sampler (not 20) and rely on the fact that we
   only need *approximate* gradients to align the DiT with the policy
   — the diffusion MSE leg keeps the DiT close to its visually-faithful
   solution, and the joint phase only nudges it.
2. **`lambda_ce / lambda_mse` ratio.** Sensitive. Defaults: λ_mse = 1.0
   (keep the DiT close to its P2 trajectory), λ_ce = 0.3 (gently pull
   the DiT toward policy-useful subgoals). Both flags are CLI-tunable.
   If λ_ce is too high, the DiT collapses to mode-locked predictions
   that happen to score well on the most common actions. If too low,
   the joint phase has no effect.

### Run

```bash
bash openfly/run_train_joint_refine.sh \
  --p3_ckpt     logs/openfly/paligemma_subgoal/<run>/last.pt \
  --dit_path    logs/openfly/subgoal_dit/<run>/best.pt \
  --epochs      1 \
  --batch_size  2 \           # smaller — backprop through DDIM is expensive
  --ddim_steps  4 \
  --lambda_mse  1.0 \
  --lambda_ce   0.3 \
  --backbone_lr 1e-6 \         # only used if PixArt-init
  --adapter_lr  1e-5
```

### Decision rule

- **P3.5 SR > P3 SR + 2%** → joint refine is useful. Use the
  refined DiT for P5 RL.
- **P3.5 SR ≈ P3 SR** → joint refine didn't help. Use the P3 DiT (or
  rerun with different `λ` ratios — try `λ_ce = 0.1` for gentler pull
  or `0.7` for stronger).
- **P3.5 SR < P3 SR** → joint loss is destabilising the model.
  Likely either `λ_ce` too high, or the DiT collapsed into a degenerate
  high-action-success mode. Revert to P3 checkpoint.

## Why we don't do end-to-end from the start

Three reasons, each load-bearing:

1. **Compute.** Backprop through 20-step DDIM + 3B PaliGemma + policy
   doesn't fit on one A100. The sequential phases let us use full GPU
   memory per model. P3.5 fits because we drop DDIM steps to 4 and
   only run a single epoch.
2. **Credit assignment.** When P3 succeeds or fails, we can attribute
   it to *the policy* (the only thing that changed). With end-to-end,
   any change in metrics could be the DiT, the policy, or their
   interaction.
3. **Stability of the diffusion objective.** Diffusion MSE has its
   own optimisation dynamics — long warmup, gentle LR. Mixing it with
   action CE from step 0 makes it harder to debug. Starting from a
   reasonable DiT checkpoint and only doing a short joint phase
   sidesteps this.

The result is a hybrid: most of the engineering simplicity of
sequential, the alignment win of joint, no need to figure out a stable
joint training recipe from scratch. This is the same hybrid π0.7
ships ("multiple objectives, but with shared, pretrained
backbones").

## File map

| Concern | File |
|---|---|
| P3 BC + subgoals trainer | [`openfly/train_paligemma_subgoal.py`](../openfly/train_paligemma_subgoal.py) |
| P3 launcher | [`openfly/run_train_paligemma_subgoal.sh`](../openfly/run_train_paligemma_subgoal.sh) |
| P3.5 joint-refine trainer | [`openfly/train_joint_refine.py`](../openfly/train_joint_refine.py) |
| P3.5 launcher | [`openfly/run_train_joint_refine.sh`](../openfly/run_train_joint_refine.sh) |
| Policy backbone (used by both) | [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) |
| World model (vanilla) | [`openfly/models/subgoal_dit.py`](../openfly/models/subgoal_dit.py) |
| World model (PixArt-Σ-init) | [`openfly/models/subgoal_dit_pixart.py`](../openfly/models/subgoal_dit_pixart.py) |
