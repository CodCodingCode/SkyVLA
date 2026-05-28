# Training pipeline

SkyVLA trains in four phases (P1 → P2 → P3 → P3.5). Each phase produces
a checkpoint the next phase consumes. The pipeline is layered so you
can stop after P1 and have a baseline BC policy, or stop after P3 to
have the full BC+subgoals model without joint refinement.

```
P1: BC baseline                            ─►  best.pt   (PaliGemma + LoRA + 8-class head)
P2: SubgoalDiT world model                 ─►  best.pt   (PixArt-Σ + adapter heads; feature-space ε-pred)
P3: BC + frozen DiT subgoals                ─►  best.pt   (P1's head retrained with subgoal pathway active)
P3.5 (optional): joint refine               ─►  best.pt   (unfreezes DiT and policy together with MSE + CE)
```

All phases share the same dataset loader (`openfly.dataset`), the same
PaliGemma-3B feature extractor, the same OpenFly 8-class action space
(`TRAINABLE_ACTION_IDS = (0,1,2,3,4,5,8,9)`), and the same 25/75 subgoal
pairing recipe (π0.7 Appendix C — see [implementation.md](implementation.md)).

## Setup

```bash
source ~/SkyVLA/openfly/activate.sh
export PIXART=/home/ubuntu/assets/pretrained/hf_cache/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/snapshots/76fb7eb5a9314bc1e4e479d2f13447517fca9be4
```

`activate.sh` sets `OPENFLY_ROOT`, `OPENFLY_IMAGE_ROOT`, `PYTHONPATH`,
and activates the `openfly` conda env. `PIXART` points at the PixArt-Σ
HF snapshot the DiT finetunes from. Every command below assumes these
are set.

Hardware reference (validated): A100-SXM4-40GB. Training uses 14-30 GB
GPU memory depending on phase; PaliGemma in fp16 + LoRA fits with room
to spare.

---

## P1 — BC baseline

**Script**: [`openfly/train_paligemma.py`](../openfly/train_paligemma.py)
**Output**: `logs/openfly/paligemma/<ts>/best.pt`
**What it trains**: PaliGemma LoRA adapters (q/k/v/o, rank 16) + a
fresh 8-class action head + pose encoder + history-frame [CLS] pool +
text→image cross-attention. **No subgoals.** This is the BC anchor.

```bash
python -m openfly.train_paligemma \
  --split train \
  --epochs 3 --batch_size 4 \
  --lora_lr 1e-5 --head_lr 3e-4 \
  --history_frames 2 \
  --lora_rank 16 --lora_alpha 32.0 \
  --warmup_steps 500
```

Notable flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--lora_lr` | 1e-5 | LR for PaliGemma LoRA adapters. |
| `--head_lr` | 3e-4 | LR for action head + projections (~30× lora_lr). |
| `--history_frames` | 2 | Past frames pooled to one `[CLS]` token each. |
| `--max_samples` | 0 | Cap dataset for short runs. 0 = use everything available. |
| `--env_filter` | None | Restrict to one env (e.g. `env_airsim_16`). |

Loss: cross-entropy over 8-class action logits + small auxiliary
goal-delta regression. Validation: top-1 accuracy on a held-out
fraction of the train split. (Phase P1 doesn't use `unseen.json` —
that's reserved for the final benchmark.)

---

## P2 — SubgoalDiT world model

**Script**: [`openfly/train_subgoal_dit.py`](../openfly/train_subgoal_dit.py)
**Output**: `logs/openfly/subgoal_dit/<ts>/best.pt`
**What it trains**: a feature-space diffusion model that predicts the
SigLIP token grid of a *future* frame given the current frame's SigLIP
tokens + a Gemma text summary + pose-delta + last-action + horizon.
Target space is PaliGemma's SigLIP encoder output (256 tokens × 2048
dim), NOT pixel space.

**Backbone**: PixArt-Σ 610M, finetuned full (not frozen). The pretrained
weights provide image-domain priors; we override the T5 caption
projection with a fresh `Linear(2048 → 1152)` for Gemma summaries (see
[implementation.md](implementation.md) for the wrapper-bug history).

**Recommended baseline command** (~10h on A100-40GB with current
defaults; tune `--per_env_max_episodes` and `--epochs` to time-box):

```bash
python -m openfly.train_subgoal_dit \
  --pretrained_path "$PIXART" \
  --split train --val_split seen --val_ood_split unseen \
  --per_env_max_episodes 10000 \
  --val_max_episodes 100 --val_ood_max_episodes 100 \
  --val_ddim_steps 4 \
  --epochs 20 --batch_size 4 \
  --dropout 0.10 \
  --early_stop_patience 3
```

Notable flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--backbone_lr` | 1e-6 | LR for the 610M pretrained PixArt-Σ weights. Tiny — heavy LoRA-style finetune. |
| `--adapter_lr` | 1e-4 | LR for the new from-scratch heads (text_to_caption, extra_modulation, token_in/out). ~100× backbone_lr. |
| `--subgoal_pairing` | mixed | π0.7 recipe: 25% semantic (end-of-segment) + 75% uniform 0-4s ahead. `--subgoal_semantic_prob` and `--subgoal_uniform_max` tune the mix. |
| `--per_env_max_episodes` | 0 (no cap) | Cap episodes-per-env when loading train.json. Without this, `env_ue_bigcity` + `env_airsim_sh` together dominate (~36% of episodes) and the model overfits to their visual styles. Recommended **10000** for full data with balance; 2000–3000 for quick diagnostics. |
| `--val_split` | unseen | Primary held-out val split. Random-split-on-train leaked at the per-step level (adjacent frames in train + val); we now use a separate OpenFly split. |
| `--val_ood_split` | (off) | Optional **second** val split. Runs a parallel val pass each epoch and logs `val_ood_loss` / `val_ood_cos`. Use to distinguish in-distribution (`val_split=seen`) from out-of-distribution (`val_ood_split=unseen`) generalization. |
| `--val_max_episodes` / `--val_ood_max_episodes` | 0 (all) | Cap each val pass. **100 each is plenty** — val cost is dominated by DDIM sampling, not statistics. |
| `--val_ddim_steps` | **4** | DDIM steps for val_cos. Matches the policy's deploy setting (`PaliGemmaVLNPolicy.subgoal_sample_steps=4`). Bumping to 20 measures the DiT's denoising *ceiling*, not deploy quality — see the [val_cos](#val_cos) note. |
| `--dropout` | 0.0 | Dropout on adapter outputs (PixArt path: `token_in`, `text_to_caption`, `extra_cond`; random-init path: residual attn + MLP inside each DiTBlock). 0.10–0.15 is the useful range; the pretrained PixArt backbone itself is *not* retrofitted. |
| `--augment_input` | False | Color jitter / brightness / contrast / noise on RGB inputs (current + history only; subgoal target stays clean). **Currently a net loss** — see [Gotchas](#gotchas--hard-won-lessons). |
| `--ema_decay` | 0.9999 | EMA shadow weights. Best.pt + val both use EMA params. |
| `--early_stop_patience` | 0 (off) | Stop if `val_cos` (primary) hasn't improved for N consecutive epochs. Pair with a generous `--epochs` ceiling. |
| `--resume` | None | Resume from a prior `last.pt` (loads weights + optimizer + EMA + epoch counter). Use `--warmup_steps 0` when resuming so the LR schedule doesn't restart. |

**Resume**: P2 takes a long time. Use `--resume <path>/last.pt` to pick
up where a previous run stopped. Set `--warmup_steps 0` when resuming
so the LR doesn't ramp down again.

### val_cos

Cosine similarity between sampled subgoal tokens and the ground-truth
future SigLIP tokens. Higher = the world model predicts the right
representation. Random ≈ 0; current best ≈ 0.60 (measured at 20 DDIM
steps).

**Important caveat on past numbers**: every `val_cos ≈ 0.61` you'll see
in older `history.json` files was measured with `--val_ddim_steps 20`,
i.e. 5× the sampling fidelity the policy actually uses at inference.
The trainer default is now **4**, matching the policy. So
deploy-quality val_cos on the same checkpoint will be *lower* than the
20-step number — that's not a regression, it's the honest measurement.
Use 20 only as a one-off "denoising ceiling" diagnostic.

**Capping dataset for time-boxed runs**: either `--max_samples N`
(truncate global) or `--per_env_max_episodes K` (balanced per env). On
this machine ~25% of train.json steps have local frames after the
`require_images` filter, so the relationship between cap and usable
steps roughly tracks:

| `--per_env_max_episodes` | usable train steps | epoch wall time (B=4) |
| --- | --- | --- |
| 2000 | ~22k | ~15 min |
| 3000 | ~30k | ~22 min |
| 10000 | ~115k | ~45 min |
| 0 (no cap) | ~180k | ~55 min (but unbalanced) |

### Gotchas / hard-won lessons

These are the failure modes that ate hours of compute before being
diagnosed. Don't rediscover them.

1. **Sample-level random val splits leak.** The previous trainer carved
   val out of train via `random_split(full_ds, [train_size, val_size])`.
   At the per-step level, frame `k` lands in train and frame `k+1`
   lands in val of the *same* trajectory — visually near-identical,
   ~100% val accuracy in epoch 1. **Fix**: validate on a separate
   OpenFly split (`--val_split seen` or `unseen`), never on a random
   slice of train.

2. **Val cost is dominated by DDIM step count × episodes × number of
   val splits.** A naive `--val_ddim_steps 20 --val_max_episodes 200`
   on two val splits was ~30k PixArt forwards per epoch — about 4h of
   the 5h epoch wall time was *just* val. **Fix**: `--val_ddim_steps 4`
   (matches deploy) + `--val_max_episodes 100`. 10× faster val,
   sub-statistic noise on 100 episodes.

3. **`--val_ddim_steps` defaulted to 20 while the policy deploys at 4.**
   That made every old val_cos number an inflated ceiling. The default
   is now 4. See the [val_cos](#val_cos) note for what this means
   for historical comparison.

4. **Without `--per_env_max_episodes`, `env_ue_bigcity` + `env_airsim_sh`
   dominate.** Together they account for ~36% of train.json episodes.
   The DiT learns those two envs' textures and can't generalize. **Fix**:
   `--per_env_max_episodes 10000` caps both at 10k, leaves the other 9
   envs uncapped (their natural counts), and produces a ~88k-episode
   balanced training set instead of a 100k unbalanced one.

5. **Aggressive `--augment_input` hurt val_cos on both splits** (0.61 →
   0.0004 in one run with dropout=0.15 + aug). The hypothesis: PaliGemma
   was pretrained on a specific image-normalization regime; color jitter
   / brightness / additive noise on RGB inputs produce SigLIP tokens
   that don't match what the DiT is supposed to predict, and the
   EMA-averaged weights track the perturbed-input regime instead of the
   clean target. **Default it off** until we have a milder augmentation
   that preserves SigLIP-feature stability.

6. **`require_images` filters silently drop ~75% of train.json on this
   machine.** Only ~400k of ~1.6M frame files are downloaded locally;
   the dataset skips trajectories with missing frames. The reported
   "train steps" in the trainer log are what's *actually* usable, not
   what's in train.json.

7. **Old PixArt runs' `args.json` was sometimes missing.** The trainer
   now persists `args.json` to the run directory at startup, so every
   run's config is recoverable. If you're comparing to a run before
   that change, check the checkpoint's internal `args` dict (the
   trainer also embeds args in `last.pt` / `best.pt`).

---

## P3 — BC + frozen DiT subgoals

**Script**: [`openfly/train_paligemma_subgoal.py`](../openfly/train_paligemma_subgoal.py)
**Output**: `logs/openfly/paligemma_subgoal/<ts>/best.pt`
**What it trains**: the *same* PaliGemma policy as P1 but with the
subgoal pathway active. At each step a future subgoal is mixed in:
50% of the time it's the oracle (real future frame encoded by
PaliGemma), 50% it's sampled from the frozen P2 DiT (4-step DDIM). The
policy learns to use both — at inference time we only have the DiT,
but training on both keeps the subgoal channel useful even when DiT
quality is moderate.

```bash
python -m openfly.train_paligemma_subgoal \
  --p3_ckpt none \
  --dit_path logs/openfly/subgoal_dit/<ts>/best.pt \
  --pretrained_path "$PIXART" \
  --epochs 3 --batch_size 4 \
  --lora_lr 1e-5 --head_lr 3e-4 \
  --history_frames 2 \
  --dit_mix_prob 0.5 --ddim_steps 4 \
  --warmup_steps 500
```

Notable flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_path` | required | The frozen P2 SubgoalDiT checkpoint. |
| `--pretrained_path` | None | Same PixArt snapshot the DiT was trained against. Needed to reconstruct the wrapper architecture. |
| `--dit_mix_prob` | 0.5 | Prob of using DiT-sampled subgoals during training. 0.5 = even oracle/DiT mix. At inference always 1.0 (DiT only). |
| `--ddim_steps` | 4 | DDIM steps for inference-time subgoal sampling. Matches recorder defaults. |
| `--lora_lr` | 1e-5 | Same as P1 — we're still adapting PaliGemma LoRA. |
| `--head_lr` | 3e-4 | The action head gets fully re-trained because its input dim grew (now includes subgoal tokens). |

**Checkpoint loading**: P3 reuses P1's LoRA + head if you point
`--p3_ckpt` at a P1 `best.pt`. Pass `--p3_ckpt none` to train P3 from
fresh PaliGemma-init. The loader **shape-filters** mismatched keys
silently — if you change the architecture between phases, surprise-
random-init weights will leak through. Always check the
`loaded=N missing=M unexpected=K` log line and explicitly compare
state-dict shapes if any of M/K is nonzero. See the action-head spin
bug history in [RECORDING_DEMOS.md](RECORDING_DEMOS.md#troubleshooting).

---

## P3.5 — Joint refine (optional)

**Script**: [`openfly/train_joint_refine.py`](../openfly/train_joint_refine.py)
**Output**: `logs/openfly/joint_refine/<ts>/best.pt`
**What it trains**: P3 *and* the DiT jointly, end-to-end. Loss is
`λ_mse · MSE(DiT_pred, oracle_subgoal) + λ_ce · CE(policy_logits, expert_action)`.
DDIM sample is gradient-aware so the policy's CE signal flows back
through the DiT's parameters. Use this only when P3 has converged
and you want to squeeze the last bit of consistency between the world
model and the policy.

```bash
python -m openfly.train_joint_refine \
  --p3_ckpt   logs/openfly/paligemma_subgoal/<ts>/best.pt \
  --dit_path  logs/openfly/subgoal_dit/<ts>/best.pt \
  --pretrained_path "$PIXART" \
  --epochs 1 --batch_size 2 \
  --lambda_mse 1.0 --lambda_ce 0.3 \
  --warmup_steps 200 --grad_clip 0.5
```

Notable flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--lambda_mse` | 1.0 | World-model loss weight. Keep at 1.0 unless DiT is drifting. |
| `--lambda_ce` | 0.3 | Policy loss weight. <1 because CE gradient is much bigger than MSE on raw scale. |
| `--batch_size` | 2 | Smaller than P3 because we're holding gradients through both models. |
| `--grad_clip` | 0.5 | Tighter than P2/P3 — joint loss can spike. |

Most projects skip P3.5; the marginal gain rarely justifies the wall
clock. Try it only after the eval shows P3+DiT is close to oracle.

---

## Checkpoints

Every trainer writes to `logs/openfly/<phase>/<YYYYMMDD_HHMMSS>/`:

| File | Contents |
| --- | --- |
| `best.pt` | Weights at the epoch with the highest validation metric. |
| `last.pt` | Most recent weights — use this for `--resume`. |
| `history.json` | Per-epoch summary (train_loss, val_loss, val_cos, val_acc, n_steps, time_s). |

Each `*.pt` contains:

- `model` / `dit` — model state_dict
- `optimizer` — AdamW state
- `ema` (P2 only) — EMA shadow weights, if `--ema_decay` was set
- `epoch`, `global_step`, `val_*` — bookkeeping
- `args` — the argparse Namespace the run was launched with

The `best.pt` and `last.pt` writes are atomic (`.tmp` → `rename`) so an
interrupted save can't leave a corrupt file.

---

## Time budgets (A100-40GB, batch_size 4)

Measured rates from runs on this box:

| Phase | Steps/min | Per-step training time |
| --- | --- | --- |
| P1 BC | ~110 | 0.55 s |
| P2 DiT | ~80 | 0.75 s (PixArt-Σ is the biggest backbone) |
| P3 BC+subgoals | ~70 | 0.85 s (PaliGemma + frozen DiT 4-step DDIM per step) |
| P3.5 joint | ~35 | 1.7 s (both models hold gradients) |

For a 5-hour budget per phase:

- P1: ~33,000 steps ≈ ~5 full epochs on 25k usable samples
- P2: ~24,000 steps. With `--max_samples 20000` that's 3 epochs.
- P3: ~21,000 steps. Roughly 4 epochs on 20k samples.
- P3.5: ~10,500 steps. Half an epoch — only useful as a polish pass.

---

## Common patterns

### Time-box a phase

Use `--max_samples N` to fit a budget. `N / batch_size` = steps per
epoch. Knowing the per-phase rate above, pick `N` and `--epochs` so
the total stays under your wall-clock target.

### Don't lose work to long runs

Always use `nohup` + `tee` (or just `nohup ... > log 2>&1 &`) so the
training survives SSH disconnects:

```bash
nohup python -m openfly.train_subgoal_dit ... > /home/ubuntu/SkyVLA/logs/openfly/subgoal_dit/run.log 2>&1 &
```

### Sanity-check checkpoint compatibility

Before loading a phase-N checkpoint into phase N+1, run:

```python
import torch
ck = torch.load("logs/openfly/<phase>/<ts>/best.pt", map_location="cpu", weights_only=False)
own = MyNewModel().state_dict()
mismatch = [k for k in own if k in ck["model"] and tuple(own[k].shape) != tuple(ck["model"][k].shape)]
missing = [k for k in own if k not in ck["model"]]
print("missing:", missing); print("mismatch:", mismatch)
```

If either list is non-empty, the loader will silently drop those
weights and you'll get a partially random model. Match the architecture
exactly OR retrain from scratch.

### Resume vs restart

| Situation | Choice |
| --- | --- |
| Same architecture, just want more training | `--resume last.pt` |
| Architecture change broke state-dict compat | Retrain from scratch — there's no partial-load story |
| Suspect local-minimum stall | Restart with a different seed, same setup |
| Optimizer state corrupt / hyperparam change | Load weights only (manual `load_state_dict`, drop optimizer) |
