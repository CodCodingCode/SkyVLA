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

```bash
python -m openfly.train_subgoal_dit \
  --pretrained_path "$PIXART" \
  --epochs 4 --batch_size 4 \
  --lr 1e-4 --backbone_lr 1e-6 --adapter_lr 1e-4 \
  --warmup_steps 1000 \
  --subgoal_pairing mixed --subgoal_semantic_prob 0.25 --subgoal_uniform_max 4 \
  --val_split unseen
```

Notable flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--backbone_lr` | 1e-6 | LR for the 610M pretrained PixArt-Σ weights. Tiny — heavy LoRA-style finetune. |
| `--adapter_lr` | 1e-4 | LR for the new from-scratch heads (text_to_caption, extra_modulation, token_in/out). ~100× backbone_lr. |
| `--subgoal_pairing` | mixed | π0.7 recipe: 25% semantic (end-of-segment) + 75% uniform 0-4s ahead. `--subgoal_semantic_prob` and `--subgoal_uniform_max` tune the mix. |
| `--val_split` | unseen | Validates on held-out scenes. Random-split-on-train leaked at the per-step level so val_cos inflated; we now use unseen.json. |
| `--val_ddim_steps` | 20 | DDIM steps for val_cos sampling. Matches inference. |
| `--num_timesteps` | 1000 | Cosine schedule. |
| `--ema_decay` | 0.9999 | EMA shadow weights (saved alongside live weights). |
| `--resume` | None | Resume from a prior `last.pt` (loads weights + optimizer + EMA + epoch counter). |

**Resume**: P2 takes a long time. Use `--resume <path>/last.pt` to pick
up where a previous run stopped. Set `--warmup_steps 0` when resuming
so the LR doesn't ramp down again.

**val_cos**: Cosine similarity between sampled subgoal tokens (4-step
DDIM at inference, 20-step during val) and the ground-truth future
SigLIP tokens. Higher = the world model predicts the right
representation. Random ≈ 0; current best ≈ 0.60. Anything above ~0.85
would be very strong.

**Capping dataset for time-boxed runs**: `--max_samples N` truncates
the train split. Each epoch costs ~(N / batch_size) gradient steps at
~80 steps/min on an A100-40GB at batch_size=4. So for a 5-hour 3-epoch
budget: `--max_samples 20000` ≈ 5,000 steps/epoch × 3 = ~3h training +
~10 min validation.

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
