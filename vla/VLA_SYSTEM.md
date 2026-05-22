# VLA System: PaliGemma feature extractor design notes

`vla/vla_policy.py` keeps the portable, framework-agnostic pieces from the original Isaac Sim curriculum so the new OpenFly stack can reuse them without re-implementing PaliGemma + LoRA wiring. The OpenFly training entry point [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) imports `PaliGemmaFeatureExtractor` from this file, then drops the parts that were specific to the four-camera Isaac arena.

## What this file contains today

- `LoRALinear` — a small custom LoRA adapter (no PEFT dependency).
- `PaliGemmaFeatureExtractor` — frozen `google/paligemma-3b-pt-224` with LoRA on `q_proj` / `v_proj`. Provides `forward_tokens`, `forward_tokens_with_grad`, and a token cache used during PPO replay.
- `VLAActorModel` / `VLACriticModel` and `HierarchicalVLAActor` / `HierarchicalVLACritic` — the original 4-camera Isaac actor / critic. They are **not** used by the OpenFly stack, but the class definitions stay here as a reference for how the cross-attention head and the LSTM were wired together.

## What changed when we moved to OpenFly

| Layer | Isaac (`HierarchicalVLAActor`) | OpenFly (`PaliGemmaVLNPolicy`) |
|-------|--------------------------------|--------------------------------|
| RGB input | 4 cameras × 224×224×3 | 1 camera × 224×224×3 plus 2 history frames |
| Depth | 4 × 224×224 with dropout | None |
| Cross-attention key | 4×256 SigLIP image tokens | T×256 SigLIP tokens (T = 1 + history) |
| LSTM input | scene_summary + obj_logits + 9-d flight state | scene_summary + 4-d pose embedding |
| Action head | Frozen Stage-2 waypoint MLP → 4-d thrust | `nn.Linear(hidden, 10)` discrete macros |
| Auxiliary heads | Object classifier (cube / sphere / cylinder) + body-frame target MSE | Optional body-frame goal regression |
| Curriculum | Two-phase precision schedule | Offline cross-entropy on `train.json` |

## PaliGemma details (unchanged)

- Base model: `google/paligemma-3b-pt-224` (224×224 input).
- All base parameters are frozen; only LoRA matrices and downstream heads are trainable.
- LoRA: rank 8, alpha 16, applied to every `q_proj` / `v_proj` linear in the language model.
- Image preprocessing: NHWC float [0, 1] → NCHW → bilinear resize to 224 → SigLIP normalization `(x - 0.5) / 0.5` → cast to fp16.
- Feature extraction returns both the SigLIP image tokens (`(B, 256, 2048)`, spatially coherent) and the Gemma last hidden state (`(B, seq, 2048)`, text-rich).

## Why keep the legacy classes around

Two reasons:

1. The LoRA / token-cache scaffolding around `PaliGemmaFeatureExtractor` is non-trivial; rewriting it in `openfly/` would duplicate a couple hundred lines for no benefit.
2. The legacy `HierarchicalVLAActor` documents the architectural shape that `PaliGemmaVLNPolicy` was simplified from. Anyone reading the new model can compare the two implementations side-by-side to understand which design decisions were Isaac-specific.

If you ever need a single-camera continuous-action baseline (for sim-to-real or for CityNav metric exploration), the existing `VLAActorModel` is still a good starting point — it just hasn't been wired into a runnable training script in this repository.

## See also

- [`openfly/README.md`](../openfly/README.md) — eval and training reference for the new stack.
- [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) — the OpenFly-specific model.
- [`openfly/train_paligemma.py`](../openfly/train_paligemma.py) — the offline behaviour-cloning trainer.
