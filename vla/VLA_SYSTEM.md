# `vla/vla_policy.py` — PaliGemma feature extractor + LoRA

`vla/vla_policy.py` ships the portable PaliGemma + LoRA scaffolding that the OpenFly training and RL code depends on. Nothing in this file is OpenFly-specific — it is the reusable backbone that the rest of the repo plugs into.

## What this file contains

- `LoRALinear` — a small custom LoRA adapter (no PEFT dependency). Wraps a frozen `nn.Linear`, adds a rank-`r` `A·B` correction with Kaiming/zero init, and scales by `alpha / rank`. Used by [`openfly/models/openfly_agent_rl.py`](../openfly/models/openfly_agent_rl.py) to LoRA-adapt the OpenFly-Agent 7B for PPO.
- `PaliGemmaFeatureExtractor` — frozen `google/paligemma-3b-pt-224` with LoRA on every `q_proj` / `v_proj` linear in the language model. Exposes:
  - `preprocess_images(rgb)` — NHWC float [0, 1] → NCHW 224×224 → SigLIP `(x - 0.5) / 0.5` normalisation in `fp16`.
  - `forward(...)` / `get_features(...)` — single-token features (last non-padding hidden state), with a mini-batched cache.
  - `forward_tokens(...)` / `forward_tokens_with_grad(...)` / `get_token_features(...)` — full Gemma sequence + SigLIP image tokens, used by the OpenFly PaliGemma BC policy.
  - `forward_with_grad(...)` — same as `forward()` but with gradients flowing through the LoRA adapters; meant for small LoRA-update mini-batches.
  - `clear_cache()` — flush the internal feature cache between rollouts.

## PaliGemma config

- Base model: `google/paligemma-3b-pt-224` (224×224 input).
- All base parameters are frozen; only LoRA matrices and downstream heads are trainable.
- LoRA: rank 8, alpha 16, applied to every `q_proj` / `v_proj` linear in the language model (~`replaced` count is printed on load).
- Feature extraction returns both the SigLIP image tokens (`(B, 256, 2048)`, spatially coherent — image features are un-rescaled by `sqrt(2048)` so they match the Gemma scale) and the Gemma last hidden state (`(B, seq, 2048)`, text-rich).

## How OpenFly uses it

[`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) instantiates `PaliGemmaFeatureExtractor`, calls `preprocess_images` + `forward_tokens(_with_grad)` for each rollout step, and stacks an MLP action head on top of the pooled features to predict OpenFly's 10-class discrete action space. [`openfly/models/openfly_agent_rl.py`](../openfly/models/openfly_agent_rl.py) uses `LoRALinear` to add a rank-`r` adapter and a value head to OpenVLA / OpenFly-Agent for PPO.

## See also

- [`openfly/README.md`](../openfly/README.md) — eval and training reference for the OpenFly stack.
- [`openfly/models/paligemma_vln.py`](../openfly/models/paligemma_vln.py) — the OpenFly BC / GRPO model.
- [`openfly/models/openfly_agent_rl.py`](../openfly/models/openfly_agent_rl.py) — the OpenFly-Agent PPO wrapper.
- [`openfly/train_paligemma.py`](../openfly/train_paligemma.py) — offline behaviour-cloning trainer.
