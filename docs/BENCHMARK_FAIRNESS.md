# Benchmark Fairness

Short reference for "is what we're doing fair, and what can we honestly claim about the numbers?" — read this before posting OpenFly results anywhere public.

## TL;DR

Training on a benchmark's **train** split and reporting the held-out **seen / unseen** numbers is the standard protocol followed by the OpenFly paper and every leaderboard entry we compare against. It is **not** cheating. What is cheating, or at least misleading, is:

1. Training on the eval splits.
2. Reporting oracle-goal heuristic numbers as a model result.
3. Calling fine-tuned numbers "zero-shot generalisation".

## What we train on, what we eval on

| Component | Train data | Eval data | Notes |
|-----------|------------|-----------|-------|
| Heuristic policy | None — closed-form oracle | OpenFly `seen` / `unseen` | Receives the episode goal directly. Always label as oracle. |
| OpenFly-Agent (7B) | OpenFly `train.json` (RLDS / TFDS) | OpenFly `seen` / `unseen` | Upstream FSDP fine-tune wrapped via [`openfly/run_train_agent.sh`](../openfly/run_train_agent.sh). |
| Custom PaliGemma BC | OpenFly `train.json` only | OpenFly `seen` / `unseen` | Cross-entropy on the discrete expert actions. |
| PaliGemma + DAgger | OpenFly `train.json` + on-policy rollouts in `train` AirSim scenes | OpenFly `seen` / `unseen` | DAgger relabels visited states with the goal-oracle heuristic. Train-split AirSim scenes only — eval scenes are untouched. |
| PaliGemma + GRPO | DAgger checkpoint + online rollouts in `train` scenes | OpenFly `seen` / `unseen` | Reward is computed from `train` episode goals; KL anchor keeps the policy close to DAgger init. |
| OpenFly-Agent + PPO/LoRA | HF checkpoint + online rollouts in `train` scenes | OpenFly `seen` / `unseen` | LoRA on `q_proj`/`v_proj` plus value head; backbone frozen. |

The heuristic policy never sees an image and gets the goal coordinate from the episode definition; numbers from it are an upper bound on geometry, not a vision-language result.

## What is fair to claim

- "Custom PaliGemma BC policy fine-tuned on OpenFly train; reports `seen=X`, `unseen=Y`." — Yes.
- "Hierarchical VLA: PaliGemma + LoRA + LSTM head trained on `train.json`, low-level discrete actions decoded directly." — Yes; this is a valid hierarchical method and worth flagging because the action space is shared with the upstream OpenFly-Agent.
- "Reproduces the OpenFly-Agent baseline within ε using upstream checkpoint plus our eval harness." — Yes, when reporting `--policy openfly-agent`.
- "SFT → DAgger → GRPO/PPO on the AirSim `train` scenes; eval on held-out `unseen` scenes." — Yes. The RL stage only sees `train`-split AirSim instances (the env wrapper filters to `env_filter=env_airsim_16` by default), and the goal-oracle heuristic used during DAgger only labels states the policy actually visited.

## What is not fair

- "Zero-shot OpenFly results." — Not after fine-tuning on `train.json`. Use the upstream OpenFly-Agent checkpoint with no further training if you want a true zero-shot number on a new scene.
- "Beats OpenFly-Agent on unseen." — Only if both runs use the same eval harness, the same number of episodes, and the same `--max_steps` budget. The harness writes those into the result JSON; quote them.
- "Heuristic SR equals the model SR." — The oracle has access to the goal coordinate. It is a sim sanity check, not a baseline.

## Practical: how to label results

Suggested wording for a results table or blog post:

> Custom PaliGemma BC, **fine-tuned on OpenFly `train.json` for 10 epochs**, evaluated on `seen` and `unseen` splits with the official simulation bridges. Metric: success rate within 20 m, navigation error in metres, SPL.

That is accurate, calibrates expectations, and makes the contribution legible without overclaiming generalisation.

## Cross-references

- Implementation: [`openfly/train_paligemma.py`](../openfly/train_paligemma.py), [`openfly/eval_benchmark.py`](../openfly/eval_benchmark.py), [`openfly/policies.py`](../openfly/policies.py).
- Eval reference: [`../openfly/README.md`](../openfly/README.md).
- Architecture lineage: [`vla/VLA_SYSTEM.md`](../vla/VLA_SYSTEM.md).
