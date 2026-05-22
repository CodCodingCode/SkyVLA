# Next steps

The repository targets the OpenFly outdoor aerial VLN benchmark. This document tracks where the codebase is today and the natural next steps for improving the trained models and the eval coverage.

## Where you are now

| Component | Status |
|-----------|--------|
| OpenFly evaluation harness | Implemented, runs the `heuristic`, `openfly-agent`, `paligemma`, `dagger`, `grpo`, and `ppo` policies |
| OpenFly-Agent (OpenVLA 7B) wrapper | [`openfly/run_train_agent.sh`](../openfly/run_train_agent.sh) calls upstream FSDP training |
| Custom PaliGemma BC policy | [`openfly/train_paligemma.py`](../openfly/train_paligemma.py) — offline cross-entropy on `train.json` |
| RL pipeline | DAgger / GRPO / PPO trainers under [`openfly/`](../openfly/) with shared Gymnasium env, rewards, and rollout collector |

## Recommended order of work

1. **Verify the simulator end-to-end.** After `openfly/setup.sh` and downloading at least one AirSim scene, run the heuristic policy on five episodes and confirm a non-zero SR / OSR. This catches missing system packages, AirSim port collisions, and pose-ratio bugs before any model is trained.
2. **Bring up the OpenFly-Agent baseline.** Pull the upstream checkpoint and run `--policy openfly-agent` on the same five episodes. The 7B model is a good reference point and exercises the FSDP / flash-attn install.
3. **Train the custom PaliGemma BC policy.** Fetch the `train.json` annotations and the trajectory frames (set `OPENFLY_IMAGE_ROOT` to the extracted directory). A first pass with `--max_samples 10000 --epochs 1` validates the data pipeline; a full run at `--epochs 10 --batch_size 8` produces a usable checkpoint.
4. **Score the trained PaliGemma checkpoint.** Run `--policy paligemma --paligemma_ckpt <path>` on `seen` and `unseen` splits and compare against the OpenFly-Agent and heuristic numbers.
5. **Iterate on the BC pipeline.** Likely improvements once the loop runs:
   - Use a larger history window (`--history_frames 4`).
   - Add the auxiliary body-frame goal regression (already implemented; tune `--aux_goal_weight`).
   - Feed the upstream OpenFly action vector (`openfly.actions.action_id_to_vector`) as a regression target to align with the OpenFly-Agent's continuous head.
6. **Online RL on top of the BC initialisation.** Implemented as a three-stage pipeline:
   1. Smoke-test the Gymnasium env: `python -m openfly.scripts.smoke_rl_env --episodes 3 --split seen`.
   2. DAgger between SFT and RL: `bash openfly/run_train_dagger.sh --sft_ckpt <BC>.pt --iterations 3 --episodes_per_iter 200`.
   3. **Track B (PaliGemma)** — GRPO with a KL anchor: `bash openfly/run_train_grpo.sh --init_ckpt <DAgger>.pt --steps 200 --group_size 4`.
   4. **Track A (OpenFly-Agent 7B)** — PPO + LoRA + value head: `bash openfly/run_train_ppo_agent.sh --iterations 30 --episodes_per_iter 4`. Heavier than Track B; do this only after GRPO validates the env/reward pipeline.

   Evaluate the resulting checkpoints with `--policy grpo` or `--policy ppo` (see [`openfly/README.md`](../openfly/README.md) "RL track"). Validation gates G0–G5 from the implementation plan are tracked in the benchmark JSON under `logs/openfly/`.

## Eval coverage

The OpenFly harness already supports the seen / unseen / eval_test splits and per-environment filters. Useful next additions:

- Per-scene break-downs in the summary JSON (currently aggregated across envs).
- Episode video logging via OpenCV when `--save_video` is passed.

## References

- [`docs/A100_SETUP.md`](A100_SETUP.md) — provisioning and step-by-step bring-up on an x86 + NVIDIA box (the only architecture that runs the OpenFly AirSim binaries).
- [`openfly/README.md`](../openfly/README.md) — full eval and training reference.
- [`vla/VLA_SYSTEM.md`](../vla/VLA_SYSTEM.md) — design notes for the PaliGemma feature extractor reused by the new training stack.
- [`docs/BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md) — what is claimable from each leaderboard number.
