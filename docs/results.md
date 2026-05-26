---
layout: default
title: Results
description: Per-env unseen results across the experiment matrix.
permalink: /results/
---

# Results

These tables are produced by running

```bash
python -m openfly.scripts.aggregate_results \
  --logs_dir logs/benchmarks --per-env \
  --output docs/results.md
```

and pasting the resulting Markdown into the placeholders below. Until
the first runs land they stay marked as *TBD*. The same numbers also
appear in
[`docs/RESEARCH.md`](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/RESEARCH.md).

## Per-env unseen success rate (primary table)

| Method | GTA SR | smallcity SR | sjtu02 SR | seen SR | seen NE (m) |
|--------|-------:|-------------:|----------:|--------:|------------:|
| B0 Heuristic | TBD | TBD | TBD | TBD | TBD |
| B1 SFT | TBD | TBD | TBD | TBD | TBD |
| B2 RL dense | TBD | TBD | TBD | TBD | TBD |
| B3 RL cold sparse | TBD | TBD | TBD | TBD | TBD |
| **B4 RL curriculum** | **TBD** | **TBD** | **TBD** | TBD | TBD |

## Sample budget vs unseen SR (curriculum)

| Stage | Reward preset | Rollout episodes | Wall time | Best unseen SR |
|-------|---------------|-----------------:|----------:|---------------:|
| easy | `easy` | TBD | TBD | TBD |
| medium | `medium` | TBD | TBD | TBD |
| hard | `hard` | TBD | TBD | TBD |

## Failure modes (per env)

Each failed episode is bucketed by
[`openfly.scripts.analyse_failures`](https://github.com/CodCodingCode/SkyVLA/blob/main/openfly/scripts/analyse_failures.py)
into one of the following modes, using the per-episode fields stored
in the benchmark JSON:

- **wandered** — terminated early, far from the goal
- **stalled** — used the full step budget, never reached the goal
- **oracle_only** — passed within the success radius but never emitted
  the stop action
- **near_miss** — stopped close to the goal but outside the radius
- **image_error** — sim/bridge crash, not the policy's fault

```bash
python -m openfly.scripts.analyse_failures \
  --inputs 'logs/benchmarks/openfly_unseen_*.json' \
  --output docs/failure_modes.md
```

| Env | N | Failed | wandered | stalled | oracle_only | near_miss | image_error |
|-----|--:|------:|--------:|--------:|------------:|----------:|------------:|
| `env_game_gtav` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `env_ue_smallcity` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `env_gs_sjtu02` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

When the table is filled, the headline question becomes *which mode
shrinks under the curriculum*. A drop in **stalled** under B5 versus B2
means the policy learned to commit to a direction; a drop in
**oracle_only** means it learned to issue the stop action; persistent
**wandered** rates on `env_game_gtav` while `env_ue_smallcity` improves
is the cross-renderer story the
[research page](research) calls out.

## Interpreting these numbers

A few rules of thumb the
[fairness doc](https://github.com/CodCodingCode/SkyVLA/blob/main/docs/BENCHMARK_FAIRNESS.md)
spells out and that apply here:

1. The heuristic policy is an oracle — it gets the goal directly. Use
   it only as a geometry sanity check.
2. Identical `--max_steps`, `--success_dist`, and episode count are
   required when comparing checkpoints.
3. Improvements on `seen` are interesting; the headline claim must
   come from the per-env `unseen` table.

## Reproducing a row

```bash
for ENV in env_game_gtav env_ue_smallcity env_gs_sjtu02; do
  bash openfly/run_eval.sh --split unseen --policy paligemma \
    --paligemma_ckpt <ckpt> --env_filter "$ENV" --max_episodes 50
done
bash openfly/run_eval.sh --split seen --policy paligemma \
  --paligemma_ckpt <ckpt> --max_episodes 200
```

Each invocation writes a `logs/benchmarks/openfly_*.json` file with the
new `per_env`, `checkpoint`, and `max_steps` metadata that the
aggregator picks up.
