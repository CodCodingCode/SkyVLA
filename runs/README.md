# Preserved training runs

Both training runs were stopped here. Checkpoints live in
`skyvla_isaac/snatch/checkpoints/` (the one directory `.gitignore` exempts from
the `*.pt` rule); trimmed logs are the `*.log.gz` files beside this README.

| checkpoint | run | ended at | headline metric |
|---|---|---|---|
| `carry_8m_model_22800.pt` | `drone_snatch_carry_v3` | iter 22,800 | arrive 31%* at **8.14 m**, grasp 99.9% |
| `place_deposit_throw_21950.pt` | `drone_snatch_place_only_v4` | iter 21,950 | **deposit 90.2%** (peak 92.7%) |

Only the best checkpoint of each stage is kept. Two were deleted:

- **`place_latest_v5_model_12500.pt`** — `place_only_v5` was the more recent place run
  but a clear regression, never clearing 57.5% deposit where v4 sat at ~90%. Its log
  survives in `place_v5_tail.log.gz` if you want to see what the attempt did.
- **`place_deposit_throw_21950.pt`** — byte-identical duplicate of
  `place_deposit_throw_21950.pt` (same md5). One copy is enough.

`carry_transport_21350.pt` is a genuinely different, earlier carry snapshot, so it stays.

Caveat on the surviving place policy: 90% deposit comes with `impact_v` 1.35 m/s and
`landing_err` 10.4 cm — it drops the cube rather than setting it down, which is what the
`throw` in its name records. The deposit succeeds; it is not gentle.

\* the carry run was mid-rung when stopped: it had cleared the 55% gate to reach
8.14 m and was re-earning it at the new distance (it stalled ~46%, then slid to 31%).
The last distance it *held* 55% at was 7.08 m. Full ladder in `carry_v3_ladder.txt`:
1.5 m to 8.14 m over 11 advances.

## Distance ladder (carry)

Separation expands by ~15% each time the arrival EMA clears 0.55, then the EMA resets
so the gate is re-earned at the new distance. Rung cost in iterations:
`46, 242, 669, 634, 451, 684, 339, 422, 493, 982, 849`.

## Resuming

```bash
DIR=logs/isaac/drone_snatch_carry_v4 \
INIT=skyvla_isaac/snatch/checkpoints/carry_8m_model_22800.pt \
SEP=8.14 SEP_MAX=10.0 SPEED=1.5 \
  bash skyvla_isaac/scripts/run_snatch_carry.sh
```

Set `SEP` to the last value in `carry_v3_ladder.txt` or the curriculum restarts short.
If the 8 m rung stalls again, raise `SPEED` one rung (1.5 -> 2.5): at 8 m the transit
eats ~5.4 s of an 18.7 s episode, so the budget — not the policy — is the binding limit.
