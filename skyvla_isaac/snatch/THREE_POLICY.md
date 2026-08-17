# SNATCH — the three-policy transport architecture (A → B)

How pick-from-A / carry-to-B / deposit-on-B is *supposed* to work: one env, three
policies, two handoffs. This is the process document — what each stage owns, what it
must be handed, what it must hand on, and which gate promotes it. Companion to
[`DESIGN.md`](DESIGN.md) (module contracts, single-stage) and the README's
"Transport lineage" section (exact launch commands).

Everything here is opt-in behind `--two_platform`. Without it the env is the
single-platform grasp task and nothing below applies.

---

## 0. The picture

```
                    ┌──────────── ONE env, ONE obs space, ONE action space ────────────┐
                    │            snatch/pick_place_env.py (two_platform=True)          │
                    └─────────────────────────────────────────────────────────────────┘
                              ▲                    ▲                    ▲
        ┌─────────────────────┘        ┌───────────┘        ┌───────────┘
        │  π_snatch                    │  π_carry           │  π_place
        │  model_9250.pt               │  drone_snatch_     │  drone_snatch_
        │  (committed)                 │  carry/*.pt        │  place/*.pt
        └─────────────────────┬────────┴──────────┬─────────┴──────────┐
                              │  warm start       │  warm start        │
                              └──────────────────►└───────────────────►┘

  phase 0  SNATCH ────────────► phase 1  CARRY ────────────► phase 2  PLACE
  fly in from altitude,         haul the held cube           descend onto B,
  cage the cube, lift it        across the A→B gap           open jaws, let go
        │                             │                             │
   handoff gate:                 handoff gate:                 terminal:
   _held for N consecutive       _arrived                      _deposited
   steps (default 10)            (held ∧ cube over B)          (at rest on B, jaws open)

        A ▉▉▉                                            ▉▉▉ B
       platform A (pick)          ← plat_sep →          platform B (deliver)
       random bearing each reset, separation is a curriculum knob
```

Three `.pt` files. Each is warm-started from the one before it, trains in the *same*
env with a different success predicate, and is frozen when its own gate is met.
`run_pipeline.py` loads all three and switches between them per-env at runtime.

---

## 1. Why three policies and not one

One policy over the whole task was tried implicitly (single-platform place) and the
failure mode is instructive. Four reasons the split is the right architecture here:

1. **Credit assignment over a 600-step episode.** A single terminal deposit reward is
   ~0.1% likely under random exploration at 1.5 m and *zero* at 20 m. Splitting gives
   each policy a dense, locally-earnable objective.
2. **The stages disagree about what to do with the gripper.** Snatch is rewarded for
   `grip_cmd → 1` near the cube; place is rewarded for `grip_cmd → 0` over B
   (`pick_place_env.py:838`). One network has to *hide* the release behind a latent
   phase variable it was never given. Two networks just have different weights.
3. **Different state distributions ⇒ different observation normalizers.** Each stage
   runs `empirical_normalization=True`, so its running mean/var is fitted to the states
   it actually sees (fly-in altitude vs. long-haul transit vs. terminal descent). This is
   why `run_pipeline.py` loads *three rsl_rl runners* and not three state_dicts into one
   (`run_pipeline.py:11-13`) — the normalizer is part of the policy.
4. **The distance ladder only invalidates the middle stage.** Raising `--speed` for a
   longer rung rescales what `action=1.0` means. If everything is one policy, every
   speed rung re-breaks the grasp. Split, only π_carry has to be re-laddered — π_snatch
   and π_place stay at their trained 1.5 m/s precision regime.

**The cost** — and it should be stated plainly — is that the handoff becomes an explicit
piece of engineering that must itself be correct (§4), and three checkpoints must be
kept mutually compatible (§2).

---

## 2. The shared substrate — invariants that must hold across all three

The whole architecture rests on the three policies being *interchangeable at the
interface*. Any change that breaks one of these silently invalidates every saved
checkpoint:

| Invariant | Value | Where |
|---|---|---|
| Action | `(N,5)` = `[vx, vy, vz, yaw_rate, grip]`, all in `[-1,1]` | `_apply_action`, `pick_place_env.py:427` |
| Observation | `[camera latents ‖ state(14)]`; state = `pose_est(3), base_v(3), grip(1), grip_τ(1), block_est(3), goal_rel(3)` | `_get_observations`, `:502-515` |
| Net dims | actor/critic `[256,128,64]`, elu | `build_runner` in `train_snatch.py` / `eval_snatch.py` / `run_pipeline.py` |
| Control rate | 200 Hz physics, decimation 4 → **50 Hz policy** | `:16, :43` |
| Physics grasp | `--no_latch` (`grasp_latch=False`) — real friction cage, no kinematic cheat | every stage's launch script |
| Cube | 5 cm, 50 g (`--cube_mass 0.05`) | `:169`, launch scripts |
| Goal semantics | `goal_rel` = *the cube at rest on B's top* − drone pos | `:1177-1180` |

Two consequences worth internalising:

- **`goal_rel` is what makes the split legal.** All three policies see the same
  "where should the cube end up" vector, so π_snatch's fly-in is not blind to B and
  π_carry does not need a new input to know where it is going. The stages differ in
  *reward*, not in *observability*.
- **`cfg.speed` is global.** `_apply_action` applies one scalar to all three policies
  (`run_pipeline.py:21-26`). Mixing checkpoints trained at different `--speed` values in
  one pipeline run is invalid. Either keep all three at one rung, or make `speed`
  per-env before mixing rungs.

### What each stage flips

Everything else about the env is identical. This table *is* the difference between the
three training runs:

| flag | snatch | carry | place | effect |
|---|---|---|---|---|
| `--two_platform` | off | **on** | **on** | adds platform B, re-anchors the goal to B |
| `--release_only` | off | off | **on** | success becomes `_deposited`, not `_arrived` (`:558`) |
| `--carry_demo` | 0.25 | **0.35** | **0.5** | fraction of resets spawning already-holding (`:1132-1163`) |
| `--plat_sep` / `--plat_sep_max` | — | **1.5 → 4.0** (rung 1) | 1.5 → 2.0 (short) | transport distance curriculum |
| `--reset_std` | 0.2 | 0.2 | **0.35** | re-widen exploration for the new release behaviour |
| `--entropy_coef` | 0.001 | 0.001 | **0.002** | ditto |
| `--start_stage` | 2 | 2 | 2 | all reward terms live from step 0 (warm-started policy) |
| `--cur_p` | 0.15 | **0.0** | **0.0** | 0 = honest full fly-in, no straddle-start freebies |

---

## 3. Stage contracts

Each stage is defined by four things: **what it is handed**, **what counts as success**,
**what it must hand on**, and **the gate that promotes it**. Nothing else about a stage
is normative — reward shaping is free to change as long as these hold.

### Stage 1 — π_snatch  (`skyvla_isaac/snatch/checkpoints/model_9250.pt`, committed)

|  |  |
|---|---|
| **Handed** | drone airborne up to `--side_spawn 5.0` m from the cube, jaws open, cube at rest on A |
| **Success** | `_held = _lifted ∧ (d_reach < 0.10)` — cube ≥ 6 cm off the table and inside the cage (`:539-540`) |
| **Hands on** | drone holding the cube in a *settled* grasp, at carry altitude, anywhere over A |
| **Gate** | grasp ≥ ~85% end-to-end clean at `cur_p 0`, and the VIO-drift sweep recorded |
| **Status** | **done** — 87.4% grasp / 86.1% place / 19.3 cm lift; 94.7% end-to-end clean, ~51% under full modeled VIO drift |

π_snatch is *frozen*. It is the only stage with a committed checkpoint under
`snatch/checkpoints/`, and it is the warm start for everything downstream. Do not
retrain it to fix a carry problem.

### Stage 2 — π_carry  (`logs/isaac/drone_snatch_carry/`)

|  |  |
|---|---|
| **Handed** | the output of π_snatch — plus, for 35% of resets, a synthetic version of exactly that state (`--carry_demo 0.35`: airborne at 0.38–0.65 m, cube shelf-seated in the shut cage) |
| **Success** | `_arrived = _held ∧ (‖cube_xy − B_xy‖ < arrive_radius 0.30)` (`:547`) |
| **Hands on** | drone hovering over B, cube **still held**, ready to descend |
| **Gate** | `arrive_rate` EMA ≥ 0.55 sustained at `plat_sep_max` for the rung (`plat_sep_thresh`, `:132`) |
| **Status** | **training now**, rung 1 (1.5 → 4 m, speed 1.5) |

The reward that actually does the work here is `carry_prog = held·(prev_d_goal − d_goal)`
— a per-step potential difference, which is scale-free and therefore still has gradient
at 20 m. The `tanh(d_goal/…)` term saturates flat once B is metres away, so it is
rescaled to the *current* separation (`d_scale = max(0.35, 0.25·sep)`, `:826-827`).
That rescale is why the gap can widen without the shaping going dead.

**`--carry_demo` is the load-bearing trick.** A third of the envs start in the handoff
distribution, so transit income flows from step one instead of waiting on a grasp to
happen first. Note it is a no-op unless `grasp_latch=False` (`:1138`) — with the latch
on, carry demos silently never spawn.

### Stage 3 — π_place  (`logs/isaac/drone_snatch_place/`) — **not yet trained**

|  |  |
|---|---|
| **Handed** | the output of π_carry — over B, holding the cube; 50% of resets spawn in that state |
| **Success** | `_deposited` = cube within `place_radius 0.20` of B's centre **∧** at rest height ±2.5 cm **∧** speed < 0.10 m/s **∧** `¬_held` (`:551-556`) |
| **Hands on** | nothing — terminal |
| **Gate** | `deposit_rate` EMA ≥ 0.55 at 1.5–2 m, then re-run the ladder if deposit-at-distance is wanted |

This is the **first success predicate in the repo that requires *not* holding the cube**.
Every previous "place" number, including the README's 86.1%, was a still-gripped hover.
The reward encodes the distinction carefully (`:830-840`):

```
lower   = held · over_b · (1 − tanh(seat_err/0.12))     descend while over B      ×15
release = held · over_b · (seat_err<0.04) · (1−grip)    open jaws, seated         ×25
stray   = (¬held) ∧ (d_plat_b > place_radius)           dropped it anywhere else  ×−20
```

The release payment is *gated on being seated over B*, so "drop it early" earns nothing,
and dropping away from B is taxed outright. Keep that gating structure if you touch the
reward — an ungated release term trains a policy that jettisons the cube on takeoff.

Train it **short-range on purpose** (1.5–2 m). Deposit is a precision skill; learn it
where transit is nearly free, then ladder it if needed. Nothing in the place reward
depends on separation.

---

## 4. The handoff contract (runtime)

`run_pipeline.py` maintains a per-env `phase ∈ {0,1,2}` and gathers the action of the
corresponding policy (`:126-137`):

```python
acts = torch.stack([p(obs) for p in policies], dim=0)     # (3,N,5)
act  = acts.gather(0, phase.view(1,n,1).expand(...))[0]
```

Two rules govern transitions, and both matter:

1. **Debounce on the way in.** `0 → 1` requires `_held` for `--hold_steps` (default 10)
   *consecutive* steps = 0.2 s at 50 Hz. A single-frame contact is not a grasp.
2. **Never step backwards inside an episode.** A momentary slip must not restart the
   fly-in — the drone is already 3 m from A and π_snatch would fly it back. Phase resets
   to 0 only on episode `done`.

`1 → 2` fires on `_arrived` with no debounce, which is defensible because
`arrive_radius` (30 cm) is loose relative to the descent π_place then performs, but it is
the obvious next place to add hysteresis if the pipeline is seen oscillating at the pad.

### The honest caveat: the current handoff reads privileged state

`base._held` and `base._arrived` are computed from ground-truth sim poses
(`_d_reach` from the true cube position, `_d_plat_b` from the true B position). A real
drone has neither. **As written, the pipeline is a sim-only evaluation harness, not a
deployable controller** — and it should be labelled as such rather than quietly treated
as the system.

The deployable version of the same architecture replaces the two gates with observable
quantities that already exist in the observation vector:

| gate | privileged form (today) | observable form (intended) |
|---|---|---|
| grasped | `_lifted ∧ d_reach < 0.10` | gripper travel `grip` at the closed stop **∧** `grip_τ` sustained above the free-close baseline **∧** `block_est` tracking the drone's own motion |
| arrived | `‖cube_xy − B_xy‖ < 0.30` | `‖goal_rel_xy‖ < 0.30` — already in the obs, already drift-affected, no new sensing needed |

The arrival gate is nearly free to convert (`goal_rel` is the last 3 entries of the
state block, already in the policy's own input). The
grasp gate needs the grip-torque threshold calibrated once against the free-close
baseline. Until both are converted, every pipeline number should be read as an
*upper bound* on the real-world end-to-end rate.

---

## 5. Training order and the promotion rule

Strictly sequential. Each stage trains only after the one before it has passed its gate,
and warm-starts from that stage's checkpoint:

```
π_snatch (frozen, model_9250.pt)
    └─ INIT ─► π_carry     rung 1, until arrive_rate EMA ≥ 0.55
                   └─ INIT ─► π_place    short range, until deposit_rate EMA ≥ 0.55
                                  └─► run_pipeline.py end-to-end
```

**Warm-start hygiene** — every stage transition must do these three things, because each
one has burned a run before:

- **Re-widen exploration.** A converged policy has `action_std ≈ 0.2` and cannot discover
  a new behaviour. `--reset_std` (0.2 carry, 0.35 place) resets it at load.
- **Start at stage 2.** `--start_stage 2` turns all reward terms on from step 0. The
  hover→grab→carry stage machine (`_update_stage`, `:730-753`) exists to bootstrap a
  *fresh* policy; re-running it on a warm start makes an expert re-earn hovering.
- **Expect the dip.** Changing the success predicate changes the reward landscape, so
  grasp collapses on resume and re-climbs. Observed on the current carry run: 0.119 at
  resume → 0.010 by iter 9252 → 0.261 by iter 9644. **Judge a warm start by the slope
  over ~500 iterations, not by the first 50.**

### Two independent curricula, and they must not be confused

| | `_update_stage` (`:730`) | `_update_sep` (`:775`) |
|---|---|---|
| controls | which reward terms are live (hover→grab→carry) | the A→B separation `plat_sep` |
| driven by | hover / grasp EMAs | **the stage's own success** — arrival for carry, deposit for place |
| thresholds | 0.70 hover, 0.45 grasp, dwell 3000 | 0.55, +0.25 m per expansion, dwell 3000 |
| in this architecture | pinned off via `--start_stage 2` | the live one |

`_update_sep` keying off `_success` is what makes the distance ladder self-pacing *and*
stage-correct: the same code widens the gap on arrivals during carry and on deposits
during place, because `_success` is whichever the stage declared.

---

## 6. The distance ladder

`env_spacing` and `episode_length_s` are sized off `--plat_sep_max` **at scene
construction** (`train_snatch.py:152-154`), so the ceiling cannot be raised mid-run.
Finish a rung, kill it, relaunch with `SEP`/`SEP_MAX` raised and `INIT` pointed at the
last checkpoint:

| rung | separation | `--speed` |
|---|---|---|
| 1 | 1.5 → 4 m | 1.5 |
| 2 | 4 → 10 m | 2.5 |
| 3 | 10 → 20 m | 4.0 |
| 4 | 20 → 40 m | 6.0 |

**Raise `--speed` exactly one rung at a time.** It rescales what `action=1.0` means, so a
large jump invalidates the warm start's learned action magnitudes. And once π_carry
leaves speed 1.5, either π_snatch and π_place must be fine-tuned at the new speed too, or
`cfg.speed` must become per-env — otherwise `run_pipeline.py` is flying the precision
stages at cruise speed (`run_pipeline.py:21-26`).

Also note B's bearing is re-randomised every reset (`:1165-1171`) and the separation is
jittered to 85–100% of `plat_sep`. There is no fixed +X habit to overfit.

---

## 7. Evaluation protocol

Three levels, and the third is the only one that means anything:

1. **Per-stage, honest settings** — `--cur_p 0` (full fly-in, no straddle start),
   `--carry_demo 0` (no synthetic handoff spawns), `--no_latch` (real physics).
   Every stage must be measured with its own demos *off*: `carry_demo` is a training
   crutch and inflates any metric measured with it on.
2. **VIO-drift sweep** — `eval_snatch.py --drift_scales 0 0.25 0.5 1.0 2.0`. This is the
   headline sim2real number, not the clean rate.
3. **End-to-end pipeline** — `run_pipeline.py`, which reports grasped / arrived over B /
   **deposited on B**. `cfg.carry_demo_p = 0` and `curriculum_p = 0` are hardcoded there
   (`run_pipeline.py:87-88`) precisely so the whole task must actually be flown.

> **Gap in the tooling today:** `eval_snatch.py` has no `--two_platform` /
> `--release_only` / `--plat_sep` flags, so there is currently *no* honest per-stage eval
> for carry or place — only the training-time EMAs (which run with demos on) and the
> full pipeline (which needs all three checkpoints to exist). Adding those three flags to
> `eval_snatch.py` is a prerequisite for trusting stage 2 and 3 numbers.

---

## 8. Failure modes

| symptom | reading | action |
|---|---|---|
| grasp collapses to ~0.01 on warm start | expected — success predicate changed | watch the slope over 500 iters before intervening |
| `grasp_rate` recovers, `arrive_rate` flat at ~0 | policy re-learned the pick, transit not discovered | check `carry_prog` has gradient; verify `d_scale` is tracking `_sep` |
| `arrive_rate` good, `deposit_rate` ~0 | it is hovering while gripping — the classic false place | this is exactly what `--release_only` exists to fix; do not read `place_success` from a non-release run as a deposit |
| cube dropped en route | friction cage under-models a real grip when off-centre | tighten the approach centring in π_snatch, not the carry reward |
| `plat_sep` never expands | `_success` EMA below 0.55, or dwell not elapsed | it is self-pacing by design — do not force `plat_sep` up manually |
| restart loop reprints every 10 s | import/env failure, not a crash | check the log shows real iterations (CLAUDE.md) |
| segfault after `TRAIN_SNATCH_OK` | benign Vulkan teardown on this headless host | ignore; checkpoints are already written |
| `value_function loss ~6e5` | unnormalized reward scale (mean reward ~3e4) | not a divergence signal on its own |

---

## 9. Status — 2026-08-17

| stage | checkpoint | state |
|---|---|---|
| π_snatch | `snatch/checkpoints/model_9250.pt` | **converged, committed, frozen** |
| π_carry | `logs/isaac/drone_snatch_carry/model_*.pt` | **training** — iter ~9650/49250, rung 1 (1.5 m, ceiling 4 m), grasp 0.26 and climbing, arrive ~0.002 |
| π_place | — | **not started** — blocked on π_carry's gate |
| pipeline | `run_pipeline.py` | untested end-to-end (needs all three checkpoints) |

Known gaps, in the order they should be closed:

1. π_carry has not passed its gate; π_place cannot start until it does.
2. `eval_snatch.py` cannot evaluate stage 2 or 3 (§7).
3. The runtime handoff reads privileged sim state (§4) — the arrival gate is nearly free
   to convert, the grasp gate needs one calibration.
4. `cfg.speed` is global, so mixing distance rungs across the three policies is invalid
   (§6).
5. No `--rc_distance_mode` fly-in beyond 10 m has ever converged — rungs 3 and 4 are
   unproven territory, not a scheduled task.
