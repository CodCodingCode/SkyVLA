"""Pure-torch unit tests for snatch/rewards.py (Agent A4).

Runnable with plain `python` (no Isaac):

    PYTHONPATH=$PWD python skyvla_isaac/snatch/tests/test_rewards.py

Prints REWARD_TESTS_OK on success.
"""
from __future__ import annotations

import torch

from skyvla_isaac.snatch.rewards import (
    compute_reward, grasp_trigger,
    NAV_COEF, ALIGN_COEF, ALIGN_PX_NORM, ALT_COEF, ALT_TARGET,
    GRASP_BONUS, TRANSPORT_COEF, PLACE_BONUS, CRASH_PENALTY, TIME_PENALTY,
    GRASP_OFFSET_PX, GRASP_ALT_M,
)


def _zero_state(n=1):
    """A state where every shaping term is exactly zero except time."""
    z = torch.zeros(n)
    return {
        "d_block": z.clone(),
        "pixel_offset": z.clone(),
        "alt_above_block": torch.full((n,), ALT_TARGET),  # zero alt error
        "grasp_success": torch.zeros(n, dtype=torch.bool),
        "carrying": torch.zeros(n, dtype=torch.bool),
        "d_goal": z.clone(),
        "place_success": torch.zeros(n, dtype=torch.bool),
        "crashed": torch.zeros(n, dtype=torch.bool),
    }


def approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


def test_baseline_is_only_time_penalty():
    r = compute_reward(_zero_state())
    assert approx(r.item(), -TIME_PENALTY), f"baseline should be -time, got {r.item()}"


def test_nav_negative_and_scaled():
    s = _zero_state(); s["d_block"] = torch.tensor([2.0])
    r = compute_reward(s)
    expected = -NAV_COEF * 2.0 - TIME_PENALTY
    assert approx(r.item(), expected), f"nav {r.item()} != {expected}"
    assert r.item() < -TIME_PENALTY, "nav term must be negative"


def test_align_negative_and_scaled():
    s = _zero_state(); s["pixel_offset"] = torch.tensor([320.0])  # == ALIGN_PX_NORM
    r = compute_reward(s)
    expected = -(320.0 / ALIGN_PX_NORM) * ALIGN_COEF - TIME_PENALTY
    assert approx(r.item(), expected), f"align {r.item()} != {expected}"
    # full-width offset costs exactly ALIGN_COEF
    assert approx(expected + TIME_PENALTY, -ALIGN_COEF)


def test_alt_error_symmetric():
    lo = _zero_state(); lo["alt_above_block"] = torch.tensor([ALT_TARGET - 0.1])
    hi = _zero_state(); hi["alt_above_block"] = torch.tensor([ALT_TARGET + 0.1])
    rlo, rhi = compute_reward(lo).item(), compute_reward(hi).item()
    assert approx(rlo, rhi), "alt penalty must be symmetric about target"
    expected = -abs(0.1) * ALT_COEF - TIME_PENALTY
    assert approx(rlo, expected), f"alt {rlo} != {expected}"
    # at target, no alt penalty
    assert approx(compute_reward(_zero_state()).item(), -TIME_PENALTY)


def test_grasp_bonus():
    s = _zero_state(); s["grasp_success"] = torch.tensor([True])
    r = compute_reward(s)
    assert approx(r.item(), GRASP_BONUS - TIME_PENALTY), f"grasp {r.item()}"
    assert r.item() > 9.0, "grasp bonus must dominate shaping"


def test_place_bonus():
    s = _zero_state(); s["place_success"] = torch.tensor([True])
    r = compute_reward(s)
    assert approx(r.item(), PLACE_BONUS - TIME_PENALTY), f"place {r.item()}"
    assert PLACE_BONUS > GRASP_BONUS, "place should exceed grasp"


def test_carrying_gates_transport():
    # Not carrying: transport term must be zero regardless of d_goal.
    s0 = _zero_state(); s0["d_goal"] = torch.tensor([5.0])
    assert approx(compute_reward(s0).item(), -TIME_PENALTY), "transport must be gated off"
    # Carrying: transport term active and negative.
    s1 = _zero_state(); s1["d_goal"] = torch.tensor([5.0])
    s1["carrying"] = torch.tensor([True])
    r1 = compute_reward(s1)
    expected = -TRANSPORT_COEF * 5.0 - TIME_PENALTY
    assert approx(r1.item(), expected), f"transport {r1.item()} != {expected}"
    assert r1.item() < compute_reward(s0).item(), "carrying far from goal must cost more"


def test_crash_dominates():
    s = _zero_state()
    s["crashed"] = torch.tensor([True])
    # stack every positive bonus against the crash; crash must still win.
    s["grasp_success"] = torch.tensor([True])
    s["place_success"] = torch.tensor([True])
    r = compute_reward(s)
    expected = GRASP_BONUS + PLACE_BONUS - CRASH_PENALTY - TIME_PENALTY
    assert approx(r.item(), expected), f"crash {r.item()} != {expected}"
    assert r.item() < 0.0, "crash must dominate all bonuses -> net negative"
    assert r.item() < -20.0, "crash penalty should be clearly dominant"


def test_vectorized_batch():
    n = 8
    s = _zero_state(n)
    s["d_block"] = torch.arange(n, dtype=torch.float32)
    r = compute_reward(s)
    assert r.shape == (n,), f"expected ({n},) got {tuple(r.shape)}"
    # monotonically decreasing in d_block
    assert torch.all(r[1:] <= r[:-1]), "reward must decrease with d_block"


def test_scalar_inputs_accepted():
    s = {
        "d_block": 1.0, "pixel_offset": 0.0, "alt_above_block": ALT_TARGET,
        "grasp_success": False, "carrying": False, "d_goal": 0.0,
        "place_success": False, "crashed": False,
    }
    r = compute_reward(s)
    assert approx(r.item(), -NAV_COEF * 1.0 - TIME_PENALTY), f"scalar path {r.item()}"


def test_grasp_trigger_heuristic():
    # centered + low -> close
    assert bool(grasp_trigger(torch.tensor(10.0), torch.tensor(0.2)))
    # off-center -> no close
    assert not bool(grasp_trigger(torch.tensor(50.0), torch.tensor(0.2)))
    # too high -> no close
    assert not bool(grasp_trigger(torch.tensor(10.0), torch.tensor(0.4)))
    # boundary: exactly at thresholds is NOT a close (strict <)
    assert not bool(grasp_trigger(torch.tensor(GRASP_OFFSET_PX), torch.tensor(0.1)))
    assert not bool(grasp_trigger(torch.tensor(5.0), torch.tensor(GRASP_ALT_M)))


def test_grasp_trigger_vectorized():
    po = torch.tensor([5.0, 25.0, 5.0])
    alt = torch.tensor([0.1, 0.1, 0.5])
    out = grasp_trigger(po, alt)
    assert out.tolist() == [True, False, False], out.tolist()


def test_grasp_trigger_learned_mode():
    po = torch.tensor([200.0, 200.0])    # offset irrelevant in learned mode
    alt = torch.tensor([1.0, 1.0])
    out = grasp_trigger(po, alt, mode="learned",
                        contact_close=torch.tensor([True, False]))
    assert out.tolist() == [True, False], out.tolist()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("REWARD_TESTS_OK")
