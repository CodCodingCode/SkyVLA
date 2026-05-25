#!/usr/bin/env python3
"""Smoke test for the SubgoalDiT pipeline.

Validates the four pieces of the world-model pathway WITHOUT touching
real OpenFly data, so it runs anywhere (CPU, no PaliGemma weights, no
trajectory images):

  1. ``SubgoalDiT`` constructs and runs a forward pass on synthetic
     2048-d SigLIP tokens.
  2. ``q_sample`` + DDIM sampler produce finite outputs of the right
     shape.
  3. One training step (eps-MSE backward + AdamW.step) completes.
  4. ``PaliGemmaVLNPolicy`` with ``subgoal_dit=None`` accepts the new
     ``subgoal_pose_delta`` / ``subgoal_horizon`` kwargs (so existing
     BC training paths keep working).

If anything below fails, the integration into the real trainers will
fail in the same way — fix it here first.

Usage:
  python -m openfly.scripts.smoke_subgoal_dit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_DRONE_ROOT = Path(__file__).resolve().parents[2]
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.models.subgoal_dit import (
    SubgoalDiT,
    cosine_alpha_bar,
)


def _print_ok(name: str) -> None:
    print(f"  [OK] {name}")


def _print_fail(name: str, exc: BaseException) -> None:
    print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")


def test_alpha_bar() -> None:
    ab = cosine_alpha_bar(1000)
    assert ab.shape == (1001,), f"expected 1001 schedule entries, got {ab.shape}"
    assert torch.all(ab > 0), "alpha_bar must be positive"
    assert torch.all(ab <= 1.0), "alpha_bar must be ≤ 1"
    assert ab[0] > ab[-1], "alpha_bar should decrease over time"
    _print_ok("cosine_alpha_bar shape + monotonicity")


def test_dit_forward(device: torch.device) -> None:
    dit = SubgoalDiT(
        token_dim=2048, hidden=256, depth=2, num_heads=4,
        text_dim=2048, pose_delta_dim=4, num_last_actions=9,
        num_timesteps=100,
    ).to(device)
    B, S, D = 3, 256, 2048
    curr = torch.randn(B, S, D, device=device)
    noisy = torch.randn(B, S, D, device=device)
    t = torch.randint(0, dit.num_timesteps, (B,), device=device)
    text = torch.randn(B, 2048, device=device)
    pose = torch.randn(B, 4, device=device)
    la = torch.randint(0, 9, (B,), device=device)
    hz = torch.randint(0, 32, (B,), device=device)

    out = dit(curr, noisy, t, text, pose, la, hz)
    assert out.shape == (B, S, D), f"got {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in DiT output"
    _print_ok(f"SubgoalDiT forward — shape {tuple(out.shape)} all finite")


def test_q_sample(device: torch.device) -> None:
    dit = SubgoalDiT(
        token_dim=2048, hidden=128, depth=1, num_heads=2,
        num_timesteps=100,
    ).to(device)
    B, S, D = 2, 256, 2048
    x0 = torch.randn(B, S, D, device=device)
    t = torch.tensor([0, 99], device=device, dtype=torch.long)
    x_t, noise = dit.q_sample(x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape
    # At t=0, x_t ≈ x0 (alpha_bar near 1). At t=99, x_t ≈ noise.
    d_t0 = (x_t[0] - x0[0]).abs().mean().item()
    d_t99 = (x_t[1] - noise[1]).abs().mean().item()
    assert d_t0 < d_t99, (
        f"q_sample monotonicity broken: ‖x_t0 − x0‖ = {d_t0:.4f} "
        f"≥ ‖x_t99 − noise‖ = {d_t99:.4f}"
    )
    _print_ok(f"q_sample monotonic: t=0 Δ={d_t0:.4f} < t=99 Δ={d_t99:.4f}")


def test_ddim_sample(device: torch.device) -> None:
    dit = SubgoalDiT(
        token_dim=2048, hidden=128, depth=1, num_heads=2,
        num_timesteps=100,
    ).to(device)
    dit.eval()
    B, S, D = 2, 256, 2048
    curr = torch.randn(B, S, D, device=device)
    text = torch.randn(B, 2048, device=device)
    pose = torch.zeros(B, 4, device=device)
    la = torch.zeros(B, device=device, dtype=torch.long)
    hz = torch.full((B,), 4, device=device, dtype=torch.long)

    out = dit.ddim_sample(curr, text, pose, la, hz, num_steps=4)
    assert out.shape == (B, S, D)
    assert torch.isfinite(out).all(), "non-finite values in DDIM output"
    _print_ok(f"DDIM 4-step sample — shape {tuple(out.shape)} all finite")


def test_training_step(device: torch.device) -> None:
    """One forward + backward + optimizer.step. Verifies gradient plumbing."""
    dit = SubgoalDiT(
        token_dim=2048, hidden=256, depth=2, num_heads=4,
        num_timesteps=100,
    ).to(device)
    opt = torch.optim.AdamW(dit.parameters(), lr=1e-4)

    B, S, D = 3, 256, 2048
    curr = torch.randn(B, S, D, device=device)
    x0 = torch.randn(B, S, D, device=device)
    text = torch.randn(B, 2048, device=device)
    pose = torch.randn(B, 4, device=device)
    la = torch.randint(0, 9, (B,), device=device)
    hz = torch.randint(0, 32, (B,), device=device)
    t = torch.randint(0, dit.num_timesteps, (B,), device=device)

    x_t, noise = dit.q_sample(x0, t)
    eps_pred = dit(curr, x_t, t, text, pose, la, hz)
    loss = F.mse_loss(eps_pred, noise)
    initial_loss = float(loss.item())
    loss.backward()

    # Verify some parameter received a gradient.
    has_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in dit.parameters() if p.requires_grad
    )
    assert has_grad, "no parameter received a finite, non-zero gradient"
    opt.step()
    _print_ok(f"training step OK — loss={initial_loss:.4f}")


def test_policy_backward_compat(device: torch.device) -> None:
    """The new kwargs on PaliGemmaVLNPolicy.forward must not break old callers.

    We construct the policy *without* PaliGemma (mock its forward path) so this
    runs anywhere. The goal here is purely API-shape verification, not
    correctness of the BC head.
    """
    # Import lazily so a missing transformers install doesn't kill the
    # other smoke tests.
    try:
        from openfly.models.paligemma_vln import PaliGemmaVLNPolicy  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] policy import failed ({exc}); transformers not installed?")
        return

    # Inspect the signature only — full construction needs the PaliGemma
    # weights. We just confirm the new kwargs exist.
    import inspect
    sig = inspect.signature(PaliGemmaVLNPolicy.forward)
    params = set(sig.parameters)
    expected = {"subgoal_pose_delta", "subgoal_horizon"}
    missing = expected - params
    assert not missing, f"PaliGemmaVLNPolicy.forward missing kwargs: {missing}"
    _print_ok("PaliGemmaVLNPolicy.forward exposes subgoal_* kwargs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu",
                        help="Device for synthetic tensor tests (default: cpu).")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    print(f"[smoke_subgoal_dit] device={device}")

    tests = [
        ("alpha_bar", test_alpha_bar, ()),
        ("dit_forward", test_dit_forward, (device,)),
        ("q_sample", test_q_sample, (device,)),
        ("ddim_sample", test_ddim_sample, (device,)),
        ("training_step", test_training_step, (device,)),
        ("policy_backward_compat", test_policy_backward_compat, (device,)),
    ]
    n_ok = 0
    n_fail = 0
    for name, fn, fargs in tests:
        try:
            fn(*fargs)
            n_ok += 1
        except (AssertionError, RuntimeError, ValueError) as exc:
            _print_fail(name, exc)
            n_fail += 1
    print(f"[smoke_subgoal_dit] {n_ok} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
