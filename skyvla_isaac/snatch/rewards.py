"""SNATCH reward shaping + grasp-trigger heuristic (Agent A4).

Pure-torch, no Isaac import — unit-testable on its own. The orchestrator's
``snatch/pick_place_env.py`` assembles the state dict ``s`` each step and calls
:func:`compute_reward`.

Reward = 4-component shaping from the SNATCH spec (DESIGN.md, "snatch/rewards.py"):

    r_nav       = -0.1 * d_block                       # approach the block
    r_align     = -(pixel_offset / 320) * 0.5          # center block in bottom cam
    r_alt       = -|alt_above_block - 0.2| * 0.3       # hold ~0.2 m grasp standoff
    r_grasp     = +10 * grasp_success                  # one-shot grasp bonus
    r_transport = -0.1 * d_goal   if carrying else 0   # carry toward goal
    r_place     = +15 * place_success                  # one-shot delivery bonus
    r_crash     = -50 * crashed                        # dominant failure penalty
    r_time      = -0.005                               # per-step time pressure

All terms are vectorized over the (N,) env batch.
"""
from __future__ import annotations

import torch

# ---- shaping coefficients (single source of truth; tests import these) -------
NAV_COEF = 0.1
ALIGN_COEF = 0.5
ALIGN_PX_NORM = 320.0          # half the 640px bottom-cam width
ALT_COEF = 0.3
ALT_TARGET = 0.2               # m, caging-grasp standoff above block
GRASP_BONUS = 10.0
TRANSPORT_COEF = 0.1
PLACE_BONUS = 15.0
CRASH_PENALTY = 50.0
TIME_PENALTY = 0.005

# ---- grasp-trigger heuristic thresholds -------------------------------------
GRASP_OFFSET_PX = 20.0         # block centered within 20 px of bottom-cam center
GRASP_ALT_M = 0.25             # drone within 0.25 m above block


def _as_tensor(x, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar/bool/array key to a float tensor on ref's device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=ref.device)


def compute_reward(s: dict) -> torch.Tensor:
    """Vectorized SNATCH reward.

    Args:
        s: dict with keys (each a scalar or (N,) tensor):
            d_block         float  distance drone->block (m)
            pixel_offset    float  block offset from bottom-cam center (px)
            alt_above_block float  drone altitude above block (m)
            grasp_success   bool   grasp closed this step (one-shot)
            carrying        bool   block currently held
            d_goal          float  distance block->goal (m)
            place_success   bool   delivered at goal this step (one-shot)
            crashed         bool   collided / out-of-bounds this step

    Returns:
        (N,) reward tensor (float32) on the same device as the inputs.
    """
    # Find a reference tensor for device/shape; fall back to any float key.
    ref = None
    for k in ("d_block", "pixel_offset", "alt_above_block", "d_goal"):
        v = s.get(k)
        if isinstance(v, torch.Tensor):
            ref = v
            break
    if ref is None:
        # No tensor present — make a 1-element tensor on CPU.
        ref = torch.zeros(1)

    d_block = _as_tensor(s["d_block"], ref)
    pixel_offset = _as_tensor(s["pixel_offset"], ref)
    alt_above_block = _as_tensor(s["alt_above_block"], ref)
    grasp_success = _as_tensor(s["grasp_success"], ref)
    carrying = _as_tensor(s["carrying"], ref)
    d_goal = _as_tensor(s["d_goal"], ref)
    place_success = _as_tensor(s["place_success"], ref)
    crashed = _as_tensor(s["crashed"], ref)

    r_nav = -NAV_COEF * d_block
    r_align = -(pixel_offset / ALIGN_PX_NORM) * ALIGN_COEF
    r_alt = -torch.abs(alt_above_block - ALT_TARGET) * ALT_COEF
    r_grasp = GRASP_BONUS * grasp_success
    r_transport = (-TRANSPORT_COEF * d_goal) * carrying     # gated by carrying
    r_place = PLACE_BONUS * place_success
    r_crash = -CRASH_PENALTY * crashed
    r_time = torch.full_like(d_block, -TIME_PENALTY)

    reward = (
        r_nav + r_align + r_alt + r_grasp
        + r_transport + r_place + r_crash + r_time
    )
    return reward


def grasp_trigger(
    pixel_offset,
    alt,
    mode: str = "heuristic",
    contact_close: bool = False,
) -> torch.Tensor:
    """Decide whether to command the caging gripper to close.

    Two paths (both supported; ``heuristic`` is the spec default):

    - ``mode="heuristic"``: close when the block is centered in the bottom cam
      (offset < 20 px) AND the drone is within 0.25 m above it. Deterministic,
      no learned contact needed — robust for the 4-jaw cage which tolerates
      sloppy centering.
    - ``mode="learned"``: the policy emits the gripper action directly and the
      caging contact does the work; ``contact_close`` carries the policy's
      decision. The cage geometry makes a learned-contact grasp ALSO viable
      (the jaws funnel the block in), so we expose it for ablations.

    Args:
        pixel_offset: scalar or (N,) block offset from cam center (px).
        alt: scalar or (N,) drone altitude above block (m).
        mode: "heuristic" (default) or "learned".
        contact_close: (learned mode) policy/contact close signal, bool/(N,).

    Returns:
        Bool tensor (same shape as inputs) — True => command gripper close.
    """
    po = torch.as_tensor(pixel_offset, dtype=torch.float32)
    a = torch.as_tensor(alt, dtype=torch.float32)
    if mode == "heuristic":
        return (po < GRASP_OFFSET_PX) & (a < GRASP_ALT_M)
    elif mode == "learned":
        return torch.as_tensor(contact_close, dtype=torch.bool).expand_as(
            torch.zeros_like(po, dtype=torch.bool)
        )
    raise ValueError(f"unknown grasp_trigger mode: {mode!r}")
