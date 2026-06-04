"""SNATCH domain-randomization + sim2real-noise module (Agent A3).

This is the HEADLINE of the SNATCH paper. There are NO real flights, so the
sim-to-real story is: we *model* the dominant hardware/perception gaps directly
in simulation and measure how policy success degrades as we dial them up. Every
function here corresponds to a concrete real-hardware gap:

  - motor thrust scale / time-constant  -> ESC + propeller manufacturing spread
  - payload mass                        -> unknown/variable grasped object mass
  - wind + gust spikes                  -> outdoor disturbance the controller never
                                           saw in a still sim
  - ground effect                       -> extra lift near the ground/landing pad
  - lighting                            -> exposure changes the camera sees
  - depth noise + dropout               -> RealSense/Pi-cam depth sensor artefacts
  - control latency / obs buffer        -> USB + compute + comms pipeline delay
  - VIO drift  (apply_vio_drift)        -> THE key gap knob: onboard visual-inertial
                                           odometry slowly drifts from ground truth
  - detection noise (apply_detection_noise) -> bottom-cam block detector is biased,
                                           noisy, and occasionally drops a frame

Everything is PURE TORCH and vectorized over ``num_envs`` so it is unit-testable
without launching Isaac. Functions never mutate their inputs in place; they
return fresh tensors.

Design notes
------------
* ``sample_dr_params`` is called ONCE per episode-reset to draw a fixed
  per-env physical configuration (thrust, tau, payload, ...) plus the *scales*
  of the stochastic noise processes. The per-step noise functions
  (``apply_vio_drift``, ``apply_detection_noise``, ``add_depth_noise``) then draw
  fresh randomness each control step using those scales.
* Two "gap knobs" are exposed for the sweep so an evaluator can turn the gap from
  zero (oracle sim) to large and plot success-vs-gap:
    - ``vio_drift_scale``        (localization gap)
    - ``detection_noise_scale``  (perception gap)
  Setting a scale to 0 makes the corresponding function an exact identity, which
  the tests assert.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor

# --------------------------------------------------------------------------- #
# Constants for the SNATCH domain-randomization table (see DESIGN.md).
# All ranges are taken straight from the spec.
# --------------------------------------------------------------------------- #
_THRUST_SCALE_RANGE = (0.85, 1.15)        # motor thrust +/-15%
_MOTOR_TAU_RANGE = (0.020, 0.050)         # motor time constant 20-50 ms
_PAYLOAD_RANGE = (0.0, 0.300)             # payload 0-300 g (kg)
_WIND_STD = 0.5                           # wind ~ N(0, 0.5 N) per axis
_GUST_MAG_RANGE = (0.0, 2.0)              # gust spike magnitude 0-2 N
_GUST_DURATION = 0.5                      # gust lasts ~0.5 s
_GUST_PROB_PER_S = 1.0                    # expected ~1 gust onset / second / env
_GROUND_EFFECT_RANGE = (0.5, 2.0)        # upward force U(0.5, 2.0) N
_GROUND_EFFECT_ALT = 0.5                  # only active below 0.5 m altitude
_LIGHTING_RANGE = (200.0, 2000.0)         # lighting 200-2000 lux
_DEPTH_NOISE_STD = 0.01                   # camera depth noise sigma = 0.01 m
_DEPTH_DROPOUT = 0.02                     # 2% depth pixel dropout
_LATENCY_RANGE = (0.050, 0.150)           # control latency 50-150 ms
_LATENCY_STEPS_RANGE = (1, 5)             # -> 1-5 step obs buffer
_CONTROL_DT = 0.02                        # 50 Hz policy step (DESIGN.md)

# VIO-drift defaults (the headline knob). These are the *nominal* magnitudes at
# vio_drift_scale = 1.0; the sweep multiplies them.
_VIO_BIAS_STD = 0.05          # per-episode constant bias std (m)
_VIO_WALK_STD = 0.01          # random-walk increment std per second (m / sqrt(s))
_VIO_RAMP_TAU = 30.0          # seconds over which the bias term ramps in

# Detection-noise defaults (perception knob), at detection_noise_scale = 1.0.
_DET_BIAS_STD = 0.02          # per-episode constant detector bias std (m)
_DET_NOISE_STD = 0.03         # per-step gaussian detection noise std (m)
_DET_DROPOUT = 0.05           # 5% chance the detector drops the block this step


def _uniform(num_envs: int, lo: float, hi: float, device) -> Tensor:
    """Vectorized U(lo, hi) of shape (num_envs,)."""
    return torch.rand(num_envs, device=device) * (hi - lo) + lo


def sample_dr_params(
    num_envs: int,
    device,
    vio_drift_scale: float = 1.0,
    detection_noise_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """Draw a per-env domain-randomization configuration.

    Call this once per episode reset. Returns a dict of tensors, each leading
    dim == ``num_envs`` (scalars-per-env are shape ``(num_envs,)``, vectors are
    ``(num_envs, 3)``). The two ``*_scale`` knobs are broadcast into per-env
    tensors so a sweep can pass a single float and have it apply to every env,
    and so zero exactly disables a noise source.

    Each entry maps to a real-hardware gap (see module docstring).
    """
    g = device

    # --- physical drone randomization (dynamics gap) ---
    thrust_scale = _uniform(num_envs, *_THRUST_SCALE_RANGE, g)        # ESC/prop spread
    motor_tau = _uniform(num_envs, *_MOTOR_TAU_RANGE, g)              # motor lag
    payload = _uniform(num_envs, *_PAYLOAD_RANGE, g)                  # unknown mass (kg)

    # --- disturbances (outdoor gap) ---
    # Constant-ish wind: a steady vector drawn per env (N per axis).
    wind = torch.randn(num_envs, 3, device=g) * _WIND_STD
    # Gust: pre-draw magnitude + direction; the *onset* is sampled per step by
    # the env using gust_prob_per_step. Direction is a random unit vector.
    gust_mag = _uniform(num_envs, *_GUST_MAG_RANGE, g)
    gust_dir = torch.randn(num_envs, 3, device=g)
    gust_dir = gust_dir / (gust_dir.norm(dim=-1, keepdim=True) + 1e-8)
    gust_prob_per_step = torch.full(
        (num_envs,), _GUST_PROB_PER_S * _CONTROL_DT, device=g
    )
    gust_duration_steps = torch.full(
        (num_envs,), max(1.0, round(_GUST_DURATION / _CONTROL_DT)), device=g
    )

    # --- ground effect (near-ground lift gap) ---
    ground_effect_mag = _uniform(num_envs, *_GROUND_EFFECT_RANGE, g)
    ground_effect_alt = torch.full((num_envs,), _GROUND_EFFECT_ALT, device=g)

    # --- visual (lighting / depth-sensor gap) ---
    lighting = _uniform(num_envs, *_LIGHTING_RANGE, g)
    depth_noise_std = torch.full((num_envs,), _DEPTH_NOISE_STD, device=g)
    depth_dropout = torch.full((num_envs,), _DEPTH_DROPOUT, device=g)

    # --- latency (pipeline-delay gap) ---
    latency_s = _uniform(num_envs, *_LATENCY_RANGE, g)
    latency_steps = torch.randint(
        _LATENCY_STEPS_RANGE[0], _LATENCY_STEPS_RANGE[1] + 1, (num_envs,), device=g
    )

    # --- VIO drift (HEADLINE localization gap) ---
    vio_scale = torch.full((num_envs,), float(vio_drift_scale), device=g)
    # Per-episode constant bias direction*magnitude (scaled). Drawn here so the
    # bias is fixed for the episode; the random walk is added per step.
    vio_bias = torch.randn(num_envs, 3, device=g) * _VIO_BIAS_STD * vio_scale.unsqueeze(-1)
    vio_walk_std = torch.full((num_envs,), _VIO_WALK_STD, device=g) * vio_scale

    # --- detection noise (perception gap) ---
    det_scale = torch.full((num_envs,), float(detection_noise_scale), device=g)
    det_bias = torch.randn(num_envs, 3, device=g) * _DET_BIAS_STD * det_scale.unsqueeze(-1)
    det_noise_std = torch.full((num_envs,), _DET_NOISE_STD, device=g) * det_scale
    det_dropout = torch.full((num_envs,), _DET_DROPOUT, device=g) * det_scale

    return {
        # dynamics
        "thrust_scale": thrust_scale,
        "motor_tau": motor_tau,
        "payload": payload,
        # disturbances
        "wind": wind,
        "gust_mag": gust_mag,
        "gust_dir": gust_dir,
        "gust_prob_per_step": gust_prob_per_step,
        "gust_duration_steps": gust_duration_steps,
        # ground effect
        "ground_effect_mag": ground_effect_mag,
        "ground_effect_alt": ground_effect_alt,
        # visual
        "lighting": lighting,
        "depth_noise_std": depth_noise_std,
        "depth_dropout": depth_dropout,
        # latency
        "latency_s": latency_s,
        "latency_steps": latency_steps,
        # VIO drift (headline knob)
        "vio_drift_scale": vio_scale,
        "vio_bias": vio_bias,
        "vio_walk_std": vio_walk_std,
        # detection
        "detection_noise_scale": det_scale,
        "det_bias": det_bias,
        "det_noise_std": det_noise_std,
        "det_dropout": det_dropout,
    }


def apply_vio_drift(pose: Tensor, t: Tensor, params: Dict[str, Tensor]) -> Tensor:
    """Apply modeled VIO localization drift to the drone pose estimate.

    THIS IS THE KEY SIM2REAL GAP KNOB. Real onboard visual-inertial odometry
    does not give ground-truth pose: it accumulates a slowly-ramping bias plus a
    random walk that grows with time. We model the *estimated* pose as

        pose_est = pose + bias * (1 - exp(-t / tau)) + cumulative_random_walk

    where the random-walk term has std proportional to ``sqrt(t)`` (Brownian),
    so drift magnitude grows with both ``t`` and ``vio_drift_scale``. Setting
    ``vio_drift_scale = 0`` (-> zero bias and zero walk std) makes this an exact
    identity, giving the "oracle localization" baseline of the sweep.

    Args:
        pose: (N, 3) or (N, >=3) true pose; only the first 3 (position) columns
            are perturbed (orientation is left untouched).
        t: scalar or (N,) episode time in seconds.
        params: from ``sample_dr_params`` (uses ``vio_bias``, ``vio_walk_std``).

    Returns:
        Pose tensor of the same shape with drift added to the position columns.
    """
    bias = params["vio_bias"]                       # (N, 3)
    walk_std = params["vio_walk_std"]               # (N,)
    device = pose.device

    t = torch.as_tensor(t, dtype=pose.dtype, device=device)
    if t.dim() == 0:
        t = t.expand(pose.shape[0])
    t = t.clamp_min(0.0)

    # Ramp-in of the constant bias: 0 at t=0, -> bias as t grows.
    ramp = (1.0 - torch.exp(-t / _VIO_RAMP_TAU)).unsqueeze(-1)       # (N,1)
    bias_term = bias * ramp                                          # (N,3)

    # Brownian random walk: accumulated variance ~ t, so std ~ sqrt(t).
    walk_sigma = (walk_std * torch.sqrt(t)).unsqueeze(-1)            # (N,1)
    walk_term = torch.randn(pose.shape[0], 3, device=device) * walk_sigma

    drift = bias_term + walk_term                                    # (N,3)

    out = pose.clone()
    out[:, :3] = out[:, :3] + drift
    return out


def apply_detection_noise(block_pos: Tensor, params: Dict[str, Tensor]) -> Tensor:
    """Apply modeled bottom-camera block-detection noise.

    The real bottom-cam block detector has (a) a small per-episode systematic
    bias (extrinsics/calibration error), (b) per-frame gaussian jitter, and
    (c) occasional dropouts (motion blur / occlusion / detector miss). On a
    dropout we return the LAST-KNOWN estimate proxy by emitting a sentinel of
    NaN-free fallback: we hold the biased position but flag via large noise is
    avoided — instead we return the input block position unchanged for that env
    (i.e. "stale" estimate). ``detection_noise_scale = 0`` -> exact identity.

    Args:
        block_pos: (N, 3) ground-truth block position in the bottom-cam frame.
        params: from ``sample_dr_params``.

    Returns:
        (N, 3) noised block-position estimate.
    """
    bias = params["det_bias"]                       # (N,3)
    noise_std = params["det_noise_std"]             # (N,)
    dropout = params["det_dropout"]                 # (N,)
    device = block_pos.device
    n = block_pos.shape[0]

    noise = torch.randn(n, 3, device=device) * noise_std.unsqueeze(-1)
    est = block_pos + bias + noise

    # Dropout: with prob `dropout`, the detector failed this frame -> hold the
    # (un-noised) input as a stale estimate. At dropout=0 this never fires.
    drop_mask = (torch.rand(n, device=device) < dropout).unsqueeze(-1)
    est = torch.where(drop_mask, block_pos, est)
    return est


def add_depth_noise(depth: Tensor, params: Dict[str, Tensor]) -> Tensor:
    """Add gaussian depth noise + bernoulli pixel dropout to a depth image.

    Models RealSense / Pi-cam depth-sensor artefacts: zero-mean gaussian range
    noise (sigma ~ 0.01 m) and a fraction of pixels that return no reading
    (dropout -> 0.0). With ``depth_noise_std = 0`` and ``depth_dropout = 0`` this
    is an exact identity.

    Args:
        depth: (N, H, W) metric depth.
        params: from ``sample_dr_params`` (``depth_noise_std``, ``depth_dropout``).

    Returns:
        (N, H, W) noised depth (same shape/dtype).
    """
    std = params["depth_noise_std"].view(-1, 1, 1)      # (N,1,1)
    dropout = params["depth_dropout"].view(-1, 1, 1)    # (N,1,1)

    noisy = depth + torch.randn_like(depth) * std
    keep = (torch.rand_like(depth) >= dropout).to(depth.dtype)
    return noisy * keep


def apply_obs_latency(buffer: Tensor, params: Dict[str, Tensor]) -> Tensor:
    """Return a delayed observation from a ring/history buffer (pipeline delay).

    Models USB + compute + comms latency: the policy acts on a stale obs. The
    buffer holds the last L observations along dim=1, ordered oldest..newest
    (index -1 == most recent). Per env we index back ``latency_steps`` frames
    (clamped to the buffer length). ``latency_steps`` is drawn in
    ``sample_dr_params`` from the 1-5 step / 50-150 ms range.

    Args:
        buffer: (N, L, D) history of observations, index -1 = newest.
        params: from ``sample_dr_params`` (uses ``latency_steps``).

    Returns:
        (N, D) the delayed observation per env.
    """
    n, L, _ = buffer.shape
    delay = params["latency_steps"].to(buffer.device).clamp(0, L - 1)  # (N,)
    idx = (L - 1) - delay                                              # (N,) oldest..newest
    out = buffer[torch.arange(n, device=buffer.device), idx]          # (N, D)
    return out


def ground_effect_force(alt: Tensor, params: Dict[str, Tensor]) -> Tensor:
    """Upward ground-effect force, active only below the threshold altitude.

    Near the ground (alt < ~0.5 m) the rotor downwash recirculates and produces
    extra lift the still-air sim wouldn't otherwise have. We model a force that
    is ``ground_effect_mag`` Newtons up at alt=0, decays linearly to 0 at the
    threshold altitude, and is exactly 0 above it.

    Args:
        alt: (N,) or (N,1) drone altitude (m).
        params: from ``sample_dr_params`` (``ground_effect_mag``,
            ``ground_effect_alt``).

    Returns:
        (N, 3) force vector (only +z populated).
    """
    mag = params["ground_effect_mag"]               # (N,)
    thr = params["ground_effect_alt"]               # (N,)
    device = mag.device

    alt = torch.as_tensor(alt, dtype=mag.dtype, device=device).reshape(-1)
    frac = (1.0 - alt / thr).clamp(min=0.0)         # 1 at ground, 0 at threshold
    fz = mag * frac                                 # zero above threshold
    force = torch.zeros(alt.shape[0], 3, device=device, dtype=mag.dtype)
    force[:, 2] = fz
    return force
