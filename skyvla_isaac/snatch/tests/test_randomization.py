"""Pure-torch tests for snatch/randomization.py (Agent A3).

Runnable with plain ``python`` (no Isaac):

    PYTHONPATH=$PWD \
        python skyvla_isaac/snatch/tests/test_randomization.py

Prints TESTS_OK iff every assert passes.
"""

import os
import sys

# Make the repo importable when run as a bare script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from skyvla_isaac.snatch.randomization import (
    add_depth_noise,
    apply_detection_noise,
    apply_obs_latency,
    apply_vio_drift,
    ground_effect_force,
    sample_dr_params,
)

DEVICE = "cpu"
N = 4096


def _approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def test_sample_shapes():
    p = sample_dr_params(N, DEVICE)
    scalars = [
        "thrust_scale", "motor_tau", "payload", "gust_mag", "gust_prob_per_step",
        "gust_duration_steps", "ground_effect_mag", "ground_effect_alt",
        "lighting", "depth_noise_std", "depth_dropout", "latency_s",
        "latency_steps", "vio_drift_scale", "vio_walk_std",
        "detection_noise_scale", "det_noise_std", "det_dropout",
    ]
    vecs = ["wind", "gust_dir", "vio_bias", "det_bias"]
    for k in scalars:
        assert p[k].shape == (N,), f"{k} shape {p[k].shape}"
    for k in vecs:
        assert p[k].shape == (N, 3), f"{k} shape {p[k].shape}"
    # latency_steps within 1..5
    assert int(p["latency_steps"].min()) >= 1
    assert int(p["latency_steps"].max()) <= 5
    # ranges sane
    assert 0.85 <= float(p["thrust_scale"].min()) and float(p["thrust_scale"].max()) <= 1.15
    assert 0.0 <= float(p["payload"].min()) and float(p["payload"].max()) <= 0.300
    # gust_dir unit norm
    norms = p["gust_dir"].norm(dim=-1)
    assert _approx(norms.mean(), 1.0, 1e-4)
    print("  ok test_sample_shapes")


def test_vio_drift_zero_scale_identity():
    p = sample_dr_params(N, DEVICE, vio_drift_scale=0.0)
    pose = torch.randn(N, 7, device=DEVICE)  # pos(3)+quat(4); only pos perturbed
    out = apply_vio_drift(pose, t=10.0, params=p)
    assert torch.allclose(out, pose, atol=1e-7), "zero vio scale must be identity"
    # orientation columns must never change even with drift
    p2 = sample_dr_params(N, DEVICE, vio_drift_scale=5.0)
    out2 = apply_vio_drift(pose, t=10.0, params=p2)
    assert torch.allclose(out2[:, 3:], pose[:, 3:]), "drift must not touch orientation"
    print("  ok test_vio_drift_zero_scale_identity")


def test_vio_drift_grows_with_t_and_scale():
    torch.manual_seed(0)
    pose = torch.zeros(N, 3, device=DEVICE)

    def mean_drift(scale, t):
        p = sample_dr_params(N, DEVICE, vio_drift_scale=scale)
        d = apply_vio_drift(pose, t=t, params=p) - pose
        return float(d.norm(dim=-1).mean())

    # grows with t (more time -> more bias ramp + more random walk)
    d_small_t = mean_drift(1.0, 1.0)
    d_large_t = mean_drift(1.0, 60.0)
    assert d_large_t > d_small_t, f"drift should grow with t: {d_small_t} !< {d_large_t}"

    # grows with vio_drift_scale
    d_small_s = mean_drift(1.0, 30.0)
    d_large_s = mean_drift(4.0, 30.0)
    assert d_large_s > d_small_s, f"drift should grow with scale: {d_small_s} !< {d_large_s}"
    print("  ok test_vio_drift_grows_with_t_and_scale")


def test_detection_noise_zero_identity_and_dropout():
    p0 = sample_dr_params(N, DEVICE, detection_noise_scale=0.0)
    block = torch.randn(N, 3, device=DEVICE)
    out0 = apply_detection_noise(block, p0)
    assert torch.allclose(out0, block, atol=1e-7), "zero detection scale must be identity"

    # With scale on, output should differ from input on most envs.
    p1 = sample_dr_params(N, DEVICE, detection_noise_scale=1.0)
    out1 = apply_detection_noise(block, p1)
    frac_changed = float((out1 != block).any(dim=-1).float().mean())
    assert frac_changed > 0.9, f"expected most envs perturbed, got {frac_changed}"
    print("  ok test_detection_noise_zero_identity_and_dropout")


def test_depth_noise_zero_identity_and_dropout_fraction():
    depth = torch.rand(64, 32, 32, device=DEVICE) + 0.5

    # zero noise + zero dropout -> identity
    p = sample_dr_params(64, DEVICE)
    p["depth_noise_std"] = torch.zeros(64, device=DEVICE)
    p["depth_dropout"] = torch.zeros(64, device=DEVICE)
    out = add_depth_noise(depth, p)
    assert torch.allclose(out, depth, atol=1e-7), "zero depth noise must be identity"

    # dropout fraction ~ expected (use a known dropout, zero gaussian to isolate)
    p["depth_dropout"] = torch.full((64,), 0.2, device=DEVICE)
    p["depth_noise_std"] = torch.zeros(64, device=DEVICE)
    out2 = add_depth_noise(depth, p)
    frac_zero = float((out2 == 0).float().mean())
    assert _approx(frac_zero, 0.2, 0.02), f"dropout frac {frac_zero} != ~0.2"
    print("  ok test_depth_noise_zero_identity_and_dropout_fraction")


def test_obs_latency():
    n, L, D = 100, 6, 3
    # buffer index -1 is newest; make value encode the frame index
    buffer = torch.arange(L, dtype=torch.float32).view(1, L, 1).expand(n, L, D).contiguous()
    p = sample_dr_params(n, DEVICE)
    p["latency_steps"] = torch.full((n,), 2, dtype=torch.long)
    out = apply_obs_latency(buffer, p)
    # newest=L-1=5; delay 2 -> frame 3
    assert torch.all(out == 3.0), f"expected frame 3, got {out[0]}"

    # zero delay -> newest frame
    p["latency_steps"] = torch.zeros(n, dtype=torch.long)
    out0 = apply_obs_latency(buffer, p)
    assert torch.all(out0 == float(L - 1))
    assert out.shape == (n, D)
    print("  ok test_obs_latency")


def test_ground_effect():
    p = sample_dr_params(N, DEVICE)
    # above threshold -> zero
    alt_high = torch.full((N,), 1.0, device=DEVICE)
    f_high = ground_effect_force(alt_high, p)
    assert f_high.shape == (N, 3)
    assert torch.all(f_high == 0.0), "ground effect must be zero above 0.5 m"

    # below threshold -> positive +z, zero x/y
    alt_low = torch.full((N,), 0.1, device=DEVICE)
    f_low = ground_effect_force(alt_low, p)
    assert torch.all(f_low[:, 2] > 0.0), "ground effect must be positive below 0.5 m"
    assert torch.all(f_low[:, :2] == 0.0), "ground effect only acts on +z"

    # exactly at threshold -> zero
    f_thr = ground_effect_force(torch.full((N,), 0.5, device=DEVICE), p)
    assert torch.all(f_thr[:, 2] == 0.0)
    print("  ok test_ground_effect")


def main():
    torch.manual_seed(1234)
    test_sample_shapes()
    test_vio_drift_zero_scale_identity()
    test_vio_drift_grows_with_t_and_scale()
    test_detection_noise_zero_identity_and_dropout()
    test_depth_noise_zero_identity_and_dropout_fraction()
    test_obs_latency()
    test_ground_effect()
    print("TESTS_OK")


if __name__ == "__main__":
    main()
