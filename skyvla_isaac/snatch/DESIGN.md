# SNATCH — shared design contract (sim-only)

Sim-trained Neural Aerial Transport and Capture. Single-DOF caging gripper, direct
velocity control in Isaac Lab, dual-camera visuomotor policy, sim-to-real **gap
analysis** (no real flights — VIO drift + perception noise are MODELED in sim).

This file is the integration contract. Each module is built to these signatures so the
orchestrator can wire them into `snatch/pick_place_env.py`. Build to the contract; do not
change shared signatures without flagging it.

## Drone + action
- Free-flying quadrotor, **single-DOF caging gripper** (jaw close only — NO lower/raise DOF;
  the real BOM has one servo). Drone descends *bodily* to grasp.
- **Action (5):** `[vx, vy, vz, yaw_rate, gripper]`, vx/vy/vz clip ±3 m/s, yaw_rate ±1 rad/s,
  gripper continuous 0..1 (>0.5 = close). An in-sim velocity-tracking controller converts
  [vx,vy,vz,yaw_rate] to body wrench (no PX4 in the loop).
- physics_dt = 0.005 (200 Hz), control decimation -> 50 Hz policy.

## Observation (dict; env assembles, modules provide pieces)
- `top_depth`  : (N, H_t, W_t) metric depth, top RealSense (depth 848x480, fwd, FoV 87)
- `bottom_depth`: (N, H_b, W_b) metric depth, bottom Pi-cam (640x480, down, FoV 120)
- `state` (N, 11): pos(3), vel(3), quat... -> actually [pos(3), lin_vel(3), gripper_state(1),
  gripper_torque(1), block_pos_est(3)] where block_pos_est is the bottom-cam DETECTION
  (noised), and pose terms have VIO drift applied. **No privileged ground truth in obs.**
- Policy = perception encoders -> latents, concat with `state` -> MLP -> action.

## Module contracts

### snatch/perception.py  (Agent A2)
- `top_camera_cfg()`, `bottom_camera_cfg()` -> isaaclab CameraCfg (depth data_type, the
  resolutions/FoV/mounts above).
- `class DepthEncoder(nn.Module)`: ResNet-18 (torchvision, depth->3ch repeat), input depth
  clipped [0.1,5.0]m normalized [0,1]; `forward(depth)->(N,512)`.
- `build_obs_latents(top_depth, bottom_depth, enc_top, enc_bottom)->(N,1024)`.
- A smoke script `scripts/snatch_cam_smoke.py` that spawns cams in a trivial scene and prints
  depth tensor shapes. Encoders are separate instances (no shared weights).

### snatch/randomization.py  (Agent A3)
- Pure-torch where possible (unit-testable without Isaac). Provide functions returning
  per-env perturbations and noise, each `(...)->Tensor`:
  - `sample_dr_params(num_envs, device)->dict` (motor thrust ±15%, motor tau 20-50ms,
    payload 0-300g, wind N(0,0.5)/axis, gust spikes, ground-effect force <0.5m alt,
    lighting, camera depth noise σ=0.01 + 2% dropout, control latency 50-150ms / 1-5 step
    buffer).
  - `apply_vio_drift(pose, t, params)->pose` — slowly-accumulating bias + walk (the sim2real
    localization gap; this is the HEADLINE gap-analysis knob, magnitude scalable).
  - `apply_detection_noise(block_pos, params)->block_pos` — bias+noise+dropout on bottom-cam
    block estimate.
  - `add_depth_noise(depth, params)->depth`, `apply_obs_latency(buffer, params)->obs`.
- `snatch/tests/test_randomization.py` — pure-torch asserts (shapes, zero-noise identity,
  drift grows with t). Runnable with plain `python` (no Isaac).

### snatch/rewards.py  (Agent A4)
- `compute_reward(s)->(N,)` where `s` is a dict with: d_block, pixel_offset, alt_above_block,
  grasp_success(bool), carrying(bool), d_goal, place_success(bool), crashed(bool). Implement
  the 4-component shaping from the SNATCH spec (nav -0.1·d, align, alt error to 0.2m, grasp
  +10, transport -0.1·d_goal while carrying, place +15, crash -50, time -0.005). Keep it a
  pure function of `s` (testable).
- Heuristic grasp trigger helper `grasp_trigger(pixel_offset, alt)->bool`
  (offset<20px AND alt<0.25m) AND a CAGING-contact path note (we CAN learn contact with the
  cage; expose both, default heuristic per spec).
- `scripts/train_snatch.py` + `scripts/eval_snatch.py` using **rsl_rl** (already working in
  this repo — NOT skrl) against env id the orchestrator will register. eval reports pick %,
  place %, end-to-end %, mean grasp pos error, AND a **VIO-noise sweep** (success vs drift
  magnitude) — the core sim2real gap result.

### assets (Agent A1)
- `assets/drone_snatch.urdf` — quadrotor, Z-up, **single-DOF caging gripper** (4-jaw cage,
  jaw close only, NO lower joint), rigid mount, + two camera mount frames (top fwd, bottom
  down). Collision on base + 4 fingers + cube only.
- `scripts/convert_snatch.py` (URDF->USD, prints CONVERT_OK) and
  `scripts/snatch_grasp_test.py` (scripted close-then-lift gate: cube_z must track base_z up,
  d_reach ~0.01). Reuse the proven cage geometry from drone_with_gripper.urdf but DROP the
  `lower` joint.

## Env (orchestrator owns) — snatch/pick_place_env.py
Wires the four modules: cameras+encoders (A2), DR+noise (A3), reward (A4), gripper asset
(A1). Direct velocity control, 5-action, dict obs above.
