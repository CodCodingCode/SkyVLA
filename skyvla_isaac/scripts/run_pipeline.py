"""Run the full three-policy SNATCH pipeline: snatch -> carry -> place.

Loads THREE checkpoints and switches between them per-env on task state:

    phase 0  snatch  fly in, cage and lift the cube off platform A
      | held for --hold_steps consecutive steps
    phase 1  carry   haul it across to platform B
      | _arrived (still held, cube within arrive_radius of B's centre)
    phase 2  place   descend onto B, open the jaws, deposit

Each stage keeps its own observation normalizer (they were trained on different
state distributions), so the three are loaded as three separate rsl_rl runners
rather than three state_dicts into one.

    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
      .venv311/bin/python skyvla_isaac/scripts/run_pipeline.py --no_cams \
        --snatch skyvla_isaac/snatch/checkpoints/model_9250.pt \
        --carry  logs/isaac/drone_snatch_carry/model_XXXX.pt \
        --place  logs/isaac/drone_snatch_place/model_XXXX.pt

CAVEAT -- shared speed. cfg.speed is a single global scalar applied in
_apply_action (v_des = a[:,:3] * cfg.speed), so all three policies fly at the
same scale here. That is correct while every stage is trained at the same
--speed (distance rungs 1-2). If you raise the carry policy's speed for the
20-40 m rungs, the snatch/place policies must be fine-tuned at that speed too,
or cfg.speed must be made per-env before this runner is meaningful.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--snatch", required=True, help="stage-1 grasp checkpoint")
parser.add_argument("--carry", required=True, help="stage-2 transport checkpoint")
parser.add_argument("--place", required=True, help="stage-3 deposit checkpoint")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--plat_sep", type=float, default=1.5)
parser.add_argument("--plat_sep_max", type=float, default=2.0)
parser.add_argument("--speed", type=float, default=None)
parser.add_argument("--side_spawn", type=float, default=5.0)
parser.add_argument("--hold_steps", type=int, default=10,
                    help="consecutive steps of _held before handing off to the carry policy")
parser.add_argument("--no_cams", action="store_true")
parser.add_argument("--trace_out", type=str, default=None,
                    help="record drone/cube/phase per step to this .npz for offline plotting "
                         "(RTX video is unavailable on hosts without Vulkan)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_cams
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
from isaaclab.utils.dict import class_to_dict  # noqa: E402
from skyvla_isaac.snatch.pick_place_env import (  # noqa: E402
    DroneSnatchEnv, DroneSnatchEnvCfg)


def build_runner(env, device):
    """Same agent cfg as training -- rsl_rl needs matching net dims to load."""
    agent_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=64, max_iterations=1, save_interval=50,
        experiment_name="drone_snatch", logger="tensorboard",
        empirical_normalization=True,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0, noise_std_type="log",
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64], activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.01, num_learning_epochs=10, num_mini_batches=4,
            learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0),
    )
    return OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=device)


def main():
    cfg = DroneSnatchEnvCfg()
    cfg.use_cameras = not args.no_cams
    cfg.grasp_latch = False                      # real physics, matches every trained stage
    cfg.staged_curriculum = True
    cfg.start_stage = 2
    cfg.curriculum_p_start = cfg.curriculum_p_end = 0.0    # honest full fly-in
    cfg.carry_demo_p = 0.0                       # no mid-carry spawns: run the WHOLE task
    cfg.two_platform = True
    cfg.plat_sep = args.plat_sep
    cfg.plat_sep_max = max(args.plat_sep_max, args.plat_sep)
    # NEITHER deposit flag. _reset_target must hand the CARRY policy the goal it was
    # trained against (a hover point at surface_z + 0.25); the PLACE policy was trained
    # against the resting-height goal instead, so the runner rewrites _target[:, 2]
    # per-env at the carry->place handoff below. Setting release_only here would give
    # the carry policy the wrong goal for the whole haul.
    cfg.release_only = False
    cfg.place_only = False
    cfg.grip_closed = False                      # per-phase, handled in the loop
    if args.speed is not None:
        cfg.speed = args.speed
    cfg.side_spawn_max = args.side_spawn
    cfg.scene.env_spacing = max(cfg.scene.env_spacing,
                                2.0 * cfg.plat_sep_max + 2.0,
                                2.0 * args.side_spawn + 2.0)
    cfg.episode_length_s = max(cfg.episode_length_s,
                               12.0 + cfg.plat_sep_max / max(cfg.speed, 0.3),
                               10.0 + args.side_spawn / max(cfg.speed, 0.3))
    cfg.scene.num_envs = args.num_envs
    env = RslRlVecEnvWrapper(DroneSnatchEnv(cfg, render_mode=None))
    base = env.unwrapped
    dev = base.device

    def _with_norm(pol, ckpt):
        """Re-apply the checkpoint's observation normalizer BY HAND.

        These checkpoints were written by rsl_rl 5.x, which stored the running obs
        mean/std in a top-level "obs_norm_state_dict". rsl_rl 3.1.2 (installed with
        Isaac Lab 2.3.2) deprecated `empirical_normalization`, moved normalization
        inside the policy, and its load() ignores that key entirely -- so the weights
        load fine while the normalizer is silently dropped and the actor is fed RAW
        observations. Measured effect: model_9250 scores 0% grasp in every config.
        """
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        nz = ck.get("obs_norm_state_dict")
        if not nz or "_mean" not in nz:
            print(f"[pipeline]   no obs_norm in {ckpt} -- using policy as-is", flush=True)
            return pol
        mean = nz["_mean"].to(dev).reshape(1, -1)
        std = nz["_std"].to(dev).reshape(1, -1).clamp(min=1e-6)
        print(f"[pipeline]   obs_norm restored (dim {mean.shape[-1]})", flush=True)

        def _p(o):
            td = hasattr(o, "keys") and "policy" in o.keys()
            if td:
                o = o.clone()
                o["policy"] = (o["policy"] - mean) / std
                return pol(o)
            return pol((o - mean) / std)
        return _p

    policies = []
    for name, ckpt in (("snatch", args.snatch), ("carry", args.carry), ("place", args.place)):
        runner = build_runner(env, dev)
        runner.load(ckpt)
        policies.append(_with_norm(runner.get_inference_policy(device=dev), ckpt))
        print(f"[pipeline] {name:6s} <- {ckpt}", flush=True)

    n = args.num_envs
    phase = torch.zeros(n, dtype=torch.long, device=dev)
    hold_run = torch.zeros(n, dtype=torch.long, device=dev)
    episodes = torch.zeros((), dtype=torch.long, device=dev)
    ep_grasped = torch.zeros(n, dtype=torch.bool, device=dev)
    ep_arrived = torch.zeros(n, dtype=torch.bool, device=dev)
    ep_deposited = torch.zeros(n, dtype=torch.bool, device=dev)
    tot = {"grasp": 0, "arrive": 0, "deposit": 0, "eps": 0}

    # Isaac Lab 2.3.2 / rsl_rl 3.1.2 return a bare obs tensor here; older versions
    # returned (obs, extras). Accept either so this runs across both.
    def _obs(r):
        return r[0] if isinstance(r, tuple) else r

    def _step(a):
        r = env.step(a)
        # (obs, rew, dones, extras) or (obs, rew, terminated, truncated, extras)
        if len(r) == 5:
            return r[0], (r[2].bool() | r[3].bool())
        return r[0], r[2].bool()

    def _retarget_obs(o, ph):
        """Each policy was trained with a DIFFERENT MEANING for the goal channel
        (obs[:, -3:] = target - drone_pos), so handing over requires rewriting it:

          snatch  goal sampled within +-0.25m of the CUBE at 0.55m (goal_offset_diam)
          carry   goal = platform B's hover point (surface_z + 0.25)
          place   goal = the cube at rest ON B (surface_z + cube/2)

        Feeding the snatch policy B's goal 2m away puts those 3 dims far outside its
        training distribution -- measured: grasp collapses to 0.1%. Phases 1/2 already
        read the env target, which _reset_target and the handoff below keep correct."""
        op = getattr(base, "_obj_p", None)
        bp = getattr(base, "_base_p", None)
        if op is None or bp is None:
            return o
        g = torch.empty_like(bp)
        g[:, :2] = op[:, :2]                      # hover point over the CUBE
        g[:, 2] = cfg.surface_z + 0.25
        o = o.clone()
        # Isaac Lab 2.3.2 hands back a TensorDict keyed "policy"; older versions a tensor.
        td = hasattr(o, "keys") and "policy" in o.keys()
        t = o["policy"] if td else o
        t = t.clone()
        t[:, -3:] = torch.where((ph == 0).unsqueeze(-1), g - bp, t[:, -3:])
        if td:
            o["policy"] = t
            return o
        return t

    obs = _obs(env.get_observations())
    # Offline trace: this host has no Vulkan, so an mp4 of a successful delivery cannot be
    # rendered. Record the geometry instead and plot it after the fact.
    tr = None
    if args.trace_out:
        import numpy as np
        tr = {k: np.zeros((args.steps, n, 3), dtype=np.float32) for k in ("drone", "cube")}
        tr["phase"] = np.zeros((args.steps, n), dtype=np.int8)
        tr["dep"] = np.zeros((args.steps, n), dtype=bool)
        tr["done"] = np.zeros((args.steps, n), dtype=bool)
        tr["platB"] = np.zeros((args.steps, n, 2), dtype=np.float32)
    for _t in range(args.steps):
        with torch.inference_mode():
            obs_p = _retarget_obs(obs, phase)
            acts = torch.stack([p(obs_p) for p in policies], dim=0)   # (3,N,5)
        act = acts.gather(0, phase.view(1, n, 1).expand(1, n, acts.shape[-1]))[0]
        # The carry policy NEVER controlled its own grip: it was trained with --grip_closed,
        # so a[4] was overridden to +1 every step and its 5th output head is untrained noise.
        # Let that noise through here and the cage can pop open mid-haul and drop the cube.
        # Weld it shut for exactly the phase that was trained that way.
        act = torch.where((phase == 1).unsqueeze(-1),
                          torch.cat([act[:, :4], torch.ones_like(act[:, 4:5])], dim=-1), act)
        obs, dones = _step(act)

        held = base._held
        hold_run = torch.where(held, hold_run + 1, torch.zeros_like(hold_run))
        # 0 -> 1 on a settled grasp; 1 -> 2 on arrival over B. Never step backwards
        # within an episode: a momentary slip should not restart the fly-in.
        to_carry = (phase == 0) & (hold_run >= args.hold_steps)
        to_place = (phase == 1) & base._arrived
        phase = torch.where(to_carry, torch.ones_like(phase), phase)
        phase = torch.where(to_place, torch.full_like(phase, 2), phase)
        if bool(to_place.any()):
            # Hand the place policy the goal IT was trained on: the cube at rest ON B,
            # not the carry policy's hover point 25cm above it.
            base._target[to_place, 2] = cfg.surface_z + 0.5 * cfg.cube_size

        if tr is not None:
            tr["drone"][_t] = base._base_p.detach().cpu().numpy()
            tr["cube"][_t] = base._obj_p.detach().cpu().numpy()
            tr["phase"][_t] = phase.detach().cpu().numpy()
            tr["dep"][_t] = base._deposited.detach().cpu().numpy()
            tr["done"][_t] = dones.detach().cpu().numpy()
            tr["platB"][_t] = base._plat_b_xy.detach().cpu().numpy()
        ep_grasped |= held
        ep_arrived |= base._arrived
        ep_deposited |= base._deposited

        if bool(dones.any()):
            d = dones
            tot["eps"] += int(d.sum().item())
            tot["grasp"] += int((ep_grasped & d).sum().item())
            tot["arrive"] += int((ep_arrived & d).sum().item())
            tot["deposit"] += int((ep_deposited & d).sum().item())
            phase[d] = 0
            hold_run[d] = 0
            ep_grasped[d] = False
            ep_arrived[d] = False
            ep_deposited[d] = False

    e = max(tot["eps"], 1)
    print("\n=== SNATCH 3-policy pipeline (A -> B transport) ===")
    print(f"  separation      : {base._sep:.2f} m")
    print(f"  episodes        : {tot['eps']}")
    print(f"  grasped         : {tot['grasp']/e:.1%}")
    print(f"  arrived over B  : {tot['arrive']/e:.1%}")
    print(f"  DEPOSITED on B  : {tot['deposit']/e:.1%}   <- end-to-end")
    if tr is not None:
        import numpy as np
        np.savez_compressed(args.trace_out, surface_z=cfg.surface_z,
                            cube_size=cfg.cube_size, sep=base._sep, **tr)
        print(f"[pipeline] trace -> {args.trace_out}  (deposits recorded: {int(tr['dep'].any(0).sum())} envs)")
    print("PIPELINE_OK")
    env.close()


main()
