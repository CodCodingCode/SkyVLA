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
    cfg.release_only = True                      # deposit is the pipeline's success metric
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

    policies = []
    for name, ckpt in (("snatch", args.snatch), ("carry", args.carry), ("place", args.place)):
        runner = build_runner(env, dev)
        runner.load(ckpt)
        policies.append(runner.get_inference_policy(device=dev))
        print(f"[pipeline] {name:6s} <- {ckpt}", flush=True)

    n = args.num_envs
    phase = torch.zeros(n, dtype=torch.long, device=dev)
    hold_run = torch.zeros(n, dtype=torch.long, device=dev)
    episodes = torch.zeros((), dtype=torch.long, device=dev)
    ep_grasped = torch.zeros(n, dtype=torch.bool, device=dev)
    ep_arrived = torch.zeros(n, dtype=torch.bool, device=dev)
    ep_deposited = torch.zeros(n, dtype=torch.bool, device=dev)
    tot = {"grasp": 0, "arrive": 0, "deposit": 0, "eps": 0}

    obs, _ = env.get_observations()
    for _ in range(args.steps):
        with torch.inference_mode():
            acts = torch.stack([p(obs) for p in policies], dim=0)     # (3,N,5)
        act = acts.gather(0, phase.view(1, n, 1).expand(1, n, acts.shape[-1]))[0]
        obs, _, dones, _ = env.step(act)

        held = base._held
        hold_run = torch.where(held, hold_run + 1, torch.zeros_like(hold_run))
        # 0 -> 1 on a settled grasp; 1 -> 2 on arrival over B. Never step backwards
        # within an episode: a momentary slip should not restart the fly-in.
        to_carry = (phase == 0) & (hold_run >= args.hold_steps)
        to_place = (phase == 1) & base._arrived
        phase = torch.where(to_carry, torch.ones_like(phase), phase)
        phase = torch.where(to_place, torch.full_like(phase, 2), phase)

        ep_grasped |= held
        ep_arrived |= base._arrived
        ep_deposited |= base._deposited

        if bool(dones.any()):
            d = dones.bool()
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
    print("PIPELINE_OK")
    env.close()


main()
