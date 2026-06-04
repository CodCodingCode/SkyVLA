"""Evaluate a trained SNATCH policy + run the VIO-noise sweep (sim2real gap).

Loads a checkpoint, runs deterministic rollouts, and reports:
  - pick %        (episodes that achieved a grasp)
  - place %       (episodes that achieved a delivery)
  - end-to-end %  (episodes that picked AND placed)
  - mean grasp position error (cm)
  - VIO-noise sweep: re-run eval across a range of vio_drift_scale values and
    print success vs drift. This sweep is the core sim2real GAP RESULT.

  conda activate isaac; OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/ubuntu/SkyVLA \
    python skyvla_isaac/scripts/eval_snatch.py \
      --checkpoint logs/isaac/drone_snatch/model_1499.pt

NOTE: imports the orchestrator's env (snatch/pick_place_env.py) and
snatch.randomization.apply_vio_drift; runs end-to-end once those land.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--drift_scales", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0],
                    help="vio_drift_scale values for the sim2real gap sweep")
parser.add_argument("--no_cams", action="store_true",
                    help="evaluate the state-based variant (match a --no_cams trained policy)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_cams   # visuomotor policy needs the depth cams
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
from skyvla_isaac.snatch.randomization import apply_vio_drift  # noqa: E402,F401


def build_runner(env, device):
    """Same agent cfg as training (rsl_rl needs matching net dims to load)."""
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
    runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=device)
    return runner


def run_eval(env, policy, base, steps):
    """Deterministic rollout -> per-episode pick / place / end-to-end + grasp err.

    Relies on the env exposing per-step bool buffers (contract with the
    orchestrator's env):
        base._grasped       (N,) bool — block currently grasped
        base._placed        (N,) bool — block delivered at goal this step
        base._grasp_pos_err (N,) float — m, |gripper - block| at grasp moment
    """
    obs, _ = env.get_observations()
    n = base.num_envs
    dev = base.device
    ever_grasp = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_place = torch.zeros(n, dtype=torch.bool, device=dev)
    pick_ep = place_ep = e2e_ep = ep_cnt = 0
    grasp_err_sum, grasp_err_cnt = 0.0, 0

    for _ in range(steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, dones, _ = env.step(act)

        grasped = getattr(base, "_grasped", torch.zeros(n, dtype=torch.bool, device=dev))
        placed = getattr(base, "_placed", torch.zeros(n, dtype=torch.bool, device=dev))
        ever_grasp |= grasped.bool()
        ever_place |= placed.bool()

        gerr = getattr(base, "_grasp_pos_err", None)
        if gerr is not None:
            new = grasped.bool()
            if new.any():
                grasp_err_sum += float(gerr[new].sum().item())
                grasp_err_cnt += int(new.sum().item())

        dm = dones.bool()
        if dm.any():
            pick_ep += int((ever_grasp & dm).sum().item())
            place_ep += int((ever_place & dm).sum().item())
            e2e_ep += int((ever_grasp & ever_place & dm).sum().item())
            ep_cnt += int(dm.sum().item())
            ever_grasp[dm] = False
            ever_place[dm] = False

    ep_cnt = max(1, ep_cnt)
    mean_grasp_err_cm = (grasp_err_sum / max(1, grasp_err_cnt)) * 100.0
    return {
        "episodes": ep_cnt,
        "pick_pct": pick_ep / ep_cnt,
        "place_pct": place_ep / ep_cnt,
        "e2e_pct": e2e_ep / ep_cnt,
        "grasp_err_cm": mean_grasp_err_cm,
    }


def make_env(num_envs, drift_scale):
    cfg = DroneSnatchEnvCfg()
    cfg.use_cameras = not args.no_cams
    cfg.scene.num_envs = num_envs
    # Headline sim2real knob: scale the VIO drift magnitude applied to the
    # pose terms in the observation. The orchestrator's env reads this and
    # routes it into snatch.randomization.apply_vio_drift.
    if hasattr(cfg, "vio_drift_scale"):
        cfg.vio_drift_scale = drift_scale
    env = DroneSnatchEnv(cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    return env


def main():
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] VIO-noise sweep over vio_drift_scale={args.drift_scales}")
    results = []
    for scale in args.drift_scales:
        env = make_env(args.num_envs, scale)
        base = env.unwrapped
        runner = build_runner(env, base.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=base.device)
        m = run_eval(env, policy, base, args.steps)
        m["vio_drift_scale"] = scale
        results.append(m)
        print(f"[eval] drift={scale:<4}  pick={m['pick_pct']:.1%}  "
              f"place={m['place_pct']:.1%}  e2e={m['e2e_pct']:.1%}  "
              f"grasp_err={m['grasp_err_cm']:.1f}cm  (eps={m['episodes']})")
        env.close()

    # Core sim2real gap table: success vs drift.
    print("\n=== SNATCH VIO-noise sweep (sim2real gap) ===")
    print(f"{'vio_drift_scale':>16} | {'pick%':>7} | {'place%':>7} | "
          f"{'e2e%':>7} | {'grasp_err_cm':>12}")
    print("-" * 64)
    for m in results:
        print(f"{m['vio_drift_scale']:>16} | {m['pick_pct']*100:>6.1f}% | "
              f"{m['place_pct']*100:>6.1f}% | {m['e2e_pct']*100:>6.1f}% | "
              f"{m['grasp_err_cm']:>12.1f}")
    if results:
        base_e2e = results[0]["e2e_pct"]
        worst = results[-1]["e2e_pct"]
        drop = base_e2e - worst
        print(f"\n[eval] end-to-end success drop {base_e2e:.1%} -> {worst:.1%} "
              f"(Delta={drop:.1%}) across drift {args.drift_scales[0]}->{args.drift_scales[-1]}")
    print("EVAL_SNATCH_OK")


main()
sim_app.close()
