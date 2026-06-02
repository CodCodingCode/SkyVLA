#!/usr/bin/env python
"""Self-contained PPO for drone PICK-AND-PLACE (continuous control, pure torch).

Trains a GaussianActorCritic on PickPlaceEnv: the drone must reach an object,
close its 2-DoF gripper, carry the object to a target zone, and release it.
Single-env on-policy PPO with GAE — same structure as train_ppo_coverage.py,
adapted to a continuous (6-D) action space and state-based observations.

Run (habitat env, torch installed there):
  conda activate habitat; export CUDA_HOME=/usr; export PYTHONUTF8=1
  python -m indoor_uav.scripts.train_pick_place \
      --scene <glb> --run_dir logs/pickplace/run1 --total_steps 300000

W&B is on by default (project skyvla-indoor-uav), key from .wandb_key; disable
with --wandb_mode disabled for a smoke run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from indoor_uav.tasks.pick_place_env import PickPlaceEnv
from indoor_uav.policy.pick_place_ac import GaussianActorCritic

_TAG = "[train_pick_place]"


def _gae(rewards, values, dones, last_v, gamma, lam):
    T = len(rewards)
    adv = torch.zeros(T); gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        nxt = last_v if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--total_steps", type=int, default=300000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--ep_len", type=int, default=220)
    ap.add_argument("--ppo_epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--sim_res", type=int, default=160)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default="skyvla-indoor-uav")
    ap.add_argument("--wandb_mode", default="online")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PickPlaceEnv(args.scene, sim_res=args.sim_res, max_steps=args.ep_len,
                       load_urdf=False, seed=args.seed)
    policy = GaussianActorCritic(obs_dim=env.observation_space.shape[0],
                                 act_dim=env.action_space.shape[0]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        kf = Path("/home/ubuntu/SkyVLA/.wandb_key")
        if "WANDB_API_KEY" not in os.environ and kf.exists():
            os.environ["WANDB_API_KEY"] = kf.read_text().strip()
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=out.name,
                                   resume="allow", mode=args.wandb_mode, config=vars(args))
            print(f"{_TAG} wandb: {getattr(wandb_run, 'url', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{_TAG} WARN wandb init failed: {exc!r}")

    def to_t(o):
        return torch.from_numpy(o).float().unsqueeze(0).to(device)

    gstep = 0; update = 0
    obs, _ = env.reset()
    ep_ret = 0.0; ep_rets, ep_succ, ep_holds = [], [], []
    ep_grasped = False
    print(f"{_TAG} start scene={args.scene} total_steps={args.total_steps} device={device}")

    while gstep < args.total_steps:
        st_buf, act_buf, lp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        for _ in range(args.rollout):
            st = to_t(obs)
            with torch.no_grad():
                a, lp, v = policy.act(st)
            act = a.squeeze(0).cpu().numpy()
            nobs, r, term, trunc, info = env.step(act)
            done = term or trunc
            st_buf.append(st); act_buf.append(a); lp_buf.append(lp); val_buf.append(v)
            rew_buf.append(r); done_buf.append(float(done))
            ep_ret += r; gstep += 1
            ep_grasped = ep_grasped or info["holding"]
            obs = nobs
            if done:
                ep_rets.append(ep_ret); ep_succ.append(1.0 if info["success"] else 0.0)
                ep_holds.append(1.0 if ep_grasped else 0.0)
                ep_ret = 0.0; ep_grasped = False
                obs, _ = env.reset()
            if gstep >= args.total_steps:
                break

        st_b = torch.cat(st_buf); act_b = torch.cat(act_buf)
        lp_b = torch.cat(lp_buf).detach()
        val_b = torch.cat(val_buf).detach().cpu()
        rew_b = torch.tensor(rew_buf); done_b = torch.tensor(done_buf)
        with torch.no_grad():
            last_v = policy.forward(to_t(obs))[1].cpu()
        adv, ret = _gae(rew_b, val_b, done_b, last_v, args.gamma, args.lam)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device); ret = ret.to(device)

        n = st_b.shape[0]; idx = np.arange(n); last = {}
        for _ in range(args.ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, args.minibatch):
                mb = idx[s:s + args.minibatch]
                new_lp, ent, v = policy.evaluate(st_b[mb], act_b[mb])
                ratio = torch.exp(new_lp - lp_b[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
                pg = -torch.min(s1, s2).mean()
                vf = F.mse_loss(v, ret[mb])
                loss = pg + args.vf_coef * vf - args.ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                last = {"pg": float(pg.item()), "vf": float(vf.item()),
                        "ent": float(ent.mean().item()), "gn": float(gn.item())}
        update += 1

        mret = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
        msucc = float(np.mean(ep_succ[-20:])) if ep_succ else 0.0
        mhold = float(np.mean(ep_holds[-20:])) if ep_holds else 0.0
        print(f"{_TAG} upd {update} gstep {gstep} ep_ret~{mret:.2f} "
              f"grasp_rate~{mhold:.2f} success~{msucc:.2f} "
              f"pg={last.get('pg',0):.3f} vf={last.get('vf',0):.3f}", flush=True)
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({"rl/ep_return": mret, "rl/grasp_rate": mhold,
                           "rl/success_rate": msucc, "rl/pg_loss": last.get("pg", 0),
                           "rl/value_loss": last.get("vf", 0), "rl/entropy": last.get("ent", 0),
                           "rl/grad_norm": last.get("gn", 0)}, step=gstep)
            except Exception:  # noqa: BLE001
                pass

        if update % args.ckpt_every == 0:
            tmp = out / "last.pt.tmp"
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                        "gstep": gstep, "args": vars(args)}, tmp)
            os.replace(tmp, out / "last.pt")

    torch.save({"policy": policy.state_dict(), "gstep": gstep, "args": vars(args)},
               out / "final.pt")
    print(f"{_TAG} done gstep={gstep}")
    if wandb_run is not None:
        try:
            import wandb; wandb.finish()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
