#!/usr/bin/env python
"""Self-contained PPO for the GS coverage task (no sb3 — pure torch).

Trains an ActorCritic to explore an indoor scene so as to maximise Gaussian-
Splatting reconstruction coverage. Single-env, on-policy PPO with GAE — small
and fully under our control, matching the bespoke GS reward. Resilient
checkpointing + W&B (default-on, project skyvla-indoor-uav) per repo convention.

Scene backend:
  --backend synthetic           -> SyntheticRoom (deps-free, default for bring-up)
  --backend habitat --scene X   -> HabitatRoom (run from the habitat env, or feed
                                    a scene the openfly env can load)

Run (synthetic, openfly env):
  python -m indoor_uav.scripts.train_ppo_coverage --total_steps 20000 --run_dir <dir>
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from indoor_uav.tasks import GSCoverageEnv
from indoor_uav.policy import ActorCritic

_TAG = "[train_ppo_coverage]"


def _make_env(args, device):
    if args.backend == "habitat":
        from indoor_uav.sim.habitat_room import HabitatRoom
        sim = HabitatRoom(args.scene, width=args.sim_res, height=args.sim_res, device=device)
    else:
        from indoor_uav.sim import SyntheticRoom
        sim = SyntheticRoom(width=args.sim_res, height=args.sim_res, device=device, seed=args.seed)
    return GSCoverageEnv(sim, max_steps=args.ep_len, obs_res=args.obs_res,
                         device=device, seed=args.seed)


def _to_t(obs, device):
    rgb = torch.from_numpy(obs["rgb"]).unsqueeze(0).to(device)
    state = torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
    return rgb, state


def _gae(rewards, values, dones, last_v, gamma, lam):
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        nxt = last_v if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["synthetic", "habitat"], default="synthetic")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--total_steps", type=int, default=20000)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--ep_len", type=int, default=64)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--sim_res", type=int, default=160)
    ap.add_argument("--obs_res", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default="skyvla-indoor-uav")
    ap.add_argument("--wandb_mode", default="online")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _make_env(args, device)
    policy = ActorCritic(obs_res=args.obs_res).to(device)
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

    gstep = 0
    update = 0
    obs, _ = env.reset()
    ep_ret, ep_cov, ep_rets, ep_covs = 0.0, 0.0, [], []
    print(f"{_TAG} start backend={args.backend} total_steps={args.total_steps} "
          f"rollout={args.rollout} device={device}")

    while gstep < args.total_steps:
        # ---- collect a rollout ----
        rgb_buf, st_buf, act_buf, lp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
        for _ in range(args.rollout):
            rgb, state = _to_t(obs, device)
            with torch.no_grad():
                a, lp, v = policy.act(rgb, state)
            nobs, r, term, trunc, info = env.step(int(a.item()))
            done = term or trunc
            rgb_buf.append(rgb); st_buf.append(state)
            act_buf.append(a); lp_buf.append(lp); val_buf.append(v)
            rew_buf.append(r); done_buf.append(float(done))
            ep_ret += r; ep_cov = info["coverage"]; gstep += 1
            obs = nobs
            if done:
                ep_rets.append(ep_ret); ep_covs.append(ep_cov)
                ep_ret = 0.0; obs, _ = env.reset()
            if gstep >= args.total_steps:
                break

        # ---- advantages ----
        rgb_b = torch.cat(rgb_buf); st_b = torch.cat(st_buf)
        act_b = torch.cat(act_buf); lp_b = torch.cat(lp_buf).detach()
        val_b = torch.cat(val_buf).detach().cpu()
        rew_b = torch.tensor(rew_buf); done_b = torch.tensor(done_buf)
        with torch.no_grad():
            rgb, state = _to_t(obs, device)
            last_v = policy.forward(rgb, state)[1].cpu()
        adv, ret = _gae(rew_b, val_b, done_b, last_v, args.gamma, args.lam)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device)
        ret = ret.to(device)

        # ---- PPO update ----
        n = rgb_b.shape[0]
        idx = np.arange(n)
        last = {}
        for _ in range(args.ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, args.minibatch):
                mb = idx[s:s + args.minibatch]
                new_lp, ent, v = policy.evaluate(rgb_b[mb], st_b[mb], act_b[mb])
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
                        "ent": float(ent.mean().item()), "gn": float(gn.item()),
                        "clip_frac": float(((ratio - 1).abs() > args.clip).float().mean().item())}
        update += 1

        mret = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
        mcov = float(np.mean(ep_covs[-20:])) if ep_covs else 0.0
        print(f"{_TAG} upd {update} gstep {gstep} ep_ret~{mret:.3f} ep_cov~{mcov:.3f} "
              f"pg={last.get('pg',0):.3f} ent={last.get('ent',0):.3f}", flush=True)
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({"rl/ep_return": mret, "rl/ep_coverage": mcov,
                           "rl/pg_loss": last.get("pg", 0), "rl/value_loss": last.get("vf", 0),
                           "rl/entropy": last.get("ent", 0), "rl/grad_norm": last.get("gn", 0),
                           "rl/clip_frac": last.get("clip_frac", 0)}, step=gstep)
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
