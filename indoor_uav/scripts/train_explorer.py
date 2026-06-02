"""PPO training of the physics drone to MAXIMIZE coverage (run in habitat env).

Self-contained PPO (no sb3) over PhysicsCoverageEnv: physics drone (velocity
control, real collisions, zero-gravity hover) + RGB + coverage/frontier map memory.
Reward is coverage gain + frontier progress (geometric for now; the GS-quality
reward swaps in at the env level). Resilient ckpt + W&B (project skyvla-indoor-uav).

Curriculum-ready via --manifest (size-balanced scene list) — cycles scenes.

Run (habitat env, has torch+gymnasium):
  python -m indoor_uav.scripts.train_explorer --scene <glb> --run_dir <dir> --total_steps 50000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from indoor_uav.tasks.physics_coverage_env import PhysicsCoverageEnv
from indoor_uav.policy.explorer_net import ExplorerNet

_TAG = "[train_explorer]"


def _scene_list(args):
    if args.manifest:
        man = json.load(open(args.manifest))
        ids = man.get("scene_ids", [])
        root = "/home/ubuntu/assets/indoor_scenes/versioned_data/hm3d-0.2/hm3d"
        import glob
        out = []
        for sid in ids:
            hits = glob.glob(f"{root}/*/*{sid}*/*.basis.glb")
            if hits:
                out.append(hits[0])
        if out:
            return out
    return [args.scene]


def _to_t(obs, device):
    return (torch.from_numpy(obs["rgb"]).unsqueeze(0).to(device),
            torch.from_numpy(obs["map"]).unsqueeze(0).to(device),
            torch.from_numpy(obs["state"]).unsqueeze(0).to(device))


def _gae(rew, val, done, last_v, gamma, lam):
    T = len(rew); adv = torch.zeros(T); gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nxt = last_v if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--total_steps", type=int, default=50000)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--ep_len", type=int, default=256)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--reward_mode", choices=["geometric", "gs"], default="geometric")
    ap.add_argument("--sim_res", type=int, default=128)
    ap.add_argument("--obs_res", type=int, default=64)
    ap.add_argument("--map_res", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default="skyvla-indoor-uav")
    ap.add_argument("--wandb_mode", default="online")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = _scene_list(args)
    print(f"{_TAG} {len(scenes)} scene(s); device={device}")

    def make_env(i):
        return PhysicsCoverageEnv(scenes[i % len(scenes)], sim_res=args.sim_res,
                                  obs_res=args.obs_res, map_res=args.map_res,
                                  max_steps=args.ep_len, reward_mode=args.reward_mode,
                                  seed=args.seed + i)

    env_i = 0
    env = make_env(env_i)
    policy = ExplorerNet().to(device)
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
            print(f"{_TAG} wandb: {getattr(wandb_run,'url','?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{_TAG} WARN wandb init: {exc!r}")

    gstep = 0; update = 0
    obs, _ = env.reset()
    ep_ret = 0.0; ep_cov = 0.0; rets = []; covs = []

    while gstep < args.total_steps:
        rb = {"rgb": [], "map": [], "st": [], "a": [], "lp": [], "v": [], "r": [], "d": []}
        for _ in range(args.rollout):
            rgb, mp, st = _to_t(obs, device)
            with torch.no_grad():
                a, lp, v = policy.act(rgb, mp, st)
            nobs, r, term, trunc, info = env.step(int(a.item()))
            done = term or trunc
            rb["rgb"].append(rgb); rb["map"].append(mp); rb["st"].append(st)
            rb["a"].append(a); rb["lp"].append(lp); rb["v"].append(v)
            rb["r"].append(r); rb["d"].append(float(done))
            ep_ret += r; ep_cov = info["coverage"]; gstep += 1; obs = nobs
            if done:
                rets.append(ep_ret); covs.append(ep_cov); ep_ret = 0.0
                env_i += 1
                if len(scenes) > 1:  # curriculum: rotate scenes
                    env.close(); env = make_env(env_i)
                obs, _ = env.reset()
            if gstep >= args.total_steps:
                break

        rgb_b = torch.cat(rb["rgb"]); map_b = torch.cat(rb["map"]); st_b = torch.cat(rb["st"])
        a_b = torch.cat(rb["a"]); lp_b = torch.cat(rb["lp"]).detach()
        v_b = torch.cat(rb["v"]).detach().cpu()
        r_b = torch.tensor(rb["r"]); d_b = torch.tensor(rb["d"])
        with torch.no_grad():
            rgb, mp, st = _to_t(obs, device)
            last_v = policy.forward(rgb, mp, st)[1].cpu()
        adv, ret = _gae(r_b, v_b, d_b, last_v, args.gamma, args.lam)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device); ret = ret.to(device)

        n = rgb_b.shape[0]; idx = np.arange(n); last = {}
        for _ in range(args.ppo_epochs):
            np.random.shuffle(idx)
            for s0 in range(0, n, args.minibatch):
                mb = idx[s0:s0 + args.minibatch]
                nlp, ent, v = policy.evaluate(rgb_b[mb], map_b[mb], st_b[mb], a_b[mb])
                ratio = torch.exp(nlp - lp_b[mb])
                s1 = ratio * adv[mb]; s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
                pg = -torch.min(s1, s2).mean(); vf = F.mse_loss(v, ret[mb])
                loss = pg + args.vf_coef * vf - args.ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                last = {"pg": float(pg.item()), "vf": float(vf.item()),
                        "ent": float(ent.mean().item()), "gn": float(gn.item())}
        update += 1
        mret = float(np.mean(rets[-20:])) if rets else 0.0
        mcov = float(np.mean(covs[-20:])) if covs else 0.0
        print(f"{_TAG} upd {update} gstep {gstep} ep_ret~{mret:.2f} ep_cov~{mcov:.3f} "
              f"pg={last.get('pg',0):.3f} ent={last.get('ent',0):.2f}", flush=True)
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({"rl/ep_return": mret, "rl/ep_coverage": mcov,
                           "rl/pg_loss": last.get("pg",0), "rl/value_loss": last.get("vf",0),
                           "rl/entropy": last.get("ent",0), "rl/grad_norm": last.get("gn",0)}, step=gstep)
            except Exception:  # noqa: BLE001
                pass
        if update % args.ckpt_every == 0:
            tmp = out / "last.pt.tmp"
            torch.save({"policy": policy.state_dict(), "gstep": gstep, "args": vars(args)}, tmp)
            os.replace(tmp, out / "last.pt")

    torch.save({"policy": policy.state_dict(), "gstep": gstep}, out / "final.pt")
    print(f"{_TAG} done gstep={gstep}")
    if wandb_run is not None:
        try:
            import wandb; wandb.finish()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
