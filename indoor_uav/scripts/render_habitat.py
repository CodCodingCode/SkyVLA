#!/usr/bin/env python
"""Dump RGB-D frames from a Habitat scene to .npz (run in the 'habitat' env).

Bridges the isolated sim env and the cu130 training env: render a short patrol
of navigable poses, save (rgb, depth, pose_c2w, K) per frame; the GS pipeline
in the 'openfly' env loads them via load_frames() and builds the Gaussian map.

Usage (habitat env):
  python -m indoor_uav.scripts.render_habitat --scene <scene.glb> --out frames.npz --n 12
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import habitat_sim
    import quaternion

    bk = habitat_sim.SimulatorConfiguration()
    bk.scene_id = args.scene; bk.enable_physics = False
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid; s.sensor_type = stype
        s.resolution = [args.res, args.res]; s.hfov = args.hfov
        specs.append(s)
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ag]))
    sim.seed(args.seed)

    f = 0.5 * args.res / np.tan(np.deg2rad(args.hfov) / 2)
    K = np.array([[f, 0, args.res / 2], [0, f, args.res / 2], [0, 0, 1]], dtype=np.float32)
    hab2cv = np.diag([1, -1, -1, 1]).astype(np.float32)

    rgbs, depths, poses = [], [], []
    for _ in range(args.n):
        st = habitat_sim.AgentState()
        st.position = sim.pathfinder.get_random_navigable_point()
        st.rotation = quaternion.from_euler_angles(0, np.random.uniform(0, 2 * np.pi), 0)
        sim.get_agent(0).set_state(st)
        obs = sim.get_sensor_observations()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quaternion.as_rotation_matrix(st.rotation)
        T[:3, 3] = st.position
        poses.append((T @ hab2cv).astype(np.float32))
        rgbs.append(np.ascontiguousarray(obs["rgb"][..., :3]).astype(np.uint8))
        depths.append(np.ascontiguousarray(obs["depth"]).astype(np.float32))
    sim.close()

    np.savez_compressed(
        args.out, rgb=np.stack(rgbs), depth=np.stack(depths),
        pose=np.stack(poses), K=K,
    )
    print(f"[render_habitat] wrote {args.n} frames -> {args.out}")
    return 0


def load_frames(npz_path: str, device):
    """Load dumped frames in the training env -> list of (rgb,depth,pose,K) tensors."""
    import numpy as np
    import torch

    z = np.load(npz_path)
    K = torch.from_numpy(z["K"]).to(device).float()
    out = []
    for i in range(z["rgb"].shape[0]):
        out.append((
            torch.from_numpy(z["rgb"][i]).to(device).float() / 255.0,
            torch.from_numpy(z["depth"][i]).to(device).float(),
            torch.from_numpy(z["pose"][i]).to(device).float(),
            K,
        ))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
