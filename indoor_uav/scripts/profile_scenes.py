#!/usr/bin/env python
"""Profile HM3D scenes on geometric + semantic axes, then build a BALANCED manifest.

Run in the 'habitat' env (needs habitat_sim to read navmesh/bounds):
  python -m indoor_uav.scripts.profile_scenes --split train --out scenes_profile.json

For each scene we record:
  * floor_area_m2   — navigable surface area (the primary difficulty/size axis)
  * volume_m3       — bbox volume (multi-room / large-home proxy)
  * extent          — bbox dims (x,y,z)
  * n_objects/n_cats/n_furniture — from .semantic.txt if annotated (else 0/None)
  * annotated       — whether HM3D-Sem labels exist for this scene

Then `--balance` stratifies scenes into floor-area bins (small/medium/large/xl)
and writes a manifest that samples evenly across bins, so training episodes draw
uniformly across home SIZES/TYPES instead of over-fitting whatever dominates the
raw distribution. This mirrors the old --per_env_max_index_samples balancing,
but over scene-geometry strata instead of source envs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Structural categories excluded from the "furniture" count (not objects you
# navigate *to* — they're the room shell).
_STRUCT = {
    "wall", "ceiling", "floor", "carpet", "window", "window frame", "door",
    "door frame", "ceiling lower", "unknown", "misc", "", "stairs", "railing",
    "beam", "column", "baseboard", "trim",
}

# Floor-area strata (m^2). Tunable; chosen to span studio -> large multi-room home.
_BINS = [("small", 0, 60), ("medium", 60, 120), ("large", 120, 200), ("xl", 200, 1e9)]


def _semantic_counts(txt_path: str):
    if not os.path.exists(txt_path):
        return 0, 0, 0, False
    cats: dict[str, int] = {}
    for ln in open(txt_path).read().splitlines()[1:]:
        p = ln.split(",")
        if len(p) >= 3:
            c = p[2].strip().strip('"')
            cats[c] = cats.get(c, 0) + 1
    n_obj = sum(cats.values())
    n_furn = sum(v for k, v in cats.items() if k.lower() not in _STRUCT)
    return n_obj, len(cats), n_furn, n_obj > 0


def profile_split(scene_root: str, split: str) -> list[dict]:
    import habitat_sim

    pat = os.path.join(scene_root, split, "*", "*.basis.glb")
    scenes = sorted(glob.glob(pat))
    out = []
    for sc in scenes:
        sid = os.path.basename(sc).replace(".basis.glb", "")
        n_obj, n_cat, n_furn, annotated = _semantic_counts(
            sc.replace(".basis.glb", ".semantic.txt")
        )
        bk = habitat_sim.SimulatorConfiguration()
        bk.scene_id = sc
        bk.enable_physics = False
        sim = habitat_sim.Simulator(
            habitat_sim.Configuration(bk, [habitat_sim.agent.AgentConfiguration()])
        )
        pf = sim.pathfinder
        lo, hi = pf.get_bounds()
        ext = [float(hi[i] - lo[i]) for i in range(3)]
        out.append({
            "scene_id": sid,
            "path": sc,
            "floor_area_m2": round(float(pf.navigable_area), 1),
            "volume_m3": round(ext[0] * ext[1] * ext[2], 1),
            "extent": [round(e, 1) for e in ext],
            "n_objects": n_obj,
            "n_categories": n_cat,
            "n_furniture": n_furn,
            "annotated": annotated,
        })
        sim.close()
    return out


def _bin_of(area: float) -> str:
    for name, lo, hi in _BINS:
        if lo <= area < hi:
            return name
    return _BINS[-1][0]


def build_balanced(profiles: list[dict], per_bin: int | None, seed: int) -> dict:
    import random

    rng = random.Random(seed)
    bins: dict[str, list] = {b[0]: [] for b in _BINS}
    for p in profiles:
        bins[_bin_of(p["floor_area_m2"])].append(p["scene_id"])
    counts = {k: len(v) for k, v in bins.items()}
    # default: match the smallest non-empty bin so every size is equally represented
    nonempty = [c for c in counts.values() if c > 0]
    target = per_bin or (min(nonempty) if nonempty else 0)
    chosen: list[str] = []
    for name, ids in bins.items():
        rng.shuffle(ids)
        chosen.extend(ids[:target])
    rng.shuffle(chosen)
    return {
        "strata": [b[0] for b in _BINS],
        "bin_edges_m2": [[b[1], b[2]] for b in _BINS],
        "raw_counts": counts,
        "per_bin_target": target,
        "n_selected": len(chosen),
        "seed": seed,
        "scene_ids": chosen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root",
                    default="/home/ubuntu/assets/indoor_scenes/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="scenes_profile.json")
    ap.add_argument("--balance", action="store_true",
                    help="also write a size-stratified balanced manifest")
    ap.add_argument("--per_bin", type=int, default=0,
                    help="scenes per area-bin (0 = match smallest non-empty bin)")
    ap.add_argument("--manifest", default="scenes_balanced.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    profiles = profile_split(args.scene_root, args.split)
    json.dump({"split": args.split, "n_scenes": len(profiles), "scenes": profiles},
              open(args.out, "w"), indent=2)
    print(f"[profile] {len(profiles)} scenes -> {args.out}")
    if profiles:
        areas = sorted(p["floor_area_m2"] for p in profiles)
        n_annot = sum(p["annotated"] for p in profiles)
        print(f"  floor_area_m2: min={areas[0]} median={areas[len(areas)//2]} max={areas[-1]}")
        print(f"  annotated (HM3D-Sem): {n_annot}/{len(profiles)}")

    if args.balance:
        man = build_balanced(profiles, args.per_bin or None, args.seed)
        json.dump(man, open(args.manifest, "w"), indent=2)
        print(f"[balance] raw per-bin {man['raw_counts']} "
              f"-> {man['per_bin_target']}/bin -> {man['n_selected']} scenes "
              f"-> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
