"""Regenerate HM3D navmeshes for a FLYING agent (run in the 'habitat' env).

The navmeshes HM3D ships are built for a ~1.5 m walking agent: "navigable" means
floor a person could stand on. A drone occupies *air* and is small, so it needs a
navmesh recomputed with drone-scale agent params. We recompute per scene and save
a separate ``<scene>.drone.navmesh`` so the original is untouched.

Drone params (tunable): a small radius (the quadrotor's half-width + margin) and a
short height, with a generous max-climb so vertical transitions between levels are
allowed. The result is a much larger navigable set than the walker mesh — the
flyable free space — which GSCoverageEnv.is_free() then queries.

NOTE on what a navmesh can/can't express: Recast navmeshes are a 2.5D walkable
surface, not a true 3D free-space volume. For a drone this gives "navigable
footprint at the configured clearance," which is the standard, fast approximation
used for aerial agents in Habitat. Full 3D occupancy (fly over furniture at
arbitrary height) would need a voxel map; we note that as a future upgrade and use
the recomputed navmesh as a solid, cheap free-space oracle for now.

Usage:
  python -m indoor_uav.scripts.make_drone_navmesh --split train [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import os


def regen(scene_glb: str, *, radius: float, height: float, max_climb: float,
          cell_size: float) -> tuple[bool, float, float]:
    """Recompute a drone navmesh for one scene. Returns (ok, walk_area, drone_area)."""
    import habitat_sim

    bk = habitat_sim.SimulatorConfiguration()
    bk.scene_id = scene_glb
    bk.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(bk, [habitat_sim.agent.AgentConfiguration()])
    )
    walk_area = float(sim.pathfinder.navigable_area) if sim.pathfinder.is_loaded else 0.0

    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = radius        # drone half-width + safety margin
    ns.agent_height = height        # short — it's a flyer, not a walker
    ns.agent_max_climb = max_climb   # allow vertical transitions between levels
    ns.cell_size = cell_size
    ok = sim.recompute_navmesh(sim.pathfinder, ns)
    drone_area = float(sim.pathfinder.navigable_area) if ok else 0.0
    if ok:
        out = scene_glb.replace(".basis.glb", ".drone.navmesh")
        sim.pathfinder.save_nav_mesh(out)
    sim.close()
    return ok, walk_area, drone_area


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root",
                    default="/home/ubuntu/assets/indoor_scenes/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--split", default="train")
    ap.add_argument("--radius", type=float, default=0.20, help="drone half-width + margin (m)")
    ap.add_argument("--height", type=float, default=0.30, help="agent height (m)")
    ap.add_argument("--max_climb", type=float, default=2.0, help="vertical transition allowance (m)")
    ap.add_argument("--cell_size", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    args = ap.parse_args()

    scenes = sorted(glob.glob(os.path.join(args.scene_root, args.split, "*", "*.basis.glb")))
    if args.limit:
        scenes = scenes[: args.limit]
    print(f"[drone_navmesh] {len(scenes)} scenes in split '{args.split}'", flush=True)

    ok_n = 0
    for i, sc in enumerate(scenes, 1):
        sid = os.path.basename(sc).replace(".basis.glb", "")
        try:
            ok, wa, da = regen(sc, radius=args.radius, height=args.height,
                               max_climb=args.max_climb, cell_size=args.cell_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(scenes)}] {sid} ERROR {exc!r}", flush=True)
            continue
        ok_n += int(ok)
        if i <= 5 or i % 50 == 0:
            mult = (da / wa) if wa > 0 else 0.0
            print(f"[{i}/{len(scenes)}] {sid} ok={ok} walk={wa:.0f} drone={da:.0f} "
                  f"({mult:.1f}x) m^2", flush=True)
    print(f"[drone_navmesh] done: {ok_n}/{len(scenes)} navmeshes regenerated", flush=True)
    return 0 if ok_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
