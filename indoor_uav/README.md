# indoor_uav — Gaussian-Splatting-aware indoor UAV navigation

Pivot from the OpenFly outdoor VLN / diffusion-subgoal stack (preserved on git
branch `openfly-subgoal-dit-archive`) to **indoor UAV navigation with a
Gaussian-Splatting reconstruction reward**.

Core idea: drive exploration/navigation to maximize *reconstruction quality*
(what the GS map can see), not just geometric coverage. Every navigation signal
is a **forward rasterize** of an existing Gaussian map — milliseconds, no GS
fitting in any inner loop — so the same reward serves a greedy next-best-view
planner or an RL policy.

## Layout

```
indoor_uav/
  sim/        IndoorSim contract + backends
    base.py            # Frame + IndoorSim ABC (RGB-D from any 6-DOF pose)
    synthetic_room.py  # dependency-free ray-cast room (CI / bring-up) — WORKS NOW
    habitat_room.py    # Habitat-Sim backend (photorealistic scenes) [optional env]
  gs/         Gaussian map + analytic reward (the differentiator)
    gaussian_map.py    # incremental GS map; render + add_from_rgbd (no fitting)
    coverage.py        # view_coverage, exploration_gain, fisher_view_info, best_next_view
  scripts/
    smoke_gs.py        # GS reward unit test (GPU, no sim) — ALL PASS
    smoke_pipeline.py  # end-to-end sim -> GS -> coverage -> NBV — ALL PASS
    setup_habitat.sh   # isolated 'habitat' env + indoor scene download
    render_habitat.py  # dump RGB-D frames (habitat env) -> load in training env
```

## Environments (intentionally decoupled)

- **`openfly`** — training/GS stack (torch 2.12+cu130, gsplat 1.5.3). Runs the
  GS map, rewards, planners, policy learning. gsplat JIT-compiles its CUDA
  backend on first import (needs `ninja` on PATH; cached afterwards).
- **`habitat`** — *optional* isolated sim env (python 3.9, habitat-sim
  headless) for photorealistic scenes. Kept separate so its pinned deps never
  collide with the cu130 stack; the two talk via the `IndoorSim` interface or
  dumped RGB-D frames (`render_habitat.py`).

## Verified on this box (1× A100 40GB, cu130)

- gsplat 1.5.3 builds + rasterizes against cu130 (sm_80); JIT backend cached.
- `smoke_gs`: coverage facing geometry > away; exploration gain empty 1.0 vs
  covered 0.96 — 5/5 PASS.
- `smoke_pipeline`: synthetic room -> GS map (18k Gaussians) -> coverage ->
  NBV correctly picks the unvisited view (0.601 > 0.280) — 5/5 PASS.
- **Real scenes**: Habitat-Sim 0.3.3 + git-lfs meshes (apartment_1 /
  van-gogh-room / skokloster-castle). 12 real RGB-D frames from apartment_1 ->
  197k-Gaussian map -> coverage rises monotonically 0.045 -> 0.546 across the
  patrol. Full real-scene sim -> GS -> reward path confirmed.

## Quickstart

```bash
conda activate openfly
python -m indoor_uav.scripts.smoke_gs        # GS reward unit test
python -m indoor_uav.scripts.smoke_pipeline  # full synthetic nav loop
```

## Status / next

Done: sim abstraction + synthetic backend, incremental GS map, analytic
coverage / exploration / FisherRF-style info-gain, greedy NBV baseline,
end-to-end validation.

Next: photorealistic Habitat backend + indoor scenes (`setup_habitat.sh`); a
`tasks/` gymnasium env (nav-to-landing with the GS reward); then greedy-NBV
eval vs a learned policy (the open "why learn over NBV" question: long-horizon
non-myopia, amortized onboard inference, semantic/language conditioning).
