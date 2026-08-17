# GS rendering cache — render & test the splat scene FAST

**TL;DR — never rebuild the Gaussian-splat room you already built.** The room is
static geometry; cache it once and reuse it. Then re-render videos with **no Isaac
Sim at all**.

This is the canonical reference for how GS-backdrop rendering is cached. The
machinery lives in [`gs/cache.py`](cache.py) (+ `GaussianMap.state_dict` /
`from_state_dict` in [`gs/gaussian_map.py`](gaussian_map.py)).

## Why

`render_rollout_gs.py` used to do this on **every** run:

1. Boot Isaac Sim (~30–60 s).
2. **Rebuild the splat room**: a ~44-view RGB-D orbit (88 `sim.step(render=True)`
   calls) + fusion — slow, and *identical every run* because the room doesn't
   depend on the policy/checkpoint.
3. Roll out the policy and composite the live drone+cube over the splat.

Steps 2 (always) and even 1+3 (when only the look changes) are wasted work when
you're iterating. Caching kills them.

## The two caches

| Cache | File | Holds | Built by | Skips |
|-------|------|-------|----------|-------|
| **scene** | `cache/room_splat.pt` | the splat (`GaussianMap`) + `K` + render W×H | `render_rollout_gs.py` (once) | the 44-view room orbit |
| **rollout** | `cache/rollout_<ckpt>.npz` | per-frame foreground RGBA + follow-cam pose, for one checkpoint | `render_rollout_gs.py --save_rollout` | **all of Isaac** on re-render |

Both live under `skyvla_isaac/gs/cache/` (gitignored). Writes are atomic
(tmp + `os.replace`) and format-versioned, so a stale/half-written cache is
rejected, not silently mis-rendered. A measured GS render is **~1.5 ms/frame**
at 720×540, so a 450-frame backdrop renders in <1 s once the scene is loaded.

## How to use

### 1. Build it once (Isaac, slow — do this a single time per room)

> **Currently blocked on this host.** `render_rollout_gs.py` imports *both*
> `isaaclab` and `gsplat`, and no interpreter here has both: `.venv311` has Isaac
> without gsplat, `.venv` has gsplat without Isaac. Install `gsplat` into
> `.venv311` before running this step. Step 2 below (the Isaac-free replay) is
> unaffected and works today.

```bash
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1 PYTHONPATH=/home/ubuntu/SkyVLA \
.venv311/bin/python skyvla_isaac/scripts/render_rollout_gs.py \
    --checkpoint logs/isaac/drone_pick_place/model_5999.pt \
    --out videos/isaac_pickplace_gs.mp4 \
    --save_rollout        # also cache the rollout for Isaac-free replay
```
First run builds **and saves** `cache/room_splat.pt`. Every later run **loads** it
and skips the orbit automatically. Use `--rebuild_gs` to force a fresh orbit (e.g.
you changed the room geometry or the render resolution).

### 2. Re-render FAST, no Isaac (the iteration loop)
```bash
# any interpreter with torch + gsplat
PYTHONUTF8=1 PYTHONPATH=/home/ubuntu/SkyVLA \
.venv/bin/python skyvla_isaac/scripts/render_gs_cache.py \
    --rollout skyvla_isaac/gs/cache/rollout_model_5999.npz \
    --out videos/replay.mp4
```
Re-composites the cached drone+cube over the cached splat — change `--fps`,
`--fill R G B`, `--cover_thresh` and re-render in seconds.

Orbit the bare room to sanity-check the scene (no rollout needed):
```bash
.venv/bin/python skyvla_isaac/scripts/render_gs_cache.py --out videos/gs_orbit.mp4 \
    --steps 180 --radius 2.6 --height 1.6 --look_z 1.2
```

## Flags added to `render_rollout_gs.py`
- `--gs_cache PATH` — scene-cache path (default `cache/room_splat.pt`).
- `--rebuild_gs` — force a fresh room orbit even if the cache exists.
- `--save_rollout` — also write the rollout cache for Isaac-free replay.
- `--rollout_cache PATH` — rollout-cache path (default `cache/rollout_<ckpt>.npz`).

## Gotchas
- **Resolution must match.** The scene cache stores the W×H it was captured at.
  If you change `cfg.cam_w/cam_h`, rebuild with `--rebuild_gs` (you'll get a WARN
  otherwise).
- **Rollout cache is per-checkpoint.** A new policy needs a fresh rollout
  (re-run with `--save_rollout`); the *scene* cache is reused across all of them.
- **Disk.** `/dev/vda1` is tight on this host (see top-level `CLAUDE.md`). The
  rollout `.npz` compresses well (background is zeroed before saving) but a long
  450-frame run is still tens of MB — `rm skyvla_isaac/gs/cache/rollout_*.npz`
  when done. The scene cache is small (hundreds of KB).
- **`PYTHONUTF8=1`** — gsplat JIT-compiles its CUDA kernels on first use; torch
  reads the `.cu` sources with the locale codec and crashes without UTF-8.
- **Programmatic use:** `from skyvla_isaac.gs import save_scene, load_scene,
  save_rollout, load_rollout` (or `GaussianMap.state_dict()` /
  `GaussianMap.from_state_dict(sd, device)` for just the splat).
