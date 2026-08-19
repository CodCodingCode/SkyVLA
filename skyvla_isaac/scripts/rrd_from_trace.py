"""Log a real Isaac pipeline delivery to a Rerun .rrd (3D, scrubbable).

Geometry comes from run_pipeline.py --trace_out, i.e. the actual Isaac PhysX rollout of
snatch -> carry -> place, not a reimplementation.
"""
import numpy as np, rerun as rr, argparse
S = "/tmp/claude-1000/-home-ubuntu/2e3c21ac-6c1a-4db2-8cad-ba02519b309f/scratchpad"
ap = argparse.ArgumentParser()
ap.add_argument("--trace", default=f"{S}/trace.npz")
ap.add_argument("--out", default=f"{S}/delivery.rrd")
ap.add_argument("--top", type=int, default=4, help="how many successful episodes to log")
a = ap.parse_args()

d = np.load(a.trace)
dep, done, ph = d["dep"], d["done"], d["phase"]
drone, cube, platB = d["drone"], d["cube"], d["platB"]
sz, cs = float(d["surface_z"]), float(d["cube_size"])
T, N = dep.shape

cands = []
for e in np.where(dep.any(0))[0]:
    td = int(np.where(dep[:, e])[0][0])
    pr = np.where(done[:td, e])[0]
    t0 = int(pr[-1]) + 1 if len(pr) else 0
    cands.append((td - t0, int(e), t0, td))
cands.sort(reverse=True)
print(f"{len(cands)} successful episodes; logging top {a.top}")

rr.init("skyvla_delivery", spawn=False)
rr.save(a.out)

PHASE = ["approach", "transit", "deposit"]
COL = [(150, 158, 170), (90, 170, 210), (95, 190, 150)]

for rank, (ln, e, t0, td) in enumerate(cands[:a.top]):
    root = f"ep{rank}_env{e}"
    bx, by = float(platB[td, e, 0]), float(platB[td, e, 1])
    # static scene: the two pads (1x1x0.30 boxes resting on the floor)
    rr.log(f"{root}/pad_A", rr.Boxes3D(centers=[[0, 0, 0.15]], half_sizes=[[.5, .5, .15]],
           colors=[(122, 88, 54)], labels=["pad A - pick"]), static=True)
    rr.log(f"{root}/pad_B", rr.Boxes3D(centers=[[bx, by, 0.15]], half_sizes=[[.5, .5, .15]],
           colors=[(53, 99, 122)], labels=["pad B - place"]), static=True)
    rr.log(f"{root}/floor", rr.Boxes3D(centers=[[bx/2, by/2, -0.02]],
           half_sizes=[[4.5, 3.0, 0.02]], colors=[(40, 44, 50)]), static=True)

    dr = drone[t0:td+1, e]; cu = cube[t0:td+1, e]; pp = ph[t0:td+1, e]
    for k in range(len(dr)):
        rr.set_time("step", sequence=k)
        rr.set_time("sim_time", duration=k * 0.02)
        rr.log(f"{root}/drone", rr.Points3D([dr[k]], radii=[0.07], colors=[(70, 130, 170)]))
        rr.log(f"{root}/cube", rr.Boxes3D(centers=[cu[k]], half_sizes=[[cs/2]*3],
                                          colors=[(190, 62, 45)]))
        rr.log(f"{root}/drone_path", rr.LineStrips3D([dr[:k+1]], colors=[(70, 130, 170)], radii=[0.012]))
        rr.log(f"{root}/cube_path", rr.LineStrips3D([cu[:k+1]], colors=[(190, 62, 45)], radii=[0.018]))
        rr.log(f"{root}/phase", rr.TextLog(PHASE[pp[k]], level=rr.TextLogLevel.INFO))
        rr.log(f"{root}/altitude/cube", rr.Scalars(float(cu[k][2])))
        rr.log(f"{root}/altitude/drone", rr.Scalars(float(dr[k][2])))
        rr.log(f"{root}/dist_to_padB", rr.Scalars(float(np.hypot(cu[k][0]-bx, cu[k][1]-by))))
    print(f"  ep{rank}: env {e}, {len(dr)} steps ({len(dr)*0.02:.1f}s), pad B at ({bx:+.2f},{by:+.2f})")

print("saved", a.out)
