"""Phase 2: draw a recorded PLACE take (film_place_rollout.py .npz) to MP4.

Isaac's RTX camera cannot run on this host (vkCreateInstance fails -> the Camera sensor
hangs inside env construction), so the frames are drawn with matplotlib instead. The physics
in the .npz is the real Isaac rollout; only the pixels are synthetic.

  .venv311/bin/python skyvla_isaac/scripts/film_place_draw.py --take /tmp/take.npz --out videos/place.mp4
"""
import argparse, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p = argparse.ArgumentParser()
p.add_argument("--take", required=True)
p.add_argument("--out", required=True)
p.add_argument("--fps", type=int, default=25)      # sim is 50 Hz -> 25 fps is 0.5x slow motion
p.add_argument("--dpi", type=int, default=110)
p.add_argument("--title", default="SkyVLA place stage - model_20000")
args = p.parse_args()

d = np.load(args.take)
MODE = str(d["mode"]) if "mode" in d else "place"
drone, cube, jaw = d["drone"], d["cube"], d["jaw"]
if MODE == "snatch":
    rel = ~d["held"]; dep = d["held"]          # "attached to the drone" plays the same role
    pad_a = d["pad_a"]; pad_b = None
else:
    rel, dep = d["released"], d["deposited"]
    pad_a, pad_b = d["pad_a"], d["pad_b"]
sz, cs = float(d["surface_z"]), float(d["cube_size"])
rest = sz + 0.5 * cs
T = len(drone)
if MODE == "snatch":
    pr = float(d["grasp_clear"]); lip = 0.0
    t_dep = int(d["t_grasp"]); impact = 0.0
else:
    pr, lip = float(d["place_radius"]), float(d["lip_h"])
    t_dep = int(d["t_dep"])
    impact = float(d["impact"][t_dep]) if t_dep < len(d["impact"]) else float(d["impact"][-1])


def box(c, s, sz_z=None):
    """axis-aligned box faces centred at c with side s (z side sz_z if given)"""
    hx = hy = s / 2.0
    hz = (sz_z if sz_z is not None else s) / 2.0
    x, y, z = c
    v = np.array([[x-hx, y-hy, z-hz], [x+hx, y-hy, z-hz], [x+hx, y+hy, z-hz], [x-hx, y+hy, z-hz],
                  [x-hx, y-hy, z+hz], [x+hx, y-hy, z+hz], [x+hx, y+hy, z+hz], [x-hx, y+hy, z+hz]])
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]], [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]


ctr = 0.5 * ((pad_b[:2] if pad_b is not None else cube[0, :2]) + drone[0, :2])
R = max(0.75, np.abs(np.concatenate([drone[:, :2], cube[:, :2]]) - ctr).max() * 1.35)
tmp = tempfile.mkdtemp(prefix="placedraw_")
for i in range(T):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(ctr[0]-R, ctr[0]+R); ax.set_ylim(ctr[1]-R, ctr[1]+R); ax.set_zlim(sz-0.02, sz+0.95)
    ax.set_box_aspect((1, 1, 0.62))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0)); pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    # ground, pads
    g = 1.6 * R
    ax.add_collection3d(Poly3DCollection(
        [[[ctr[0]-g, ctr[1]-g, sz], [ctr[0]+g, ctr[1]-g, sz],
          [ctr[0]+g, ctr[1]+g, sz], [ctr[0]-g, ctr[1]+g, sz]]],
        facecolor="#e9eef2", edgecolor="none", alpha=0.55, zorder=0))
    for c, col in ([(pad_a, "#8a6a44")] if pad_b is None
                   else [(pad_a, "#8a6a44"), (pad_b, "#2f7d4f")]):
        ax.add_collection3d(Poly3DCollection(
            [[[c[0]-0.5, c[1]-0.5, sz+0.001], [c[0]+0.5, c[1]-0.5, sz+0.001],
              [c[0]+0.5, c[1]+0.5, sz+0.001], [c[0]-0.5, c[1]+0.5, sz+0.001]]],
            facecolor=col, edgecolor="#00000022", alpha=0.9, zorder=1))
    if pad_b is not None:
        th = np.linspace(0, 2*np.pi, 60)                  # the deposit gate on pad B
        ax.plot(pad_b[0]+pr*np.cos(th), pad_b[1]+pr*np.sin(th), sz+0.002,
                color="#ffffff", lw=1.6, alpha=0.85, zorder=2)
    # trails
    ax.plot(drone[:i+1, 0], drone[:i+1, 1], drone[:i+1, 2], color="#2f6fb2", lw=1.3, alpha=0.55, zorder=3)
    ax.plot(cube[:i+1, 0], cube[:i+1, 1], cube[:i+1, 2], color="#c62828", lw=1.3, alpha=0.55, zorder=3)
    # drone body + cage jaws
    dp = drone[i]
    ax.add_collection3d(Poly3DCollection(box(dp, 0.13, 0.05), facecolor="#3b7dd8",
                                         edgecolor="#1b4f8f", alpha=0.95, zorder=5))
    ap = max(0.0, float(jaw[i])) + 0.012
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([dp[0]+sx*0.045, dp[0]+sx*(0.045+ap*1.4)],
                    [dp[1]+sy*0.045, dp[1]+sy*(0.045+ap*1.4)],
                    [dp[2]-0.025, dp[2]-0.075], color="#9aa7b4", lw=2.0, zorder=5)
    # cube
    ax.add_collection3d(Poly3DCollection(box(cube[i], cs), facecolor="#d33b3b",
                                         edgecolor="#7d1f1f", alpha=1.0, zorder=6))
    # HUD
    held = not bool(rel[i])
    hcm = (cube[i, 2] - rest) * 100
    rcm = (np.linalg.norm(cube[i, :2] - pad_b[:2]) * 100 if pad_b is not None
           else np.linalg.norm(drone[i, :2] - cube[i, :2]) * 100)
    if MODE == "snatch":
        state = "LIFTED" if dep[i] else ("ON THE TABLE" if hcm < 1.0 else "IN THE CAGE")
        scol = "#1b5e20" if dep[i] else "#2f6fb2"
    else:
        state = ("IN THE CAGE" if held else ("AT REST ON PAD B" if dep[i] else "FALLING"))
        scol = "#1b5e20" if dep[i] else ("#2f6fb2" if held else "#b71c1c")
    fig.text(0.045, 0.945, args.title, fontsize=15, weight="bold", color="#20303f")
    fig.text(0.045, 0.905, f"t = {i*0.02:5.2f} s      cube: {state}", fontsize=12.5, color=scol, weight="bold")
    lab2 = ("cage to cube          " if MODE == "snatch" else "cube distance to centre")
    gate = (f"   (lift gate {pr*100:.0f})" if MODE == "snatch" else f"   (gate {pr*100:.0f})")
    fig.text(0.045, 0.868, f"cube height above table {hcm:6.1f} cm{gate if MODE=='snatch' else ''}",
             fontsize=11.5, color="#43525f", family="monospace")
    fig.text(0.045, 0.838, f"{lab2} {rcm:6.1f} cm{'' if MODE=='snatch' else gate}",
             fontsize=11.5, color="#43525f", family="monospace")
    fig.text(0.045, 0.808, f"jaw aperture            {jaw[i]*1000:6.1f} mm", fontsize=11.5,
             color="#43525f", family="monospace")
    if i >= t_dep:
        fig.text(0.045, 0.755,
                 (f"GRASPED + LIFTED   {hcm:.1f} cm off the table" if MODE == "snatch"
                  else f"DEPOSITED   contact speed {impact:.2f} m/s  (gentle floor 0.40)"),
                 fontsize=12.5, color="#1b5e20", weight="bold")
    fig.text(0.70, 0.055, ("brown pad = pick platform A" if MODE == "snatch"
                           else "green pad = delivery target B\nwhite ring = deposit gate"),
             fontsize=10, color="#5b6b78")
    fig.savefig(os.path.join(tmp, f"f{i:05d}.png"), facecolor="white")
    plt.close(fig)

print(f"[draw] {T} frames -> {args.out}")
cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
       "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-crf", "18", args.out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg ok" if r.returncode == 0 else "ffmpeg error:\n" + r.stderr[-700:])
print(f"[draw] DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")
