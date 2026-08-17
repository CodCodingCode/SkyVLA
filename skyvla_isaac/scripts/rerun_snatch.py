"""Roll out a SNATCH checkpoint and log it to a Rerun .rrd for interactive 3D viewing.

    pip install rerun-sdk
    python skyvla_isaac/scripts/rerun_snatch.py \
        --checkpoint skyvla_isaac/snatch/checkpoints/model_9250.pt \
        --episodes 4 --out videos/snatch_9250.rrd
    rerun videos/snatch_9250.rrd        # or drag the file into rerun.io/viewer

IMPORTANT -- this is NOT Isaac Sim. Isaac Lab does not install on this aarch64 host, so
the checkpoint is driven through a lightweight reimplementation of the env's control law
(`DroneSnatchEnv._apply_action` / `_get_observations` / `_get_dones`), integrated at the
same 200 Hz physics / 50 Hz policy rate. The policy weights, observation layout, empirical
normalization, action semantics, velocity-tracking wrench, gripper latch rule and success
tests are the real ones; what is approximated is rigid-body contact:

  * the drone is a point mass with gravity-compensated velocity tracking (no attitude
    dynamics beyond commanded yaw rate, no wind/ground-effect DR -- training ran clean),
  * the cube is kinematic: rigidly carried while latched, free-falling to the table/floor
    otherwise -- no friction cage, no knock-over, no cube-jaw contact forces,
  * `grip_tau` is modeled as the actuator's clipped position error (blocked at the seated
    jaw position while latched) rather than a measured contact torque.

So treat the .rrd as "what this policy commands and where that takes it", not as a
physics-accurate replay. For ground truth use scripts/render_snatch.py under Isaac.

Pass --checkpoint more than once to overlay several checkpoints on identical episodes
(same cube/target seeds) -- the training-progression view.
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import rerun as rr

# --- constants mirrored from skyvla_isaac/snatch/pick_place_env.py (DroneSnatchEnvCfg) ---
DT = 1.0 / 200.0          # sim.dt
DECIM = 4                 # decimation -> 50 Hz policy
EP_SECONDS = 10.0         # episode_length_s
SPEED = 1.5               # cfg.speed
KV = 18.0                 # cfg.kv (velocity-tracking gain)
YAW_RATE_SCALE = 1.0
GRAV = 9.81
SURFACE_Z = 0.30          # table top
CUBE_SIZE = 0.05
CUBE_HALF = CUBE_SIZE / 2
GRASP_CLEAR = 0.06        # cube off-surface height to count as lifted
TIP_DZ = 0.07             # cage sits base-0.07 (_tip_w)
GRIP_TRAVEL = 0.035
LATCH_R = 0.035           # cfg.latch_r
OBJ_SPAWN_DIAM = 0.8
GOAL_OFFSET_DIAM = 0.5
GOAL_Z = SURFACE_Z + 0.25
GRIP_STIFF, GRIP_EFFORT, JAW_VLIM = 2000.0, 80.0, 1.0
TABLE_HALF = 0.5          # 1.0 x 1.0 m top
SPAWN = np.array([0.0, 0.0, 1.0])   # robot init_state.pos (fly-in start, cur_p=0)
BODY_HALF = np.array([0.09, 0.09, 0.03])
CAGE_HALF = np.array([0.027, 0.027, 0.02])
NORM_EPS = 1e-2           # rsl_rl EmpiricalNormalization default


class Policy:
    """The rsl_rl actor MLP + empirical obs normalizer, evaluated in numpy.

    torch is used only to unpickle the checkpoint (converted through .tolist(), since the
    system torch and the numpy that rerun requires have incompatible C ABIs on this host).
    """

    def __init__(self, path):
        import torch                              # local: keep the numpy ABI clash contained
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        self.layers, i = [], 0
        while f"actor.{i}.weight" in sd:
            self.layers.append((np.array(sd[f"actor.{i}.weight"].tolist(), dtype=np.float64),
                                np.array(sd[f"actor.{i}.bias"].tolist(), dtype=np.float64)))
            i += 2
        nrm = ck["obs_norm_state_dict"]
        self.mean = np.array(nrm["_mean"].tolist(), dtype=np.float64).reshape(-1)
        self.std = np.array(nrm["_std"].tolist(), dtype=np.float64).reshape(-1)
        self.iter = int(ck.get("iter", -1))
        self.obs_dim = self.layers[0][0].shape[1]
        self.tag = os.path.splitext(os.path.basename(path))[0]

    def act(self, obs):
        """Deterministic action: the actor mean, clamped like _pre_physics_step does."""
        x = (obs - self.mean) / (self.std + NORM_EPS)
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:         # activation="elu"
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return np.clip(x, -1.0, 1.0)


class Rollout:
    """One episode of the SNATCH task under the approximate plant described in the docstring."""

    def __init__(self, rng):
        off = (rng.random(2) - 0.5) * OBJ_SPAWN_DIAM
        self.cube = np.array([off[0], off[1], SURFACE_Z + CUBE_HALF])
        self.cube_v = np.zeros(3)
        self.target = np.array([
            self.cube[0] + (rng.random() - 0.5) * GOAL_OFFSET_DIAM,
            self.cube[1] + (rng.random() - 0.5) * GOAL_OFFSET_DIAM,
            GOAL_Z])
        self.p, self.v, self.yaw = SPAWN.copy(), np.zeros(3), 0.0
        self.jaw, self.tau = 0.0, 0.0
        self.latched, self.latch_off = False, np.zeros(3)
        self.ever_grasped, self.ever_placed = False, False

    @property
    def tip(self):
        return self.p - np.array([0.0, 0.0, TIP_DZ])

    def obs(self):
        goal_rel = self.target - self.p
        return np.concatenate([self.p, self.v, [self.jaw], [self.tau], self.cube, goal_rel])

    def step(self, a):
        """Advance DECIM physics substeps under action a = [vx, vy, vz, yaw_rate, grip]."""
        v_des = a[:3] * SPEED
        a_max = 4.0 * GRAV                       # force clamp fmax = 4*m*g -> accel cap
        jaw_tgt = (a[4] * 0.5 + 0.5) * GRIP_TRAVEL
        for _ in range(DECIM):
            acc = np.clip(KV * (v_des - self.v), -a_max, a_max)
            self.v += acc * DT
            self.p += self.v * DT
            self.yaw += a[3] * YAW_RATE_SCALE * DT
            floor = (SURFACE_Z if abs(self.p[0]) < TABLE_HALF and abs(self.p[1]) < TABLE_HALF
                     else 0.0) + 0.025
            if self.p[2] < floor:                # body rests on the surface it touches
                self.p[2], self.v[2] = floor, max(self.v[2], 0.0)
            # gripper: implicit actuator tracks the commanded jaw travel; the seated cube
            # blocks it at 0.026 (the carry-demo joint pos) once latched
            limit = 0.026 if self.latched else GRIP_TRAVEL
            reach = min(jaw_tgt, limit)
            self.jaw += np.clip(reach - self.jaw, -JAW_VLIM * DT, JAW_VLIM * DT)
            self.tau = float(np.clip(abs(GRIP_STIFF * (jaw_tgt - self.jaw)), 0.0, GRIP_EFFORT))
            if self.latched:
                self.cube = self.tip + self.latch_off
                self.cube_v = self.v.copy()
            else:                                # free cube: fall onto table or floor
                self.cube_v[2] -= GRAV * DT
                self.cube = self.cube + self.cube_v * DT
                rest = (SURFACE_Z if abs(self.cube[0]) < TABLE_HALF and abs(self.cube[1]) < TABLE_HALF
                        else 0.0) + CUBE_HALF
                if self.cube[2] <= rest:
                    self.cube[2], self.cube_v = rest, np.zeros(3)

        # latch state machine (env._get_dones, cfg.grasp_latch=True)
        tip = self.tip
        horiz = float(np.linalg.norm((self.cube - tip)[:2]))
        inside = horiz < LATCH_R and abs(self.cube[2] - tip[2]) < 0.05
        closing = a[4] > 0.0
        if inside and closing and not self.latched:
            off = self.cube - tip
            off[:2] = np.clip(off[:2], -0.012, 0.012)
            self.latch_off = off
        self.latched = (self.latched or (inside and closing)) and (a[4] > -0.2)

        d_reach = float(np.linalg.norm(self.cube - tip))
        d_goal = float(np.linalg.norm(self.cube - self.target))
        lifted = self.cube[2] > SURFACE_Z + CUBE_HALF + GRASP_CLEAR
        held = bool(lifted and d_reach < 0.10)
        success = bool(held and d_goal < 0.18)
        self.ever_grasped |= held
        self.ever_placed |= success
        return dict(d_reach=d_reach, d_goal=d_goal, horiz=horiz, held=held,
                    lifted=bool(lifted), success=success)


def log_static(world):
    # Z-up: without this the viewer assumes its own up-axis and the default camera ends up
    # staring past a ~1 m scene (endless grid, no visible content).
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(f"{world}/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3, vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
        colors=[[220, 70, 70], [70, 220, 70], [70, 120, 240]], labels=["x", "y", "z"]), static=True)
    rr.log(f"{world}/table", rr.Boxes3D(centers=[[0.0, 0.0, SURFACE_Z / 2]],
                                        half_sizes=[[TABLE_HALF, TABLE_HALF, SURFACE_Z / 2]],
                                        colors=[[115, 82, 56]], labels=["table"]), static=True)
    grid = []
    for i in range(-3, 4):
        grid += [[[-1.5, i * 0.5, 0.0], [1.5, i * 0.5, 0.0]],
                 [[i * 0.5, -1.5, 0.0], [i * 0.5, 1.5, 0.0]]]
    rr.log(f"{world}/floor", rr.LineStrips3D(grid, colors=[[60, 60, 68]], radii=0.002), static=True)


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="path to a model_*.pt; repeat to overlay several policies")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(repo, "videos/snatch_rollout.rrd"))
    args = ap.parse_args()

    steps = int(EP_SECONDS / (DT * DECIM))
    policies = [Policy(c) for c in args.checkpoint]
    for p in policies:
        assert p.obs_dim == 14, f"{p.tag}: expected the 14-dim state obs (--no_cams policy), got {p.obs_dim}"
    multi = len(policies) > 1

    rr.init("skyvla_snatch", spawn=False)
    rr.save(args.out)
    try:
        from rerun import blueprint as rbl
        # pin the camera on the table (the env's own ViewerCfg eye/lookat) -- the scene is
        # only ~1 m across, so a default camera can easily start outside it
        eye = rbl.archetypes.EyeControls3D(position=[1.6, 1.6, 1.2],
                                           look_target=[0.0, 0.0, 0.35], eye_up=[0.0, 0.0, 1.0])
        rr.send_blueprint(rbl.Blueprint(rbl.Horizontal(
            rbl.Spatial3DView(origin="/", name="SNATCH rollout", eye_controls=eye),
            rbl.Vertical(rbl.TimeSeriesView(origin="/metrics/distance", name="distances (m)"),
                         rbl.TimeSeriesView(origin="/metrics/gripper", name="gripper"),
                         rbl.TimeSeriesView(origin="/metrics/action", name="action")),
            column_shares=[2, 1])))
    except Exception as e:                       # blueprint API drift shouldn't kill the run
        print(f"[rerun] default blueprint ({e})")

    log_static("world")
    summary = {p.tag: [0, 0] for p in policies}
    frame = 0

    for ep in range(args.episodes):
        # identical episode (cube + target) for every checkpoint -> honest comparison
        runs = [(p, Rollout(np.random.default_rng(args.seed + ep))) for p in policies]
        init = runs[0][1]
        rr.set_time("step", sequence=frame)
        rr.log("events", rr.TextLog(
            f"episode {ep}: cube ({init.cube[0]:+.2f}, {init.cube[1]:+.2f}) "
            f"target ({init.target[0]:+.2f}, {init.target[1]:+.2f})"))

        paths = {p.tag: [] for p in policies}
        for t in range(steps):
            rr.set_time("step", sequence=frame)
            rr.set_time("sim_time", duration=t * DT * DECIM)
            for pol, r in runs:
                tag = pol.tag
                a = pol.act(r.obs())
                m = r.step(a)
                root = f"world/{tag}" if multi else "world"
                paths[tag].append(r.p.copy())
                held_col = [90, 220, 120] if r.latched else [225, 90, 80]
                rr.log(f"{root}/drone", rr.Boxes3D(
                    centers=[r.p], half_sizes=[BODY_HALF], colors=[[110, 170, 255]],
                    rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], radians=r.yaw)],
                    labels=[tag] if multi else None))
                rr.log(f"{root}/cage", rr.Boxes3D(
                    centers=[r.tip], half_sizes=[CAGE_HALF],
                    colors=[[90, 220, 120] if r.latched else
                            ([235, 190, 70] if a[4] > 0 else [150, 150, 160])],
                    rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], radians=r.yaw)],
                    fill_mode="majorwireframe"))
                rr.log(f"{root}/cube", rr.Boxes3D(centers=[r.cube],
                                                  half_sizes=[[CUBE_HALF] * 3], colors=[held_col]))
                rr.log(f"{root}/target", rr.Points3D([r.target], colors=[[80, 230, 160]],
                                                     radii=0.18 / 2, labels=["drop zone"]))
                rr.log(f"{root}/path", rr.LineStrips3D([paths[tag]], colors=[[110, 170, 255]],
                                                       radii=0.004))
                rr.log(f"metrics/distance/{tag}/tip_to_cube", rr.Scalars(m["d_reach"]))
                rr.log(f"metrics/distance/{tag}/cube_to_goal", rr.Scalars(m["d_goal"]))
                rr.log(f"metrics/distance/{tag}/cube_height", rr.Scalars(r.cube[2]))
                rr.log(f"metrics/gripper/{tag}/grip_cmd", rr.Scalars(float(a[4])))
                rr.log(f"metrics/gripper/{tag}/jaw", rr.Scalars(r.jaw))
                rr.log(f"metrics/gripper/{tag}/latched", rr.Scalars(float(r.latched)))
                for i, name in enumerate(["vx", "vy", "vz", "yaw_rate"]):
                    rr.log(f"metrics/action/{tag}/{name}", rr.Scalars(float(a[i])))
            frame += 1

        for pol, r in runs:
            summary[pol.tag][0] += int(r.ever_grasped)
            summary[pol.tag][1] += int(r.ever_placed)
            rr.log("events", rr.TextLog(
                f"episode {ep} [{pol.tag}]: grasped={r.ever_grasped} placed={r.ever_placed}"))

    print(f"\nrrd -> {args.out}  ({args.episodes} episodes x {steps} steps @ 50 Hz)")
    for pol in policies:
        g, p = summary[pol.tag]
        print(f"  {pol.tag:<12} (iter {pol.iter:>5}): "
              f"grasped {g}/{args.episodes}, placed {p}/{args.episodes}")


if __name__ == "__main__":
    main()
