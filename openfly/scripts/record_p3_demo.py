#!/usr/bin/env python3
"""Record an MP4 of the P3 (BC + subgoals) policy navigating in the real
Unreal sim (``env_ue_smallcity``).

The policy is ``PaliGemmaVLNPolicy`` with subgoal tokens sampled per step
from a frozen ``SubgoalDiT`` / ``PixArtSubgoalDiT``. Frames are pulled
from a live UnrealCV connection at the bridge's native resolution
(typically 1920×1080); only the policy-input copy is downsampled.

Outputs dual MP4s — first-person view with HUD + top-down trajectory map.

Examples::

    # Cold launch: fork CitySample.sh, run, tear down.
    python -m openfly.scripts.record_p3_demo \\
      --p3_ckpt logs/openfly/paligemma_subgoal/<ts>/best.pt \\
      --dit_path logs/openfly/subgoal_dit/<ts>/best.pt \\
      --pretrained_path /path/to/pixart/snapshot \\
      --episodes 3 --out videos/p3_realsim.mp4

    # Faster: launch the sim once in another shell, then reconnect.
    bash $OPENFLY_ROOT/envs/ue/env_ue_smallcity/CitySample.sh &
    python -m openfly.scripts.record_p3_demo --no_launch \\
      --p3_ckpt ... --dit_path ... --pretrained_path ... \\
      --episodes 3 --out videos/p3_realsim.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_DRONE_ROOT = Path(__file__).resolve().parents[2]
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.actions import ACTION_NAMES, distance3d
from openfly.envs import AirSimVLNEnv, AirSimVLNEnvConfig
from openfly.episodes import load_episodes
from openfly.models.paligemma_vln import PaliGemmaVLNPolicy
from openfly.scripts.record_rl_demo import (
    _SceneCtx,
    _draw_hud,
    _topdown_frame,
)
from openfly.train_paligemma_subgoal import (
    _body_frame_pose_delta,
    _build_processor,
    _dit_subgoal_tokens,
    _load_world_model,
    _tokenise_batch,
    _oracle_subgoal_tokens,
)


# ---------------------------------------------------------------------------
# Real-sim bridge adapter
# ---------------------------------------------------------------------------

def _drift_camera(
    bridge: Any, src: list[float], dst: list[float], yaw_dst_deg: float,
    steps: int, *, label: str = "drift",
) -> None:
    """Move the camera from ``src`` to ``dst`` in ``steps`` small hops,
    fetching a frame after each.

    OpenFly's CitySample is ~2 km across; UnrealCV holds TCP requests open
    while UE streams new level chunks. A direct teleport of >100m can
    therefore stall ``get_camera_data`` indefinitely because UE never
    finishes loading before the client times out. Drifting in small
    increments keeps each request's streaming delta small enough to
    return quickly. Without this preamble the recorder's very first
    ``get_camera_data`` after make_bridge will hang forever.
    """
    src_a = np.asarray(src, dtype=np.float64)
    dst_a = np.asarray(dst, dtype=np.float64)
    dist = float(np.linalg.norm(dst_a - src_a))
    print(f"[{label}] drift {dist:.0f}m in {steps} steps "
          f"from ({src[0]:.0f},{src[1]:.0f}) -> ({dst[0]:.0f},{dst[1]:.0f})",
          flush=True)
    for k in range(1, steps + 1):
        t = k / steps
        x, y, z = (src_a * (1.0 - t) + dst_a * t).tolist()
        # Keep yaw at 0 during drift; the recorder sets the real yaw on the
        # first env.reset.
        bridge.set_camera_pose(x, y, z, 0.0, yaw_dst_deg * t, 0.0)
        time.sleep(0.05)
        _ = bridge.get_camera_data("lit")


class _LitCameraAdapter:
    """Wraps a real OpenFly bridge (UEBridge / AirsimBridge / GSBridge) so the
    env can call ``get_camera_data()`` with no args (matching the mock's
    signature) while we still pull the ``'lit'`` channel from the live sim.

    Also caches the most recent native-resolution frame so the recorder can
    use it for the MP4 — the env will downscale internally for policy input.
    """

    def __init__(self, bridge: Any, camera_type: str = "lit") -> None:
        self._bridge = bridge
        self._camera_type = camera_type
        self.last_native: np.ndarray | None = None

    def set_camera_pose(self, x, y, z, pitch, yaw_deg, roll):
        self._bridge.set_camera_pose(x, y, z, pitch, yaw_deg, roll)

    def get_camera_data(self, camera_type: str | None = None) -> np.ndarray:
        ct = camera_type or self._camera_type
        rgb = self._bridge.get_camera_data(ct)
        if rgb is None:
            raise RuntimeError(f"bridge.get_camera_data({ct!r}) returned None")
        rgb = np.asarray(rgb)
        self.last_native = rgb
        return rgb

    def disconnect(self) -> None:
        try:
            client = getattr(self._bridge, "_client", None)
            if client is not None and hasattr(client, "disconnect"):
                client.disconnect()
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            print(f"[record_p3_demo] bridge disconnect warning: {exc}")


def _real_sim_hud(
    fpv_native: np.ndarray,
    *,
    instruction: str,
    pose: list[float],
    goal: list[float],
    action_id: int,
    reward_so_far: float,
    step: int,
    episode_idx: int,
    sr_running: float,
    sim_env: str,
) -> np.ndarray:
    """HUD for the real-sim mode: re-uses ``_draw_hud`` shape/layout but
    overrides the banner so it reads ``policy=P3+DiT real-sim`` instead of the
    mock's ``heuristic policy synthetic render``.
    """
    out = _draw_hud(
        fpv_native,
        instruction=instruction,
        pose=pose,
        goal=goal,
        action_id=action_id,
        reward_so_far=reward_so_far,
        step=step,
        episode_idx=episode_idx,
        sr_running=sr_running,
    )
    H, W = out.shape[:2]
    cv2.rectangle(out, (0, 0), (W, 24), (15, 15, 15), -1)
    cv2.putText(
        out,
        f"OpenFly real sim ({sim_env}) - policy=P3+DiT - native UE render",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 220),
        1,
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Policy wrapper — wraps the trained P3 + DiT into a single per-step "predict"
# ---------------------------------------------------------------------------

class _SubgoalPolicyDriver:
    """Loads the P3 policy + frozen DiT and exposes ``act(obs, instruction,
    info, last_action)`` returning an OpenFly action id.

    Encodes the current frame through PaliGemma to get curr_siglip +
    text_embed, samples a subgoal via 4-step DDIM through the DiT, then
    runs ``policy.predict_action`` with the resulting subgoal_tokens.
    Single call per env step.
    """

    def __init__(
        self,
        p3_ckpt: str,
        dit_path: str,
        pretrained_path: str | None,
        paligemma_model: str,
        history_frames: int,
        lora_rank: int,
        lora_alpha: float,
        device: torch.device,
        ddim_steps: int = 4,
    ):
        self.device = device
        self.ddim_steps = ddim_steps
        self.history_frames = history_frames

        print("[record_p3_demo] loading world model …")
        self.dit = _load_world_model(
            dit_path, pretrained_path=pretrained_path, device=device,
        )

        print("[record_p3_demo] loading P3 policy …")
        self.model = PaliGemmaVLNPolicy(
            history_frames=history_frames,
            paligemma_model_name=paligemma_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        ).to(device)

        # Same shape-filter as the eval script (older checkpoints predate
        # the frame_embed slot count and progress_head dims).
        state = torch.load(p3_ckpt, map_location=device, weights_only=False)
        own = self.model.state_dict()
        filtered = {
            k: v for k, v in state["model"].items()
            if k in own and tuple(own[k].shape) == tuple(v.shape)
        }
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print(
            f"[record_p3_demo] P3 ckpt — loaded={len(filtered)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        self.model.eval()

        self.processor = _build_processor(paligemma_model)

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, np.ndarray],
        instruction: str,
        last_action_id: int,
    ) -> tuple[int, str]:
        """Run one inference step. Returns (action_id, action_name)."""
        device = self.device

        # Build a (B=1) batch.
        rgb = torch.from_numpy(np.ascontiguousarray(obs["rgb"]))[None].to(device)
        history = torch.from_numpy(np.ascontiguousarray(obs["rgb_history"]))[None].to(device)
        pose = torch.from_numpy(np.asarray(obs["pose"], dtype=np.float32))[None].to(device)
        # Synthetic next_pose / progress / subgoal_pose for inference — the
        # policy accepts these for signature parity but doesn't use the
        # subgoal_pose for anything when we already supply subgoal_tokens.
        next_pose_t = pose.clone()
        # last_action: the dataset stores this as a LOGIT INDEX in [0..7] plus
        # the START sentinel (NUM_OPENFLY_ACTIONS = 8). Passing the raw OpenFly
        # action id here (which can be 9 — left strafe — via
        # TRAINABLE_ACTION_IDS) overruns ``last_action_emb = nn.Embedding(9)``
        # in the DiT and triggers a device-side assert that surfaces async
        # several layers later (typically at GeLU). last_action_id == -1
        # means "no previous action" → START sentinel.
        from openfly.actions import (
            NUM_TRAINABLE_ACTIONS, action_id_to_logit_index, TRAINABLE_ACTION_IDS,
        )
        if last_action_id < 0 or last_action_id not in TRAINABLE_ACTION_IDS:
            la_idx = NUM_TRAINABLE_ACTIONS  # 8 = START sentinel (matches dataset)
        else:
            la_idx = action_id_to_logit_index(int(last_action_id))
        la = torch.tensor([la_idx], dtype=torch.long, device=device)
        progress = None  # let policy default to zeros — at inference we don't have a clean progress signal

        # Encode the prompt — uses the same template as the BC trainer.
        input_ids, attention_mask = _tokenise_batch(
            self.processor, [instruction], rgb_dummy=rgb[0], device=device,
        )

        # Encode current frame to get curr_siglip + Gemma text summary,
        # needed for DiT sampling. ``preprocess_images`` expects float in
        # [0, 1] (it normalises to [-1, 1] for SigLIP); the env emits
        # uint8, so cast first. ``predict_action`` below keeps the
        # uint8 tensor — the policy converts internally.
        rgb_float = rgb.to(torch.float32) / 255.0
        pv = self.model.paligemma.preprocess_images(rgb_float)
        gemma_feats, curr_siglip = self.model.paligemma.forward_tokens(
            pv, input_ids, attention_mask,
        )
        self.model.paligemma.clear_cache()
        token_per_frame = 256
        text_feats = (
            gemma_feats[:, token_per_frame:]
            if gemma_feats.shape[1] > token_per_frame else gemma_feats
        )
        text_mask_part = (
            attention_mask[:, token_per_frame:]
            if attention_mask.shape[1] > token_per_frame else attention_mask
        )
        seq_len = text_mask_part.sum(dim=1).clamp(min=1) - 1
        text_embed = text_feats[0, seq_len].float()

        # Subgoal: sample from the frozen DiT. We don't have a real
        # subgoal_pose at inference (it's deterministic given the
        # current action sequence, but we haven't picked an action yet);
        # use zeros for pose_delta and a default horizon of 4.
        pose_delta = torch.zeros(1, 4, device=device)
        horizon = torch.full((1,), 4, dtype=torch.long, device=device)
        subgoal_tokens = _dit_subgoal_tokens(
            self.dit,
            curr_tokens=curr_siglip.float(),
            text_embed=text_embed,
            pose_delta=pose_delta,
            last_action=la,
            horizon=horizon,
            num_steps=self.ddim_steps,
        )

        # Predict action with the subgoal pathway active.
        action_logit_id = self.model.predict_action(
            instruction_input_ids=input_ids,
            instruction_attention_mask=attention_mask,
            rgb_current=rgb,
            rgb_history=history,
            pose=pose,
            last_action=la,
            next_pose=next_pose_t,
            progress=progress,
            subgoal_tokens=subgoal_tokens,
        )

        # The model's head emits a LOGIT index in [0, 8) over
        # TRAINABLE_ACTION_IDS. Map back to the env's raw OpenFly action id.
        from openfly.actions import logit_index_to_action_id
        action_id = logit_index_to_action_id(int(action_logit_id))
        return int(action_id), ACTION_NAMES.get(int(action_id), str(action_id))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p3_ckpt", type=str, required=True,
                   help="P3 (BC + subgoals) checkpoint, e.g. best.pt.")
    p.add_argument("--dit_path", type=str, required=True,
                   help="P2 SubgoalDiT checkpoint. Same one used during P3 training.")
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="PixArt-Σ HF snapshot dir if the DiT is a PixArtSubgoalDiT.")
    p.add_argument("--ddim_steps", type=int, default=4)

    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--split", default="unseen",
                   help="Defaults to unseen since env_ue_smallcity only has "
                        "episodes there.")
    p.add_argument("--env_filter", default="env_ue_smallcity",
                   help="Auto-overridden to --sim_env so we never try to "
                        "pull episodes from an env we can't render.")
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--image_size", type=int, default=256,
                   help="Policy-input image size. The FPV MP4 writer uses "
                        "the bridge's native resolution; image_size only "
                        "controls what PaliGemma sees and the top-down map.")
    p.add_argument("--sim_env", default="env_ue_smallcity",
                   help="Scene to launch. env_ue_smallcity is the only one "
                        "with a working launcher unpacked on this machine.")
    p.add_argument("--drift_steps", type=int, default=12,
                   help="Number of small camera hops used to drift from the "
                        "previous pose to the next episode's start pose. "
                        "Direct teleports >100m stall UnrealCV; drifting "
                        "keeps each streaming delta bounded.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history_frames", type=int, default=2)
    p.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument(
        "--out",
        default=str(_DRONE_ROOT / "videos" / "openfly_p3_demo.mp4"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _maybe_h264_reencode(raw_path: Path, final_path: Path) -> None:
    """If ffmpeg is on PATH, re-encode raw mp4v to H.264 for cleaner playback.
    Best-effort: if anything fails, keep the raw output and warn.
    """
    if not raw_path.is_file():
        return
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
                str(final_path),
            ],
            check=True,
        )
        raw_path.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        # ffmpeg missing or failed — keep the raw .mp4v output, just rename
        print(f"[record_p3_demo] ffmpeg re-encode skipped ({exc}); keeping raw mp4v")
        raw_path.rename(final_path)


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topdown_path = out_path.with_name(out_path.stem + "_topdown.mp4")

    device = torch.device(args.device)

    # The only scene with a working launcher on this machine is env_ue_smallcity;
    # rebind episode filter so we never load episodes we can't render.
    if args.env_filter != args.sim_env:
        print(
            f"[record_p3_demo] forcing env_filter={args.sim_env!r} "
            f"(was {args.env_filter!r})"
        )
        args.env_filter = args.sim_env
    if args.split == "seen" and args.sim_env in {
        "env_ue_smallcity", "env_game_gtav", "env_gs_sjtu02",
    }:
        print(
            f"[record_p3_demo] switching split=seen->unseen "
            f"({args.sim_env} only has episodes in unseen.json)"
        )
        args.split = "unseen"

    episodes = load_episodes(
        args.split, max_episodes=args.episodes, env_filter=args.env_filter,
    )
    if not episodes:
        raise RuntimeError(
            f"No episodes for split={args.split!r} env_filter={args.env_filter!r}"
        )
    print(f"[record_p3_demo] using {len(episodes)} episodes from split={args.split}")

    driver = _SubgoalPolicyDriver(
        p3_ckpt=args.p3_ckpt,
        dit_path=args.dit_path,
        pretrained_path=args.pretrained_path,
        paligemma_model=args.paligemma_model,
        history_frames=args.history_frames,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
        ddim_steps=args.ddim_steps,
    )

    # Launch the sim through the upstream UEBridge — its __init__ includes a
    # 15s settle inside the thread that launches CitySample, then
    # ``_camera_init`` (cameras/spawn + camera/1 size 1920x1080 + initial
    # pose). That settle is critical: without it the very first heavy vset
    # call hangs forever (the UE world is still streaming chunks while
    # UnrealCV holds the TCP request open).
    from openfly.platform import load_eval_module, make_bridge, openfly_root
    eval_mod = load_eval_module()
    os.chdir(openfly_root() / "train")
    print(f"[record_p3_demo] launching real sim for {args.sim_env} "
          f"(this takes ~30-90s) …")
    t_sim = time.time()
    raw_bridge, _ = make_bridge(args.sim_env, eval_mod)
    print(f"[record_p3_demo] real sim up in {time.time()-t_sim:.1f}s")
    real_bridge = _LitCameraAdapter(raw_bridge)

    # UEBridge._camera_init leaves the camera at (150, 400, 15) (smallcity's
    # default init). The first episode's start is typically ~500-1000m away;
    # teleporting directly there hangs UnrealCV. Drift from the init pose to
    # ep[0]'s start so UE has time to stream the path incrementally.
    init_pose = [150.0, 400.0, 15.0]
    ep0_start = [float(c) for c in episodes[0]["pos"][0]]
    yaw0 = math.degrees(float(episodes[0]["yaw"][0]))
    _drift_camera(real_bridge, init_pose, ep0_start, yaw0,
                  args.drift_steps, label="record_p3_demo.preroll")
    probe = real_bridge.last_native
    assert probe is not None, "drift should have set last_native"
    fpv_h, fpv_w = int(probe.shape[0]), int(probe.shape[1])
    print(f"[record_p3_demo] native FPV={fpv_w}x{fpv_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fpv_raw = out_path.with_suffix(".raw.mp4")
    td_raw = topdown_path.with_suffix(".raw.mp4")
    fpv_writer = cv2.VideoWriter(
        str(fpv_raw), fourcc, args.fps, (fpv_w, fpv_h)
    )
    td_writer = cv2.VideoWriter(
        str(td_raw), fourcc, args.fps, (args.image_size, args.image_size)
    )
    if not fpv_writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open — check ffmpeg backend")

    n_total = 0
    n_success = 0
    aggregate: list[dict[str, Any]] = []

    for ep_idx, ep in enumerate(episodes):
        scene = _SceneCtx(
            image_size=args.image_size,
            bounds=(0, 0, 0, 0),
            goal=tuple(ep["pos"][-1]),
            start=tuple(ep["pos"][0]),
            instruction=ep.get("gpt_instruction", ""),
            waypoints=[tuple(p) for p in ep["pos"]],
        )
        # Drift between episodes too: each episode's start can be far from the
        # previous episode's end. The env.reset(options={"episode": ep}) below
        # will teleport to scene.start; do a gradual drift first so UE has
        # time to stream the path. ep_idx==0 was already handled by the
        # preroll drift before the loop.
        if ep_idx > 0:
            cur_native_pose = list(prev_end_pose)  # set at end of prior iteration
            yaw_new_deg = math.degrees(float(ep["yaw"][0]))
            _drift_camera(real_bridge, cur_native_pose,
                          [float(c) for c in ep["pos"][0]],
                          yaw_new_deg, args.drift_steps,
                          label=f"record_p3_demo.ep{ep_idx}.preroll")

        # The env's voxel-PCD collision check doesn't know about UE-side
        # collisions (the sim itself ignores them as we teleport via
        # set_camera_pose), so leave it off.
        cfg = AirSimVLNEnvConfig(
            split=args.split,
            env_filter=args.env_filter,
            max_steps=args.max_steps,
            image_size=args.image_size,
            history_frames=args.history_frames,
            reset_sleep_s=0.0,
            bridge_init_sleep_s=0.0,
            seed=args.seed + ep_idx,
            collision_check=False,
        )
        env = AirSimVLNEnv(cfg, episodes=[ep], bridge=real_bridge)
        obs, info = env.reset(options={"episode": ep})

        reward_so_far = 0.0
        trail: list[list[float]] = [obs["pose"].tolist()]
        last_action_id = -1  # -1 → START sentinel in act() (matches dataset)
        success: bool | None = None
        t0 = time.time()

        for step in range(args.max_steps):
            action_id, action_name = driver.act(
                obs, info.get("instruction", scene.instruction), last_action_id,
            )

            assert real_bridge.last_native is not None, \
                "bridge.last_native should be set by env.reset/step's get_camera_data call"
            fpv = _real_sim_hud(
                real_bridge.last_native,
                instruction=info.get("instruction", scene.instruction),
                pose=obs["pose"].tolist(),
                goal=obs["goal"].tolist(),
                action_id=action_id,
                reward_so_far=reward_so_far,
                step=step,
                episode_idx=ep_idx,
                sr_running=(n_success / max(n_total, 1)),
                sim_env=args.sim_env,
            )
            fpv_writer.write(fpv)

            td = _topdown_frame(
                scene=scene,
                trail=trail,
                pose=obs["pose"].tolist(),
                image_size=args.image_size,
                success=None,
                reward=reward_so_far,
            )
            td_writer.write(td)

            obs, reward, terminated, truncated, info = env.step(action_id)
            reward_so_far += float(reward)
            trail.append(obs["pose"].tolist())
            last_action_id = action_id

            if terminated or truncated:
                success = bool(info.get("success", False))
                # Draw one more frame so the final pose is visible in the video.
                final_td = _topdown_frame(
                    scene=scene,
                    trail=trail,
                    pose=obs["pose"].tolist(),
                    image_size=args.image_size,
                    success=success,
                    reward=reward_so_far,
                )
                td_writer.write(final_td)
                break

        d_final = distance3d(obs["pose"].tolist(), obs["goal"].tolist())
        n_total += 1
        if success:
            n_success += 1
        aggregate.append({
            "episode": ep_idx,
            "instruction": scene.instruction[:80],
            "steps": step + 1,
            "final_distance_m": round(d_final, 2),
            "success": bool(success) if success is not None else False,
            "reward": round(reward_so_far, 3),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(
            f"[record_p3_demo] ep {ep_idx}: steps={step + 1} "
            f"final_d={d_final:.1f}m success={success} reward={reward_so_far:.2f} "
            f"({time.time() - t0:.1f}s)"
        )
        # Stash where the camera ended this episode so the next ep can drift
        # from it instead of teleporting.
        prev_end_pose = [float(c) for c in obs["pose"].tolist()[:3]]

    fpv_writer.release()
    td_writer.release()

    print("[record_p3_demo] tearing down real sim …")
    real_bridge.disconnect()
    for kw in ("CitySample", "CrashReport"):
        subprocess.run(["pkill", "-9", "-f", kw], check=False)
    subprocess.run(["pkill", "-9", "Xvfb"], check=False)

    _maybe_h264_reencode(fpv_raw, out_path)
    _maybe_h264_reencode(td_raw, topdown_path)

    sr = n_success / max(n_total, 1)
    print()
    print(f"[record_p3_demo] DONE. SR={sr:.2%} ({n_success}/{n_total})")
    print(f"[record_p3_demo] FPV video       : {out_path}")
    print(f"[record_p3_demo] top-down video  : {topdown_path}")
    print()
    print("[record_p3_demo] per-episode summary:")
    for row in aggregate:
        print(f"  ep{row['episode']:>2d}  steps={row['steps']:>3d}  "
              f"d={row['final_distance_m']:>6.1f}m  success={str(row['success']):>5s}  "
              f"r={row['reward']:>+6.2f}  ({row['elapsed_s']:>4.1f}s)  "
              f"{row['instruction']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
